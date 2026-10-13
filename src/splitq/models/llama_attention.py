# Adapted from the local BalanceQ / FlatQuant research code. See LICENSE.
import torch
from torch import nn
from .attention_transforms import AttentionTransformsMixin
from ..quantization.transforms import (
    SVDSingleTransMatrix,
    SVDDecomposeTransMatrix,
    make_isolation_transform,
)
from ..quantization.utils import get_main_decompose_dim
from ..quantization.linear import SplitQLinear
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SplitQLlamaAttention(AttentionTransformsMixin, nn.Module):
    def __init__(self, args, module, *, initialize_svd=True):
        nn.Module.__init__(self)
        self.config = module.config
        self.layer_idx = module.layer_idx
        self.hidden_size = module.config.hidden_size
        self.num_heads = module.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = module.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = module.config.attention_dropout
        if hasattr(module, "rotary_emb"):
            self.rotary_emb = module.rotary_emb
        self.args = args
        self.q_proj = SplitQLinear(args, module.q_proj, initialize_svd=initialize_svd)
        self.k_proj = SplitQLinear(args, module.k_proj, initialize_svd=initialize_svd)
        self.v_proj = SplitQLinear(args, module.v_proj, initialize_svd=initialize_svd)
        self.o_proj = SplitQLinear(args, module.o_proj, initialize_svd=initialize_svd)
        self.register_buffer("vision_mask", torch.tensor([]))
        self.register_buffer(
            "input_isolated_mask", torch.zeros(self.hidden_size, dtype=torch.bool)
        )
        self.is_split = True
        self.v_saliency_in_list = []
        self.t_saliency_in_list = []
        self._ori_mode = False
        self._eval_mode = False
        self.diag_init = "sq_style"
        if self.diag_init == "sq_style":
            self.ln_smax = (
                torch.ones_like(self.q_proj.linear.weight.abs().max(dim=0)[0]) * 1e-05
            )

    def add_fq_trans(self, in_mask=None, down_mask=None):
        in_count = 0
        self.input_isolated_mask = in_mask
        if self.is_split:
            in_count = in_mask.sum().item()
            self.q_proj.isolated_mask = in_mask
            self.k_proj.isolated_mask = in_mask
            self.v_proj.isolated_mask = in_mask
        (SingleTransMatrix, DecomposeTransMatrix) = (
            SVDSingleTransMatrix,
            SVDDecomposeTransMatrix,
        )
        (ln_dim_left, ln_dim_right) = get_main_decompose_dim(
            self.q_proj.linear.weight.shape[1] - in_count,
        )
        self.ln_trans = DecomposeTransMatrix(ln_dim_left, ln_dim_right, add_diag=True)
        self.o_trans = SingleTransMatrix(self.config.num_attention_heads)
        self.iso_ln_trans = None
        if self.is_split:
            self.iso_ln_trans = make_isolation_transform(
                in_count,
            )
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.value_trans = SingleTransMatrix(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        (bsz, q_len, _) = hidden_states.size()
        if self._ori_mode:
            (query_states, key_states, value_states) = self._ori_forward_after_ln(
                hidden_states
            )
        else:
            (query_states, key_states, value_states) = self._trans_forward_after_ln(
                hidden_states
            )
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed `position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be removed and `position_embeddings` will be mandatory."
            )
            (cos, sin) = self.rotary_emb(value_states, position_ids)
        else:
            (cos, sin) = position_embeddings
        (query_states, key_states) = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            (key_states, value_states) = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        if self._ori_mode:
            attn_output = self.o_proj._ori_forward(attn_output)
        elif self.o_trans is None and self.value_trans is not None:
            init_shape = attn_output.shape
            attn_output = attn_output.reshape(
                -1,
                self.config.num_attention_heads,
                self.config.hidden_size // self.config.num_attention_heads,
            )
            attn_output = torch.matmul(
                attn_output, self.value_trans.get_matrix(inv_t=True).T.to(attn_output)
            ).reshape(init_shape)
            attn_output = self.o_proj(attn_output)
        else:
            init_shape = attn_output.shape
            attn_output = attn_output.reshape(
                -1,
                self.config.num_attention_heads,
                self.config.hidden_size // self.config.num_attention_heads,
            )
            attn_output = torch.matmul(
                self.o_trans.get_matrix().T.to(attn_output), attn_output
            ).reshape(init_shape)
            if not self._eval_mode:
                attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
                attn_v_og_it = self.value_trans.get_matrix(inv_t=True)
                attn_output = self.o_proj(
                    attn_output, qa_trans=[attn_o_og_it, attn_v_og_it]
                )
            else:
                attn_output = self.o_proj(attn_output)
        attn_weights = None
        return (attn_output, attn_weights, past_key_value)
