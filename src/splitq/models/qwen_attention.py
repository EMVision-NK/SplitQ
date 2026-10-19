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
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from collections.abc import Callable


class SplitQQwenAttention(AttentionTransformsMixin, nn.Module):
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
        config = self.config
        self.is_causal = True
        self.rope_parameters = config.rope_parameters
        self.scaling = self.head_dim ** (-0.5)
        self.q_proj = SplitQLinear(args, module.q_proj, initialize_svd=initialize_svd)
        self.k_proj = SplitQLinear(args, module.k_proj, initialize_svd=initialize_svd)
        self.v_proj = SplitQLinear(args, module.v_proj, initialize_svd=initialize_svd)
        self.o_proj = SplitQLinear(args, module.o_proj, initialize_svd=initialize_svd)
        self.layer_type = (
            config.layer_types[module.layer_idx]
            if hasattr(config, "layer_types")
            else None
        )
        self.sliding_window = (
            config.sliding_window if self.layer_type == "sliding_attention" else None
        )
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
        self.input_isolated_mask = None
        if self.is_split:
            in_count = in_mask.sum().item()
            self.input_isolated_mask = in_mask
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
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        position_embeddings=None,
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
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        (cos, sin) = position_embeddings
        (query_states, key_states) = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            self.config.rope_parameters["mrope_section"],
        )
        if past_key_values is not None:
            (key_states, value_states) = past_key_values.update(
                key_states, value_states, self.layer_idx
            )
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        (attn_output, attn_weights) = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
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
        if not output_attentions:
            attn_weights = None
        return (attn_output, attn_weights)
