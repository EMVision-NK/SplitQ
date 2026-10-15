# Adapted from the local BalanceQ / FlatQuant research code. See LICENSE.
"""Shared attention transforms; model-specific RoPE and cache APIs remain in the adapters."""

import torch
from .mocd import MOCDMixin
from ..quantization.utils import get_init_scale


class AttentionTransformsMixin(MOCDMixin):
    def _trans_forward_after_ln(self, hidden_states):
        (x_main_ts, x_isolated) = self._split_logic(
            hidden_states, self.input_isolated_mask
        )
        if self.ln_trans is not None:
            x_main_ts = self.ln_trans(x_main_ts)
        if x_isolated is not None:
            x_isolated = self.iso_ln_trans(x_isolated)
        query_states = self.q_proj(
            x_main_ts,
            x_isolated,
            qa_trans=self.ln_trans,
            iso_qa_trans=self.iso_ln_trans,
        )
        key_states = self.k_proj(
            x_main_ts,
            x_isolated,
            qa_trans=self.ln_trans,
            iso_qa_trans=self.iso_ln_trans,
        )
        value_states = self.v_proj(
            x_main_ts,
            x_isolated,
            qa_trans=self.ln_trans,
            iso_qa_trans=self.iso_ln_trans,
            out_trans=self.value_trans,
        )
        return (query_states, key_states, value_states)

    def _ori_forward_after_ln(self, hidden_states):
        self.in_channel_marker(hidden_states)
        if self.diag_init == "sq_style" and hasattr(self, "ln_smax"):
            self.ln_smax = torch.maximum(
                self.ln_smax,
                hidden_states.reshape(-1, hidden_states.shape[-1])
                .abs()
                .max(0)[0]
                .clone()
                .detach(),
            )
        query_states = self.q_proj._ori_forward(hidden_states)
        key_states = self.k_proj._ori_forward(hidden_states)
        value_states = self.v_proj._ori_forward(hidden_states)
        return (query_states, key_states, value_states)

    def reparameterize(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.value_trans is not None:
            self.value_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()
        if self.iso_ln_trans is not None:
            self.iso_ln_trans.to_eval_mode()
        self.q_proj.reparameterize(
            qa_trans=self.ln_trans, iso_qa_trans=self.iso_ln_trans
        )
        self.k_proj.reparameterize(
            qa_trans=self.ln_trans, iso_qa_trans=self.iso_ln_trans
        )
        self.v_proj.reparameterize(
            qa_trans=self.ln_trans,
            iso_qa_trans=self.iso_ln_trans,
            out_trans=self.value_trans,
        )
        if self.o_trans is not None and self.value_trans is not None:
            attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
            attn_v_og_it = self.value_trans.get_matrix(inv_t=True)
            self.o_proj.reparameterize(qa_trans=[attn_o_og_it, attn_v_og_it])
        self._eval_mode = True

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "ln_smax")
        qkvw_smax = (
            torch.cat(
                [
                    self.q_proj.linear.weight,
                    self.k_proj.linear.weight,
                    self.v_proj.linear.weight,
                ],
                dim=0,
            )
            .abs()
            .max(dim=0)[0]
        )
        if self.ln_trans is not None:
            main_qkvw_smax = qkvw_smax
            main_ln_smax = self.ln_smax
            if self.is_split:
                main_qkvw_smax = qkvw_smax[~self.input_isolated_mask]
                main_ln_smax = self.ln_smax[~self.input_isolated_mask]
            self.ln_trans.diag_scale.data = get_init_scale(
                main_qkvw_smax, main_ln_smax, alpha
            )
        del self.ln_smax
        self.diag_init = None

    def _split_logic(self, x, isolated_mask):
        x_main = x.clone()
        x_isolated = None
        if self.is_split and isolated_mask.any():
            x_main = x[..., ~isolated_mask]
            x_isolated = x[..., isolated_mask]
        return (x_main, x_isolated)
