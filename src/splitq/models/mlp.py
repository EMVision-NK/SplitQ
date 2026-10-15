# Adapted from the local BalanceQ / FlatQuant research code. See LICENSE.
import torch
from torch import nn
from .mocd import MOCDMixin
from ..quantization.transforms import (
    SVDDecomposeTransMatrix,
    make_isolation_transform,
)
from ..quantization.utils import get_init_scale, get_main_decompose_dim
from ..quantization.linear import SplitQLinear


class SplitQMLP(MOCDMixin, nn.Module):
    def __init__(
        self, args, module, layer_idx=None, config=None, *, initialize_svd=True
    ):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        config = config or module.config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = module.act_fn
        self.up_proj = SplitQLinear(args, module.up_proj, initialize_svd=initialize_svd)
        self.gate_proj = SplitQLinear(
            args, module.gate_proj, initialize_svd=initialize_svd
        )
        self.down_proj = SplitQLinear(
            args, module.down_proj, initialize_svd=initialize_svd
        )
        self.register_buffer("vision_mask", torch.tensor([]))
        self.register_buffer(
            "input_isolated_mask", torch.zeros(self.hidden_size, dtype=torch.bool)
        )
        self.register_buffer(
            "down_isolated_mask", torch.zeros(self.intermediate_size, dtype=torch.bool)
        )
        self.v_saliency_in_list = []
        self.t_saliency_in_list = []
        self.v_saliency_down_list = []
        self.t_saliency_down_list = []
        self.is_split = True
        self._ori_mode = False
        self.diag_init = "sq_style"
        if self.diag_init == "sq_style":
            self.up_smax = (
                torch.ones_like(self.up_proj.linear.weight.abs().max(dim=0)[0]) * 1e-05
            )
            self.down_smax = (
                torch.ones_like(self.down_proj.linear.weight.abs().max(dim=0)[0])
                * 1e-05
            )

    def add_fq_trans(self, in_mask=None, down_mask=None):
        in_count = 0
        down_count = 0
        self.input_isolated_mask = None
        self.down_isolated_mask = None
        if self.is_split:
            in_count = in_mask.sum().item()
            down_count = down_mask.sum().item()
            self.input_isolated_mask = in_mask
            self.down_isolated_mask = down_mask
            self.up_proj.isolated_mask = in_mask
            self.gate_proj.isolated_mask = in_mask
            self.down_proj.isolated_mask = down_mask
        (up_dim_left, up_dim_right) = get_main_decompose_dim(
            self.up_proj.linear.weight.shape[1] - in_count,
        )
        self.up_gate_trans = SVDDecomposeTransMatrix(
            up_dim_left, up_dim_right, add_diag=True
        )
        (down_dim_left, down_dim_right) = get_main_decompose_dim(
            self.down_proj.linear.weight.shape[1] - down_count,
        )
        self.down_trans = SVDDecomposeTransMatrix(
            down_dim_left, down_dim_right, add_diag=True
        )
        (self.iso_up_gate_trans, self.iso_down_trans) = (None, None)
        if self.is_split:
            self.iso_up_gate_trans = make_isolation_transform(
                in_count,
            )
            self.iso_down_trans = make_isolation_transform(
                down_count,
            )

    def _trans_forward(self, x):
        (x_main_ts, x_isolated) = self._split_logic(x, self.input_isolated_mask)
        if self.up_gate_trans is not None:
            x_main_ts = self.up_gate_trans(x_main_ts)
        if x_isolated is not None:
            x_isolated = self.iso_up_gate_trans(x_isolated)
        up_states = self.up_proj(
            x_main_ts,
            x_isolated,
            qa_trans=self.up_gate_trans,
            iso_qa_trans=self.iso_up_gate_trans,
        )
        gate_states = self.gate_proj(
            x_main_ts,
            x_isolated,
            qa_trans=self.up_gate_trans,
            iso_qa_trans=self.iso_up_gate_trans,
        )
        gate_states = self.act_fn(gate_states)
        x_act_fn = gate_states * up_states
        (x_ts_2_main, x_ts_2_isolated) = self._split_logic(
            x_act_fn, self.down_isolated_mask
        )
        if self.down_trans is not None:
            x_ts_2_main = self.down_trans(x_ts_2_main)
        if x_ts_2_isolated is not None:
            x_ts_2_isolated = self.iso_down_trans(x_ts_2_isolated)
        down_states = self.down_proj(
            x_ts_2_main,
            x_ts_2_isolated,
            qa_trans=self.down_trans,
            iso_qa_trans=self.iso_down_trans,
        )
        return down_states

    def _ori_forward(self, x):
        """Original MLP: Down(SiLU(Gate(x)) * Up(x))."""
        self.in_channel_marker(x)
        if self.diag_init == "sq_style":
            self.up_smax = torch.maximum(
                self.up_smax,
                x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach(),
            )
        x = self.act_fn(self.gate_proj._ori_forward(x)) * self.up_proj._ori_forward(x)
        if self.diag_init == "sq_style":
            self.down_smax = torch.maximum(
                self.down_smax,
                x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach(),
            )
        self.down_channel_marker(x)
        down_states = self.down_proj._ori_forward(x)
        return down_states

    def forward(self, x):
        if self._ori_mode:
            return self._ori_forward(x)
        return self._trans_forward(x)

    def reparameterize(self):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
        if self.down_trans is not None:
            self.down_trans.to_eval_mode()
        if self.iso_up_gate_trans is not None:
            self.iso_up_gate_trans.to_eval_mode()
        if self.iso_down_trans is not None:
            self.iso_down_trans.to_eval_mode()
        self.gate_proj.reparameterize(
            qa_trans=self.up_gate_trans, iso_qa_trans=self.iso_up_gate_trans
        )
        self.up_proj.reparameterize(
            qa_trans=self.up_gate_trans, iso_qa_trans=self.iso_up_gate_trans
        )
        self.down_proj.reparameterize(
            qa_trans=self.down_trans, iso_qa_trans=self.iso_down_trans
        )
        # Keep activation diagonal scaling at the calibration sites.

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "up_smax") and hasattr(self, "down_smax")
        upw_smax = (
            torch.cat([self.up_proj.linear.weight, self.gate_proj.linear.weight], dim=0)
            .abs()
            .max(dim=0)[0]
        )
        downw_smax = self.down_proj.linear.weight.abs().max(dim=0)[0]
        if self.up_gate_trans is not None:
            main_upw_smax = upw_smax
            main_up_smax = self.up_smax
            if self.is_split:
                main_upw_smax = upw_smax[~self.input_isolated_mask]
                main_up_smax = self.up_smax[~self.input_isolated_mask]
            self.up_gate_trans.diag_scale.data = get_init_scale(
                main_upw_smax, main_up_smax, alpha
            )
        if self.down_trans is not None:
            main_downw_smax = downw_smax
            main_down_smax = self.down_smax
            if self.is_split:
                main_downw_smax = downw_smax[~self.down_isolated_mask]
                main_down_smax = self.down_smax[~self.down_isolated_mask]
            self.down_trans.diag_scale.data = get_init_scale(
                main_downw_smax, main_down_smax, alpha
            )
        del self.up_smax, self.down_smax
        self.diag_init = None

    def _split_logic(self, x, isolated_mask):
        x_main = x
        x_isolated = None
        if self.is_split and isolated_mask is not None and isolated_mask.any():
            x_main = x[..., ~isolated_mask]
            x_isolated = x[..., isolated_mask]
        return (x_main, x_isolated)
