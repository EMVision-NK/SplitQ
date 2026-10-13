# Adapted from the local BalanceQ / FlatQuant research code. See LICENSE.
import torch
from torch import nn
from torch.nn import functional as F
from .quantizers import ActivationQuantizer, WeightQuantizer
from .utils import kronecker_matmul


class SplitQLinear(nn.Module):
    """MOCD linear paths with CWS and MAC compensation in ACC."""

    def __init__(self, args, linear: nn.Linear, layer_name="", *, initialize_svd=True):
        super(SplitQLinear, self).__init__()
        self.args = args
        self.linear = linear
        (self.out_features, self.in_features) = linear.weight.shape
        self.layer_name = layer_name
        self.weight_quantizer = WeightQuantizer()
        self.weight_quantizer.configure(
            args.w_bits, perchannel=True, sym=True, mse=False
        )
        self.iso_weight_quantizer = WeightQuantizer()
        self.iso_weight_quantizer.configure(8, perchannel=True, sym=True, mse=False)
        self.cws_l1_weight_quantizer = WeightQuantizer()
        self.cws_l1_weight_quantizer.configure(
            args.cws_w_bits, perchannel=True, sym=True, mse=False
        )
        self.cws_l2_weight_quantizer = WeightQuantizer()
        self.cws_l2_weight_quantizer.configure(
            args.cws_w_bits, perchannel=True, sym=True, mse=False
        )
        self.mac_l1_weight_quantizer = WeightQuantizer()
        self.mac_l1_weight_quantizer.configure(
            args.mac_w_bits, perchannel=True, sym=True, mse=False
        )
        self.mac_l2_weight_quantizer = WeightQuantizer()
        self.mac_l2_weight_quantizer.configure(
            args.mac_w_bits, perchannel=True, sym=True, mse=False
        )
        self.act_quantizer = ActivationQuantizer(
            bits=args.a_bits,
            sym=True,
            lac=True,
            groupsize=-1,
        )
        self.act_mac_quantizer = ActivationQuantizer(
            bits=args.mac_a_bits,
            sym=True,
            lac=True,
            groupsize=-1,
        )
        # CWS/MAC support optional W4A4 or higher quantization with largely preserved performance; 16 bypasses fake quantization.
        self.act_cws_hidden_quantizer = (
            ActivationQuantizer(
                args.cws_a_bits,
                sym=True,
                lac=True,
                groupsize=-1,
            )
            if args.cws_a_bits < 16
            else None
        )
        self.act_mac_hidden_quantizer = (
            ActivationQuantizer(
                args.mac_a_bits,
                sym=True,
                lac=True,
                groupsize=-1,
            )
            if args.mac_a_bits < 16
            else None
        )
        self.act_iso_quantizer = ActivationQuantizer(
            bits=8,
            sym=True,
            lac=True,
            groupsize=-1,
        )
        init_value = 4.0
        self.clip_factor_w_max = nn.Parameter(
            torch.ones((self.out_features, 1)) * init_value
        )
        self.clip_factor_w_min = nn.Parameter(
            torch.ones((self.out_features, 1)) * init_value
        )
        self.sigmoid = nn.Sigmoid()
        with torch.no_grad():
            self.cws_rank = min(
                self.out_features,
                self.in_features,
                max(
                    16,
                    round(
                        args.cws_rank_ratio * min(self.out_features, self.in_features)
                    ),
                ),
            )
            self.mac_rank = min(
                self.out_features,
                self.in_features,
                max(
                    64,
                    round(
                        args.mac_rank_ratio * min(self.out_features, self.in_features)
                    ),
                ),
            )
            self.shared_rank = max(self.cws_rank, self.mac_rank)
            W = self.linear.weight.data.detach().float()
            if initialize_svd:
                U_svd, sigma_svd, V_svd = torch.svd_lowrank(
                    W, q=self.shared_rank, niter=15
                )
            else:
                U_svd = W.new_zeros(self.out_features, self.shared_rank)
                sigma_svd = W.new_zeros(self.shared_rank)
                V_svd = W.new_zeros(self.in_features, self.shared_rank)
            Vt_svd = V_svd.t()
            self.register_buffer("U_svd_shared", U_svd.contiguous())
            self.register_buffer("sigma_svd_shared", sigma_svd.contiguous())
            self.register_buffer("Vt_svd_shared", Vt_svd.contiguous())
        self.gate_logits_cws = nn.Parameter(torch.ones(self.cws_rank) * -1.0)
        self.is_mac = args.mac
        self.is_cws = args.cws
        self.gate_logits_mac = nn.Parameter(torch.ones(self.mac_rank) * -1.0)
        self.register_buffer("isolated_mask", None)
        self._eval_mode = False

    def apply_wclip(self, weight):
        wmin = weight.min(1, keepdim=True)[0]
        wmax = weight.max(1, keepdim=True)[0]
        wmax = wmax * self.sigmoid(self.clip_factor_w_max)
        wmin = wmin * self.sigmoid(self.clip_factor_w_min)
        return torch.clamp(weight, min=wmin, max=wmax)

    def apply_trans(self, weight, qa_trans):
        if isinstance(qa_trans, list):
            weight = kronecker_matmul(
                weight, qa_trans[0].to(weight), qa_trans[1].to(weight)
            )
        else:
            weight = qa_trans(weight, inv_t=True)
        return weight

    def _get_shared_svd(self, rank):
        U = self.U_svd_shared[:, :rank]
        sigma = self.sigma_svd_shared[:rank]
        Vt = self.Vt_svd_shared[:rank, :]
        return (U, sigma, Vt)

    def _build_cws_weight(self, isolated_mask):
        (U, sigma, Vt) = self._get_shared_svd(self.cws_rank)
        gate_t = 0.4 * torch.sigmoid(self.gate_logits_cws).to(
            dtype=sigma.dtype, device=sigma.device
        )
        S = sigma * gate_t
        L1 = U * S
        L2 = Vt.clone()
        if isolated_mask is not None:
            L2 = L2[:, ~isolated_mask]
        return (L1, L2)

    def _build_mac_weight(self, isolated_mask, gate_logits):
        (U, sigma, Vt) = self._get_shared_svd(self.mac_rank)
        gate_t = 0.9 + 0.1 * torch.sigmoid(gate_logits).to(
            dtype=sigma.dtype, device=sigma.device
        )
        S = sigma * gate_t
        L1 = U * S
        L2 = Vt.clone()
        if isolated_mask is not None:
            L2 = L2[:, ~isolated_mask]
        return (L1, L2)

    @staticmethod
    def _low_rank_forward(x, l2, l1, hidden_quantizer):
        hidden = F.linear(x, l2, None)
        if hidden_quantizer is not None:
            hidden = hidden_quantizer(hidden)
        return F.linear(hidden, l1, None)

    def _train_forward(
        self, x_main, x_isolated=None, qa_trans=None, iso_qa_trans=None, out_trans=None
    ):
        work_weight = self.linear.weight.detach()
        main_weight = work_weight.clone()
        iso_weight = work_weight.clone()
        if self.isolated_mask is not None and x_isolated is not None:
            main_weight = main_weight[:, ~self.isolated_mask]
            iso_weight = iso_weight[:, self.isolated_mask]
        (L1_cws, L2_cws) = self._build_cws_weight(self.isolated_mask)
        (L1_mac, L2_mac) = self._build_mac_weight(
            self.isolated_mask, self.gate_logits_mac
        )
        if qa_trans is not None:
            main_weight = self.apply_trans(main_weight, qa_trans)
            L2_cws = self.apply_trans(L2_cws, qa_trans)
            L2_mac = self.apply_trans(L2_mac, qa_trans)
        if iso_qa_trans is not None and x_isolated is not None:
            iso_weight = self.apply_trans(iso_weight, iso_qa_trans)
        main_weight = self.apply_wclip(main_weight)
        if iso_qa_trans is not None and x_isolated is not None:
            main_weight = torch.cat((main_weight, iso_weight), dim=-1)
        if out_trans is not None:
            main_weight = out_trans(main_weight.T).T
            L1_cws = out_trans(L1_cws.T).T
            L1_mac = out_trans(L1_mac.T).T
        if iso_qa_trans is not None and x_isolated is not None:
            iso_len = iso_weight.shape[-1]
            main_len = main_weight.shape[-1] - iso_len
            (main_weight, iso_weight) = torch.split(
                main_weight, [main_len, iso_len], dim=-1
            )
        if self.is_cws:
            self.cws_l1_weight_quantizer.find_params(L1_cws)
            L1_cws = self.cws_l1_weight_quantizer(L1_cws)
            self.cws_l2_weight_quantizer.find_params(L2_cws)
            L2_cws = self.cws_l2_weight_quantizer(L2_cws)
        q_hidden_states = self.act_quantizer(x_main)
        if self.is_cws == True:
            main_weight = main_weight - L1_cws @ L2_cws
            self.weight_quantizer.find_params(main_weight)
            q_weight = self.weight_quantizer(main_weight)
        else:
            self.weight_quantizer.find_params(main_weight)
            q_weight = self.weight_quantizer(main_weight)
        main_output = F.linear(q_hidden_states, q_weight, None)
        if self.is_cws == True:
            main_output = main_output + self._low_rank_forward(
                q_hidden_states,
                L2_cws,
                L1_cws,
                self.act_cws_hidden_quantizer,
            )
        if self.is_mac == True:
            self.mac_l1_weight_quantizer.find_params(L1_mac)
            l1_weight = self.mac_l1_weight_quantizer(L1_mac)
            self.mac_l2_weight_quantizer.find_params(L2_mac)
            l2_weight = self.mac_l2_weight_quantizer(L2_mac)
            # MAC compensates activation residuals for both modalities.
            q_mac = (
                self.act_mac_quantizer(x_main - q_hidden_states)
                if self.args.mac_a_bits < 16
                else x_main - q_hidden_states
            )
            main_output = main_output + self._low_rank_forward(
                q_mac,
                l2_weight,
                l1_weight,
                self.act_mac_hidden_quantizer,
            )
        if x_isolated is not None:
            # Text- and vision-specific channels can share this branch for efficiency.
            q_v = self.act_iso_quantizer(x_isolated)
            self.iso_weight_quantizer.find_params(iso_weight)
            qv_weight = self.iso_weight_quantizer(iso_weight)
            main_output = F.linear(q_v, qv_weight, None) + main_output
        bias = self.linear.bias
        if bias is not None:
            bias = bias.detach()
            if out_trans is not None:
                bias = out_trans(bias)
            main_output = main_output + bias
        return main_output

    def _ori_forward(self, hidden_states):
        return self.linear(hidden_states)

    def forward(
        self, x_main, x_isolated=None, qa_trans=None, iso_qa_trans=None, out_trans=None
    ):
        if not self._eval_mode:
            return self._train_forward(
                x_main,
                x_isolated=x_isolated,
                qa_trans=qa_trans,
                iso_qa_trans=iso_qa_trans,
                out_trans=out_trans,
            )
        else:
            return self._eval_forward(x_main, x_isolated=x_isolated)

    def _eval_forward(self, x_main, x_isolated=None):
        target_dtype = self.linear.weight.dtype
        if x_main.is_cuda and torch.is_autocast_enabled():
            # Cast auxiliary factors directly to the GEMM dtype to avoid double rounding.
            target_dtype = torch.get_autocast_gpu_dtype()
        q_hidden_states = self.act_quantizer(x_main)
        # Add FP32 bias after the autocast GEMM, as in calibration.
        output = F.linear(q_hidden_states, self.linear.weight, None)
        if self.is_cws == True:
            if self.L2_cws.dtype != target_dtype or self.L1_cws.dtype != target_dtype:
                self.L2_cws = self.L2_cws.to(dtype=target_dtype)
                self.L1_cws = self.L1_cws.to(dtype=target_dtype)
            output = output + self._low_rank_forward(
                q_hidden_states,
                self.L2_cws,
                self.L1_cws,
                self.act_cws_hidden_quantizer,
            )
        if self.is_mac == True:
            if self.L2_mac.dtype != target_dtype or self.L1_mac.dtype != target_dtype:
                self.L2_mac = self.L2_mac.to(dtype=target_dtype)
                self.L1_mac = self.L1_mac.to(dtype=target_dtype)
            # Compensate all text and vision tokens.
            x_mac = (
                self.act_mac_quantizer(x_main - q_hidden_states)
                if self.args.mac_a_bits < 16
                else x_main - q_hidden_states
            )
            output = output + self._low_rank_forward(
                x_mac,
                self.L2_mac,
                self.L1_mac,
                self.act_mac_hidden_quantizer,
            )
        if x_isolated is not None:
            if self.isolated_weight.dtype != target_dtype:
                self.isolated_weight = self.isolated_weight.to(dtype=target_dtype)
            q_v = self.act_iso_quantizer(x_isolated)
            low_rank_output = F.linear(q_v, self.isolated_weight, None)
            output = output + low_rank_output
        if self.linear.bias is not None:
            output = output + self.linear.bias
        return output

    @torch.no_grad()
    def reparameterize(self, qa_trans=None, iso_qa_trans=None, out_trans=None):
        weight = self.linear.weight
        ori_dtype = weight.dtype
        ori_device = weight.device
        aux_dtype = ori_dtype if ori_dtype == torch.float32 else torch.float16
        (L1_cws, L2_cws) = self._build_cws_weight(self.isolated_mask)
        (L1_mac, L2_mac) = self._build_mac_weight(
            self.isolated_mask, self.gate_logits_mac
        )
        # Use the same transform precision and quantization order as calibration.
        main_weight = weight.clone()
        iso_weight = None
        if self.isolated_mask is not None:
            main_weight = main_weight[:, ~self.isolated_mask]
            iso_weight = weight[:, self.isolated_mask]
        if qa_trans is not None:
            main_weight = self.apply_trans(main_weight, qa_trans)
            L2_cws = self.apply_trans(L2_cws, qa_trans)
            L2_mac = self.apply_trans(L2_mac, qa_trans)
        main_weight = self.apply_wclip(main_weight)
        if iso_qa_trans is not None:
            iso_weight = self.apply_trans(iso_weight, iso_qa_trans)
            main_weight = torch.cat((main_weight, iso_weight), dim=-1)
        if out_trans is not None:
            main_weight = out_trans(main_weight.T).T
            L1_cws = out_trans(L1_cws.T).T
            L1_mac = out_trans(L1_mac.T).T
        if iso_qa_trans is not None:
            iso_len = iso_weight.shape[-1]
            main_len = main_weight.shape[-1] - iso_len
            (main_weight, iso_weight) = torch.split(
                main_weight, [main_len, iso_len], dim=-1
            )
        if self.is_cws:
            self.cws_l1_weight_quantizer.find_params(L1_cws)
            L1_cws = self.cws_l1_weight_quantizer(L1_cws)
            self.cws_l2_weight_quantizer.find_params(L2_cws)
            L2_cws = self.cws_l2_weight_quantizer(L2_cws)
        if self.is_cws:
            main_weight = main_weight - L1_cws @ L2_cws
        # Quantize before casting to the inference storage dtype.
        self.weight_quantizer.find_params(main_weight)
        main_weight = self.weight_quantizer(main_weight)
        if iso_weight is not None:
            self.iso_weight_quantizer.find_params(iso_weight)
            iso_weight = self.iso_weight_quantizer(iso_weight)
            iso_weight = iso_weight.to(dtype=aux_dtype, device=ori_device)
        self.register_buffer("isolated_weight", iso_weight)
        self.register_buffer("L1_cws", L1_cws.to(dtype=aux_dtype, device=ori_device))
        self.register_buffer("L2_cws", L2_cws.to(dtype=aux_dtype, device=ori_device))
        self.mac_l1_weight_quantizer.find_params(L1_mac)
        L1_mac = self.mac_l1_weight_quantizer(L1_mac)
        self.mac_l2_weight_quantizer.find_params(L2_mac)
        L2_mac = self.mac_l2_weight_quantizer(L2_mac)
        self.register_buffer("L1_mac", L1_mac.to(dtype=aux_dtype, device=ori_device))
        self.register_buffer("L2_mac", L2_mac.to(dtype=aux_dtype, device=ori_device))
        if out_trans is not None and self.linear.bias is not None:
            self.linear.bias.data = out_trans(self.linear.bias.data)
        self.linear.weight.data = main_weight.to(dtype=ori_dtype, device=ori_device)
        self._eval_mode = True
