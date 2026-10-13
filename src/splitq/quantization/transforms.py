# Adapted from the local BalanceQ / FlatQuant research code. See LICENSE.
import torch
from torch import nn
from .utils import kronecker_matmul, get_init_weight, get_decompose_dim


class SVDSingleTransMatrix(nn.Module):
    def __init__(self, size):
        super(SVDSingleTransMatrix, self).__init__()
        orthogonal_map = "cayley"
        self.linear_u = nn.Linear(size, size, bias=False, dtype=torch.float32)
        self.linear_u.weight.data = get_init_weight(size).to(self.linear_u.weight)
        self.linear_u = nn.utils.parametrizations.orthogonal(
            self.linear_u, orthogonal_map=orthogonal_map, use_trivialization=False
        )
        self.linear_v = nn.Linear(size, size, bias=False, dtype=torch.float32)
        self.linear_v.weight.data = get_init_weight(size).to(self.linear_v.weight)
        self.linear_v = nn.utils.parametrizations.orthogonal(
            self.linear_v, orthogonal_map=orthogonal_map, use_trivialization=False
        )
        self.linear_diag = torch.nn.Parameter(
            torch.ones(size, dtype=torch.float32), requires_grad=True
        )
        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        init_shape = inp.shape
        matirx = self.get_matrix(inv_t=inv_t).to(inp)
        inp = inp.reshape(-1, matirx.shape[0])
        return inp.matmul(matirx).reshape(init_shape)

    def get_matrix(self, inv_t=False):
        if not self._eval_mode:
            (orthog_u, orthog_v) = (self.linear_u.weight, self.linear_v.weight)
            linear_diag = self.linear_diag
            if inv_t:
                linear_diag = 1 / linear_diag
            return orthog_u @ torch.diag(linear_diag) @ orthog_v.t()
        else:
            if inv_t:
                return self.matrix_inv_t
            return self.matrix

    def to_eval_mode(self):
        if not self._eval_mode:
            matrix = (
                self.linear_u.weight
                @ torch.diag(self.linear_diag)
                @ self.linear_v.weight.t()
            )
            matrix_inv_t = (
                self.linear_u.weight
                @ torch.diag(1 / self.linear_diag)
                @ self.linear_v.weight.t()
            )
            self.matrix = nn.Parameter(matrix, requires_grad=False)
            self.matrix_inv_t = nn.Parameter(matrix_inv_t, requires_grad=False)
            self._eval_mode = True
            del self.linear_u, self.linear_diag, self.linear_v

    def __repr__(self):
        res = f"SVDSingleTransMatrix(eval_mode={self._eval_mode}"
        if hasattr(self, "matrix"):
            res += f", matrix.shape={self.matrix.shape})"
        else:
            res += f", matrix.shape={self.linear_u.weight.shape})"
        return res


class SVDDecomposeTransMatrix(nn.Module):
    def __init__(self, left_size, right_size, add_diag=False, diag_init_para=None):
        super(SVDDecomposeTransMatrix, self).__init__()
        orthogonal_map = "cayley"
        self.linear_u_left = nn.Linear(
            left_size, left_size, bias=False, dtype=torch.float32
        )
        self.linear_u_left.weight.data = get_init_weight(left_size).to(
            self.linear_u_left.weight
        )
        self.linear_u_left = nn.utils.parametrizations.orthogonal(
            self.linear_u_left, orthogonal_map=orthogonal_map, use_trivialization=False
        )
        self.linear_v_left = nn.Linear(
            left_size, left_size, bias=False, dtype=torch.float32
        )
        self.linear_v_left.weight.data = get_init_weight(left_size).to(
            self.linear_v_left.weight
        )
        self.linear_v_left = nn.utils.parametrizations.orthogonal(
            self.linear_v_left, orthogonal_map=orthogonal_map, use_trivialization=False
        )
        self.linear_diag_left = torch.nn.Parameter(
            torch.ones(left_size, dtype=torch.float32), requires_grad=True
        )
        self.linear_u_right = nn.Linear(
            right_size, right_size, bias=False, dtype=torch.float32
        )
        self.linear_u_right.weight.data = get_init_weight(right_size).to(
            self.linear_u_right.weight
        )
        self.linear_u_right = nn.utils.parametrizations.orthogonal(
            self.linear_u_right, orthogonal_map=orthogonal_map, use_trivialization=False
        )
        self.linear_v_right = nn.Linear(
            right_size, right_size, bias=False, dtype=torch.float32
        )
        self.linear_v_right.weight.data = get_init_weight(right_size).to(
            self.linear_v_right.weight
        )
        self.linear_v_right = nn.utils.parametrizations.orthogonal(
            self.linear_v_right, orthogonal_map=orthogonal_map, use_trivialization=False
        )
        self.linear_diag_right = torch.nn.Parameter(
            torch.ones(right_size, dtype=torch.float32), requires_grad=True
        )
        self.add_diag = add_diag
        self.use_diag = True
        if self.add_diag:
            if diag_init_para is None:
                self.diag_scale = torch.nn.Parameter(
                    torch.ones(left_size * right_size, dtype=torch.float32),
                    requires_grad=True,
                )
            else:
                self.diag_scale = torch.nn.Parameter(diag_init_para, requires_grad=True)
        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        if self.add_diag and self.use_diag:
            if inv_t:
                inp = inp / self.diag_scale.to(inp)
            else:
                inp = inp * self.diag_scale.to(inp)
        if not self._eval_mode:
            (matrix_u_left, matrix_u_right) = (
                self.linear_u_left.weight,
                self.linear_u_right.weight,
            )
            (matrix_v_left, matrix_v_right) = (
                self.linear_v_left.weight,
                self.linear_v_right.weight,
            )
            (linear_diag_left, linear_diag_right) = (
                self.linear_diag_left,
                self.linear_diag_right,
            )
            if inv_t:
                (linear_diag_left, linear_diag_right) = (
                    1 / linear_diag_left,
                    1 / linear_diag_right,
                )
        else:
            (matrix_left, matrix_right) = (self.matrix_left, self.matrix_right)
            if inv_t:
                (matrix_left, matrix_right) = (
                    self.matrix_left_inv,
                    self.matrix_right_inv,
                )
            return kronecker_matmul(inp, matrix_left.to(inp), matrix_right.to(inp))
        (matrix_left, matrix_right) = (
            matrix_u_left @ torch.diag(linear_diag_left) @ matrix_v_left.t(),
            matrix_u_right @ torch.diag(linear_diag_right) @ matrix_v_right.t(),
        )
        return kronecker_matmul(inp, matrix_left.to(inp), matrix_right.to(inp))

    def to_eval_mode(self):
        if not self._eval_mode:
            matrix_left = (
                self.linear_u_left.weight
                @ torch.diag(self.linear_diag_left)
                @ self.linear_v_left.weight.t()
            )
            matrix_right = (
                self.linear_u_right.weight
                @ torch.diag(self.linear_diag_right)
                @ self.linear_v_right.weight.t()
            )
            matrix_left_inv = (
                self.linear_u_left.weight
                @ torch.diag(1 / self.linear_diag_left)
                @ self.linear_v_left.weight.t()
            )
            matrix_right_inv = (
                self.linear_u_right.weight
                @ torch.diag(1 / self.linear_diag_right)
                @ self.linear_v_right.weight.t()
            )
            self.matrix_left = nn.Parameter(matrix_left, requires_grad=False)
            self.matrix_right = nn.Parameter(matrix_right, requires_grad=False)
            self.matrix_left_inv = nn.Parameter(matrix_left_inv, requires_grad=False)
            self.matrix_right_inv = nn.Parameter(matrix_right_inv, requires_grad=False)
            del (
                self.linear_u_left,
                self.linear_diag_left,
                self.linear_v_left,
                self.linear_u_right,
                self.linear_diag_right,
                self.linear_v_right,
            )
            self._eval_mode = True

    def __repr__(self):
        res = f"SVDDecomposeTransMatrix(_eval_mode={self._eval_mode}"
        if hasattr(self, "matrix_left"):
            res += f", matrix.shape={self.matrix_left.shape}, matrix_right.shape={self.matrix_right.shape}, )"
        else:
            res += f", matrix.shape={self.linear_u_left.weight.shape}, linear_right.shape={self.linear_u_right.weight.shape}, )"
        return res


def make_isolation_transform(size):
    left, right = get_decompose_dim(size)
    return SVDDecomposeTransMatrix(left, right, add_diag=False)
