# Adapted from the local BalanceQ / FlatQuant research code. See LICENSE.
import math
from functools import lru_cache
import torch
import numpy as np
from scipy.linalg import qr


def get_init_scale(w_smax, x_smax, alpha=0.5):
    return (w_smax.pow(1 - alpha) / x_smax.pow(alpha)).clamp(min=1e-05)


def get_random_orthg(size):
    H = np.random.randn(size, size)
    (Q, R) = qr(H)
    Q_modified = Q @ np.diag(np.sign(np.diag(R)))
    return torch.from_numpy(Q_modified)


def get_init_weight(dim):
    return get_random_orthg(dim)


def kronecker_matmul(x, hadL, hadR):
    """Equivalent to

    had = torch.kron(hadL, hadR)
    x = x.reshape(-1, had.shape[0])
    x = x.matmul(had).reshape(init_shape)
    """
    init_shape = x.shape
    x = x.reshape(-1, hadL.shape[0], hadR.shape[0])
    x = torch.matmul(x, hadR)
    x = torch.matmul(hadL.T, x)
    return x.reshape(init_shape)


def get_decompose_dim(n):
    if n < 1:
        raise ValueError("Transform dimension must be positive")
    left = math.isqrt(n)
    while n % left:
        left -= 1
    return left, n // left


@lru_cache(maxsize=None)
def get_main_decompose_dim(n):
    """Find main-path Kronecker factors."""
    if n < 6:
        raise ValueError(f"No asymmetric main factorization for {n}")
    pairs = [
        (abs((n // left) / left - 2.5), left, n // left)
        for left in range(2, math.isqrt(n) + 1)
        if n % left == 0 and 2 * left <= n // left <= 3 * left
    ]
    if not pairs:
        raise ValueError(f"No 2..3-ratio main factorization for {n}")
    _, left, right = min(pairs)
    return left, right
