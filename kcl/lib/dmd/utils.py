import warnings
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SVDInfo:
    U: torch.Tensor
    s: torch.Tensor
    V: torch.Tensor
    rank: int

    def __iter__(self):
        yield self.U
        yield self.s
        yield self.V
        yield self.rank


def calc_optimal_rank(svs, shape):
    """Optimal hard threshold for singular values (Gavish and Donoho, 2014).

    Follows the implementation in mathLab/PyDMD (MIT License), see THIRD_PARTY_NOTICES.md.

    :param svs: singular values of the matrix
    :param shape: shape of the matrix
    :return: number of singular values to keep
    """
    def omega(x):
        return 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43

    if isinstance(svs, torch.Tensor):
        beta = torch.divide(*sorted(shape))
        tau = torch.median(svs) * omega(beta)
        rank = torch.sum(svs > tau)
    else:
        beta = np.divide(*sorted(shape))
        tau = np.median(svs) * omega(beta)
        rank = np.sum(svs > tau)
    return rank


def reduced_svd(X):
    """SVD of a tall matrix through the eigendecomposition of ``X^T X``."""
    V, s, _ = torch.svd(X.T @ X)
    sigma = s.sqrt()
    U = X @ V / sigma
    return U, s, V


def compute_svd_torch(X, svd_rank=0, reduced=False):
    """Truncated SVD of ``X``.

    :param svd_rank: 0 = optimal hard threshold; a positive integer = fixed rank; a float in
        (0, 1) = smallest rank whose energy reaches ``svd_rank``; -1 = no truncation
    :return: :class:`SVDInfo` with the truncated ``U``, ``s``, ``V`` and the rank used
    """
    if reduced:
        U, s, V = reduced_svd(X)
    else:
        U, s, V = torch.linalg.svd(X, full_matrices=False)
        V = V.conj().T

    if svd_rank == 0:
        rank = calc_optimal_rank(s, X.shape)
        if rank == 0:
            warnings.warn("SVD optimal rank is 0; the largest singular values are indistinguishable "
                          "from noise. Using rank 1.", RuntimeWarning)
            rank = 1
    elif 0 < svd_rank < 1:
        cumulative_energy = torch.cumsum((s**2 / (s**2).sum()).flatten(), dim=0)
        rank = torch.searchsorted(cumulative_energy, svd_rank) + 1
    elif svd_rank >= 1 and isinstance(svd_rank, int):
        rank = min(svd_rank, U.shape[1])
    else:
        rank = X.shape[1]

    return SVDInfo(U[:, :rank], s[:rank], V[:, :rank], rank)
