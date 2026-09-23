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
    """
    Compute the optimal rank of truncation for a given dynamical system matrix
    using the algorithm from Gavish and Donoho (2014).

    :param numpy.ndarray svs: the singular values of the matrix;
    :param tuple shape: the shape of the matrix
    :return: the optimal rank of truncation.

    References:
    Gavish, Matan, and David L. Donoho, The optimal hard threshold for
    singular values is, IEEE Transactions on Information Theory 60.8
    (2014): 5040-5053.
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


def compute_tlsq_torch(X, Y, tlsq_rank):
    """
    Compute Total Least Square.

    :param numpy.ndarray X: the first matrix;
    :param numpy.ndarray Y: the second matrix;
    :param int tlsq_rank: the rank for the truncation; If 0, the method
        does not compute any noise reduction; if positive number, the
        method uses the argument for the SVD truncation used in the TLSQ
        method.
    :return: the denoised matrix X, the denoised matrix Y
    :rtype: numpy.ndarray, numpy.ndarray

    References:
    https://arxiv.org/pdf/1703.11004.pdf
    https://arxiv.org/pdf/1502.03854.pdf
    """
    # Do not perform tlsq
    if tlsq_rank == 0:
        return X, Y

    V = torch.linalg.svd(torch.append(X, Y, axis=0), full_matrices=False)[-1]
    rank = min(tlsq_rank, V.shape[0])
    VV = torch.matmul(V[:rank, :].conj().T, V[:rank, :])

    return torch.matmul(X, VV), torch.matmul(Y, VV)


def reduced_svd(X):
    """
    Method from https://arxiv.org/pdf/2006.14371.pdf
    """
    V, s, _ = torch.svd(X.T @ X)
    sigma = s.sqrt()
    U = X @ V / sigma
    return U, s, V


def predict(U, eigvecs, eigs, x):
    modes = torch.matmul(U.to(torch.complex64), eigvecs)
    pinv_modes = torch.pinverse(modes)

    return torch.linalg.multi_dot(
        [modes, torch.diag(eigs), pinv_modes,
         x.type(torch.complex64)]
    ).real


def find_sv(X, U):
    """
    Find the singular values of X, given the left singular vectors U.
    """
    _, s, v = torch.linalg.svd(X, full_matrices=False)
    v = v.conj().T

    return s, v


def compute_svd_torch(X, svd_rank=0, reduced=False):
    """
    Truncated Singular Value Decomposition.

    :param numpy.ndarray X: the matrix to decompose.
    :param svd_rank: the rank for the truncation; If 0, the method computes
        the optimal rank and uses it for truncation; if positive interger,
        the method uses the argument for the truncation; if float between 0
        and 1, the rank is the number of the biggest singular values that
        are needed to reach the 'energy' specified by `svd_rank`; if -1,
        the method does not compute truncation. Default is 0.
    :type svd_rank: int or float
    :return: the truncated left-singular vectors matrix, the truncated
        singular values array, the truncated right-singular vectors matrix.
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
    """
    if reduced:
        U, s, V = reduced_svd(X)
    else:
        U, s, V = torch.linalg.svd(X, full_matrices=False)
        V = V.conj().T

    if svd_rank == 0:
        rank = calc_optimal_rank(s, X.shape)
        if rank == 0:
            warnings.warn(
                "SVD optimal rank is 0. The largest singular values are "
                "indistinguishable from noise. Setting rank truncation to 1.",
                RuntimeWarning,
            )
            rank = 1
    elif 0 < svd_rank < 1:
        cumulative_energy = torch.cumsum(
            (s**2 / (s**2).sum()).flatten(), dim=0
        )
        rank = torch.searchsorted(cumulative_energy, svd_rank) + 1
    elif svd_rank >= 1 and isinstance(svd_rank, int):
        rank = min(svd_rank, U.shape[1])
    else:
        rank = X.shape[1]

    U = U[:, :rank]
    V = V[:, :rank]
    s = s[:rank]

    svd_info = SVDInfo(U, s, V, rank)

    return svd_info
