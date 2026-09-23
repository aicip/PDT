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


# def compute_svd_torch(X, svd_rank=0, reduced=False):
#     """
#     Truncated Singular Value Decomposition with fallback mechanisms for large matrices.

#     :param numpy.ndarray X: the matrix to decompose.
#     :param svd_rank: the rank for the truncation; If 0, the method computes
#         the optimal rank and uses it for truncation; if positive interger,
#         the method uses the argument for the truncation; if float between 0
#         and 1, the rank is the number of the biggest singular values that
#         are needed to reach the 'energy' specified by `svd_rank`; if -1,
#         the method does not compute truncation. Default is 0.
#     :type svd_rank: int or float
#     :param bool reduced: if True, use reduced SVD method (faster but less accurate)
#     :return: the truncated left-singular vectors matrix, the truncated
#         singular values array, the truncated right-singular vectors matrix.
#     :rtype: SVDInfo
#     """
#     try:
#         # Try using the specified method first
#         if reduced:
#             print("Using reduced SVD for memory efficiency")
#             U, s, V = reduced_svd(X)
#         else:
#             # Try CUDA SVD
#             print(f"Attempting full SVD on CUDA for matrix of shape {X.shape}")
#             U, s, V = torch.linalg.svd(X, full_matrices=False)
#             V = V.conj().T
#             print("CUDA SVD successful")
    
#     except RuntimeError as e:
#         print(f"Primary SVD failed with error: {e}")
        
#         try:
#             print("Trying reduced SVD method as fallback...")
#             U, s, V = reduced_svd(X)
#             print("Reduced SVD successful")
        
#         except RuntimeError as e2:
#             print(f"Reduced SVD also failed: {e2}")
#             print("Attempting CPU SVD (slower but with more memory)...")
            
#             try:
#                 # Move to CPU, clear GPU cache
#                 X_cpu = X.cpu()
#                 torch.cuda.empty_cache()
                
#                 U_cpu, s_cpu, V_cpu = torch.linalg.svd(X_cpu, full_matrices=False)
#                 V_cpu = V_cpu.conj().T
                
#                 # Move results back to original device
#                 U = U_cpu.to(X.device)
#                 s = s_cpu.to(X.device)
#                 V = V_cpu.to(X.device)
                
#                 # Clear CPU memory
#                 del X_cpu, U_cpu, s_cpu, V_cpu
#                 print("CPU SVD successful")
                
#             except Exception as e3:
#                 print(f"CPU SVD also failed: {e3}")
#                 print("Using randomized SVD as final fallback...")
                
#                 # Try randomized SVD using torch operations
#                 n_oversamples = 10
#                 n_components = min(svd_rank if svd_rank > 0 else 10, min(X.shape))
                
#                 # Move to CPU to save GPU memory
#                 X_cpu = X.cpu()
#                 torch.cuda.empty_cache()
                
#                 # Step 1: Random projection
#                 random_matrix = torch.randn(X_cpu.shape[1], n_components + n_oversamples)
#                 projected = X_cpu @ random_matrix
                
#                 # Step 2: QR decomposition
#                 Q, _ = torch.linalg.qr(projected, mode='reduced')
                
#                 # Step 3: Project X onto Q
#                 B = Q.T @ X_cpu
                
#                 # Step 4: SVD on smaller matrix
#                 Uhat, s, Vt = torch.linalg.svd(B, full_matrices=False)
#                 U = Q @ Uhat
#                 V = Vt.T
                
#                 # Move back to original device
#                 U = U.to(X.device)
#                 s = s.to(X.device)
#                 V = V.to(X.device)
                
#                 # Clean up
#                 del X_cpu, random_matrix, projected, Q, B, Uhat, Vt
#                 print("Randomized SVD successful")

#     # Handle rank selection (same as before)
#     if svd_rank == 0:
#         rank = calc_optimal_rank(s, X.shape)
#         if rank == 0:
#             warnings.warn(
#                 "SVD optimal rank is 0. The largest singular values are "
#                 "indistinguishable from noise. Setting rank truncation to 1.",
#                 RuntimeWarning,
#             )
#             rank = 1
#     elif 0 < svd_rank < 1:
#         cumulative_energy = torch.cumsum(
#             (s**2 / (s**2).sum()).flatten(), dim=0
#         )
#         rank = torch.searchsorted(cumulative_energy, svd_rank) + 1
#     elif svd_rank >= 1 and isinstance(svd_rank, int):
#         rank = min(svd_rank, U.shape[1])
#     else:
#         rank = X.shape[1]

#     U = U[:, :rank]
#     V = V[:, :rank]
#     s = s[:rank]

#     svd_info = SVDInfo(U, s, V, rank)
    
#     # Final memory cleanup
#     torch.cuda.empty_cache()
    
#     return svd_info


# def compute_svd_torch(X, svd_rank=0, reduced=False):
#     """
#     Memory-optimized SVD computation with fallback options
#     """
#     try:
#         # Check if input size warrants using reduced SVD
#         if X.numel() > 100000000 and not reduced:  # 100M elements
#             print(f"Large matrix detected ({X.shape}), automatically using reduced SVD")
#             reduced = True
        
#         # Compute SVD based on selected method
#         if reduced:
#             print(f"Using reduced SVD for matrix of shape {X.shape}")
#             U, s, V = reduced_svd(X)
#         else:
#             print(f"Using full SVD for matrix of shape {X.shape}")
#             U, s, V = torch.linalg.svd(X, full_matrices=False)
#             V = V.conj().T
    
#     except RuntimeError as e:
#         print(f"Primary SVD failed with error: {e}")
        
#         try:
#             print("Trying reduced SVD method as fallback...")
#             U, s, V = reduced_svd(X)
#             print("Reduced SVD successful")
        
#         except RuntimeError as e2:
#             print(f"Reduced SVD also failed: {e2}")
#             print("Attempting CPU SVD (slower but with more memory)...")
            
#             try:
#                 # Move to CPU, clear GPU cache
#                 X_cpu = X.cpu()
#                 torch.cuda.empty_cache()
                
#                 U_cpu, s_cpu, V_cpu = torch.linalg.svd(X_cpu, full_matrices=False)
#                 V_cpu = V_cpu.conj().T
                
#                 # Move results back to original device
#                 U = U_cpu.to(X.device)
#                 s = s_cpu.to(X.device)
#                 V = V_cpu.to(X.device)
                
#                 # Clear CPU memory
#                 del X_cpu, U_cpu, s_cpu, V_cpu
#                 print("CPU SVD successful")
                
#             except Exception as e3:
#                 print(f"CPU SVD also failed: {e3}")
#                 print("Using minimal rank as fallback...")
                
#                 # Extreme fallback: use very small rank
#                 min_rank = min(5, min(X.shape))
                
#                 # Use the power method to compute top singular vectors
#                 XXT = X @ X.T
                
#                 # Compute top eigenvector of XXT
#                 v = torch.randn(XXT.shape[0], 1, device=X.device)
#                 for _ in range(10):  # Power iterations
#                     v = XXT @ v
#                     v = v / torch.norm(v)
                
#                 # Construct minimal rank approximation
#                 U = v
#                 s = torch.norm(X.T @ v)
#                 V = (X.T @ v) / s
                
#                 s = s.reshape(-1)
                
#                 print(f"Minimal fallback SVD completed with rank {min_rank}")

#     # Apply rank truncation (unchanged)
#     if svd_rank == 0:
#         rank = calc_optimal_rank(s, X.shape)
#         if rank == 0:
#             warnings.warn(
#                 "SVD optimal rank is 0. The largest singular values are "
#                 "indistinguishable from noise. Setting rank truncation to 1.",
#                 RuntimeWarning,
#             )
#             rank = 1
#     elif 0 < svd_rank < 1:
#         cumulative_energy = torch.cumsum(
#             (s**2 / (s**2).sum()).flatten(), dim=0
#         )
#         rank = torch.searchsorted(cumulative_energy, svd_rank) + 1
#     elif svd_rank >= 1 and isinstance(svd_rank, int):
#         rank = min(svd_rank, U.shape[1])
#     else:
#         rank = X.shape[1]

#     U = U[:, :rank]
#     V = V[:, :rank]
#     s = s[:rank]

#     svd_info = SVDInfo(U, s, V, rank)
    
#     # Final memory cleanup
#     torch.cuda.empty_cache()
    
#     return svd_info