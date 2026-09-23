"""Dynamic mode decomposition (standard DMD of Tu et al., 2014) in PyTorch."""

import torch

from kcl.lib.dmd.dmdGeneric import TorchDMDGeneric
from kcl.lib.dmd.utils import compute_svd_torch


class TorchDMD(TorchDMDGeneric):
    def __init__(self, rank=0, exact=False):
        """
        :param rank: SVD truncation rank (0 = optimal hard threshold, see ``compute_svd_torch``)
        :param exact: use exact DMD modes ``Y V S^-1 W`` instead of projected modes ``U W``
        """
        self.rank = rank
        self.exact = exact
        self.is_fit = False

    def fit(self, weights):
        """Fit the operator to a snapshot matrix ``weights`` (columns = consecutive states)."""
        X = weights[:, :-1]
        Y = weights[:, 1:]
        svd_info = compute_svd_torch(X, svd_rank=self.rank)

        atilde = torch.linalg.multi_dot([svd_info.U.T.conj(), Y, svd_info.V]) * torch.reciprocal(svd_info.s)
        self._atilde = atilde
        eigenvals, eigenvectors = torch.linalg.eig(atilde)

        if self.exact:
            modes = Y.matmul(svd_info.V) * torch.reciprocal(svd_info.s)
            modes = modes.to(torch.complex64).matmul(eigenvectors)
        else:
            modes = torch.matmul(svd_info.U.to(torch.complex64), eigenvectors)

        self._svd_info = svd_info
        self._eigenvals = eigenvals
        self._eigenvectors = eigenvectors
        self._modes = modes
        self.is_fit = True

    @property
    def eigs(self):
        return self._eigenvals

    @property
    def basis(self):
        return self._svd_info.U

    @property
    def eigenvectors(self):
        return self._eigenvectors

    @property
    def modes(self):
        return self._modes

    def predict(self, x):
        """One-step prediction ``Phi Lambda Phi^+ x``."""
        pinv_modes = torch.pinverse(self._modes)
        return torch.linalg.multi_dot([self._modes, torch.diag(self.eigs), pinv_modes, x.type(torch.complex64)]).real

    def predict_multistep(self, initial_state, steps):
        """Predict the state ``steps`` steps after ``initial_state`` (shape ``(N, 1)``).

        The prediction is ``x + Re{Phi (Lambda^steps - I) Phi^+ x}``, i.e. the predicted
        displacement is added to the initial state so that the reconstruction error of
        the initial state does not enter the prediction.
        """
        if not self.is_fit:
            raise ValueError("TorchDMD model has not been fit to data.")
        initial_state_complex = initial_state.type(torch.complex64)
        eigs_exp = torch.pow(self.eigs, steps)
        eigs_exp_0 = torch.pow(self.eigs, 0)
        modes_scaled = self._modes * eigs_exp.unsqueeze(0)
        modes_0 = self._modes * eigs_exp_0.unsqueeze(0)
        projected_state = torch.matmul(torch.pinverse(self._modes), initial_state_complex)
        future_state = torch.matmul(modes_scaled, projected_state)
        reconstructed_state = torch.matmul(modes_0, projected_state)
        predicted_state = initial_state + (future_state - reconstructed_state)
        return predicted_state.real
