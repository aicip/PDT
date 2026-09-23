import torch

from kcl.lib.dmd.dmdGeneric import TorchDMDGeneric
from kcl.lib.dmd.utils import compute_svd_torch


class TorchDMD(TorchDMDGeneric):
    def __init__(self, rank=0, exact=False):
        self.rank = rank
        self.exact = exact
        self.is_fit = False

    def fit(self, weights):
        X = weights[:, :-1]
        Y = weights[:, 1:]
        svd_info = compute_svd_torch(X, svd_rank=self.rank)

        atilde = torch.linalg.multi_dot([svd_info.U.T.conj(), Y, svd_info.V]
                                        ) * torch.reciprocal(svd_info.s)

        self._atilde = atilde
        eigenvals, eigenvectors = torch.linalg.eig(atilde)
        self._left_eigs = torch.linalg.eig(atilde.T)

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
    def left_eigs(self):
        return self._left_eigs

    def predict(self, x):
        modes = torch.matmul(self.basis.to(torch.complex64), self.eigenvectors)
        pinv_modes = torch.pinverse(modes)

        return torch.linalg.multi_dot(
            [
                modes,
                torch.diag(self.eigs), pinv_modes,
                x.type(torch.complex64)
            ]
        ).real
    
    
    def predict_multistep_org(self, initial_state, steps):
        """
        Predicts the state of the system 'steps' into the future based on the initial state.
        
        :param initial_state: The initial state or weights as a tensor.
        :param steps: The number of future steps to predict.
        :return: A tensor containing the predicted state after 'steps' steps.
        """
        if not self.is_fit:
            raise ValueError("TorchDMD model has not been fit to data.")
        
        # Convert initial state to complex for compatibility with DMD modes
        initial_state_complex = initial_state.type(torch.complex64)
        
        # Calculate the exponential of eigenvalues for the 'steps' future step
        eigs_exp = torch.pow(self.eigs, steps)

        # calculate the future state
        # future_state = torch.matmul(self._modes, torch.diag(eigs_exp)).matmul(torch.pinverse(self._modes)).matmul(initial_state_complex)
        
        # Direct application of each eigenvalue to its corresponding mode
        # Avoid creating a large diagonal matrix
        modes_scaled = self._modes * eigs_exp.unsqueeze(0)
        
        # Project initial state onto the modes
        projected_state = torch.matmul(torch.pinverse(self._modes), initial_state_complex)
        
        # Apply the scaled modes to the projected state
        future_state = torch.matmul(modes_scaled, projected_state)
        
        return future_state.real  # Convert prediction to real since weights are real numbers
    
    def predict_multistep(self, initial_state, steps):
        """
        Predicts the state of the system 'steps' into the future based on the initial state.
        
        :param initial_state: The initial state or weights as a tensor.
        :param steps: The number of future steps to predict.
        :return: A tensor containing the predicted state after 'steps' steps.
        """
        if not self.is_fit:
            raise ValueError("TorchDMD model has not been fit to data.")
        
        # Convert initial state to complex for compatibility with DMD modes
        initial_state_complex = initial_state.type(torch.complex64)
        
        # Calculate the exponential of eigenvalues for the 'steps' future step
        eigs_exp = torch.pow(self.eigs, steps)
        eigs_exp_0 = torch.pow(self.eigs, 0)
        # calculate the future state
        # future_state = torch.matmul(self._modes, torch.diag(eigs_exp)).matmul(torch.pinverse(self._modes)).matmul(initial_state_complex)
        
        # Direct application of each eigenvalue to its corresponding mode
        # Avoid creating a large diagonal matrix
        modes_scaled = self._modes * eigs_exp.unsqueeze(0)
        modes_0 = self._modes * eigs_exp_0.unsqueeze(0)
        # Project initial state onto the modes
        projected_state = torch.matmul(torch.pinverse(self._modes), initial_state_complex)
        
        # Apply the scaled modes to the projected state
        future_state = torch.matmul(modes_scaled, projected_state)
        reconstrcted_state = torch.matmul(modes_0, projected_state)
        predicted_difference = future_state - reconstrcted_state
        predicted_state = initial_state + predicted_difference
        
        return predicted_state.real  # Convert prediction to real since weights are real numbers
