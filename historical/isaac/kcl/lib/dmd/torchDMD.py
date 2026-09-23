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

    # def fit(self, weights):
    #     """
    #     Fit the DMD model to weight data with improved error handling and memory efficiency.
        
    #     :param weights: Tensor of model weight history
    #     """
    #     try:
    #         # Slice data for X and Y (snapshots)
    #         X = weights[:, :-1]
    #         Y = weights[:, 1:]
            
    #         print(f"Fitting DMD with weight matrix of shape {weights.shape}")
            
    #         # Try to compute SVD with memory optimization
    #         svd_info = compute_svd_torch(X, svd_rank=self.rank)
            
    #         # Compute A tilde (reduced order model)
    #         # Use more memory-efficient implementation
    #         U_T_Y = torch.matmul(svd_info.U.T.conj(), Y)
    #         temp = torch.matmul(U_T_Y, svd_info.V)
    #         atilde = temp * torch.reciprocal(svd_info.s)
            
    #         self._atilde = atilde
            
    #         # Compute eigendecomposition
    #         try:
    #             eigenvals, eigenvectors = torch.linalg.eig(atilde)
    #             self._left_eigs = torch.linalg.eig(atilde.T)
    #         except RuntimeError as e:
    #             print(f"Error in eigendecomposition: {e}")
    #             print("Moving to CPU for eigendecomposition...")
                
    #             # Try on CPU if CUDA fails
    #             atilde_cpu = atilde.cpu()
    #             eigenvals, eigenvectors = torch.linalg.eig(atilde_cpu)
    #             left_eigs = torch.linalg.eig(atilde_cpu.T)
                
    #             # Move back to original device
    #             eigenvals = eigenvals.to(weights.device)
    #             eigenvectors = eigenvectors.to(weights.device)
    #             self._left_eigs = (left_eigs[0].to(weights.device), left_eigs[1].to(weights.device))
                
    #             # Clean up
    #             del atilde_cpu
            
    #         # Compute DMD modes
    #         if self.exact:
    #             # Memory efficient computation of exact modes
    #             modes_temp = Y.matmul(svd_info.V) 
    #             modes = modes_temp * torch.reciprocal(svd_info.s)
    #             modes = modes.to(torch.complex64).matmul(eigenvectors)
    #             del modes_temp  # Clean up intermediate results
    #         else:
    #             # Standard projection
    #             modes = torch.matmul(svd_info.U.to(torch.complex64), eigenvectors)
            
    #         # Store results
    #         self._svd_info = svd_info
    #         self._eigenvals = eigenvals
    #         self._eigenvectors = eigenvectors
    #         self._modes = modes
            
    #         self.is_fit = True
            
    #         # Final memory cleanup
    #         torch.cuda.empty_cache()
            
    #     except Exception as e:
    #         print(f"DMD fitting failed with error: {e}")
    #         print("DMD will not be used for this epoch")
    #         self.is_fit = False
    #         # Ensure memory is freed
    #         torch.cuda.empty_cache()
    #         raise

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

    # def predict_multistep(self, initial_state, steps):
    #     """
    #     Predicts the state of the system 'steps' into the future based on the initial state.
    #     Memory-optimized implementation.
        
    #     :param initial_state: The initial state or weights as a tensor.
    #     :param steps: The number of future steps to predict.
    #     :return: A tensor containing the predicted state after 'steps' steps.
    #     """
    #     if not self.is_fit:
    #         raise ValueError("TorchDMD model has not been fit to data.")
        
    #     # Convert initial state to complex for compatibility with DMD modes
    #     initial_state_complex = initial_state.type(torch.complex64)
        
    #     # Calculate the exponential of eigenvalues for the 'steps' future step
    #     eigs_exp = torch.pow(self.eigs, steps)
    #     eigs_exp_0 = torch.pow(self.eigs, 0)
        
    #     # Memory efficient implementation:
    #     # Calculate projected_state = pinv_modes @ initial_state first
    #     projected_state = torch.matmul(torch.pinverse(self._modes), initial_state_complex)
        
    #     # Calculate future state without creating large matrices
    #     # future_state = modes @ diag(eigs_exp) @ projected_state
    #     modes_scaled = self._modes * eigs_exp.unsqueeze(0)
    #     future_state = torch.matmul(modes_scaled, projected_state)
        
    #     # Calculate reconstructed state similarly
    #     modes_0 = self._modes * eigs_exp_0.unsqueeze(0)
    #     reconstrcted_state = torch.matmul(modes_0, projected_state)
        
    #     # Calculate final prediction
    #     predicted_difference = future_state - reconstrcted_state
    #     predicted_state = initial_state + predicted_difference
        
    #     # Clean up to free memory
    #     del modes_scaled, modes_0, projected_state, future_state, reconstrcted_state
    #     torch.cuda.empty_cache()
        
    #     return predicted_state.real  # Convert prediction to real since weights are real numbers

    # def predict_multistep(self, initial_state, steps):
    #     """
    #     Memory-optimized implementation of multistep prediction
    #     """
    #     if not self.is_fit:
    #         raise ValueError("TorchDMD model has not been fit to data.")
        
    #     try:
    #         # Convert initial state to complex for compatibility with DMD modes
    #         initial_state_complex = initial_state.type(torch.complex64)
            
    #         # Calculate eigenvalue powers for the time steps
    #         eigs_exp = torch.pow(self.eigs, steps)
    #         eigs_exp_0 = torch.pow(self.eigs, 0)
            
    #         # Memory-efficient computation using QR decomposition
    #         try:
    #             # Compute QR decomposition of modes
    #             q, r = torch.linalg.qr(self._modes)
                
    #             # Solve triangular system instead of computing pseudoinverse
    #             qt_init = torch.matmul(q.conj().T, initial_state_complex)
    #             projected_state = torch.linalg.solve_triangular(r, qt_init, upper=True)
    #         except Exception as e:
    #             # Fall back to standard pseudoinverse if QR fails
    #             print(f"QR-based approach failed: {e}, falling back to standard pinverse")
    #             projected_state = torch.matmul(torch.pinverse(self._modes), initial_state_complex)
            
    #         # Calculate future state
    #         modes_scaled = self._modes * eigs_exp.unsqueeze(0)
    #         future_state = torch.matmul(modes_scaled, projected_state)
            
    #         # Calculate reconstructed state
    #         modes_0 = self._modes * eigs_exp_0.unsqueeze(0)
    #         reconstrcted_state = torch.matmul(modes_0, projected_state)
            
    #         # Calculate final prediction
    #         predicted_difference = future_state - reconstrcted_state
    #         predicted_state = initial_state + predicted_difference.real
            
    #         # Clean up to free memory
    #         del modes_scaled, modes_0, future_state, reconstrcted_state
    #         torch.cuda.empty_cache()
            
    #         return predicted_state.real  # Convert prediction to real since weights are real numbers
        
    #     except Exception as e:
    #         print(f"Error in DMD prediction: {e}")
    #         # Return initial state on error to avoid breaking training
    #         return initial_state
