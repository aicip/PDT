import torch
import torch.nn.functional as F
import os
import pickle
import wandb
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.profiler import profile, record_function, ProfilerActivity

from kcl.lib.trainers.ssl_trainer_base import SSLTrainerBase
from kcl.lib.dmd.torchDMD import TorchDMD
from kcl.utils.weight_tools import array_to_params, flat_params_as_torch


class SimSiamPredictedTrainer(SSLTrainerBase):
    """
    SimSiam trainer with Predictive Differential Training (PDT) integration.
    
    This trainer combines SimSiam self-supervised learning with PDT's
    weight prediction capabilities for accelerated training.
    """
    
    def __init__(
        self,
        *,
        svd_rank=10,
        n_past_weights=5,
        predict_start_epoch=10,
        predicted_num=5,
        predict_epoch_interval=5,
        **kwargs
    ):
        """
        Initialize SimSiam PDT trainer
        
        Args:
            svd_rank: Number of singular values for DMD
            n_past_weights: Number of past weights to use for prediction
            predict_start_epoch: Epoch to start making predictions
            predicted_num: Number of prediction steps
            predict_epoch_interval: Interval between predictions
            **kwargs: Arguments passed to SSLTrainerBase
        """
        super().__init__(**kwargs)
        
        # PDT specific parameters
        self.svd_rank = svd_rank
        self.n_past_weights = n_past_weights
        self.predicted_num = predicted_num
        self.predict_start_epoch = predict_start_epoch
        self.predict_epoch_interval = predict_epoch_interval
        self.epoch_since_last_predict = 0
        
        # PDT state
        self.first_predict = True
        self.dmd = None
        
        print(f"SimSiam PDT Trainer initialized:")
        print(f"  SVD rank: {self.svd_rank}")
        print(f"  Past weights: {self.n_past_weights}")
        print(f"  Start epoch: {self.predict_start_epoch}")
        print(f"  Prediction steps: {self.predicted_num}")
        print(f"  Prediction interval: {self.predict_epoch_interval}")

    def ssl_forward_step(self, batch, loss_func):
        """
        SimSiam-specific forward step
        
        Args:
            batch: Batch containing (x1, x2) - two augmented views
            loss_func: Loss function (should be simsiam_loss or compatible)
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Extract two augmented views from batch
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x1, x2 = batch
            x1, x2 = x1.to(self.device), x2.to(self.device)
        else:
            raise ValueError("SimSiam batch should contain two augmented views (x1, x2)")
        
        self.optim.zero_grad()
        
        # Forward pass through SimSiam model
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            # Model should return (p1, p2, z1, z2)
            p1, p2, z1, z2 = self.model(x1, x2, mode='ssl')
            
            # Compute SimSiam loss
            loss = loss_func(p1, p2, z1, z2)
        
        # Backward pass
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optim)
        self.grad_scaler.update()
        
        return loss

    def train_epoch(self, train_loader, loss_func):
        """
        Train epoch with PDT integration for SimSiam
        """
        # FLOP counting for the entire epoch including PDT
        if self.args and hasattr(self.args, 'count_flops') and self.args.count_flops:
            epoch_flops = self.batch_forward_flops * len(train_loader) if self.batch_forward_flops else 0

        # Check if we should apply PDT
        should_predict = (
            self.weights is not None and 
            self.weights.shape[1] >= max(self.svd_rank + 1, self.predict_start_epoch) and
            self.cur_epoch >= self.predict_start_epoch
        )
        
        if should_predict:
            if self.first_predict:
                self.first_predict = False
                self.epoch_since_last_predict = self.predict_epoch_interval

            if self.epoch_since_last_predict >= self.predict_epoch_interval:
                # Apply PDT: train normally first, then use prediction
                if self.args and hasattr(self.args, 'count_flops') and self.args.count_flops:
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True,
                        with_flops=True,
                        profile_memory=True
                    ) as prof:
                        with record_function("ssl_dmd_computation"):
                            weights_before = flat_params_as_torch(self.model).clone()
                            self._train_normal_ssl(train_loader, loss_func)
                            weights_after = flat_params_as_torch(self.model).clone()
                            weight_differences = weights_after - weights_before
                            self._perform_dmd_multistep_prediction_ssl(weight_differences)
                    
                    print("\nSSL DMD computation profiling results:")
                    print(prof.key_averages().table(sort_by="flops", row_limit=-1))
                    
                    dmd_flops = sum(e.flops for e in prof.key_averages())
                    epoch_flops += dmd_flops
                else:
                    # Normal PDT without profiling
                    weights_before = flat_params_as_torch(self.model).clone()
                    self._train_normal_ssl(train_loader, loss_func)
                    weights_after = flat_params_as_torch(self.model).clone()
                    weight_differences = weights_after - weights_before
                    self._perform_dmd_multistep_prediction_ssl(weight_differences)

                self.epoch_since_last_predict = 1
            else:
                # Train normally without prediction
                self._train_normal_ssl(train_loader, loss_func)
                self.epoch_since_last_predict += 1
        else:
            # Train normally (before prediction starts or insufficient weight history)
            self._train_normal_ssl(train_loader, loss_func)

        # Record FLOP usage
        if self.args and hasattr(self.args, 'count_flops') and self.args.count_flops:
            self.epoch_flops.append(epoch_flops)
            self.total_flops += epoch_flops
            
            if self.use_wandb:
                wandb.log({
                    "epoch_flops": epoch_flops,
                    "total_flops": self.total_flops,
                }, step=self.cur_epoch)

    def _train_normal_ssl(self, train_loader, loss_func):
        """
        Normal SSL training for one epoch (same as parent class but with weight storage)
        """
        # Store weights for PDT
        if self.save_weights:
            if self.weights is None:
                self.weights = flat_params_as_torch(self.model).reshape((-1, 1))
            else:
                self.weights = torch.cat([
                    self.weights,
                    flat_params_as_torch(self.model).reshape((-1, 1))
                ], dim=1)
                
        self.model.train()
        pbar = tqdm(
            total=len(train_loader),
            position=1,
            unit="batch",
            leave=False,
            desc="Epoch {} (PDT)".format(self.cur_epoch),
        )
        losses = []

        # FLOP profiling for first batch
        do_profile = (hasattr(self, 'args') and 
                     hasattr(self.args, 'count_flops') and 
                     self.args.count_flops and 
                     self.batch_forward_flops is None)

        for batch_idx, batch in enumerate(train_loader):
            if do_profile and batch_idx == 0:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_flops=True,
                    profile_memory=True
                ) as prof:
                    with record_function("ssl_iteration"):
                        loss = self.ssl_forward_step(batch, loss_func)
                
                print("\nSSL iteration profiling results:")
                print(prof.key_averages().table(sort_by="flops", row_limit=-1))
                
                self.batch_forward_flops = sum(e.flops for e in prof.key_averages())
                
                if self.use_wandb:
                    wandb.config.update({"batch_forward_flops": self.batch_forward_flops}, allow_val_change=True)
            else:
                loss = self.ssl_forward_step(batch, loss_func)

            if batch_idx % self.log_interval == 0:
                tqdm.write("Loss: {:.4f}".format(loss.item()))
            losses.append(loss.item())
            pbar.update(1)

        self.train_losses.append(losses)
        tqdm.write(f"Learning rate: {self.optim.param_groups[0]['lr']}")
        pbar.close()

    def _perform_dmd_multistep_prediction_ssl(self, epoch_gradient):
        """
        Perform DMD prediction and apply masking for SSL training
        
        Args:
            epoch_gradient: Weight difference from one epoch of training
        """
        current_weights = flat_params_as_torch(self.model).clone()

        if self.weights is not None:
            # Check gradient dimension compatibility
            if epoch_gradient.dim() != 1 or epoch_gradient.size(0) != self.weights.size(0):
                raise ValueError("Incorrect dimension of epoch_gradient for SSL training")
            
            # Initialize DMD
            self.dmd = TorchDMD(rank=self.svd_rank)
            
            # Use only recent weights if specified
            if self.n_past_weights is not None:
                weights_for_dmd = self.weights[:, -self.n_past_weights:]
            else:
                weights_for_dmd = self.weights

            # Fit DMD and predict
            weights_for_dmd.requires_grad = False
            self.dmd.fit(weights_for_dmd)

            # Predict future weights
            future_steps = self.predicted_num
            predicted_weights = self.dmd.predict_multistep(
                weights_for_dmd[:, -1].reshape(-1, 1), 
                future_steps
            )
            
            # Compute weight difference from prediction - ensure proper dimensions
            pred_last = predicted_weights[:, -1].reshape(-1)
            curr_last = weights_for_dmd[:, -1].reshape(-1)
            weight_diff = pred_last - curr_last
            epoch_gradient = epoch_gradient.reshape(-1)

            # Create mask based on PDT criteria
            mask = self._create_ssl_mask(weight_diff, epoch_gradient)
            
            # Log mask statistics
            mask_count = torch.sum(mask)
            mask_ratio = mask_count.float() / len(mask)
            
            # Calculate ratios for different conditions
            ratio_opposite = torch.sum((torch.sign(weight_diff) != torch.sign(epoch_gradient))).float() / len(mask)
            ratio_n_plus = torch.sum((torch.sign(weight_diff) == torch.sign(epoch_gradient)) & 
                                   (torch.abs(weight_diff) > (self.predicted_num + 1) * torch.abs(epoch_gradient))).float() / len(mask)
            ratio_0_1 = torch.sum((torch.sign(weight_diff) == torch.sign(epoch_gradient)) & 
                                (torch.abs(weight_diff) < torch.abs(epoch_gradient))).float() / len(mask)

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "ssl_mask_ratio_global": mask_ratio.item(),
                    "ssl_ratio_opposite": ratio_opposite.item(),
                    "ssl_ratio_n_plus": ratio_n_plus.item(),
                    "ssl_ratio_0_1": ratio_0_1.item(),
                }, step=self.cur_epoch)

            print(f"SSL PDT mask ratio: {mask_ratio.item():.4f}")

            # Save mask for analysis (opt-in)
            if getattr(self.args, 'save_masks', False):
                mask_folder = os.path.join(self.log_dir, "ssl_masks")
                os.makedirs(mask_folder, exist_ok=True)
                mask_file = os.path.join(mask_folder, f"ssl_mask_epoch_{self.cur_epoch:04d}.pt")
                torch.save(mask, mask_file)

            # Apply masked prediction
            updated_weights = torch.where(mask, pred_last, current_weights)
            
            # Update model weights
            array_to_params(self.model, updated_weights, device=self.weights.device)
            
            # Reset optimizer state (Critical for stability!)
            for param_group in self.optim.param_groups:
                for param in param_group['params']:
                    state = self.optim.state.get(param, None)
                    if state is not None:
                        state.clear()

    def _create_ssl_mask(self, weight_diff, epoch_gradient):
        """
        Create mask for SSL training based on PDT criteria
        
        Args:
            weight_diff: Difference between predicted and current weights
            epoch_gradient: Weight change from one epoch of SSL training
            
        Returns:
            torch.Tensor: Boolean mask for weight updates
        """
        # Direction criterion: same sign
        direction_criterion = (torch.sign(weight_diff) == torch.sign(epoch_gradient))
        
        # Quantity criterion: meaningful acceleration but not too extreme
        quantity_lower = torch.abs(weight_diff) >= torch.abs(epoch_gradient)
        quantity_upper = torch.abs(weight_diff) <= (self.predicted_num + 1) * torch.abs(epoch_gradient)
        quantity_criterion = quantity_lower & quantity_upper
        
        # Combine criteria
        mask = direction_criterion & quantity_criterion
        
        return mask

    def evaluate_representation_quality(self, test_loader, probe_train_loader=None, num_classes=10):
        """
        Evaluate representation quality using linear probing with proper train/test split
        """
        self.model.eval()
        
        if probe_train_loader is None:
            print("Warning: No probe_train_loader provided, using test_loader for both (optimistic evaluation)")
            probe_train_loader = test_loader
        
        # Extract features for probe training data
        train_features_list = []
        train_labels_list = []
        
        with torch.no_grad():
            for data, target in probe_train_loader:
                data = data.to(self.device)
                features = self.model.forward_encoder(data)
                train_features_list.append(features.cpu())
                train_labels_list.append(target)
        
        # Extract features for test data
        test_features_list = []
        test_labels_list = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                features = self.model.forward_encoder(data)
                test_features_list.append(features.cpu())
                test_labels_list.append(target)
        
        # Concatenate all features and labels
        train_features = torch.cat(train_features_list, dim=0)
        train_labels = torch.cat(train_labels_list, dim=0)
        test_features = torch.cat(test_features_list, dim=0)
        test_labels = torch.cat(test_labels_list, dim=0)
        
        # Fit linear classifier on train, evaluate on test
        accuracy = self._linear_probe_accuracy(train_features, train_labels, test_features, test_labels, num_classes)
        
        self.model.train()
        return accuracy
    
    def _linear_probe_accuracy(self, train_features, train_labels, test_features, test_labels, num_classes, max_iter=2000):
        """
        Train a linear classifier on frozen features and return test accuracy
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        # Convert to numpy
        train_features_np = train_features.numpy()
        train_labels_np = train_labels.numpy()
        test_features_np = test_features.numpy()
        test_labels_np = test_labels.numpy()
        
        # Train logistic regression classifier on train set
        classifier = LogisticRegression(
            random_state=42, 
            max_iter=max_iter,
            solver='lbfgs' if num_classes <= 10 else 'saga'
        )
        classifier.fit(train_features_np, train_labels_np)
        
        # Predict and compute accuracy on test set
        predictions = classifier.predict(test_features_np)
        accuracy = accuracy_score(test_labels_np, predictions)
        
        return accuracy

    def evaluate(self, eval_loader, eval_func=None):
        """
        Override evaluation to include PDT-specific metrics for SSL
        """
        # Handle the case where eval_loader is a tuple (test_loader, probe_train_loader)
        if isinstance(eval_loader, tuple) and len(eval_loader) == 2:
            test_loader, probe_train_loader = eval_loader
        else:
            test_loader = eval_loader
            probe_train_loader = None
        
        if eval_func is not None:
            # Use custom evaluation function
            return super().evaluate(eval_loader, eval_func)
        else:
            # Default: linear probing for representation quality
            accuracy = self.evaluate_representation_quality(test_loader, probe_train_loader)
            
            # Store metrics
            metrics = {
                "linear_probe_accuracy": accuracy,
                "eval_metric": accuracy,  # For compatibility with base class
                'pdt_epoch_since_last_predict': self.epoch_since_last_predict,
                'pdt_prediction_active': self.cur_epoch >= self.predict_start_epoch,
            }
            self.ssl_metrics.append(metrics)
            
            if self.use_wandb:
                wandb.log({
                    "ssl_linear_probe_accuracy": accuracy,
                }, step=self.cur_epoch)
            
            tqdm.write(f"SSL Linear probe accuracy: {accuracy:.4f}")
            return accuracy

    def save_all(self, save_path):
        """
        Save model and training state including PDT-specific information
        """
        super().save_all(save_path)
        
        # Save additional PDT-specific information
        pdt_info = {
            'model_type': 'SimSiam_PDT',
            'svd_rank': self.svd_rank,
            'n_past_weights': self.n_past_weights,
            'predict_start_epoch': self.predict_start_epoch,
            'predicted_num': self.predicted_num,
            'predict_epoch_interval': self.predict_epoch_interval,
            'epoch_since_last_predict': self.epoch_since_last_predict,
            'first_predict': self.first_predict,
        }
        
        with open(os.path.join(save_path, "ssl_pdt_info.pkl"), "wb") as f:
            pickle.dump(pdt_info, f)


# Convenience function for creating PDT trainers
def create_simsiam_pdt_trainer(
    svd_rank=10,
    n_past_weights=5,
    predict_start_epoch=10,
    predicted_num=5,
    predict_epoch_interval=5,
    **kwargs
):
    """
    Create a SimSiam PDT trainer with default parameters
    """
    return SimSiamPredictedTrainer(
        svd_rank=svd_rank,
        n_past_weights=n_past_weights,
        predict_start_epoch=predict_start_epoch,
        predicted_num=predicted_num,
        predict_epoch_interval=predict_epoch_interval,
        **kwargs
    )