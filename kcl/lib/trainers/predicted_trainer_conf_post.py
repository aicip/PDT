import numpy as np
import torch
from tqdm import tqdm
import wandb
import os
import time

from kcl.lib.dmd.torchDMD import TorchDMD
from kcl.lib.trainers.trainer_base import TrainerBase
from kcl.utils.weight_tools import array_to_params, flat_params_as_torch
from copy import deepcopy
from torch.profiler import profile, record_function, ProfilerActivity

class PredictedTrainer_conf(TrainerBase):
    def __init__(
        self,
        *,
        svd_rank=10,
        n_past_weights=None,
        predict_start_epoch = 10,
        predicted_num = 5,
        predict_epoch_interval = 5,
        mask_mode = 'both',
        log_gradient_metrics = False,
        if_sp = False,
        if_LV = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        """
        :param svd_rank: number of singular values to keep in the DMD
            calculation
        :param n_accel_eigs: number of eigenvalues to accelerate
        :param accel_const: constant to multiply the eigenvalues by
        :param mask_mode: 'both', 'accel_only', or 'consistency_only'
        :param log_gradient_metrics: if True, log gradient norm and direction stability
        """
        self.svd_rank = svd_rank
        self.n_past_weights = n_past_weights
        self.predicted_num = predicted_num
        self.predict_start_epoch = predict_start_epoch
        self.predict_epoch_interval = predict_epoch_interval   # the inerval epochs to make prediction
        self.epoch_since_last_predict = 0  # the number of epochs since last prediction
        self.mask_mode = mask_mode
        self.log_gradient_metrics = log_gradient_metrics
        self.if_sp = if_sp
        self.if_LV = if_LV
        self.threshold = 0.01

        # Log mask mode for clarity
        print(f"\n{'='*60}")
        print(f"PDT Masking Strategy: {self.mask_mode}")
        if self.log_gradient_metrics:
            print(f"Gradient Metrics Logging: ENABLED")
        print(f"{'='*60}\n")
#        print("self.if_sp: ", self.if_sp)
#        raise Exception("Stopping the program for inspection")


        #self.predicting = False
        self.first_predict = True

        # For gradient analysis (only used if log_gradient_metrics=True)
        self.prev_gradient = None  # Store previous epoch's gradient for direction stability

    def train_epoch(self, train_loader, loss_func):
        # Timing profiling: total epoch start (including potential DMD)
        if self.args.profile_timing:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            total_epoch_start = time.perf_counter()
            has_dmd_this_epoch = False

        # print("self.n_past_weights: ", self.n_past_weights)
        # print("self.predict_start_epoch: ", self.predict_start_epoch)
        # print("self.predicted_num: ", self.predicted_num)
        # print("self.predict_epoch_interval: ", self.predict_epoch_interval)

        # # calculate the FLOPs for a single batch
        # self.calculate_batch_flops(train_loader)

        # if self.args.count_flops:
        #     # calculate the baseline FLOPs for the current epoch
        #     batch_flops = self.batch_forward_flops * 3    # forward + backward(2x)
        #     epoch_flops = batch_flops * len(train_loader)
            
        #     # calculate the FLOPs for DMD
        #     if (self.weights is not None and 
        #         self.weights.shape[1] >= max(self.svd_rank + 1, self.predict_start_epoch) and
        #         self.epoch_since_last_predict >= self.predict_epoch_interval):
                
        #         # calculate FLOPs for DMD
        #         N = sum(p.numel() for p in self.model.parameters())
        #         h = self.n_past_weights

        #         #svd_flops = 6 * N * h * h + 20 * h * h * h
        #         dmd_compute_flops = 2 * N * h * h  # pseudo-inverse and dmd matrix computation
        #         predict_flops = N * h * self.predicted_num
        #         dmd_total_flops = (dmd_compute_flops + predict_flops)
        #         epoch_flops += dmd_total_flops
                
        #     self.epoch_flops.append(epoch_flops)
        #     self.total_flops += epoch_flops
            
        #     if self.use_wandb:
        #         wandb.log({
        #             "epoch_flops": epoch_flops,
        #             "total_flops": self.total_flops,
        #         }, step=self.cur_epoch)

        # Initialize epoch_flops for tracking
        epoch_flops = 0  # Will be computed after training

        if self.weights is not None and self.weights.shape[1] >= max(self.svd_rank + 1, self.predict_start_epoch):
            if self.first_predict:
                self.first_predict = False
                self.epoch_since_last_predict = self.predict_epoch_interval

            if self.epoch_since_last_predict >= self.predict_epoch_interval:
                # Mark that this epoch contains DMD
                if self.args.profile_timing:
                    has_dmd_this_epoch = True

                # Profile DMD computation if needed
                if self.args.count_flops:
                    # Original method: use torch.profiler for everything (slower but matches paper)
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True,
                        with_flops=True,
                        profile_memory=True
                    ) as prof:
                        with record_function("dmd_computation"):
                            weights_before = flat_params_as_torch(self.model).clone()
                            self._train_normal(train_loader, loss_func)
                            weights_after = flat_params_as_torch(self.model).clone()
                            weight_differences = weights_after - weights_before

                            # Log gradient metrics if enabled
                            self._log_gradient_metrics(weight_differences)

                            self._perform_dmd_multistep_prediction_and_update(weight_differences)

                    print("\nDMD computation profiling results:")
                    print(prof.key_averages().table(sort_by="flops", row_limit=-1))

                    # Profiler measured the entire epoch (SGD + DMD), so use it directly
                    epoch_flops = sum(e.flops for e in prof.key_averages())
                elif self.args.count_flops_fast:
                    # Fast method: use analytical formulas for DMD FLOPs (negligible overhead)
                    # First calculate SGD FLOPs (will be set after _train_normal profiles first batch)
                    weights_before = flat_params_as_torch(self.model).clone()
                    self._train_normal(train_loader, loss_func)
                    weights_after = flat_params_as_torch(self.model).clone()
                    weight_differences = weights_after - weights_before

                    # Log gradient metrics if enabled
                    self._log_gradient_metrics(weight_differences)

                    # Now batch_forward_flops is set, calculate SGD FLOPs
                    sgd_flops = self.batch_forward_flops * len(train_loader) if self.batch_forward_flops is not None else 0

                    # Calculate DMD FLOPs analytically
                    dmd_flops = self._calculate_dmd_flops_analytical()

                    epoch_flops = sgd_flops + dmd_flops

                    self._perform_dmd_multistep_prediction_and_update(weight_differences)
                else:
                    weights_before = flat_params_as_torch(self.model).clone()
                    self._train_normal(train_loader, loss_func)
                    weights_after = flat_params_as_torch(self.model).clone()
                    weight_differences = weights_after - weights_before  # Weight difference computed here

                    # Log gradient metrics if enabled
                    self._log_gradient_metrics(weight_differences)

                    self._perform_dmd_multistep_prediction_and_update(weight_differences)

                self.epoch_since_last_predict = 1
                # train normally after prediction
            else:
                # train normally (no DMD this epoch)
                # Calculate weight differences for gradient logging
                if self.log_gradient_metrics:
                    weights_before = flat_params_as_torch(self.model).clone()
                    self._train_normal(train_loader, loss_func)
                    weights_after = flat_params_as_torch(self.model).clone()
                    weight_differences = weights_after - weights_before
                    self._log_gradient_metrics(weight_differences)
                else:
                    self._train_normal(train_loader, loss_func)

                # Calculate FLOPs after training (so batch_forward_flops is set)
                if self.args.count_flops or self.args.count_flops_fast:
                    epoch_flops = self.batch_forward_flops * len(train_loader) if self.batch_forward_flops is not None else 0

                self.epoch_since_last_predict += 1
        else:
            # Before predict_start_epoch or not enough history
            # Calculate weight differences for gradient logging
            if self.log_gradient_metrics:
                weights_before = flat_params_as_torch(self.model).clone()
                self._train_normal(train_loader, loss_func)
                weights_after = flat_params_as_torch(self.model).clone()
                weight_differences = weights_after - weights_before
                self._log_gradient_metrics(weight_differences)
            else:
                self._train_normal(train_loader, loss_func)

            # Calculate FLOPs after training (so batch_forward_flops is set)
            if self.args.count_flops or self.args.count_flops_fast:
                epoch_flops = self.batch_forward_flops * len(train_loader) if self.batch_forward_flops is not None else 0

        if self.args.count_flops:
            self.epoch_flops.append(epoch_flops)
            self.total_flops += epoch_flops

            if self.use_wandb:
                wandb.log({
                    "epoch_flops": epoch_flops,
                    "total_flops": self.total_flops,
                }, step=self.cur_epoch)

        if self.args.count_flops_fast:
            self.epoch_flops.append(epoch_flops)
            self.total_flops += epoch_flops

            if self.use_wandb:
                wandb.log({
                    "flops_fast/epoch_flops": epoch_flops,
                    "flops_fast/total_flops": self.total_flops,
                }, step=self.cur_epoch)

        # Timing profiling: log PDT timing metrics
        if self.args.profile_timing:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            total_epoch_end = time.perf_counter()
            total_epoch_time_s = total_epoch_end - total_epoch_start

            if self.use_wandb:
                log_dict = {
                    "timing_pdt/total_epoch_time_s": total_epoch_time_s,
                    "timing_pdt/has_dmd": 1.0 if has_dmd_this_epoch else 0.0,
                }

                # Add SGD-only timing if available
                if hasattr(self, '_last_sgd_epoch_time'):
                    log_dict["timing_pdt/sgd_only_time_s"] = self._last_sgd_epoch_time
                    log_dict["timing_pdt/avg_sgd_batch_time_ms"] = self._last_sgd_batch_time

                    # Calculate DMD overhead if this epoch had DMD
                    if has_dmd_this_epoch:
                        dmd_overhead_s = total_epoch_time_s - self._last_sgd_epoch_time
                        log_dict["timing_pdt/dmd_overhead_s"] = dmd_overhead_s
                        log_dict["timing_pdt/dmd_overhead_ratio"] = dmd_overhead_s / total_epoch_time_s if total_epoch_time_s > 0 else 0

                wandb.log(log_dict, step=self.cur_epoch)


    def print_cuda_memory(self,title="Memory Check"):
        print(f"{title}:")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
        print(f"Allocated memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Cached memory: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    def _calculate_dmd_flops_analytical(self):
        """
        Calculate DMD FLOPs using analytical formulas (zero overhead).

        This is an alternative to torch.profiler for FLOPs counting.
        Theoretical FLOPs are very accurate for standard linear algebra operations like SVD.

        Returns:
            dmd_total_flops: Total FLOPs for DMD computation (SVD + prediction + masking)
        """
        N = sum(p.numel() for p in self.model.parameters())  # Total number of parameters
        h = self.n_past_weights if self.n_past_weights is not None else self.weights.shape[1]  # History length
        tau = self.predicted_num  # Prediction steps

        # SVD FLOPs (Golub-Reinsch algorithm for thin SVD)
        # For an N x h matrix, complexity is O(Nh^2 + h^3)
        svd_flops = 2 * N * h * h + 11 * h * h * h

        # DMD matrix computation (involves pseudo-inverse and matrix multiplication)
        # Pseudo-inverse: O(Nh^2), matrix multiply: O(Nh^2)
        dmd_matrix_flops = 2 * N * h * h

        # Multi-step prediction (matrix power operations)
        # For each prediction step, we do matrix-vector multiply: O(Nh)
        prediction_flops = N * h * tau

        # Mask computation (element-wise comparisons) - negligible
        mask_flops = 3 * N  # sign comparison + 2 magnitude comparisons

        dmd_total_flops = svd_flops + dmd_matrix_flops + prediction_flops + mask_flops

        # Log breakdown if wandb is enabled
        if self.use_wandb:
            wandb.log({
                "flops_fast/dmd_svd_flops": svd_flops,
                "flops_fast/dmd_matrix_flops": dmd_matrix_flops,
                "flops_fast/dmd_prediction_flops": prediction_flops,
                "flops_fast/dmd_total_flops": dmd_total_flops,
            }, step=self.cur_epoch)

        return dmd_total_flops

    def _log_gradient_metrics(self, weight_differences):
        """
        Log gradient norm and direction stability metrics.

        Args:
            weight_differences: Current epoch's weight update (gradient)
        """
        if not self.log_gradient_metrics or not self.use_wandb:
            return

        # Calculate gradient norm (L2 norm)
        gradient_norm = torch.norm(weight_differences).item()

        metrics = {
            "gradient/norm": gradient_norm,
        }

        # Calculate gradient direction stability (cosine similarity with previous epoch)
        if self.prev_gradient is not None:
            # Cosine similarity between current and previous gradient
            cos_sim = torch.nn.functional.cosine_similarity(
                weight_differences.view(1, -1),
                self.prev_gradient.view(1, -1),
                dim=1
            ).item()

            metrics["gradient/direction_stability"] = cos_sim

            # Also log the angle in degrees for easier interpretation
            angle_deg = torch.acos(torch.clamp(torch.tensor(cos_sim), -1.0, 1.0)) * 180 / 3.14159
            metrics["gradient/direction_angle_deg"] = angle_deg.item()

        # Log all metrics
        wandb.log(metrics, step=self.cur_epoch)

        # Update previous gradient for next epoch
        self.prev_gradient = weight_differences.clone().detach()

    def _perform_dmd_multistep_prediction_and_update(self,epoch_gradient):
        # Memory profiling: track memory before DMD
        if self.args.profile_memory:
            mem_before_dmd = torch.cuda.memory_allocated(self.device) / 1024**2 if torch.cuda.is_available() else 0
            mem_reserved_before = torch.cuda.memory_reserved(self.device) / 1024**2 if torch.cuda.is_available() else 0

        #self.print_cuda_memory("Before current weights capture")
        current_weights = flat_params_as_torch(self.model).clone()  # Get current weights after SGD
        #self.print_cuda_memory("After current weights capture")

        if self.weights is not None:
            # Check if the epoch_gradient has the correct dimension
            if epoch_gradient.dim() != 1 or epoch_gradient.size(0) != self.weights.size(0):
                raise ValueError("Incorrect dimension of epoch_gradient. Expected 1D tensor with the same size as the flattened model weights.")
            self.dmd = TorchDMD(rank=self.svd_rank)
            #self.dmd = TorchDMDFull(svd_rank=self.svd_rank, clear_snapshots=False)    # use TorchDMDFull for better reconstruction
            #self.dmd = DMD(rank=self.svd_rank)              # use the new torchDMD
            #self.dmd = RandomizedDMD(rank=self.svd_rank)       # use the RandomizedDMD
            if self.n_past_weights is not None:
                self.weights = self.weights[:, -self.n_past_weights:]

            # Memory profiling: weight snapshot storage
            if self.args.profile_memory:
                snapshot_mb = self.weights.numel() * self.weights.element_size() / 1024**2
                mem_before_svd = torch.cuda.memory_allocated(self.device) / 1024**2 if torch.cuda.is_available() else 0

            #self.print_cuda_memory("Before DMD fitting")
            self.weights.requires_grad = False

            # Timing profiling: SVD/DMD fit
            if self.args.profile_timing:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t_svd_start = time.perf_counter()

            self.dmd.fit(self.weights)

            if self.args.profile_timing:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t_svd_end = time.perf_counter()
                svd_time_ms = (t_svd_end - t_svd_start) * 1000

            # Memory profiling: SVD workspace
            if self.args.profile_memory:
                mem_after_svd = torch.cuda.memory_allocated(self.device) / 1024**2 if torch.cuda.is_available() else 0
                svd_workspace_mb = mem_after_svd - mem_before_svd

            # use the last weight as the initial condition to predict the future steps
            future_steps = self.predicted_num
            #self.print_cuda_memory("Before DMD prediction")

            # Timing profiling: Prediction
            if self.args.profile_timing:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t_pred_start = time.perf_counter()

            predicted_weights = self.dmd.predict_multistep(self.weights[:, -1].reshape(-1, 1), future_steps)

            if self.args.profile_timing:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t_pred_end = time.perf_counter()
                pred_time_ms = (t_pred_end - t_pred_start) * 1000
            #predicted_weights = self.dmd.predict_multistep_new(future_steps)   # use the predict_multistep_new function in TorchDMDFull
            #predicted_weights = self.dmd.predict(self.weights[:, -1].reshape(-1, 1), future_steps)   # use the predict function in new torchDMD and RandomizedDMD
            weight_diff = predicted_weights[:, -1].squeeze()-self.weights[:, -1].squeeze()

            # Ensure weight_diff and epoch_gradient are aligned in their dimensions
            if weight_diff.dim() == 1:
                weight_diff = weight_diff.view(-1)  # Flatten to ensure 1D tensor
            if epoch_gradient.dim() == 1:
                epoch_gradient = epoch_gradient.view(-1)  # Flatten to ensure 1D tensor

            #self.print_cuda_memory("After DMD prediction")

            # Timing profiling: Mask computation
            if self.args.profile_timing:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t_mask_start = time.perf_counter()

            # Create mask based on conditions and mask_mode
            if self.mask_mode == 'both':
                # Full PDT: Both dynamic consistency (Eq 7) and acceleration effectiveness (Eq 6)
                mask = (torch.sign(weight_diff) == torch.sign(epoch_gradient)) & \
                       (torch.abs(weight_diff) >= torch.abs(epoch_gradient)) & \
                       (torch.abs(weight_diff) <= (self.predicted_num+1) * torch.abs(epoch_gradient))
            elif self.mask_mode == 'accel_only':
                # Only acceleration effectiveness criterion (Eq 6: lower and upper bounds)
                mask = (torch.abs(weight_diff) >= torch.abs(epoch_gradient)) & \
                       (torch.abs(weight_diff) <= (self.predicted_num+1) * torch.abs(epoch_gradient))
            elif self.mask_mode == 'consistency_only':
                # Only dynamic consistency criterion (Eq 7: direction alignment)
                mask = (torch.sign(weight_diff) == torch.sign(epoch_gradient))
            else:
                raise ValueError(f"Unknown mask_mode: {self.mask_mode}. Must be 'both', 'accel_only', or 'consistency_only'")

            # Timing profiling: end of mask computation
            if self.args.profile_timing:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t_mask_end = time.perf_counter()
                mask_time_ms = (t_mask_end - t_mask_start) * 1000

            # save mask to files
            # add a folder "masks" in self.log_dir to save the masks
            mask_folder = os.path.join(self.log_dir, "masks")
            os.makedirs(mask_folder, exist_ok=True)
            mask_file = os.path.join(mask_folder, f"mask_epoch_{self.cur_epoch:04d}.pt")
            torch.save(mask, mask_file)
        
            
            # count the ture values in mask
            mask_count = torch.sum(mask)
            #print the true values in mask
            print("mask_count: ", mask_count)
            # calculate the ratio of mask_count to the total number of parameters
            mask_ratio = mask_count / len(mask)
            # calculate the ratio of other three conditions with respect to the total number of parameters
            ratio_opposite = torch.sum((torch.sign(weight_diff) != torch.sign(epoch_gradient))) / len(mask)
            ratio_n_plus = torch.sum((torch.sign(weight_diff) == torch.sign(epoch_gradient)) & (torch.abs(weight_diff) > (self.predicted_num+1) * torch.abs(epoch_gradient))) / len(mask)
            ratio_0_1 = torch.sum((torch.sign(weight_diff) == torch.sign(epoch_gradient)) & (torch.abs(weight_diff) < torch.abs(epoch_gradient))) / len(mask)

            if self.use_wandb:
                wandb.log({"mask_ratio_global": mask_ratio.item()}, step=self.cur_epoch)
                wandb.log({"ratio_opposite": ratio_opposite.item()}, step=self.cur_epoch)
                wandb.log({"ratio_n_plus": ratio_n_plus.item()}, step=self.cur_epoch)
                wandb.log({"ratio_0_1": ratio_0_1.item()}, step=self.cur_epoch)

            # Memory profiling: log detailed breakdown
            if self.args.profile_memory:
                mem_after_dmd = torch.cuda.memory_allocated(self.device) / 1024**2 if torch.cuda.is_available() else 0
                mem_reserved_after = torch.cuda.memory_reserved(self.device) / 1024**2 if torch.cuda.is_available() else 0
                peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024**2 if torch.cuda.is_available() else 0

                if self.use_wandb:
                    wandb.log({
                        "memory/snapshot_storage_mb": snapshot_mb,
                        "memory/svd_workspace_mb": svd_workspace_mb,
                        "memory/allocated_mb": mem_after_dmd,
                        "memory/reserved_mb": mem_reserved_after,
                        "memory/peak_mb": peak_mem,
                        "memory/dmd_overhead_mb": mem_after_dmd - mem_before_dmd,
                    }, step=self.cur_epoch)

            # Timing profiling: log detailed breakdown
            if self.args.profile_timing:
                total_dmd_time_ms = svd_time_ms + pred_time_ms + mask_time_ms

                if self.use_wandb:
                    wandb.log({
                        "timing/svd_ms": svd_time_ms,
                        "timing/prediction_ms": pred_time_ms,
                        "timing/masking_ms": mask_time_ms,
                        "timing/total_dmd_ms": total_dmd_time_ms,
                    }, step=self.cur_epoch)

            #print("mask shape:", mask.shape)
            #print("predicted_weights shape:", predicted_weights.shape)
            #print("current_weights shape:", current_weights.shape)

            # Apply the mask to the weight_diff to obtain the conditional weight difference
            updated_weights = torch.where(mask, predicted_weights.squeeze(), current_weights)          

            # update the model weights
            array_to_params(self.model,updated_weights,device=self.weights.device)

    def _train_normal(self, train_loader, loss_func):
        # Timing profiling: SGD epoch start
        if self.args.profile_timing:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            sgd_epoch_start = time.perf_counter()
            sgd_batch_times = []

        if self.weights is None:
            self.weights = flat_params_as_torch(self.model).reshape((-1, 1))
        else:
            self.weights = torch.cat(
                [self.weights, flat_params_as_torch(self.model).reshape((-1, 1))],
                dim=1
            )

        self.model.train()
        pbar = tqdm(
            total=len(train_loader),
            position=1,
            unit="batch",
            leave=False,
            desc="Epoch {}".format(self.cur_epoch)
        )
        losses = []

        # only calculate the FLOPs for the first batch
        do_profile = self.args.count_flops and self.batch_forward_flops is None
        do_profile_fast = self.args.count_flops_fast and self.batch_forward_flops is None

        for batch_idx, (data, target) in enumerate(train_loader):
            if do_profile and batch_idx == 0:
                # only profile the first batch
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_flops=True,
                    profile_memory=True
                ) as prof:
                    with record_function("complete_iteration"):
                        data, target = data.to(self.device), target.to(self.device)
                        self.optim.zero_grad()
                        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                            output = self.model(data)
                            loss = loss_func(output, target)
                        self.grad_scaler.scale(loss).backward()
                        self.grad_scaler.step(self.optim)
                        self.grad_scaler.update()

                print("\nComplete iteration profiling results:")
                print(prof.key_averages().table(sort_by="flops", row_limit=-1))

                self.batch_forward_flops = sum(e.flops for e in prof.key_averages())

                if self.use_wandb:
                    wandb.config.update({
                        "batch_forward_flops": self.batch_forward_flops,
                    }, allow_val_change=True)

            elif do_profile_fast and batch_idx == 0:
                # Profile first batch for count_flops_fast mode
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_flops=True,
                    profile_memory=True
                ) as prof:
                    with record_function("complete_iteration"):
                        data, target = data.to(self.device), target.to(self.device)
                        self.optim.zero_grad()
                        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                            output = self.model(data)
                            loss = loss_func(output, target)
                        self.grad_scaler.scale(loss).backward()
                        self.grad_scaler.step(self.optim)
                        self.grad_scaler.update()

                self.batch_forward_flops = sum(e.flops for e in prof.key_averages())

                if self.use_wandb:
                    wandb.config.update({
                        "batch_forward_flops": self.batch_forward_flops,
                    }, allow_val_change=True)

            else:
                # Timing profiling: batch start (measure first 100 batches)
                if self.args.profile_timing and batch_idx < 100:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    sgd_batch_start = time.perf_counter()

                data, target = data.to(self.device), target.to(self.device)
                self.optim.zero_grad()

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    output = self.model(data)
                    loss = loss_func(output, target)

                losses.append(loss.item())
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optim)
                self.grad_scaler.update()

                # Timing profiling: batch end
                if self.args.profile_timing and batch_idx < 100:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    sgd_batch_end = time.perf_counter()
                    sgd_batch_times.append((sgd_batch_end - sgd_batch_start) * 1000)

            if batch_idx % self.log_interval == 0:
                tqdm.write("Loss: {:.4f}".format(loss.item()))

            pbar.update(1)

        # Timing profiling: SGD epoch end
        if self.args.profile_timing:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            sgd_epoch_end = time.perf_counter()
            sgd_epoch_time_s = sgd_epoch_end - sgd_epoch_start
            avg_sgd_batch_time_ms = np.mean(sgd_batch_times) if len(sgd_batch_times) > 0 else 0

            # Store for later comparison in train_epoch
            self._last_sgd_epoch_time = sgd_epoch_time_s
            self._last_sgd_batch_time = avg_sgd_batch_time_ms

        # print the learning rate
        tqdm.write(f"Learning rate: {self.optim.param_groups[0]['lr']}")
        pbar.close()
        self.train_losses.append(losses)
        
        
    
        
        
        
