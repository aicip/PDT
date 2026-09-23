import numpy as np
import torch
import torch.distributed
from tqdm import tqdm
import wandb
import os
import time

from torch.profiler import profile, record_function, ProfilerActivity

#from kcl.lib.acceleration import accelerate_epoch_dynamics
from kcl.lib.dmd.torchDMD import TorchDMD
from kcl.lib.dmd.fullTorchDMD.torchDMD import TorchDMDFull
from kcl.lib.trainers.trainer_base import TrainerBase
from kcl.lib.dmd.torchdmd.methods.torchDMD import DMD
from kcl.lib.dmd.torchdmd.methods.randomDMD import RandomizedDMD
from kcl.utils.weight_tools import array_to_params, flat_params_as_torch
from kcl.utils.misc import unpack_batch
from copy import deepcopy

class PredictedTrainer_conf_adaptive(TrainerBase):
    def __init__(
        self, 
        *, 
        svd_rank=10,  
        n_past_weights=None,
        predict_start_epoch = 5,
        predicted_num = 5,
        predict_epoch_interval = 1,
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
        """
        self.svd_rank = svd_rank
        self.n_past_weights = n_past_weights
        self.predicted_num = predicted_num
        self.predict_start_epoch = predict_start_epoch
        self.predict_epoch_interval = predict_epoch_interval   # the inerval epochs to make prediction
        self.epoch_since_last_predict = 0  # the number of epochs since last prediction
        self.if_sp = if_sp
        self.if_LV = if_LV
        self.threshold = 0.01
        self.args = kwargs.get('args')
        self.orig_device = torch.device('cuda')
#        print("self.if_sp: ", self.if_sp)
#        raise Exception("Stopping the program for inspection")


        #self.predicting = False
        self.first_predict = True
        self.prev_loss = None  # To store the previous epoch loss
        self.prev_prev_loss = None  # To store the epoch before the previous loss

        if self.args.distributed:
            local_rank = self.args.dist_rank % 3  # only use the first 3 GPUs
            self.orig_device = torch.device(f'cuda:{local_rank}')
        self.storage_device = self.args.w_device
        if self.args.dist_rank_0:
            print(f"Rank {self.args.dist_rank} - Training device: {self.orig_device}, Weight storage device: {self.storage_device}")

        # Profiling initialization
        if self.args.count_flops_fast:
            self.epoch_flops = []
            self.total_flops = 0
            self.batch_forward_flops = None

    def train_epoch(self, train_loader, loss_func):
        # print("self.n_past_weights: ", self.n_past_weights)
        # print("self.predict_start_epoch: ", self.predict_start_epoch)
        # print("self.predicted_num: ", self.predicted_num)
        # print("self.predict_epoch_interval: ", self.predict_epoch_interval)

        # Initialize epoch_flops for tracking
        epoch_flops = 0

        # Timing profiling: epoch start (only on rank 0 to avoid sync overhead on other GPUs)
        if self.args.profile_timing and self.args.dist_rank_0:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            total_epoch_start = time.perf_counter()
            has_dmd_this_epoch = False

        if self.args.distributed and torch.distributed.is_initialized():
            if self.args.dist_rank == 0:
                if self.weights is not None:
                    wshape = torch.tensor(self.weights.shape[1], device=torch.device('cuda', self.args.dist_rank % 3))
                else:
                    wshape = torch.tensor(0, device=torch.device('cuda', self.args.dist_rank % 3))
                torch.distributed.broadcast(wshape, 0)
            else:
                #wshape = torch.tensor(0, device=torch.device('cuda', self.args.dist_rank))
                #local_rank = self.args.dist_rank % torch.cuda.device_count()
                local_rank = self.args.dist_rank % 3
                wshape = torch.tensor(0, device=torch.device('cuda', local_rank))
                torch.distributed.broadcast(wshape, 0)
        else:
            wshape = torch.tensor(self.weights.shape[1] if self.weights is not None else 0)
        
        if wshape.item() >= max(self.svd_rank + 1, self.predict_start_epoch):
            if self.first_predict:
                self.first_predict = False
                self.epoch_since_last_predict = self.predict_epoch_interval

            if self.epoch_since_last_predict >= self.predict_epoch_interval:
                # Mark that this epoch contains DMD (only on rank 0 where timing is tracked)
                if self.args.profile_timing and self.args.dist_rank_0:
                    has_dmd_this_epoch = True

                if self.args.count_flops_fast:
                    # Calculate SGD + DMD FLOPs
                    weights_before = flat_params_as_torch(self.model).clone().to(self.args.w_device)
                    self._train_normal(train_loader, loss_func)
                    weights_after = flat_params_as_torch(self.model).clone().to(self.args.w_device)
                    weight_differences = weights_after - weights_before

                    # SGD FLOPs
                    sgd_flops = self.batch_forward_flops * len(train_loader) if self.batch_forward_flops is not None else 0

                    # DMD FLOPs (analytical)
                    dmd_flops = self._calculate_dmd_flops_analytical()

                    epoch_flops = sgd_flops + dmd_flops

                    self._perform_dmd_multistep_prediction_and_update(weight_differences)
                else:
                    weights_before = flat_params_as_torch(self.model).clone().to(self.args.w_device)
                    self._train_normal(train_loader, loss_func)
                    weights_after = flat_params_as_torch(self.model).clone().to(self.args.w_device)
                    weight_differences = weights_after - weights_before
                    self._perform_dmd_multistep_prediction_and_update(weight_differences)

                self.epoch_since_last_predict = 1
                # train normally after prediction
            else:
                # train normally (no DMD this epoch)
                self._train_normal(train_loader, loss_func)

                if self.args.count_flops_fast:
                    epoch_flops = self.batch_forward_flops * len(train_loader) if self.batch_forward_flops is not None else 0

                self.epoch_since_last_predict += 1
        else:
            # Before predict_start_epoch or not enough history
            self._train_normal(train_loader, loss_func)

            if self.args.count_flops_fast:
                epoch_flops = self.batch_forward_flops * len(train_loader) if self.batch_forward_flops is not None else 0

        # Log FLOPs
        if self.args.count_flops_fast:
            self.epoch_flops.append(epoch_flops)
            self.total_flops += epoch_flops

            if self.use_wandb:
                wandb.log({
                    "flops_fast/epoch_flops": epoch_flops,
                    "flops_fast/total_flops": self.total_flops,
                }, step=self.cur_epoch)

        # Timing profiling: log PDT timing metrics (only on rank 0, must match epoch start check)
        if self.args.profile_timing and self.args.dist_rank_0:
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
                        log_dict["timing_pdt/dmd_overhead_ratio"] = dmd_overhead_s / total_epoch_time_s

                wandb.log(log_dict, step=self.cur_epoch)

        # Adjust parameters based on loss
        self.adjust_parameters_based_on_loss()

    def adjust_parameters_based_on_loss(self):
        if len(self.train_losses) < 2:
            return
        
        current_loss = np.mean(self.train_losses[-1])
        if self.prev_loss is None:
            self.prev_loss = current_loss
            return
        if self.prev_prev_loss is None:
            self.prev_prev_loss = self.prev_loss
            self.prev_loss = current_loss
            return

        if current_loss > self.prev_loss:
            if current_loss > self.prev_prev_loss:
                if self.predicted_num > 4:
                    self.predicted_num -= 2
                    print(f"Loss increased a lot. Decreasing predicted_num to {self.predicted_num}.")
                elif self.predicted_num == 4:
                    self.predicted_num -= 1
                    self.predict_epoch_interval += 1
                    print(f"Loss increased a lot. Decreasing predicted_num to {self.predicted_num}.")
                else:
                    self.predict_epoch_interval += 2
                    print(f"Loss increased a lot. Increasing predict_epoch_interval to {self.predict_epoch_interval}.")
            
            else:    
                if self.predicted_num > 3:
                    self.predicted_num -= 1
                    print(f"Loss increased. Decreasing predicted_num to {self.predicted_num}.")
                else:
                    self.predict_epoch_interval += 1
                    print(f"Loss increased. Increasing predict_epoch_interval to {self.predict_epoch_interval}.")
        self.prev_prev_loss = self.prev_loss
        self.prev_loss = current_loss
    
    def print_cuda_memory(self,title="Memory Check"):
        print(f"{title}:")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
        print(f"Allocated memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Cached memory: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    def _calculate_dmd_flops_analytical(self):
        """Calculate DMD FLOPs using analytical formulas (zero overhead)."""
        N = sum(p.numel() for p in self.model.parameters())
        h = self.n_past_weights if self.n_past_weights is not None else self.weights.shape[1]
        tau = self.predicted_num

        # SVD FLOPs (Golub-Reinsch algorithm): 2*N*h^2 + 11*h^3
        svd_flops = 2 * N * h * h + 11 * h * h * h

        # DMD matrix computation: 2*N*h^2
        dmd_matrix_flops = 2 * N * h * h

        # Multi-step prediction: N*h*tau
        prediction_flops = N * h * tau

        # Mask computation (element-wise comparisons): 3*N
        mask_flops = 3 * N

        dmd_total_flops = svd_flops + dmd_matrix_flops + prediction_flops + mask_flops

        if self.use_wandb:
            wandb.log({
                "flops_fast/dmd_svd_flops": svd_flops,
                "flops_fast/dmd_matrix_flops": dmd_matrix_flops,
                "flops_fast/dmd_prediction_flops": prediction_flops,
                "flops_fast/dmd_total_flops": dmd_total_flops,
            }, step=self.cur_epoch)

        return dmd_total_flops

    def _perform_dmd_multistep_prediction_and_update(self,epoch_gradient):
        if self.args.distributed and torch.distributed.is_initialized():
            if self.args.dist_rank != 0:
                #cur_device = torch.device('cuda', self.args.dist_rank)
                #local_rank = self.args.dist_rank % torch.cuda.device_count()
                local_rank = self.args.dist_rank % 3
                cur_device = torch.device('cuda', local_rank)
                # this just needs to be the same size as the model weights. It
                # will get overwritten
                current_weights = flat_params_as_torch(self.model).to(cur_device)

                #print('\n\nrecv from rank 0\n\n')
                torch.distributed.broadcast(current_weights, 0)

                array_to_params(self.model,current_weights,device=cur_device)
                return


        #self.print_cuda_memory("Before current weights capture")
        current_weights = flat_params_as_torch(self.model).clone().to(self.args.w_device)  # Get current weights after SGD
        updated_weights = None
        #self.print_cuda_memory("After current weights capture")

        # Memory profiling: track memory before DMD
        if self.args.profile_memory:
            mem_before_dmd = torch.cuda.memory_allocated(self.device) / 1024**2 if torch.cuda.is_available() else 0
            snapshot_mb = self.weights.numel() * self.weights.element_size() / 1024**2 if self.weights is not None else 0

        if self.weights is not None:
            # Check if the epoch_gradient has the correct dimension
            if epoch_gradient.dim() != 1 or epoch_gradient.size(0) != self.weights.size(0):
                raise ValueError("Incorrect dimension of epoch_gradient. Expected 1D tensor with the same size as the flattened model weights.")
            self.dmd = TorchDMD(rank=self.svd_rank)   # use the old torchDMD
            #self.dmd = TorchDMDFull(svd_rank=self.svd_rank, clear_snapshots=False)    # use TorchDMDFull for better reconstruction
            #self.dmd = DMD(rank=self.svd_rank)              # use the new torchDMD
            #self.dmd = RandomizedDMD(rank=self.svd_rank)       # use the RandomizedDMD
            if self.n_past_weights is not None:
                self.weights = self.weights[:, -self.n_past_weights:]

            #self.print_cuda_memory("Before DMD fitting")

            # Memory profiling: before SVD
            if self.args.profile_memory:
                mem_before_svd = torch.cuda.memory_allocated(self.device) / 1024**2 if torch.cuda.is_available() else 0

            # Timing profiling: SVD start
            if self.args.profile_timing:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t_svd_start = time.perf_counter()

            self.weights.requires_grad = False
            self.dmd.fit(self.weights)

            # Timing profiling: SVD end
            if self.args.profile_timing:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                svd_time_ms = (time.perf_counter() - t_svd_start) * 1000

            # Memory profiling: after SVD
            if self.args.profile_memory:
                mem_after_svd = torch.cuda.memory_allocated(self.device) / 1024**2 if torch.cuda.is_available() else 0
                svd_workspace_mb = mem_after_svd - mem_before_svd

            # use the last weight as the initial condition to predict the future steps
            future_steps = self.predicted_num
            #self.print_cuda_memory("Before DMD prediction")

            # Timing profiling: prediction start
            if self.args.profile_timing:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t_pred_start = time.perf_counter()

            predicted_weights = self.dmd.predict_multistep(self.weights[:, -1].reshape(-1, 1), future_steps)    # use the predict_multistep function in old torchDMD

            # Timing profiling: prediction end
            if self.args.profile_timing:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                pred_time_ms = (time.perf_counter() - t_pred_start) * 1000
            #predicted_weights = self.dmd.predict_multistep_new(future_steps)   # use the predict_multistep_new function in TorchDMDFull
            #predicted_weights = self.dmd.predict(self.weights[:, -1].reshape(-1, 1), future_steps)   # use the predict function in new torchDMD and RandomizedDMD
            weight_diff = predicted_weights[:, -1].squeeze()-self.weights[:, -1].squeeze()

            # Ensure weight_diff and epoch_gradient are aligned in their dimensions
            if weight_diff.dim() == 1:
                weight_diff = weight_diff.view(-1)  # Flatten to ensure 1D tensor
            if epoch_gradient.dim() == 1:
                epoch_gradient = epoch_gradient.view(-1)  # Flatten to ensure 1D tensor

            #self.print_cuda_memory("After DMD prediction")

            # Timing profiling: mask computation start
            if self.args.profile_timing:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t_mask_start = time.perf_counter()

            # Create mask based on conditions
            mask = (torch.sign(weight_diff) == torch.sign(epoch_gradient)) & (torch.abs(weight_diff) <= (self.predicted_num+1) * torch.abs(epoch_gradient))& (torch.abs(weight_diff) >= torch.abs(epoch_gradient))
            #mask = (torch.sign(weight_diff) == torch.sign(epoch_gradient)) & (torch.abs(weight_diff) <= (self.predicted_num*2) * torch.abs(epoch_gradient))& (torch.abs(weight_diff) >= torch.abs(epoch_gradient))

            # Timing profiling: mask computation end
            if self.args.profile_timing:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                mask_time_ms = (time.perf_counter() - t_mask_start) * 1000
                total_dmd_time_ms = svd_time_ms + pred_time_ms + mask_time_ms

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

            # Log timing breakdown
            if self.args.profile_timing:
                if self.use_wandb:
                    wandb.log({
                        "timing/svd_ms": svd_time_ms,
                        "timing/prediction_ms": pred_time_ms,
                        "timing/masking_ms": mask_time_ms,
                        "timing/total_dmd_ms": total_dmd_time_ms,
                    }, step=self.cur_epoch)

            # Log memory breakdown
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

            #print("mask shape:", mask.shape)
            #print("predicted_weights shape:", predicted_weights.shape)
            #print("current_weights shape:", current_weights.shape)

            # Apply the mask to the weight_diff to obtain the conditional weight difference
            updated_weights = torch.where(mask, predicted_weights.squeeze(), current_weights)  

            # update the model weights
            array_to_params(self.model,updated_weights,device=self.orig_device)
        
        # Only broadcast in distributed training (multi-GPU)
        if self.args.dist_rank_0 and torch.distributed.is_initialized():
            #print('\n\nsend from rank 0\n\n')
            if updated_weights is None:
                torch.distributed.broadcast(current_weights.to(self.orig_device), 0)
                # for i in range(1, self.args.ws):
                #     torch.distributed.broadcast(current_weights.to(self.orig_device), 0)
            else:
                torch.distributed.broadcast(updated_weights.to(self.orig_device), 0)
                # for i in range(1, self.args.ws):
                #     torch.distributed.broadcast(updated_weights.to(self.orig_device), 0)


    def _train_normal(self, train_loader, loss_func):
        if self.args.dist_rank_0:
            if self.weights is None:
                self.weights = flat_params_as_torch(self.model).reshape((-1, 1)).to(self.args.w_device)
            else:
                self.weights = torch.cat(
                    [self.weights, flat_params_as_torch(self.model).reshape((-1, 1)).to(self.args.w_device)],
                    dim=1
                )

        # Timing profiling: SGD epoch start (only on rank 0 to avoid sync overhead on other GPUs)
        if self.args.profile_timing and self.args.dist_rank_0:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            sgd_epoch_start = time.perf_counter()
            sgd_batch_times = []

        # FLOPs profiling: profile first batch (only on rank 0 to avoid overhead on other GPUs)
        do_profile_fast = self.args.count_flops_fast and self.batch_forward_flops is None and self.args.dist_rank_0

        self.model.train()
        pbar = tqdm(
            total=len(train_loader),
            position=1,
            unit="batch",
            leave=False,
            desc="Epoch {}".format(self.cur_epoch),
            disable=(not self.args.dist_rank_0)
        )
        losses = []
        for batch_idx, batch in enumerate(train_loader):
            if do_profile_fast and batch_idx == 0:
                # Profile first batch for FLOPs
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_flops=True,
                    profile_memory=True
                ) as prof:
                    with record_function("complete_iteration"):
                        data, target = unpack_batch(batch, device=self.device)
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
                # Normal training with timing (only on rank 0)
                if self.args.profile_timing and self.args.dist_rank_0 and batch_idx < 100:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    batch_start = time.perf_counter()

                data, target = unpack_batch(batch, device=self.device)
                self.optim.zero_grad()
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    output = self.model(data)
                    loss = loss_func(output, target)

                if self.args.profile_timing and self.args.dist_rank_0 and batch_idx < 100:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    sgd_batch_times.append((time.perf_counter() - batch_start) * 1000)

                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optim)
                self.grad_scaler.update()

            if batch_idx % self.log_interval == 0 and self.args.dist_rank_0:
                tqdm.write("Loss: {:.4f}".format(loss.item()))
            losses.append(loss.item())
            pbar.update(1)
        # print the learning rate
        tqdm.write(f"Learning rate: {self.optim.param_groups[0]['lr']}")
        #self.scheduler.step()
        pbar.close()
        self.train_losses.append(losses)

        # Timing profiling: SGD epoch end (only on rank 0)
        if self.args.profile_timing and self.args.dist_rank_0:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            sgd_epoch_time_s = time.perf_counter() - sgd_epoch_start
            avg_sgd_batch_time_ms = np.mean(sgd_batch_times) if len(sgd_batch_times) > 0 else 0

            # Store for later use in train_epoch
            self._last_sgd_epoch_time = sgd_epoch_time_s
            self._last_sgd_batch_time = avg_sgd_batch_time_ms
