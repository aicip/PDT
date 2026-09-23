import numpy as np
import torch
from tqdm import tqdm
import wandb
import os

#from kcl.lib.acceleration import accelerate_epoch_dynamics
from kcl.lib.dmd.torchDMD import TorchDMD
from kcl.lib.dmd.fullTorchDMD.torchDMD import TorchDMDFull
from kcl.lib.trainers.trainer_base import TrainerBase
from kcl.lib.dmd.torchdmd.methods.torchDMD import DMD
from kcl.lib.dmd.torchdmd.methods.randomDMD import RandomizedDMD
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
#        print("self.if_sp: ", self.if_sp)
#        raise Exception("Stopping the program for inspection")


        #self.predicting = False
        self.first_predict = True

    def train_epoch(self, train_loader, loss_func):
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

        if self.args.count_flops:
            epoch_flops = self.batch_forward_flops * len(train_loader) if self.batch_forward_flops is not None else 0

        if self.weights is not None and self.weights.shape[1] >= max(self.svd_rank + 1, self.predict_start_epoch):
            if self.first_predict:
                self.first_predict = False
                self.epoch_since_last_predict = self.predict_epoch_interval

            if self.epoch_since_last_predict >= self.predict_epoch_interval:
                # Profile DMD computation if needed
                if self.args.count_flops:
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
                            self._perform_dmd_multistep_prediction_and_update(weight_differences)
                    
                    print("\nDMD computation profiling results:")
                    print(prof.key_averages().table(sort_by="flops", row_limit=-1))
                    
                    dmd_flops = sum(e.flops for e in prof.key_averages())
                    epoch_flops += dmd_flops
                else:
                    weights_before = flat_params_as_torch(self.model).clone()
                    self._train_normal(train_loader, loss_func)
                    weights_after = flat_params_as_torch(self.model).clone()
                    weight_differences = weights_after - weights_before  # Weight difference computed here
                    self._perform_dmd_multistep_prediction_and_update(weight_differences)

                self.epoch_since_last_predict = 1
                # train normally after prediction                
            else:
                # train normally
                self._train_normal(train_loader, loss_func)
                self.epoch_since_last_predict += 1
        else:
            self._train_normal(train_loader, loss_func)

        if self.args.count_flops:
            self.epoch_flops.append(epoch_flops)
            self.total_flops += epoch_flops
            
            if self.use_wandb:
                wandb.log({
                    "epoch_flops": epoch_flops,
                    "total_flops": self.total_flops,
                }, step=self.cur_epoch)

    
    def print_cuda_memory(self,title="Memory Check"):
        print(f"{title}:")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
        print(f"Allocated memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Cached memory: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")


    def _perform_dmd_multistep_prediction_and_update(self,epoch_gradient):
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

            #self.print_cuda_memory("Before DMD fitting")
            self.weights.requires_grad = False
            self.dmd.fit(self.weights)  

            # use the last weight as the initial condition to predict the future steps
            future_steps = self.predicted_num
            #self.print_cuda_memory("Before DMD prediction")
            predicted_weights = self.dmd.predict_multistep(self.weights[:, -1].reshape(-1, 1), future_steps)
            #predicted_weights = self.dmd.predict_multistep_new(future_steps)   # use the predict_multistep_new function in TorchDMDFull
            #predicted_weights = self.dmd.predict(self.weights[:, -1].reshape(-1, 1), future_steps)   # use the predict function in new torchDMD and RandomizedDMD
            weight_diff = predicted_weights[:, -1].squeeze()-self.weights[:, -1].squeeze()

            # Ensure weight_diff and epoch_gradient are aligned in their dimensions
            if weight_diff.dim() == 1:
                weight_diff = weight_diff.view(-1)  # Flatten to ensure 1D tensor
            if epoch_gradient.dim() == 1:
                epoch_gradient = epoch_gradient.view(-1)  # Flatten to ensure 1D tensor

            #self.print_cuda_memory("After DMD prediction")
            # Create mask based on conditions
            mask = (torch.sign(weight_diff) == torch.sign(epoch_gradient)) & (torch.abs(weight_diff) <= (self.predicted_num+1) * torch.abs(epoch_gradient))& (torch.abs(weight_diff) >= torch.abs(epoch_gradient))
            
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

            #print("mask shape:", mask.shape)
            #print("predicted_weights shape:", predicted_weights.shape)
            #print("current_weights shape:", current_weights.shape)

            # Apply the mask to the weight_diff to obtain the conditional weight difference
            updated_weights = torch.where(mask, predicted_weights.squeeze(), current_weights)          

            # update the model weights
            array_to_params(self.model,updated_weights,device=self.weights.device)

    def _train_normal(self, train_loader, loss_func):
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
                        output = self.model(data)
                        loss = loss_func(output, target)
                        loss.backward()
                        self.optim.step()
                
                print("\nComplete iteration profiling results:")
                print(prof.key_averages().table(sort_by="flops", row_limit=-1))
                
                self.batch_forward_flops = sum(e.flops for e in prof.key_averages())
                
                if self.use_wandb:
                    wandb.config.update({
                        "batch_forward_flops": self.batch_forward_flops,
                    }, allow_val_change=True)
            
            else:
                self.optim.zero_grad()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = loss_func(output, target)
                losses.append(loss.item())
                loss.backward()
                self.optim.step()

            if batch_idx % self.log_interval == 0:
                tqdm.write("Loss: {:.4f}".format(loss.item()))
            
            pbar.update(1)
        # print the learning rate
        tqdm.write(f"Learning rate: {self.optim.param_groups[0]['lr']}")
        pbar.close()
        self.train_losses.append(losses)
        
        
    
        
        
        
