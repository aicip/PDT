import numpy as np
import torch
from tqdm import tqdm
import wandb
import time

from kcl.lib.dmd.torchDMD import TorchDMD
from kcl.lib.trainers.trainer_base import TrainerBase
from kcl.utils.weight_tools import array_to_params, flat_params_as_torch
from copy import deepcopy

class Random_accelerated(TrainerBase):
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
        
        if self.weights is not None and self.weights.shape[1] >= max(self.svd_rank + 1, self.predict_start_epoch):
            if self.first_predict:
                self.first_predict = False
                self.epoch_since_last_predict = self.predict_epoch_interval

            if self.epoch_since_last_predict >= self.predict_epoch_interval:
                weights_before = flat_params_as_torch(self.model).clone()
                self._train_normal(train_loader, loss_func)
                weights_after = flat_params_as_torch(self.model).clone()
                weight_differences = weights_after - weights_before  # Weight difference computed here
                self.random_acceleration(weight_differences)
                self.epoch_since_last_predict = 1
                # train normally after prediction                
            else:
                # train normally
                self._train_normal(train_loader, loss_func)
                self.epoch_since_last_predict += 1
        else:
            self._train_normal(train_loader, loss_func)
    
    def print_cuda_memory(self,title="Memory Check"):
        print(f"{title}:")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
        print(f"Allocated memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Cached memory: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")


    def random_acceleration(self,epoch_gradient):
        #self.print_cuda_memory("Before current weights capture")
        current_weights = flat_params_as_torch(self.model).clone()  # Get current weights after SGD
        #self.print_cuda_memory("After current weights capture")

        if self.weights is not None:
            if epoch_gradient.dim() == 1:
                epoch_gradient = epoch_gradient.view(-1)  # Flatten to ensure 1D tensor

            #self.print_cuda_memory("After DMD prediction")
            # Create mask of randomly selected 10% percentage of weights

            accelerated_weights = current_weights + (self.predicted_num-1)* epoch_gradient

            # Create a separate generator for the random mask
            mask_generator = torch.Generator(device=current_weights.device)  # Ensure the generator is on the same device
            mask_seed = 34237865  # Set this to any desired value, or generate it randomly for each run
            #mask_seed = int(time.time())  # Use the current time as the seed for the mask
            mask_generator.manual_seed(mask_seed)
            #random_mask = torch.rand(current_weights.size(), generator=mask_generator, dtype=torch.float32, device=current_weights.device) < 0.09699   # set the mask ratio
            random_mask = torch.rand(current_weights.size(), generator=mask_generator, dtype=torch.float32, device=current_weights.device) < 0.18   # set the mask ratio
            print("mask_seed: ", mask_seed)
            #print("mask_count: ", random_mask)

            # # Create a random mask selecting approximately 10% of the weights
            # random_mask = torch.rand(current_weights.size(), dtype=torch.float32, device=current_weights.device) < 0.1
            # count the ture values in mask
            mask_count = torch.sum(random_mask)
            #print the true values in mask
            print("mask_count: ", mask_count)
            #print("mask shape:", mask.shape)
            #print("predicted_weights shape:", predicted_weights.shape)
            #print("current_weights shape:", current_weights.shape)
            mask_ratio = mask_count / len(random_mask)

            if self.use_wandb:
                wandb.log({"mask_ratio_global": mask_ratio.item()}, step=self.cur_epoch)
                wandb.log({"mask_seed": mask_seed}, step=self.cur_epoch)

            # Apply the random mask to selectively update weights
            updated_weights = torch.where(random_mask,accelerated_weights, current_weights)
       

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
        for batch_idx, (data, target) in enumerate(train_loader):
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
        
        pbar.close()
        self.train_losses.append(losses)
        
        
    
        
        
        
