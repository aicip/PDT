import numpy as np
import torch
import torch.distributed
from tqdm import tqdm
import wandb

from kcl.lib.dmd.torchDMD import TorchDMD
from kcl.lib.trainers.trainer_base import TrainerBase
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

    def train_epoch(self, train_loader, loss_func):
        # print("self.n_past_weights: ", self.n_past_weights)
        # print("self.predict_start_epoch: ", self.predict_start_epoch)
        # print("self.predicted_num: ", self.predicted_num)
        # print("self.predict_epoch_interval: ", self.predict_epoch_interval)

        if self.args.distributed:
            if self.args.dist_rank == 0:
                if self.weights is not None:
                    wshape = torch.tensor(self.weights.shape[1], device=torch.device('cuda', self.args.dist_rank))
                else:
                    wshape = torch.tensor(0, device=torch.device('cuda', self.args.dist_rank))
                torch.distributed.broadcast(wshape, 0)
            else:
                wshape = torch.tensor(0, device=torch.device('cuda', self.args.dist_rank))
                torch.distributed.broadcast(wshape, 0)
        else:
            wshape = torch.tensor(self.weights.shape[1] if self.weights is not None else 0)
        
        if wshape.item() >= max(self.svd_rank + 1, self.predict_start_epoch):
            if self.first_predict:
                self.first_predict = False
                self.epoch_since_last_predict = self.predict_epoch_interval

            if self.epoch_since_last_predict >= self.predict_epoch_interval:
                weights_before = flat_params_as_torch(self.model).clone().to(self.args.w_device)
                self._train_normal(train_loader, loss_func)
                weights_after = flat_params_as_torch(self.model).clone().to(self.args.w_device)
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


    def _perform_dmd_multistep_prediction_and_update(self,epoch_gradient):
        if self.args.distributed:
            if self.args.dist_rank != 0:
                cur_device = torch.device('cuda', self.args.dist_rank)
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
            self.weights.requires_grad = False
            self.dmd.fit(self.weights)  

            # use the last weight as the initial condition to predict the future steps
            future_steps = self.predicted_num
            #self.print_cuda_memory("Before DMD prediction")
            predicted_weights = self.dmd.predict_multistep(self.weights[:, -1].reshape(-1, 1), future_steps)    # use the predict_multistep function in old torchDMD
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
            #mask = (torch.sign(weight_diff) == torch.sign(epoch_gradient)) & (torch.abs(weight_diff) <= (self.predicted_num*2) * torch.abs(epoch_gradient))& (torch.abs(weight_diff) >= torch.abs(epoch_gradient))
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
            array_to_params(self.model,updated_weights,device=self.orig_device)
        
        if self.args.dist_rank_0:
            #print('\n\nsend from rank 0\n\n')
            if updated_weights is None:
                torch.distributed.broadcast(current_weights.to(self.orig_device), 0)
            else:
                torch.distributed.broadcast(updated_weights.to(self.orig_device), 0)


    def _train_normal(self, train_loader, loss_func):
        if self.args.dist_rank_0:
            if self.weights is None:
                self.weights = flat_params_as_torch(self.model).reshape((-1, 1)).to(self.args.w_device)
            else:
                self.weights = torch.cat(
                    [self.weights, flat_params_as_torch(self.model).reshape((-1, 1)).to(self.args.w_device)],
                    dim=1
                )

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
            data, target = unpack_batch(batch,device=self.device) 
            self.optim.zero_grad()
            # output = self.model(data)
            # loss = loss_func(output, target)
            with torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16
            ):
                output = self.model(data)
                loss = loss_func(output, target)
            # losses.append(loss.item())
            # loss.backward()
            # self.optim.step()
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
