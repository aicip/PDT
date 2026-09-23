import numpy as np
import torch
import torch.distributed
from tqdm import tqdm
import wandb
import os
from collections import OrderedDict


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
        use_layerwise_dmd=True, # use layerwise DMD
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
        self.use_layerwise_dmd = use_layerwise_dmd # use layerwise DMD
#        print("self.if_sp: ", self.if_sp)
#        raise Exception("Stopping the program for inspection")


        #self.predicting = False
        self.first_predict = True
        self.prev_loss = None  # To store the previous epoch loss
        self.prev_prev_loss = None  # To store the epoch before the previous loss
        
        self.layer_weights = {}  # Dictionary to store weights for each layer
        self.dmd_failures = 0
        self.consecutive_failures = 0

        print(f"Layerwise DMD enabled: {self.use_layerwise_dmd}")  
        
        if self.args.distributed:
            local_rank = self.args.dist_rank % 3  # only use the first 3 GPUs
            self.orig_device = torch.device(f'cuda:{local_rank}')
        self.storage_device = self.args.w_device
        if self.args.dist_rank_0:
            print(f"Rank {self.args.dist_rank} - Training device: {self.orig_device}, Weight storage device: {self.storage_device}")

    def _extract_layer_weights(self, flat_weights):
        """
        Split the ViT parameters into fine-grained layer groups with memory management
        """
        if not hasattr(self, 'param_indices'):
            self.param_indices = OrderedDict()
            self.param_shapes = OrderedDict()
            idx = 0
            
            # Build fine-grained layer keys for ViT
            for name, param in self.model.named_parameters():
                # Create a finer layer key for ViT blocks
                if 'encoder.layers' in name:
                    # Extract block index and component
                    parts = name.split('.')
                    if len(parts) >= 4:  # format: encoder.layers.0.ln_1
                        block_idx = parts[2]
                        component = parts[3]  # ln_1, ln_2, self_attention, mlp, ...
                        layer_key = f"encoder.block{block_idx}.{component}"
                    else:
                        layer_key = f"encoder.block{parts[2]}"
                else:
                    layer_key = name.split('.')[0]
                
                if layer_key not in self.param_indices:
                    self.param_indices[layer_key] = []
                    self.param_shapes[layer_key] = []
                
                param_size = param.numel()
                self.param_indices[layer_key].append((idx, idx + param_size))
                self.param_shapes[layer_key].append(param.shape)
                idx += param_size
            
            print(f"Model divided into {len(self.param_indices)} layer groups for DMD")
            
            # Print the parameter count of each layer group (debug)
            if self.args.dist_rank_0:
                for layer_key, indices in self.param_indices.items():
                    total_params = sum(end - start for start, end in indices)
                    print(f"  Layer group {layer_key}: {total_params / 1e6:.2f}M parameters")
        
        # Process layer groups in small batches to save memory
        layer_weights = {}
        batch_size = 2  # 2 layer groups at a time
        layer_keys = list(self.param_indices.keys())
        
        for i in range(0, len(layer_keys), batch_size):
            batch_keys = layer_keys[i:i+batch_size]
            for layer_key in batch_keys:
                indices = self.param_indices[layer_key]
                try:
                    # Collect layer parameters in small batches to save memory
                    layer_params = []
                    for start_idx, end_idx in indices:
                        layer_params.append(flat_weights[start_idx:end_idx])
                    
                    # Concatenate only if all parameters were collected
                    if layer_params:
                        try:
                            layer_weights[layer_key] = torch.cat(layer_params)
                        except RuntimeError as e:
                            print(f"Error concatenating layer {layer_key}: {e}")
                            # Fall back to concatenating on CPU
                            try:
                                cpu_params = [p.cpu() for p in layer_params]
                                layer_weights[layer_key] = torch.cat(cpu_params).to(self.args.w_device)
                                del cpu_params  # free CPU memory
                            except:
                                print(f"Skipping layer {layer_key} due to memory constraints")
                except Exception as e:
                    print(f"Error processing layer {layer_key}: {e}")
            
            # Free memory after each batch
            torch.cuda.empty_cache()
    
        return layer_weights
        
    def _reconstruct_flat_weights(self, layer_predictions):
        """
        Reconstruct full flattened weight vector from layer-wise predictions
        """
        # Create zero vector of same size as original weights
        total_params = sum(param.numel() for param in self.model.parameters())
        reconstructed = torch.zeros(total_params, device=self.orig_device)
        
        # Fill each layer's prediction into appropriate position
        for layer_key, layer_pred in layer_predictions.items():
            if layer_key not in self.param_indices:
                continue
                
            param_idx = 0
            for start_idx, end_idx in self.param_indices[layer_key]:
                param_size = end_idx - start_idx
                reconstructed[start_idx:end_idx] = layer_pred[param_idx:param_idx+param_size]
                param_idx += param_size
                
        return reconstructed
    

    # def train_epoch(self, train_loader, loss_func):
    #     # print("self.n_past_weights: ", self.n_past_weights)
    #     # print("self.predict_start_epoch: ", self.predict_start_epoch)
    #     # print("self.predicted_num: ", self.predicted_num)
    #     # print("self.predict_epoch_interval: ", self.predict_epoch_interval)

    #     if self.args.distributed:
    #         if self.args.dist_rank == 0:
    #             if self.weights is not None:
    #                 wshape = torch.tensor(self.weights.shape[1], device=torch.device('cuda', self.args.dist_rank))
    #             else:
    #                 wshape = torch.tensor(0, device=torch.device('cuda', self.args.dist_rank))
    #             torch.distributed.broadcast(wshape, 0)
    #         else:
    #             #wshape = torch.tensor(0, device=torch.device('cuda', self.args.dist_rank))
    #             local_rank = self.args.dist_rank % torch.cuda.device_count()
    #             wshape = torch.tensor(0, device=torch.device('cuda', local_rank))
    #             torch.distributed.broadcast(wshape, 0)
    #     else:
    #         wshape = torch.tensor(self.weights.shape[1] if self.weights is not None else 0)
        
    #     if wshape.item() >= max(self.svd_rank + 1, self.predict_start_epoch):
    #         if self.first_predict:
    #             self.first_predict = False
    #             self.epoch_since_last_predict = self.predict_epoch_interval

    #         if self.epoch_since_last_predict >= self.predict_epoch_interval:
    #             weights_before = flat_params_as_torch(self.model).clone().to(self.args.w_device)
    #             self._train_normal(train_loader, loss_func)
    #             weights_after = flat_params_as_torch(self.model).clone().to(self.args.w_device)
    #             weight_differences = weights_after - weights_before  # Weight difference computed here
    #             self._perform_dmd_multistep_prediction_and_update(weight_differences)
    #             self.epoch_since_last_predict = 1
    #             # train normally after prediction                
    #         else:
    #             # train normally
    #             self._train_normal(train_loader, loss_func)
    #             self.epoch_since_last_predict += 1
    #     else:
    #         self._train_normal(train_loader, loss_func)

    #     # Adjust parameters based on loss
    #     self.adjust_parameters_based_on_loss()

    def train_epoch(self, train_loader, loss_func):
        # Handle distributed training
        if self.args.distributed:
            if self.args.dist_rank == 0:
                if self.weights is not None:
                    wshape = torch.tensor(self.weights.shape[1], device=torch.device('cuda', self.args.dist_rank % 3))
                else:
                    wshape = torch.tensor(0, device=torch.device('cuda', self.args.dist_rank % 3))
                torch.distributed.broadcast(wshape, 0)
            else:
                #local_rank = self.args.dist_rank % torch.cuda.device_count()
                local_rank = self.args.dist_rank % 3
                wshape = torch.tensor(0, device=torch.device('cuda', local_rank))
                torch.distributed.broadcast(wshape, 0)
        else:
            wshape = torch.tensor(self.weights.shape[1] if self.weights is not None else 0)
        
        # Decide whether to apply PDT based on weight history length
        if wshape.item() >= max(self.svd_rank + 1, self.predict_start_epoch):
            if self.first_predict:
                self.first_predict = False
                self.epoch_since_last_predict = self.predict_epoch_interval

            if self.epoch_since_last_predict >= self.predict_epoch_interval:
                # Maximum consecutive failures before temporarily disabling PDT
                max_consecutive_failures = 3
                
                try:
                    # Enable gradient checkpointing for memory efficiency
                    self._enable_gradient_checkpointing()
                    
                    with torch.cuda.amp.autocast():
                        # Compute weight difference directly on main device
                        weights_before = flat_params_as_torch(self.model)
                        self._train_normal(train_loader, loss_func)
                        weights_after = flat_params_as_torch(self.model)
                        
                        # Calculate difference
                        weight_differences = weights_after - weights_before
                        
                        # Clear memory before DMD computation
                        del weights_before
                        torch.cuda.empty_cache()
                        
                        # Move difference to auxiliary device
                        weight_differences = weight_differences.to(self.args.w_device, non_blocking=True)
                        
                        # Perform DMD prediction and update
                        self._perform_dmd_multistep_prediction_and_update(weight_differences)
                        
                        # Success - reset failure counter
                        self.consecutive_failures = 0
                        print(f"PDT successfully executed at epoch {self.cur_epoch}")
                        
                        # Set next attempt time
                        self.epoch_since_last_predict = 1
                except RuntimeError as e:
                    # Only increment failure count for memory errors
                    if "CUDA out of memory" in str(e):
                        self.dmd_failures += 1
                        self.consecutive_failures += 1
                        print(f"Memory error during PDT (failure #{self.dmd_failures}): {e}")
                        
                        # Adjust parameters based on consecutive failures
                        if self.consecutive_failures >= max_consecutive_failures:
                            print(f"{max_consecutive_failures} consecutive failures - temporarily increasing prediction interval")
                            self.predict_epoch_interval = min(self.predict_epoch_interval + 2, 10)
                            self.predicted_num = max(self.predicted_num - 1, 2)
                            self.consecutive_failures = 0  # Reset counter
                        
                        # Reduce prediction steps to save memory
                        if self.predicted_num > 2:
                            old_num = self.predicted_num
                            self.predicted_num -= 1
                            print(f"Reducing prediction steps from {old_num} to {self.predicted_num} to save memory")
                    else:
                        # Non-memory errors, print but don't increment failure count
                        print(f"Non-memory error in PDT: {e}")
                    
                    # Ensure memory is cleaned up
                    torch.cuda.empty_cache()
                    
                    print("Falling back to standard training for this epoch")
                    # Set next attempt time (ensure we'll try again)
                    self.epoch_since_last_predict = self.predict_epoch_interval
                
                # Log to wandb if enabled
                if self.use_wandb and self.args.dist_rank_0:
                    wandb.log({"dmd_failures": self.dmd_failures}, step=self.cur_epoch)
                    wandb.log({"prediction_steps": self.predicted_num}, step=self.cur_epoch)
                    wandb.log({"prediction_interval": self.predict_epoch_interval}, step=self.cur_epoch)
            else:
                # Standard training
                self._train_normal(train_loader, loss_func)
                self.epoch_since_last_predict += 1
        else:
            self._train_normal(train_loader, loss_func)
        
        # Adjust parameters based on loss
        self.adjust_parameters_based_on_loss()

    def _enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing for memory efficiency
        """
        if not hasattr(self, 'grad_checkpointing_enabled'):
            self.grad_checkpointing_enabled = True
            
            # Get the actual model (handle distributed case)
            if hasattr(self.model, 'module'):
                model = self.model.module
            else:
                model = self.model
            
            # Enable gradient checkpointing for transformer blocks
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
                for module in model.encoder.layers:
                    # Handle different PyTorch versions
                    if hasattr(module, 'gradient_checkpointing'):
                        module.gradient_checkpointing = True
                    elif hasattr(module, '_set_gradient_checkpointing'):
                        try:
                            module._set_gradient_checkpointing(True)
                        except:
                            pass
            
            print("Gradient checkpointing enabled for memory efficiency")

    def adjust_parameters_based_on_loss(self):
        """
        Adjust prediction parameters based on loss trends
        """
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


    # def _perform_dmd_multistep_prediction_and_update(self,epoch_gradient):
    #     if self.args.distributed:
    #         if self.args.dist_rank != 0:
    #             #cur_device = torch.device('cuda', self.args.dist_rank)
    #             local_rank = self.args.dist_rank % torch.cuda.device_count()
    #             cur_device = torch.device('cuda', local_rank)
    #             # this just needs to be the same size as the model weights. It
    #             # will get overwritten
    #             current_weights = flat_params_as_torch(self.model).to(cur_device)
                
    #             #print('\n\nrecv from rank 0\n\n')
    #             torch.distributed.broadcast(current_weights, 0)
                
    #             array_to_params(self.model,current_weights,device=cur_device)
    #             return


    #     #self.print_cuda_memory("Before current weights capture")
    #     current_weights = flat_params_as_torch(self.model).clone().to(self.args.w_device)  # Get current weights after SGD
    #     updated_weights = None
    #     #self.print_cuda_memory("After current weights capture")

    #     if self.weights is not None:
    #         # Check if the epoch_gradient has the correct dimension
    #         if epoch_gradient.dim() != 1 or epoch_gradient.size(0) != self.weights.size(0):
    #             raise ValueError("Incorrect dimension of epoch_gradient. Expected 1D tensor with the same size as the flattened model weights.")
    #         self.dmd = TorchDMD(rank=self.svd_rank)   # use the old torchDMD
    #         #self.dmd = TorchDMDFull(svd_rank=self.svd_rank, clear_snapshots=False)    # use TorchDMDFull for better reconstruction
    #         #self.dmd = DMD(rank=self.svd_rank)              # use the new torchDMD
    #         #self.dmd = RandomizedDMD(rank=self.svd_rank)       # use the RandomizedDMD
    #         if self.n_past_weights is not None:
    #             self.weights = self.weights[:, -self.n_past_weights:]

    #         #self.print_cuda_memory("Before DMD fitting")
    #         self.weights.requires_grad = False
    #         self.dmd.fit(self.weights)  

    #         # use the last weight as the initial condition to predict the future steps
    #         future_steps = self.predicted_num
    #         #self.print_cuda_memory("Before DMD prediction")
    #         predicted_weights = self.dmd.predict_multistep(self.weights[:, -1].reshape(-1, 1), future_steps)    # use the predict_multistep function in old torchDMD
    #         #predicted_weights = self.dmd.predict_multistep_new(future_steps)   # use the predict_multistep_new function in TorchDMDFull
    #         #predicted_weights = self.dmd.predict(self.weights[:, -1].reshape(-1, 1), future_steps)   # use the predict function in new torchDMD and RandomizedDMD
    #         weight_diff = predicted_weights[:, -1].squeeze()-self.weights[:, -1].squeeze()

    #         # Ensure weight_diff and epoch_gradient are aligned in their dimensions
    #         if weight_diff.dim() == 1:
    #             weight_diff = weight_diff.view(-1)  # Flatten to ensure 1D tensor
    #         if epoch_gradient.dim() == 1:
    #             epoch_gradient = epoch_gradient.view(-1)  # Flatten to ensure 1D tensor

    #         #self.print_cuda_memory("After DMD prediction")
    #         # Create mask based on conditions
    #         mask = (torch.sign(weight_diff) == torch.sign(epoch_gradient)) & (torch.abs(weight_diff) <= (self.predicted_num+1) * torch.abs(epoch_gradient))& (torch.abs(weight_diff) >= torch.abs(epoch_gradient))
    #         #mask = (torch.sign(weight_diff) == torch.sign(epoch_gradient)) & (torch.abs(weight_diff) <= (self.predicted_num*2) * torch.abs(epoch_gradient))& (torch.abs(weight_diff) >= torch.abs(epoch_gradient))

    #         # save mask to files
    #         # add a folder "masks" in self.log_dir to save the masks
    #         mask_folder = os.path.join(self.log_dir, "masks")
    #         os.makedirs(mask_folder, exist_ok=True)
    #         mask_file = os.path.join(mask_folder, f"mask_epoch_{self.cur_epoch:04d}.pt")
    #         torch.save(mask, mask_file)

    #         # count the ture values in mask
    #         mask_count = torch.sum(mask)
    #         #print the true values in mask
    #         print("mask_count: ", mask_count)
    #         # calculate the ratio of mask_count to the total number of parameters
    #         mask_ratio = mask_count / len(mask)
    #         # calculate the ratio of other three conditions with respect to the total number of parameters
    #         ratio_opposite = torch.sum((torch.sign(weight_diff) != torch.sign(epoch_gradient))) / len(mask)
    #         ratio_n_plus = torch.sum((torch.sign(weight_diff) == torch.sign(epoch_gradient)) & (torch.abs(weight_diff) > (self.predicted_num+1) * torch.abs(epoch_gradient))) / len(mask)
    #         ratio_0_1 = torch.sum((torch.sign(weight_diff) == torch.sign(epoch_gradient)) & (torch.abs(weight_diff) < torch.abs(epoch_gradient))) / len(mask)

    #         if self.use_wandb:
    #             wandb.log({"mask_ratio_global": mask_ratio.item()}, step=self.cur_epoch)
    #             wandb.log({"ratio_opposite": ratio_opposite.item()}, step=self.cur_epoch)
    #             wandb.log({"ratio_n_plus": ratio_n_plus.item()}, step=self.cur_epoch)
    #             wandb.log({"ratio_0_1": ratio_0_1.item()}, step=self.cur_epoch)

    #         #print("mask shape:", mask.shape)
    #         #print("predicted_weights shape:", predicted_weights.shape)
    #         #print("current_weights shape:", current_weights.shape)

    #         # Apply the mask to the weight_diff to obtain the conditional weight difference
    #         updated_weights = torch.where(mask, predicted_weights.squeeze(), current_weights)  

    #         # update the model weights
    #         array_to_params(self.model,updated_weights,device=self.orig_device)
        
    #     if self.args.dist_rank_0:
    #         #print('\n\nsend from rank 0\n\n')
    #         if updated_weights is None:
    #             #torch.distributed.broadcast(current_weights.to(self.orig_device), 0)
    #             for i in range(1, self.args.ws):
    #                 torch.distributed.broadcast(current_weights.to(self.orig_device), 0)
    #         else:
    #             #torch.distributed.broadcast(updated_weights.to(self.orig_device), 0)
    #             for i in range(1, self.args.ws):
    #                 torch.distributed.broadcast(updated_weights.to(self.orig_device), 0)

    def _perform_dmd_multistep_prediction_and_update(self, epoch_gradient):
        """
        Perform layerwise DMD prediction and weight update
        """
        if self.args.distributed:
            if self.args.dist_rank != 0:
                #local_rank = self.args.dist_rank % torch.cuda.device_count()
                local_rank = self.args.dist_rank % 3
                cur_device = torch.device('cuda', local_rank)
                # Just get same size as model weights (will be overwritten)
                current_weights = flat_params_as_torch(self.model).to(cur_device)
                
                torch.distributed.broadcast(current_weights, 0)
                
                array_to_params(self.model, current_weights, device=cur_device)
                return

        # Get current weights
        current_weights = flat_params_as_torch(self.model).clone().to(self.args.w_device)
        updated_weights = None
        
        if self.use_layerwise_dmd:
            # Layerwise DMD implementation
            # 1. Extract layer components from current weights
            current_layer_weights = self._extract_layer_weights(current_weights)
            
            # 2. Break down epoch_gradient into layers
            gradient_layer_weights = self._extract_layer_weights(epoch_gradient)
            
            # 3. Perform DMD for each layer
            predicted_layer_weights = {}
            layer_masks = {}
            total_params = 0
            total_masked_params = 0
            
            for layer_name, layer_weights in self.layer_weights.items():
                print(f"Processing layer group: {layer_name}, shape: {layer_weights.shape}")
                total_params += layer_weights.shape[0]
                
                # Skip layers with insufficient history
                if layer_weights.shape[1] < 3:  # Need at least 3 history points
                    print(f"  - Skipping layer {layer_name}: insufficient history ({layer_weights.shape[1]} < 3)")
                    continue
                
                # Create and fit DMD model for this layer
                try:
                    layer_dmd = TorchDMD(rank=min(self.svd_rank, layer_weights.shape[1]-1))
                    
                    # Use only recent history points
                    if self.n_past_weights is not None:
                        layer_weights_trimmed = layer_weights[:, -self.n_past_weights:]
                    else:
                        layer_weights_trimmed = layer_weights
                    
                    # Fit DMD model
                    layer_dmd.fit(layer_weights_trimmed)
                    
                    # Predict future weights
                    future_steps = self.predicted_num
                    predicted_weights = layer_dmd.predict_multistep(
                        layer_weights_trimmed[:, -1].reshape(-1, 1), 
                        future_steps
                    )
                    
                    # Calculate weight differences for this layer
                    weight_diff = predicted_weights[:, -1].squeeze() - layer_weights_trimmed[:, -1].squeeze()
                    
                    # Ensure dimensions match
                    if weight_diff.dim() == 1:
                        weight_diff = weight_diff.view(-1)
                    
                    # Get gradient for this layer
                    if layer_name in gradient_layer_weights:
                        layer_gradient = gradient_layer_weights[layer_name].view(-1)
                        
                        # Create mask for this layer
                        mask = (
                            (torch.sign(weight_diff) == torch.sign(layer_gradient)) & 
                            (torch.abs(weight_diff) <= (self.predicted_num+1) * torch.abs(layer_gradient)) & 
                            (torch.abs(weight_diff) >= torch.abs(layer_gradient))
                        )
                        
                        # Save mask
                        layer_masks[layer_name] = mask
                        
                        # Count masked parameters
                        layer_mask_count = torch.sum(mask)
                        layer_mask_ratio = layer_mask_count / len(mask)
                        print(f"  - Layer {layer_name}: mask ratio = {layer_mask_ratio:.2%}, masked parameters: {layer_mask_count}")
                        total_masked_params += layer_mask_count
                        
                        # Apply mask to get predicted weights
                        if layer_name in current_layer_weights:
                            layer_current_weights = current_layer_weights[layer_name]
                            predicted_layer_weights[layer_name] = torch.where(
                                mask, 
                                predicted_weights[:, -1].squeeze(), 
                                layer_current_weights
                            )
                    else:
                        print(f"  - Skipping layer {layer_name}: gradient not available")
                except Exception as e:
                    print(f"  - Error in layer {layer_name} DMD processing: {e}")
                    # Skip this layer on error
                    continue
                
                # Clean up memory after each layer
                torch.cuda.empty_cache()
            
            # 4. Aggregate layer predictions
            if len(predicted_layer_weights) > 0:
                # Reconstruct full weight vector
                updated_weights = self._reconstruct_flat_weights(predicted_layer_weights)
                
                # Calculate global mask ratio
                global_mask_ratio = total_masked_params / total_params if total_params > 0 else 0
                print(f"Global mask ratio: {global_mask_ratio:.2%}, total masked parameters: {total_masked_params}/{total_params}")
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({"mask_ratio_global": global_mask_ratio}, step=self.cur_epoch)
            else:
                # If all layers failed, use current weights
                updated_weights = current_weights
                print("No layer successfully processed by DMD")
        else:
            # Original global DMD implementation (unchanged)
            if self.weights is not None:
                # Check gradient dimension
                if epoch_gradient.dim() != 1 or epoch_gradient.size(0) != self.weights.size(0):
                    raise ValueError("Incorrect dimension of epoch_gradient. Expected 1D tensor with the same size as the flattened model weights.")
                self.dmd = TorchDMD(rank=self.svd_rank)
                
                if self.n_past_weights is not None:
                    self.weights = self.weights[:, -self.n_past_weights:]

                self.weights.requires_grad = False
                self.dmd.fit(self.weights)  

                # Predict future weights using last weight as initial condition
                future_steps = self.predicted_num
                predicted_weights = self.dmd.predict_multistep(
                    self.weights[:, -1].reshape(-1, 1), 
                    future_steps
                )
                
                weight_diff = predicted_weights[:, -1].squeeze() - self.weights[:, -1].squeeze()

                # Ensure dimensions are aligned
                if weight_diff.dim() == 1:
                    weight_diff = weight_diff.view(-1)
                if epoch_gradient.dim() == 1:
                    epoch_gradient = epoch_gradient.view(-1)

                # Create mask
                mask = (
                    (torch.sign(weight_diff) == torch.sign(epoch_gradient)) & 
                    (torch.abs(weight_diff) <= (self.predicted_num+1) * torch.abs(epoch_gradient)) & 
                    (torch.abs(weight_diff) >= torch.abs(epoch_gradient))
                )
                
                # Save mask to file
                mask_folder = os.path.join(self.log_dir, "masks")
                os.makedirs(mask_folder, exist_ok=True)
                mask_file = os.path.join(mask_folder, f"mask_epoch_{self.cur_epoch:04d}.pt")
                torch.save(mask, mask_file)
                
                # Count true values in mask
                mask_count = torch.sum(mask)
                print("mask_count: ", mask_count)
                # Calculate mask ratio
                mask_ratio = mask_count / len(mask)
                # Calculate other condition ratios
                ratio_opposite = torch.sum((torch.sign(weight_diff) != torch.sign(epoch_gradient))) / len(mask)
                ratio_n_plus = torch.sum((torch.sign(weight_diff) == torch.sign(epoch_gradient)) & (torch.abs(weight_diff) > (self.predicted_num+1) * torch.abs(epoch_gradient))) / len(mask)
                ratio_0_1 = torch.sum((torch.sign(weight_diff) == torch.sign(epoch_gradient)) & (torch.abs(weight_diff) < torch.abs(epoch_gradient))) / len(mask)

                if self.use_wandb:
                    wandb.log({"mask_ratio_global": mask_ratio.item()}, step=self.cur_epoch)
                    wandb.log({"ratio_opposite": ratio_opposite.item()}, step=self.cur_epoch)
                    wandb.log({"ratio_n_plus": ratio_n_plus.item()}, step=self.cur_epoch)
                    wandb.log({"ratio_0_1": ratio_0_1.item()}, step=self.cur_epoch)

                # Apply mask
                updated_weights = torch.where(mask, predicted_weights[:, -1].squeeze(), current_weights)
        
        # Update model weights
        if updated_weights is not None:
            array_to_params(self.model, updated_weights, device=self.orig_device)
        
        # Handle distributed training
        if self.args.dist_rank_0:
            if updated_weights is None:
                torch.distributed.broadcast(current_weights.to(self.orig_device), 0)
            else:
                torch.distributed.broadcast(updated_weights.to(self.orig_device), 0)

        # Clean up memory
        if self.args.dist_rank_0:
            training_gpu_id = self.args.dist_rank % 3
            print(f"GPU {training_gpu_id} memory after DMD: {torch.cuda.memory_allocated(training_gpu_id)/1e9:.2f} GB")
            torch.cuda.empty_cache()
            print(f"GPU {training_gpu_id} memory after cleanup: {torch.cuda.memory_allocated(training_gpu_id)/1e9:.2f} GB")
            
            
            storage_device_id = int(str(self.args.w_device).split(':')[-1])
            if storage_device_id != training_gpu_id:  # avoid cleaning the same device twice
                with torch.cuda.device(storage_device_id):
                    torch.cuda.empty_cache()
                print(f"Storage GPU {storage_device_id} memory cleared")


    # def _train_normal(self, train_loader, loss_func):
    #     if self.args.dist_rank_0:
    #         if self.weights is None:
    #             self.weights = flat_params_as_torch(self.model).reshape((-1, 1)).to(self.args.w_device)
    #         else:
    #             self.weights = torch.cat(
    #                 [self.weights, flat_params_as_torch(self.model).reshape((-1, 1)).to(self.args.w_device)],
    #                 dim=1
    #             )

    #     self.model.train()
    #     pbar = tqdm(
    #         total=len(train_loader),
    #         position=1,
    #         unit="batch",
    #         leave=False,
    #         desc="Epoch {}".format(self.cur_epoch),
    #         disable=(not self.args.dist_rank_0)
    #     )
    #     losses = []
    #     for batch_idx, batch in enumerate(train_loader):
    #         data, target = unpack_batch(batch,device=self.device) 
    #         self.optim.zero_grad()
    #         # output = self.model(data)
    #         # loss = loss_func(output, target)
    #         with torch.autocast(
    #             device_type=self.device.type, dtype=torch.bfloat16
    #         ):
    #             output = self.model(data)
    #             loss = loss_func(output, target)
    #         # losses.append(loss.item())
    #         # loss.backward()
    #         # self.optim.step()
    #         self.grad_scaler.scale(loss).backward()
    #         self.grad_scaler.step(self.optim)
    #         self.grad_scaler.update()

    #         if batch_idx % self.log_interval == 0 and self.args.dist_rank_0:
    #             tqdm.write("Loss: {:.4f}".format(loss.item()))
    #         losses.append(loss.item())
    #         pbar.update(1)
    #     # print the learning rate
    #     tqdm.write(f"Learning rate: {self.optim.param_groups[0]['lr']}")
    #     #self.scheduler.step()
    #     pbar.close()
    #     self.train_losses.append(losses)

    def _train_normal(self, train_loader, loss_func):
        """
        Normal training with layer-wise weight history recording
        """
        # Store global weight history (if not using layerwise DMD)
        if not self.use_layerwise_dmd:
            flat_weights = flat_params_as_torch(self.model).to(self.args.w_device)
            if self.weights is None:
                self.weights = flat_weights.reshape((-1, 1))
            else:
                self.weights = torch.cat([self.weights, flat_weights.reshape((-1, 1))], dim=1)
        else:
            # Layer-wise weight history recording
            flat_weights = flat_params_as_torch(self.model).to(self.args.w_device)
            layer_weights = self._extract_layer_weights(flat_weights)
            
            # Process in batches to save memory
            batch_size = 4  # 4 layers at a time
            layer_keys = list(layer_weights.keys())
            
            for i in range(0, len(layer_keys), batch_size):
                batch_keys = layer_keys[i:i+batch_size]
                for layer_key in batch_keys:
                    if layer_key not in self.layer_weights:
                        try:
                            self.layer_weights[layer_key] = layer_weights[layer_key].reshape(-1, 1)
                        except RuntimeError as e:
                            print(f"Error initializing history for layer {layer_key}: {e}")
                            continue
                    else:
                        try:
                            # Try to concatenate directly on the GPU
                            self.layer_weights[layer_key] = torch.cat(
                                [self.layer_weights[layer_key], layer_weights[layer_key].reshape(-1, 1)], 
                                dim=1
                            )
                        except RuntimeError as e:
                            print(f"GPU connection error, trying to connect using CPU: {e}")
                            try:
                                # Move to CPU, concatenate, move back
                                cpu_hist = self.layer_weights[layer_key].cpu()
                                cpu_new = layer_weights[layer_key].reshape(-1, 1).cpu()
                                cpu_combined = torch.cat([cpu_hist, cpu_new], dim=1)
                                self.layer_weights[layer_key] = cpu_combined.to(self.args.w_device)
                                del cpu_hist, cpu_new, cpu_combined  # free CPU memory
                            except Exception as e2:
                                print(f"Can't update history for layer {layer_key}: {e2}")
                                # Drop this layer's history and retry next time
                                if layer_key in self.layer_weights:
                                    del self.layer_weights[layer_key]
                    
                    # Limit history length
                    if layer_key in self.layer_weights and self.n_past_weights is not None:
                        if self.layer_weights[layer_key].shape[1] > self.n_past_weights:
                            self.layer_weights[layer_key] = self.layer_weights[layer_key][:, -self.n_past_weights:]
                
                # Free memory after each batch
                torch.cuda.empty_cache()
                
            # Log GPU memory usage
            if self.args.dist_rank_0:
                storage_device_id = int(str(self.args.w_device).split(':')[-1])
                print(f"GPU {storage_device_id} memory after layer history update: {torch.cuda.memory_allocated(storage_device_id)/1e9:.2f} GB")

        # Normal training process (unchanged)
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
            data, target = unpack_batch(batch, device=self.device) 
            self.optim.zero_grad()
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                output = self.model(data)
                loss = loss_func(output, target)
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optim)
            self.grad_scaler.update()

            if batch_idx % self.log_interval == 0 and self.args.dist_rank_0:
                tqdm.write("Loss: {:.4f}".format(loss.item()))
            losses.append(loss.item())
            pbar.update(1)
        tqdm.write(f"Learning rate: {self.optim.param_groups[0]['lr']}")
        pbar.close()
        self.train_losses.append(losses)

