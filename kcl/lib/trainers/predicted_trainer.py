import numpy as np
import torch
from tqdm import tqdm

from kcl.lib.dmd.torchDMD import TorchDMD
from kcl.lib.trainers.trainer_base import TrainerBase
from kcl.utils.weight_tools import array_to_params, flat_params_as_torch
from copy import deepcopy

class PredictedTrainer(TrainerBase):
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
                # perform DMD prediction and update the weights
                if self.if_sp:
                    if self.if_LV:
                        self._perform_dmd_multistep_prediction_and_update_SPLV()
                    else:
                        self._perform_dmd_multistep_prediction_and_update_sp()
                elif self.if_LV:
                    self._perform_dmd_multistep_prediction_and_update_LV()
                else:
                    self._perform_dmd_multistep_prediction_and_update()
                self.epoch_since_last_predict = 1
                # train normally after prediction
                self._train_normal(train_loader, loss_func)
            else:
                # train normally
                self._train_normal(train_loader, loss_func)
                self.epoch_since_last_predict += 1
        else:
            self._train_normal(train_loader, loss_func)

    def _perform_dmd_multistep_prediction_and_update(self):
        if self.weights is not None:
            self.dmd = TorchDMD(rank=self.svd_rank)
            if self.n_past_weights is not None:
                self.weights = self.weights[:, -self.n_past_weights:]
            
            self.weights.requires_grad = False
            self.dmd.fit(self.weights)  

            # use the last weight as the initial condition to predict the future steps
            future_steps = self.predicted_num
            predicted_weights = self.dmd.predict_multistep(self.weights[:, -1].reshape(-1, 1), future_steps)
            # update the weights
            array_to_params(self.model, predicted_weights[:, -1].squeeze(), device=self.weights.device)
    
    # perform DMD prediction and update the weights in small pieces
    def _perform_dmd_multistep_prediction_and_update_sp(self):
        if self.weights is not None:
            if self.n_past_weights is not None:
                self.weights = self.weights[:, -self.n_past_weights:]
            segment_size = 10000  # Define the size of each segment
            total_weights = self.weights.shape[0]
            future_steps = self.predicted_num
            predicted_segments = []

            # Calculate the number of full segments and the size of the final segment
            num_full_segments = total_weights // segment_size
            final_segment_size = total_weights % segment_size

            for i in range(num_full_segments):
                segment_start = i * segment_size
                segment_end = (i + 1) * segment_size
                weight_segment = self.weights[segment_start:segment_end,:]

                # Apply DMD to this segment
                dmd = TorchDMD(rank=self.svd_rank)
                dmd.fit(weight_segment)
                predicted_segment = dmd.predict_multistep(weight_segment[:, -1].reshape(-1, 1), future_steps)
                predicted_segments.append(predicted_segment[:, -1].squeeze().cpu().numpy())  # Store the last prediction
            
            # Handle the final segment if it exists
            if final_segment_size > 0:
                final_segment_start = num_full_segments * segment_size
                final_segment = self.weights[final_segment_start:, :]

                # Apply DMD to the final segment
                dmd = TorchDMD(rank=self.svd_rank)
                dmd.fit(final_segment)
                predicted_final_segment = dmd.predict_multistep(final_segment[:, -1].reshape(-1, 1), future_steps)
                predicted_segments.append(predicted_final_segment[:, -1].squeeze().cpu().numpy())  # Store the last prediction

            # Aggregate predicted segments (optional, depending on how you want to use the predictions)
            aggregated_prediction = np.concatenate(predicted_segments, axis=0)

            # Update the weights with the aggregated prediction
            array_to_params(self.model, aggregated_prediction, device=self.weights.device)

    # Reconstruct the weights only with large value
    def _find_large_value_mask(self, weights, threshold):
        mask = torch.abs(weights[:,-1]) > threshold
        return mask
    def  _perform_dmd_multistep_prediction_and_update_SPLV(self):
        if self.weights is not None:
            dmd = TorchDMD(rank=0)
            if self.n_past_weights is not None:
                self.weights = self.weights[:, -self.n_past_weights:]
            mask = self._find_large_value_mask(self.weights, self.threshold)
            # print the counts of large value
            print("The counts of large value:")
            print(mask.sum())
            # only reconstruct the weights with large value
            weights_large = self.weights[mask]
            self.weights.requires_grad = False
            segment_size = 10000  # Define the size of each segment
            predicted_segments = []
            total_weights = weights_large.shape[0]
            # Calculate the number of full segments and the size of the final segment
            num_full_segments = total_weights // segment_size
            final_segment_size = total_weights % segment_size

            for i in range(num_full_segments):
                segment_start = i * segment_size
                segment_end = (i + 1) * segment_size
                weight_segment = weights_large[segment_start:segment_end,:]

                # Apply DMD to this segment
                dmd.fit(weight_segment)
                predicted_segment = dmd.predict_multistep(weight_segment[:, -1].reshape(-1, 1), self.predicted_num)
                predicted_segments.append(predicted_segment[:, -1].squeeze().cpu().numpy())
            
            if final_segment_size > 0:
                final_segment_start = num_full_segments * segment_size
                final_segment = weights_large[final_segment_start:, :]

                # Apply DMD to the final segment
                dmd.fit(final_segment)
                predicted_final_segment = dmd.predict_multistep(final_segment[:, -1].reshape(-1, 1), self.predicted_num)
                predicted_segments.append(predicted_final_segment[:, -1].squeeze().cpu().numpy())
            
            # Aggregate predicted segments (optional, depending on how you want to use the predictions)
            aggregated_prediction = np.concatenate(predicted_segments, axis=0)
            # map the reconstructed_large_weights back to the original weights
            reconstructed_weights = deepcopy(self.weights[:,-1]).reshape(-1,1)
            reconstructed_weights[mask,:] = torch.tensor(aggregated_prediction).reshape(-1,1).to(self.weights.device)
            # update the weights
            array_to_params(self.model, reconstructed_weights[:, -1].squeeze(), device=self.weights.device)
    
    def  _perform_dmd_multistep_prediction_and_update_LV(self):
        if self.weights is not None:
            dmd = TorchDMD(rank=0)
            if self.n_past_weights is not None:
                self.weights = self.weights[:, -self.n_past_weights:]
            mask = self._find_large_value_mask(self.weights, self.threshold)
            # print the counts of large value
            print("The counts of large value:")
            print(mask.sum())
            # only reconstruct the weights with large value
            weights_large = self.weights[mask]
            self.weights.requires_grad = False
            dmd.fit(weights_large)
            future_steps = self.predicted_num
            reconstructed_weights_large = dmd.predict_multistep(weights_large[:, -1].reshape(-1, 1), future_steps)
            # map the reconstructed_large_weights back to the original weights
            reconstructed_weights = deepcopy(self.weights[:,-1]).reshape(-1,1)
            reconstructed_weights[mask,:] = reconstructed_weights_large
            # update the weights
            array_to_params(self.model, reconstructed_weights[:, -1].squeeze(), device=self.weights.device)
            

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
        
        
        
        
        
        
        
