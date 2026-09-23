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
        
        self.threshold = 0.01

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

            # if the latest test_losses is less than the one before, then perform DMD prediction and update the weights
            if np.mean(self.test_losses[-1]) < np.mean(self.test_losses[-2]):
                print("test_losses[-1]: ", np.mean(self.test_losses[-1]))
                print("test_losses[-2]: ", np.mean(self.test_losses[-2]))
                print("perform_dmd_multistep_prediction_and_update: ")
                self._perform_dmd_multistep_prediction_and_update()
                self.epoch_since_last_predict = 1
                # # train normally after prediction
                # self._train_normal(train_loader, loss_func)
            else:
                # train normally
                print("train normally: ")
                self._train_normal(train_loader, loss_func)

            # if self.epoch_since_last_predict >= self.predict_epoch_interval:
            #     # perform DMD prediction and update the weights
            #     self._perform_dmd_multistep_prediction_and_update()
            #     self.epoch_since_last_predict = 1
            #     # train normally after prediction
            #     self._train_normal(train_loader, loss_func)
            # else:
            #     # train normally
            #     self._train_normal(train_loader, loss_func)
            #     self.epoch_since_last_predict += 1
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
        
        
        
        
        
        
        
