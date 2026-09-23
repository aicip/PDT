"""Baselines compared against PDT in the paper.

* :class:`NonSelectivePredictionTrainer` -- Koopman-based predictive training that writes
  the whole predicted weight vector back without any mask (Figure 2).
* :class:`RandomAcceleratedTrainer` -- a random subset of weights gets a larger step along
  the epoch update, matching the mask ratio of PDT (Figure 5).
* :class:`RandomMaskTrainer` -- DMD prediction accepted on a random subset of weights
  instead of the PDT mask (Figure 6).
* :class:`SwitchByLossTrainer` -- prediction is applied while the validation loss is
  decreasing and training falls back to the optimizer otherwise (Figure 7).

All baselines use the same base-optimizer epoch (``TrainerBase._train_normal``) as PDT.
"""

import time

import numpy as np
import torch

from kcl.lib.dmd.torchDMD import TorchDMD
from kcl.lib.trainers.trainer_base import TrainerBase
from kcl.utils.weight_tools import array_to_params, flat_params_as_torch


class _PredictionScheduleMixin:
    """Shared bookkeeping of when a prediction step is due."""

    def _init_schedule(self, svd_rank, n_past_weights, predict_start_epoch, predicted_num, predict_epoch_interval):
        self.svd_rank = svd_rank
        self.n_past_weights = n_past_weights
        self.predicted_num = predicted_num
        self.predict_start_epoch = predict_start_epoch
        self.predict_epoch_interval = predict_epoch_interval
        self.epoch_since_last_predict = 0
        self.first_predict = True

    def _history_ready(self):
        return self.weights is not None and self.weights.shape[1] >= max(self.svd_rank + 1, self.predict_start_epoch)

    def _should_predict(self):
        if not self._history_ready():
            return False
        if self.first_predict:
            self.first_predict = False
            self.epoch_since_last_predict = self.predict_epoch_interval
        return self.epoch_since_last_predict >= self.predict_epoch_interval

    def _dmd_prediction(self):
        """Fit DMD on the last ``n_past_weights`` snapshots and predict ``predicted_num`` steps ahead."""
        if self.n_past_weights is not None:
            self.weights = self.weights[:, -self.n_past_weights:]
        self.weights.requires_grad = False
        self.dmd = TorchDMD(rank=self.svd_rank)
        self.dmd.fit(self.weights)
        return self.dmd.predict_multistep(self.weights[:, -1].reshape(-1, 1), self.predicted_num).view(-1)


class NonSelectivePredictionTrainer(_PredictionScheduleMixin, TrainerBase):
    """Predicted weights replace all parameters (no masking), then training continues."""

    def __init__(self, *, svd_rank=0, n_past_weights=5, predict_start_epoch=5, predicted_num=5,
                 predict_epoch_interval=1, **kwargs):
        super().__init__(**kwargs)
        self._init_schedule(svd_rank, n_past_weights, predict_start_epoch, predicted_num, predict_epoch_interval)

    def train_epoch(self, train_loader, loss_func):
        if self._should_predict():
            array_to_params(self.model, self._dmd_prediction(), device=self.weights.device)
            self.epoch_since_last_predict = 1
            self.epoch_extra["prediction_epoch"] = 1
            self._train_normal(train_loader, loss_func)
        else:
            self._train_normal(train_loader, loss_func)
            if self._history_ready():
                self.epoch_since_last_predict += 1


class SwitchByLossTrainer(_PredictionScheduleMixin, TrainerBase):
    """Apply the (unmasked) prediction whenever the validation loss decreased, else train."""

    def __init__(self, *, svd_rank=0, n_past_weights=5, predict_start_epoch=5, predicted_num=5,
                 predict_epoch_interval=1, **kwargs):
        super().__init__(**kwargs)
        self._init_schedule(svd_rank, n_past_weights, predict_start_epoch, predicted_num, predict_epoch_interval)

    def train_epoch(self, train_loader, loss_func):
        if self._history_ready() and len(self.test_losses) >= 2 and np.mean(self.test_losses[-1]) < np.mean(self.test_losses[-2]):
            array_to_params(self.model, self._dmd_prediction(), device=self.weights.device)
            self.epoch_extra["prediction_epoch"] = 1
            if self.is_main:
                print(f"Epoch {self.cur_epoch:04d} | validation loss decreased: prediction applied instead of training")
        else:
            self._train_normal(train_loader, loss_func)


class RandomAcceleratedTrainer(_PredictionScheduleMixin, TrainerBase):
    """Random subset of weights takes ``predicted_num`` epoch updates at once."""

    def __init__(self, *, svd_rank=0, n_past_weights=5, predict_start_epoch=5, predicted_num=5,
                 predict_epoch_interval=1, random_mask_ratio=0.18, random_mask_seed=34237865, **kwargs):
        super().__init__(**kwargs)
        self._init_schedule(svd_rank, n_past_weights, predict_start_epoch, predicted_num, predict_epoch_interval)
        self.random_mask_ratio = random_mask_ratio
        self.random_mask_seed = random_mask_seed

    def train_epoch(self, train_loader, loss_func):
        if self._should_predict():
            weights_before = flat_params_as_torch(self.model).clone()
            self._train_normal(train_loader, loss_func)
            current = flat_params_as_torch(self.model).clone()
            delta = current - weights_before
            accelerated = current + (self.predicted_num - 1) * delta
            gen = torch.Generator(device=current.device)
            gen.manual_seed(self.random_mask_seed)
            mask = torch.rand(current.size(), generator=gen, dtype=torch.float32, device=current.device) < self.random_mask_ratio
            self.epoch_extra.update({"prediction_epoch": 1, "mask_ratio": round(mask.sum().item() / len(mask), 6)})
            self._log_wandb({"mask_ratio_global": mask.sum().item() / len(mask)})
            array_to_params(self.model, torch.where(mask, accelerated, current), device=current.device)
            self.epoch_since_last_predict = 1
        else:
            self._train_normal(train_loader, loss_func)
            if self._history_ready():
                self.epoch_since_last_predict += 1


class RandomMaskTrainer(_PredictionScheduleMixin, TrainerBase):
    """DMD prediction accepted on a random subset of weights (no dynamic-consistency mask)."""

    def __init__(self, *, svd_rank=0, n_past_weights=5, predict_start_epoch=5, predicted_num=5,
                 predict_epoch_interval=1, random_mask_ratio=0.09699, random_mask_seed=None, **kwargs):
        super().__init__(**kwargs)
        self._init_schedule(svd_rank, n_past_weights, predict_start_epoch, predicted_num, predict_epoch_interval)
        self.random_mask_ratio = random_mask_ratio
        self.random_mask_seed = random_mask_seed

    def train_epoch(self, train_loader, loss_func):
        if self._should_predict():
            self._train_normal(train_loader, loss_func)
            current = flat_params_as_torch(self.model).clone()
            predicted = self._dmd_prediction()
            seed = self.random_mask_seed if self.random_mask_seed is not None else int(time.time())
            gen = torch.Generator(device=current.device)
            gen.manual_seed(seed)
            mask = torch.rand(current.size(), generator=gen, dtype=torch.float32, device=current.device) < self.random_mask_ratio
            self.epoch_extra.update({"prediction_epoch": 1, "mask_ratio": round(mask.sum().item() / len(mask), 6)})
            self._log_wandb({"mask_ratio_global": mask.sum().item() / len(mask), "mask_seed": seed})
            array_to_params(self.model, torch.where(mask, predicted, current), device=current.device)
            self.epoch_since_last_predict = 1
        else:
            self._train_normal(train_loader, loss_func)
            if self._history_ready():
                self.epoch_since_last_predict += 1
