"""PDT for multi-GPU (DistributedDataParallel) training.

Every rank trains its shard with DDP. Rank 0 keeps the weight history, fits DMD,
computes the mask and assembles the new weights; the result is broadcast to the other
ranks so that all replicas stay identical.

This trainer also contains the acceleration scheduler used for the ImageNet
experiments (``adjust_parameters_based_on_loss``): when the training loss increases from
one epoch to the next, the prediction horizon ``tau`` is reduced and/or the prediction
interval ``T_i`` is increased. The rule is evaluated on the training loss averaged over
all ranks, and the schedule state (``tau``, ``T_i``, epochs since the last prediction) is
broadcast from rank 0 after every epoch, so that all ranks take the same control path
(the prediction step contains collective operations).
"""

import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from kcl.lib.dmd.torchDMD import TorchDMD
from kcl.lib.trainers.pdt_trainer import PDTTrainer
from kcl.utils.weight_tools import array_to_params, flat_params_as_torch


class PDTDistributedTrainer(PDTTrainer):
    def __init__(self, *, w_device=None, adaptive_schedule=True, **kwargs):
        """
        :param w_device: device that stores the weight history and runs DMD (rank 0 only);
            defaults to the training device. A separate GPU can be used for very large
            models.
        :param adaptive_schedule: enable the loss-based adjustment of ``tau`` and ``T_i``
        """
        super().__init__(**kwargs)
        self.storage_device = torch.device(w_device) if w_device is not None else self.device
        self.adaptive_schedule = adaptive_schedule
        self.prev_loss = None
        self.prev_prev_loss = None
        if self.is_main:
            print(f"PDT (distributed): training device {self.device}, weight history on {self.storage_device}, "
                  f"adaptive schedule {'on' if adaptive_schedule else 'off'}")

    # ------------------------------------------------------------------ epoch
    def _history_length(self):
        """Length of the weight history, known to rank 0 and broadcast to the others."""
        if self._distributed():
            n = torch.tensor(self.weights.shape[1] if (self.is_main and self.weights is not None) else 0,
                             device=self.device)
            dist.broadcast(n, 0)
            return int(n.item())
        return self.weights.shape[1] if self.weights is not None else 0

    def _history_ready(self):
        return self._history_length() >= max(self.svd_rank + 1, self.predict_start_epoch)

    def train_epoch(self, train_loader, loss_func):
        super().train_epoch(train_loader, loss_func)
        self._end_of_epoch_schedule()

    def _end_of_epoch_schedule(self):
        if self.adaptive_schedule:
            self.adjust_parameters_based_on_loss()
        self._sync_schedule()

    def _sync_schedule(self):
        """Broadcast the schedule state from rank 0 so that every rank predicts in the same epochs."""
        if not self._distributed():
            return
        state = torch.tensor([self.predicted_num, self.predict_epoch_interval, self.epoch_since_last_predict],
                             device=self.device, dtype=torch.long)
        dist.broadcast(state, 0)
        self.predicted_num, self.predict_epoch_interval, self.epoch_since_last_predict = (
            int(v) for v in state.tolist())

    def adjust_parameters_based_on_loss(self):
        if len(self.train_losses) < 2:
            return
        current_loss = self._global_train_loss()  # identical on all ranks
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
                elif self.predicted_num == 4:
                    self.predicted_num -= 1
                    self.predict_epoch_interval += 1
                else:
                    self.predict_epoch_interval += 2
            else:
                if self.predicted_num > 3:
                    self.predicted_num -= 1
                else:
                    self.predict_epoch_interval += 1
            if self.is_main:
                print(f"Loss increased: tau={self.predicted_num}, interval={self.predict_epoch_interval}")
        self.prev_prev_loss = self.prev_loss
        self.prev_loss = current_loss

    def _sgd_epoch_with_delta(self, train_loader, loss_func):
        weights_before = flat_params_as_torch(self.model).clone().to(self.storage_device)
        self._train_normal(train_loader, loss_func)
        delta = flat_params_as_torch(self.model).to(self.storage_device) - weights_before
        self._log_gradient_metrics(delta)
        return delta

    def _train_normal(self, train_loader, loss_func):
        if self.is_main:
            snapshot = flat_params_as_torch(self.model).reshape((-1, 1)).to(self.storage_device)
            self.weights = snapshot if self.weights is None else torch.cat([self.weights, snapshot], dim=1)
        self._run_sgd_epoch(train_loader, loss_func, record_history=False)

    # ------------------------------------------------------------ prediction
    def _predict_and_update(self, epoch_gradient):
        if self._distributed() and not self.is_main:
            # receive the assembled weights from rank 0
            current_weights = flat_params_as_torch(self.model).to(self.device)
            dist.broadcast(current_weights, 0)
            array_to_params(self.model, current_weights, device=self.device)
            return

        mem = self._flag("profile_memory") and self.device.type == "cuda"
        timing = self._flag("profile_timing")
        if mem:
            mem_before = torch.cuda.memory_allocated(self.device) / 1024 ** 2

        current_weights = flat_params_as_torch(self.model).clone().to(self.storage_device)
        updated_weights = None
        if self.weights is not None:
            if epoch_gradient.dim() != 1 or epoch_gradient.size(0) != self.weights.size(0):
                raise ValueError("epoch_gradient must be a flat vector with one entry per parameter")
            if self.n_past_weights is not None:
                self.weights = self.weights[:, -self.n_past_weights:]
            self.weights.requires_grad = False
            if mem:
                snapshot_mb = self.weights.numel() * self.weights.element_size() / 1024 ** 2
                mem_before_svd = torch.cuda.memory_allocated(self.device) / 1024 ** 2

            if timing:
                self._sync()
                t0 = time.perf_counter()
            self.dmd = TorchDMD(rank=self.svd_rank)
            self.dmd.fit(self.weights)
            if timing:
                self._sync()
                svd_ms = (time.perf_counter() - t0) * 1000
            if mem:
                svd_workspace_mb = torch.cuda.memory_allocated(self.device) / 1024 ** 2 - mem_before_svd

            if timing:
                t0 = time.perf_counter()
            predicted_weights = self.dmd.predict_multistep(self.weights[:, -1].reshape(-1, 1), self.predicted_num)
            if timing:
                self._sync()
                pred_ms = (time.perf_counter() - t0) * 1000
            weight_diff = (predicted_weights[:, -1] - self.weights[:, -1]).view(-1)
            epoch_gradient = epoch_gradient.view(-1)

            if timing:
                t0 = time.perf_counter()
            mask = self.compute_mask(weight_diff, epoch_gradient)
            if timing:
                self._sync()
                mask_ms = (time.perf_counter() - t0) * 1000
            self._log_mask_stats(mask, weight_diff, epoch_gradient)

            updated_weights = torch.where(mask, predicted_weights.view(-1), current_weights)
            array_to_params(self.model, updated_weights, device=self.device)

            if mem:
                self._log_wandb({
                    "memory/snapshot_storage_mb": snapshot_mb,
                    "memory/svd_workspace_mb": svd_workspace_mb,
                    "memory/allocated_mb": torch.cuda.memory_allocated(self.device) / 1024 ** 2,
                    "memory/reserved_mb": torch.cuda.memory_reserved(self.device) / 1024 ** 2,
                    "memory/peak_mb": torch.cuda.max_memory_allocated(self.device) / 1024 ** 2,
                    "memory/dmd_overhead_mb": torch.cuda.memory_allocated(self.device) / 1024 ** 2 - mem_before,
                })
            if timing:
                self._log_wandb({"timing/svd_ms": svd_ms, "timing/prediction_ms": pred_ms,
                                 "timing/masking_ms": mask_ms, "timing/total_dmd_ms": svd_ms + pred_ms + mask_ms})

        if self._distributed():
            to_send = current_weights if updated_weights is None else updated_weights
            dist.broadcast(to_send.to(self.device), 0)
