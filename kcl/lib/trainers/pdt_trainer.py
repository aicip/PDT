"""Predictive Differential Training (PDT), single-GPU implementation.

Every ``predict_epoch_interval`` epochs after ``predict_start_epoch`` the trainer

1. trains one epoch with the base optimizer and records the epoch update
   ``delta_sgd = w_after - w_before``;
2. fits a DMD model to the last ``n_past_weights`` weight snapshots and predicts the
   weights ``predicted_num`` steps ahead (``w_pred``);
3. builds the mask of high-fidelity predictions from the dynamic-consistency and
   acceleration-effectiveness criteria (see ``compute_mask``);
4. replaces the masked coordinates of the current weights by their predictions.

Implementation notes relative to the paper (kept as they were used for all reported
experiments; see README, section "Implementation notes"):

* the acceleration upper bound is ``(predicted_num + 1) * |delta_sgd|`` and the lower
  bound is inclusive (``>=``);
* the dynamic-consistency criterion compares the sign of the *final* predicted
  displacement with the sign of ``delta_sgd``;
* both displacements are measured from the weights at the start of the epoch, which is
  the last column of the snapshot matrix.
"""

import os
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from kcl.lib.dmd.torchDMD import TorchDMD
from kcl.lib.trainers.trainer_base import TrainerBase
from kcl.utils.weight_tools import array_to_params, flat_params_as_torch


class PDTTrainer(TrainerBase):
    def __init__(
        self,
        *,
        svd_rank=0,
        n_past_weights=5,
        predict_start_epoch=5,
        predicted_num=5,
        predict_epoch_interval=1,
        mask_mode="both",
        save_masks=False,
        **kwargs,
    ):
        """
        :param svd_rank: SVD truncation rank for DMD (0 = optimal hard threshold, see
            ``kcl.lib.dmd.utils.compute_svd_torch``)
        :param n_past_weights: number of past epoch snapshots used by DMD (``h`` in the paper)
        :param predict_start_epoch: first epoch at which a prediction may be made (``T_0``)
        :param predicted_num: number of steps predicted ahead (``tau``)
        :param predict_epoch_interval: epochs between two predictions (``T_i``)
        :param mask_mode: ``both`` (full PDT), ``accel_only`` (Eq. 6 only) or
            ``consistency_only`` (Eq. 7 only); the latter two are the ablations of the paper
        :param save_masks: write the boolean mask of every prediction epoch to
            ``<log_dir>/masks/`` (large: one bool per parameter)
        """
        super().__init__(**kwargs)
        self.svd_rank = svd_rank
        self.n_past_weights = n_past_weights
        self.predicted_num = predicted_num
        self.predict_start_epoch = predict_start_epoch
        self.predict_epoch_interval = predict_epoch_interval
        self.mask_mode = mask_mode
        self.save_masks = save_masks
        self.epoch_since_last_predict = 0
        self.first_predict = True
        if self.is_main:
            print(f"PDT: tau={predicted_num}, interval={predict_epoch_interval}, "
                  f"start={predict_start_epoch}, history={n_past_weights}, mask_mode={mask_mode}")

    # ------------------------------------------------------------------ epoch
    def _history_ready(self):
        return self.weights is not None and self.weights.shape[1] >= max(self.svd_rank + 1, self.predict_start_epoch)

    def _should_predict(self):
        if not self._history_ready():
            return False
        if self.first_predict:
            self.first_predict = False
            self.epoch_since_last_predict = self.predict_epoch_interval
        return self.epoch_since_last_predict >= self.predict_epoch_interval

    def train_epoch(self, train_loader, loss_func):
        timing = self._flag("profile_timing") and self.is_main
        if timing:
            self._sync()
            epoch_start = time.perf_counter()
        predicted = False
        dmd_flops = 0

        if self._should_predict():
            predicted = True
            if self._flag("count_flops"):
                # profiler-based FLOPs: profile the whole epoch including the DMD step
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             record_shapes=True, with_flops=True, profile_memory=True) as prof:
                    with record_function("sgd_epoch_plus_dmd"):
                        delta_sgd = self._sgd_epoch_with_delta(train_loader, loss_func)
                        self._predict_and_update(delta_sgd)
                dmd_flops = sum(e.flops for e in prof.key_averages()) - (self.batch_forward_flops or 0) * len(train_loader)
            else:
                delta_sgd = self._sgd_epoch_with_delta(train_loader, loss_func)
                if self._flag("count_flops_fast"):
                    dmd_flops = self._dmd_flops_analytical()
                self._predict_and_update(delta_sgd)
            self.epoch_since_last_predict = 1
        else:
            self._sgd_epoch_with_delta(train_loader, loss_func)
            if self._history_ready():
                self.epoch_since_last_predict += 1

        self.epoch_extra.update({"prediction_epoch": int(predicted), "predicted_num": self.predicted_num,
                                 "predict_epoch_interval": self.predict_epoch_interval})
        self._log_epoch_profiling(len(train_loader), dmd_flops=dmd_flops, prefix="pdt")
        if timing:
            self._sync()
            total = time.perf_counter() - epoch_start
            log = {"timing_pdt/total_epoch_time_s": total, "timing_pdt/has_dmd": float(predicted)}
            if predicted and self._last_sgd_epoch_time is not None:
                log["timing_pdt/dmd_overhead_s"] = total - self._last_sgd_epoch_time
                log["timing_pdt/dmd_overhead_ratio"] = (total - self._last_sgd_epoch_time) / total if total > 0 else 0.0
            self._log_wandb(log)

    def _sgd_epoch_with_delta(self, train_loader, loss_func):
        """One epoch of the base optimizer; returns the flattened epoch update."""
        weights_before = flat_params_as_torch(self.model).clone()
        self._train_normal(train_loader, loss_func)
        delta = flat_params_as_torch(self.model) - weights_before
        self._log_gradient_metrics(delta)
        return delta

    # --------------------------------------------------------------- masking
    def compute_mask(self, weight_diff, epoch_gradient):
        """Mask of accepted predictions.

        ``weight_diff`` is ``w_pred - w_ref`` and ``epoch_gradient`` is ``w_sgd - w_ref``, both
        measured from the same reference weights.
        """
        same_direction = torch.sign(weight_diff) == torch.sign(epoch_gradient)
        lower = torch.abs(weight_diff) >= torch.abs(epoch_gradient)
        upper = torch.abs(weight_diff) <= (self.predicted_num + 1) * torch.abs(epoch_gradient)
        if self.mask_mode == "both":
            return same_direction & lower & upper
        if self.mask_mode == "accel_only":
            return lower & upper
        if self.mask_mode == "consistency_only":
            return same_direction
        raise ValueError(f"Unknown mask_mode: {self.mask_mode}")

    def _log_mask_stats(self, mask, weight_diff, epoch_gradient):
        n = len(mask)
        same = torch.sign(weight_diff) == torch.sign(epoch_gradient)
        mask_ratio = mask.sum().item() / n
        stats = {
            "mask_ratio_global": mask_ratio,
            "ratio_opposite": (~same).sum().item() / n,
            "ratio_n_plus": (same & (torch.abs(weight_diff) > (self.predicted_num + 1) * torch.abs(epoch_gradient))).sum().item() / n,
            "ratio_0_1": (same & (torch.abs(weight_diff) < torch.abs(epoch_gradient))).sum().item() / n,
        }
        self.epoch_extra["mask_ratio"] = round(mask_ratio, 6)
        if self.is_main:
            print(f"Epoch {self.cur_epoch:04d} | accepted predictions: {mask_ratio:.2%} of {n} parameters")
        self._log_wandb(stats)
        if self.save_masks and self.is_main:
            mask_dir = os.path.join(self.log_dir, "masks")
            os.makedirs(mask_dir, exist_ok=True)
            torch.save(mask.cpu(), os.path.join(mask_dir, f"mask_epoch_{self.cur_epoch:04d}.pt"))

    # ------------------------------------------------------------ prediction
    def _predict_and_update(self, epoch_gradient):
        if self.weights is None:
            return
        if epoch_gradient.dim() != 1 or epoch_gradient.size(0) != self.weights.size(0):
            raise ValueError("epoch_gradient must be a flat vector with one entry per parameter")

        mem = self._flag("profile_memory") and self.device.type == "cuda"
        timing = self._flag("profile_timing")
        if mem:
            mem_before = torch.cuda.memory_allocated(self.device) / 1024 ** 2

        current_weights = flat_params_as_torch(self.model).clone()  # weights after this epoch's SGD
        if self.n_past_weights is not None:
            self.weights = self.weights[:, -self.n_past_weights:]
        self.weights.requires_grad = False
        if mem:
            snapshot_mb = self.weights.numel() * self.weights.element_size() / 1024 ** 2
            mem_before_svd = torch.cuda.memory_allocated(self.device) / 1024 ** 2

        # 1. DMD fit on the snapshot matrix
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

        # 2. predict tau steps ahead from the last snapshot
        if timing:
            t0 = time.perf_counter()
        predicted_weights = self.dmd.predict_multistep(self.weights[:, -1].reshape(-1, 1), self.predicted_num)
        if timing:
            self._sync()
            pred_ms = (time.perf_counter() - t0) * 1000
        weight_diff = (predicted_weights[:, -1] - self.weights[:, -1]).view(-1)
        epoch_gradient = epoch_gradient.view(-1)

        # 3. mask
        if timing:
            t0 = time.perf_counter()
        mask = self.compute_mask(weight_diff, epoch_gradient)
        if timing:
            self._sync()
            mask_ms = (time.perf_counter() - t0) * 1000
        self._log_mask_stats(mask, weight_diff, epoch_gradient)

        # 4. assemble: predicted weights where accepted, SGD weights elsewhere
        updated_weights = torch.where(mask, predicted_weights.view(-1), current_weights)
        array_to_params(self.model, updated_weights, device=self.weights.device)

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

    # ------------------------------------------------------------- profiling
    def _dmd_flops_analytical(self):
        """Analytical FLOP count of one DMD step (thin SVD + operator + prediction + mask)."""
        N = sum(p.numel() for p in self.model.parameters())
        h = self.n_past_weights if self.n_past_weights is not None else self.weights.shape[1]
        tau = self.predicted_num
        svd_flops = 2 * N * h * h + 11 * h * h * h
        dmd_matrix_flops = 2 * N * h * h
        prediction_flops = N * h * tau
        mask_flops = 3 * N
        total = svd_flops + dmd_matrix_flops + prediction_flops + mask_flops
        self._log_wandb({"flops/dmd_svd_flops": svd_flops, "flops/dmd_matrix_flops": dmd_matrix_flops,
                         "flops/dmd_prediction_flops": prediction_flops, "flops/dmd_total_flops": total})
        return total
