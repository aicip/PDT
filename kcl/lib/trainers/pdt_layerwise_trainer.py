"""PDT-Layerwise: PDT with one DMD model per layer group.

This variant was used for the ViT-Huge experiment of the paper, where a single DMD on
the full 632M-dimensional weight vector does not fit in GPU memory. The parameter
vector is split into layer groups (one group per transformer-block component, plus one
group per remaining top-level module); each group keeps its own weight history and its
own DMD model, and the mask is computed per group with the same criteria as
:class:`PDTTrainer`. Because every group is modelled by a separate operator, this is a
block-diagonal approximation of the global dynamics, i.e. an algorithmic variant and not
only a memory optimisation.

The trainer inherits the distributed synchronisation and the acceleration scheduler of
:class:`PDTDistributedTrainer`. Running out of GPU memory anywhere in the prediction step
(including inside the DMD of a single layer group) fails the whole step: rank 0 counts
the failure, relaxes the schedule after repeated failures, and all ranks keep the weights
of the optimizer epoch. Rank 0 first broadcasts a success flag and only then the assembled
weights, so the other ranks never wait for a broadcast that does not come. Other runtime
errors inside one group's DMD (for example a failed SVD) skip only that group. Gradient
checkpointing is switched on
for transformer blocks that expose a ``gradient_checkpointing`` attribute (torchvision's
ViT blocks do not, in which case nothing changes). A group is predicted only once its
history holds at least three snapshots, so ``n_past_weights`` must be at least 3.

Note: the implementation used for the ViT-Huge experiment kept a single outer
``torch.cuda.amp.autocast()`` context around the whole prediction epoch (optimizer epoch
and DMD step). Keeping one autocast context open across optimizer updates lets the
forward passes reuse cached low-precision parameter casts made before those updates. The
released trainer scopes autocast to the individual forward passes, as every other
trainer does (see README, "Implementation notes").
"""

from collections import OrderedDict

import torch
import torch.distributed as dist

from kcl.lib.dmd.torchDMD import TorchDMD
from kcl.lib.trainers.pdt_distributed_trainer import PDTDistributedTrainer
from kcl.utils.weight_tools import array_to_params, flat_params_as_torch


class PDTLayerwiseTrainer(PDTDistributedTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_weights = {}  # layer group -> weight history (columns = epochs)
        self.dmd_failures = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        if self.n_past_weights is not None and self.n_past_weights < 3:
            raise ValueError("PDT-Layerwise needs at least three snapshots per layer group: "
                             "set --n_past_weights to 3 or more")
        if self.is_main:
            print("PDT-Layerwise: one DMD model per layer group")

    # ------------------------------------------------------- layer grouping
    def _extract_layer_weights(self, flat_weights):
        """Split a flat parameter vector into layer groups (dict of 1-D tensors)."""
        if not hasattr(self, "param_indices"):
            self.param_indices = OrderedDict()
            idx = 0
            model = self.model.module if hasattr(self.model, "module") else self.model
            for name, param in model.named_parameters():
                if "encoder.layers" in name:
                    parts = name.split(".")
                    if len(parts) >= 4:  # encoder.layers.<block>.<component>...
                        layer_key = f"encoder.{parts[2]}.{parts[3]}"
                    else:
                        layer_key = f"encoder.{parts[2]}"
                else:
                    layer_key = name.split(".")[0]
                self.param_indices.setdefault(layer_key, []).append((idx, idx + param.numel()))
                idx += param.numel()
            if self.is_main:
                print(f"Model divided into {len(self.param_indices)} layer groups for DMD")
                for layer_key, indices in self.param_indices.items():
                    total = sum(end - start for start, end in indices)
                    print(f"  {layer_key}: {total / 1e6:.2f}M parameters")

        layer_weights = {}
        for layer_key, indices in self.param_indices.items():
            parts = [flat_weights[start:end] for start, end in indices]
            try:
                layer_weights[layer_key] = torch.cat(parts)
            except RuntimeError as e:  # out of memory on the storage device
                print(f"Concatenating layer group {layer_key} on CPU: {e}")
                layer_weights[layer_key] = torch.cat([p.cpu() for p in parts]).to(self.storage_device)
        torch.cuda.empty_cache()
        return layer_weights

    def _reconstruct_flat_weights(self, layer_predictions, fallback):
        """Assemble a flat vector from per-group vectors; groups without a prediction keep ``fallback``."""
        reconstructed = fallback.clone().to(self.device)
        for layer_key, layer_pred in layer_predictions.items():
            offset = 0
            for start, end in self.param_indices[layer_key]:
                reconstructed[start:end] = layer_pred[offset:offset + (end - start)].to(self.device)
                offset += end - start
        return reconstructed

    def _enable_gradient_checkpointing(self):
        if getattr(self, "grad_checkpointing_enabled", False):
            return
        self.grad_checkpointing_enabled = True
        model = self.model.module if hasattr(self.model, "module") else self.model
        enabled = 0
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            for module in model.encoder.layers:
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = True
                    enabled += 1
        if self.is_main:
            if enabled:
                print(f"Gradient checkpointing enabled on {enabled} transformer blocks")
            else:
                print("Gradient checkpointing not available for this model (no block exposes it)")

    # ------------------------------------------------------------------ epoch
    def _history_length(self):
        if self._distributed():
            n = torch.tensor(self._local_history_length() if self.is_main else 0, device=self.device)
            dist.broadcast(n, 0)
            return int(n.item())
        return self._local_history_length()

    def _local_history_length(self):
        if not self.layer_weights:
            return 0
        return min(w.shape[1] for w in self.layer_weights.values())

    def train_epoch(self, train_loader, loss_func):
        predicted = False
        if self._should_predict():
            predicted = True
            self._enable_gradient_checkpointing()
            # flat_params_as_torch returns a new tensor (torch.cat), so no clone is needed;
            # this saves one copy of the parameter vector on very large models
            weights_before = flat_params_as_torch(self.model)
            self._train_normal(train_loader, loss_func)
            delta = (flat_params_as_torch(self.model) - weights_before).to(self.storage_device, non_blocking=True)
            del weights_before
            torch.cuda.empty_cache()
            predicted = self._predict_and_update(delta)
            if predicted:
                self.consecutive_failures = 0
                self.epoch_since_last_predict = 1
            else:
                # all ranks keep the weights of the optimizer epoch; rank 0 adjusts the
                # schedule and the state is broadcast in _end_of_epoch_schedule
                self.epoch_since_last_predict = self.predict_epoch_interval
            self._log_wandb({"dmd_failures": self.dmd_failures})
        else:
            self._train_normal(train_loader, loss_func)
            if self._history_ready():
                self.epoch_since_last_predict += 1

        self.epoch_extra.update({"prediction_epoch": int(predicted), "predicted_num": self.predicted_num,
                                 "predict_epoch_interval": self.predict_epoch_interval})
        self._end_of_epoch_schedule()

    def _handle_prediction_failure(self, error):
        """Rank 0 only: count the failure and relax the schedule after repeated out-of-memory errors."""
        if "out of memory" in str(error).lower():
            self.dmd_failures += 1
            self.consecutive_failures += 1
            print(f"Out of memory during the prediction step (failure #{self.dmd_failures}): {error}")
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.predict_epoch_interval = min(self.predict_epoch_interval + 2, 10)
                self.predicted_num = max(self.predicted_num - 1, 2)
                self.consecutive_failures = 0
            if self.predicted_num > 2:
                self.predicted_num -= 1
        else:
            print(f"Error during the prediction step: {error}")
        torch.cuda.empty_cache()
        print("Falling back to standard training for this epoch")

    def _train_normal(self, train_loader, loss_func):
        if self.is_main:
            flat = flat_params_as_torch(self.model).to(self.storage_device)
            for layer_key, w in self._extract_layer_weights(flat).items():
                col = w.reshape(-1, 1)
                if layer_key not in self.layer_weights:
                    self.layer_weights[layer_key] = col
                else:
                    try:
                        self.layer_weights[layer_key] = torch.cat([self.layer_weights[layer_key], col], dim=1)
                    except RuntimeError as e:
                        print(f"Concatenating history of {layer_key} on CPU: {e}")
                        hist = torch.cat([self.layer_weights[layer_key].cpu(), col.cpu()], dim=1)
                        self.layer_weights[layer_key] = hist.to(self.storage_device)
                if self.n_past_weights is not None and self.layer_weights[layer_key].shape[1] > self.n_past_weights:
                    self.layer_weights[layer_key] = self.layer_weights[layer_key][:, -self.n_past_weights:]
            torch.cuda.empty_cache()
        self._run_sgd_epoch(train_loader, loss_func, record_history=False)

    # ------------------------------------------------------------ prediction
    def _predict_and_update(self, epoch_gradient):
        """Predict on rank 0 and apply the result on all ranks.

        Returns True if the prediction was applied. In distributed mode rank 0 first
        broadcasts whether its prediction succeeded; the assembled weights are broadcast
        only on success, so a failure on rank 0 cannot leave the other ranks waiting.
        """
        updated_weights = None
        if self.is_main:
            try:
                updated_weights = self._predict_weights(epoch_gradient)
            except RuntimeError as e:  # e.g. out of memory in the DMD step
                self._handle_prediction_failure(e)

        if self._distributed():
            status = torch.tensor(int(updated_weights is not None), device=self.device)
            dist.broadcast(status, 0)
            if int(status.item()) == 0:
                return False
            buffer = updated_weights.to(self.device) if self.is_main else flat_params_as_torch(self.model).to(self.device)
            dist.broadcast(buffer, 0)
            array_to_params(self.model, buffer, device=self.device)
            return True

        if updated_weights is None:
            return False
        array_to_params(self.model, updated_weights, device=self.device)
        return True

    def _predict_weights(self, epoch_gradient):
        """Rank 0: fit one DMD per layer group and assemble the flat weight vector on ``self.device``.

        Returns None if no group could be predicted. An out-of-memory error in any group
        propagates and fails the whole step.
        """
        current_weights = flat_params_as_torch(self.model).clone().to(self.storage_device)
        current_groups = self._extract_layer_weights(current_weights)
        gradient_groups = self._extract_layer_weights(epoch_gradient)

        predicted_groups = {}
        total_params = 0
        masked_params = 0
        for layer_name, history in self.layer_weights.items():
            total_params += history.shape[0]
            if history.shape[1] < 3:
                continue  # not enough snapshots for this group yet
            try:
                if self.n_past_weights is not None:
                    history = history[:, -self.n_past_weights:]
                dmd = TorchDMD(rank=min(self.svd_rank, history.shape[1] - 1))
                dmd.fit(history)
                predicted = dmd.predict_multistep(history[:, -1].reshape(-1, 1), self.predicted_num)
                weight_diff = (predicted[:, -1] - history[:, -1]).view(-1)
                layer_gradient = gradient_groups[layer_name].view(-1)
                mask = self.compute_mask(weight_diff, layer_gradient)
                masked_params += mask.sum().item()
                predicted_groups[layer_name] = torch.where(mask, predicted.view(-1), current_groups[layer_name])
                if self.is_main:
                    print(f"  {layer_name}: accepted {mask.sum().item() / len(mask):.2%}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise  # fails the whole prediction step (handled by _predict_and_update)
                print(f"  {layer_name}: prediction skipped ({e})")  # e.g. SVD failure
            torch.cuda.empty_cache()

        if predicted_groups:
            updated_weights = self._reconstruct_flat_weights(predicted_groups, current_weights)
            mask_ratio = masked_params / total_params if total_params else 0.0
            self.epoch_extra["mask_ratio"] = round(mask_ratio, 6)
            print(f"Epoch {self.cur_epoch:04d} | accepted predictions: {mask_ratio:.2%} of {total_params} parameters")
            self._log_wandb({"mask_ratio_global": mask_ratio})
        else:
            updated_weights = None
            print("No layer group was predicted this epoch")
        torch.cuda.empty_cache()
        return updated_weights
