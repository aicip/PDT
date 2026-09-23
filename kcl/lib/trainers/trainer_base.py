"""Baseline trainer: standard mini-batch training with the chosen optimizer.

All PDT trainers and baselines derive from :class:`TrainerBase` and override
``train_epoch``. The base class owns the training loop, evaluation, learning-rate
schedule, checkpointing, wandb / CSV logging and the optional profiling hooks.
"""

import os
import pickle
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

from kcl.utils.metrics_logger import MetricsLogger
from kcl.utils.misc import full_seed, pretty_size, unpack_batch
from kcl.utils.weight_tools import flat_params_as_torch


class TrainerBase:
    def __init__(
        self,
        *,
        save_freq=10,
        train_epochs=1000,
        random_seed=None,
        log_dir="logs",
        device="cuda",
        log_interval=10,
        use_wandb=False,
        save_weights=False,
        log_gradient_metrics=False,
        args=None,
        **kwargs,
    ):
        """
        :param save_freq: write model, optimizer and metadata every ``save_freq`` epochs (0 = never)
        :param train_epochs: number of epochs to train
        :param random_seed: seed for torch / numpy / python
        :param log_dir: directory for checkpoints, ``metrics.csv`` and (optionally) masks
        :param device: device the model is trained on
        :param log_interval: number of mini-batches between loss print-outs
        :param use_wandb: log to Weights & Biases (``wandb.init`` must be called by the caller)
        :param save_weights: keep a history of flattened weights in the baseline trainer
            (PDT trainers always keep their own history)
        :param log_gradient_metrics: log the norm and direction stability of the epoch update
        :param args: parsed command-line arguments (profiling flags etc.)
        """
        self.args = args
        self.save_freq = save_freq
        self.train_epochs = train_epochs
        self.random_seed = random_seed
        self.log_dir = log_dir
        self.device = torch.device(device)
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.save_weights = save_weights
        self.log_gradient_metrics = log_gradient_metrics
        self.is_main = bool(getattr(args, "dist_rank_0", True))
        self.options = {k: (str(v) if isinstance(v, torch.device) else v) for k, v in vars(args).items()} if args is not None else {}

        self.model = None
        self.optim = None
        self.lr_scheduler = None
        self.grad_scaler = None
        self.dmd = None
        self.weights = None  # weight history, one column per epoch
        self.cur_epoch = 0
        self.train_losses = []
        self.test_losses = []
        self.accuracies = []
        self.prev_gradient = None

        # profiling state
        self.epoch_flops = []
        self.total_flops = 0
        self.batch_forward_flops = None
        self._last_sgd_epoch_time = None
        self._last_sgd_batch_time = None

        # per-epoch values that subclasses want in metrics.csv (e.g. mask ratio)
        self.epoch_extra = {}

        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics = MetricsLogger(self.log_dir, enabled=self.is_main)
        full_seed(random_seed)

    # ------------------------------------------------------------------ helpers
    def _flag(self, name):
        return bool(getattr(self.args, name, False))

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _distributed(self):
        return dist.is_available() and dist.is_initialized()

    def _reduce_sum(self, values):
        """Sum a list of python numbers across DDP ranks; returns a python list."""
        t = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self._distributed():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.tolist()

    def _log_wandb(self, metrics):
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=self.cur_epoch)

    # ------------------------------------------------------------- lr schedule
    def get_lr_scheduler(self, optimizer):
        args = self.args
        if not (self._flag("cosine_lr_sched") or self._flag("two_stage_training")):
            return None
        warmup = getattr(args, "lr_warmup_epochs", 0) or 0
        lr_min = getattr(args, "lr_min", 0.0) or 0.0

        if self._flag("two_stage_training"):
            # Two-stage schedule used for the ImageNet experiments: cosine annealing to
            # ``lr_min`` over the first stage, then a second cosine stage that restarts
            # at ten times ``lr_min``.
            first_stage = args.first_stage_epochs
            stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=first_stage - warmup, eta_min=lr_min)

            class SecondStageLR(torch.optim.lr_scheduler.CosineAnnealingLR):
                def __init__(self, opt, T_max, eta_min):
                    super().__init__(opt, T_max, eta_min)
                    self.base_lrs = [self.eta_min * 10 for _ in self.base_lrs]

            stage2 = SecondStageLR(optimizer, T_max=self.train_epochs - first_stage, eta_min=lr_min)
            schedulers, milestones = [stage1, stage2], [first_stage]
            if warmup > 0:
                schedulers.insert(0, torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup))
                milestones = [warmup, first_stage]
            return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

        main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.train_epochs - warmup, eta_min=lr_min)
        if warmup > 0:
            warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
            return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm, main], milestones=[warmup])
        return main

    # ------------------------------------------------------------- main loop
    def train(self, model, train_loader, optim, loss_func, test_loader=None):
        self.model = model
        self.model.to(torch.float32)
        self.optim = optim
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))
        self.lr_scheduler = self.get_lr_scheduler(optim)
        self.dmd = None

        n = sum(p.numel() for p in self.model.parameters())
        n2 = next(self.model.parameters()).element_size()
        if self.is_main:
            print(f"Training a model with {n / 1e6:.2f}M parameters ({pretty_size(n * n2)})")

        pbar = tqdm(total=self.train_epochs, unit="epoch", desc="Train", position=0, disable=not self.is_main)
        while self.cur_epoch < self.train_epochs:
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.cur_epoch)

            self.epoch_extra = {}
            self._sync()
            t0 = time.perf_counter()
            self.train_epoch(train_loader, loss_func)
            self._sync()
            epoch_time = time.perf_counter() - t0

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if test_loader is not None:
                self.test(test_loader, loss_func)

            train_loss = self._global_train_loss()
            lr = self.optim.param_groups[0]["lr"]
            log = {"train_loss": train_loss, "learning_rate": lr, "epoch_time_s": epoch_time}
            if test_loader is not None:
                log.update({"accuracy": self.accuracies[-1], "test_loss": np.mean(self.test_losses[-1])})
            self._log_wandb(log)
            self.metrics.log(
                epoch=self.cur_epoch,
                train_loss=round(train_loss, 6),
                test_loss=round(float(np.mean(self.test_losses[-1])), 6) if test_loader is not None else None,
                accuracy=round(self.accuracies[-1], 6) if test_loader is not None else None,
                lr=lr,
                epoch_time_s=round(epoch_time, 3),
                **self.epoch_extra,
            )

            if self.is_main:
                pbar.update(1)
                msg = f"Epoch {self.cur_epoch:04d} | train loss {train_loss:.4f}"
                if test_loader is not None:
                    msg += f" | test loss {np.mean(self.test_losses[-1]):.4f} | accuracy {self.accuracies[-1]:.4f}"
                msg += f" | lr {lr:.2e} | {epoch_time:.1f}s"
                tqdm.write(msg)

            if self.save_freq and self.cur_epoch % self.save_freq == 0 and self.cur_epoch != 0 and self.is_main:
                self.save_all(os.path.join(self.log_dir, f"epoch_{self.cur_epoch:04d}"))

            self.cur_epoch += 1

        pbar.close()
        if self.is_main:
            self.save_all(os.path.join(self.log_dir, "final"))

    def _global_train_loss(self):
        local = self.train_losses[-1] if self.train_losses else []
        s, n = self._reduce_sum([float(np.sum(local)), float(len(local))])
        return s / max(n, 1.0)

    def save_all(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
        torch.save(self.optim.state_dict(), os.path.join(save_path, "optimizer.pth"))
        with open(os.path.join(save_path, "meta.pkl"), "wb") as f:
            data = {
                "train_losses": self.train_losses,
                "options": self.options,
                "cur_epoch": self.cur_epoch,
                "test_loss": self.test_losses,
                "accuracies": self.accuracies,
            }
            if self._flag("count_flops") or self._flag("count_flops_fast"):
                data.update({"epoch_flops": self.epoch_flops, "total_flops": self.total_flops})
            pickle.dump(data, f)

    # ------------------------------------------------------------- evaluation
    @torch.no_grad()
    def test(self, test_loader, loss_func):
        self.model.eval()
        pbar = tqdm(total=len(test_loader), position=1, unit="batch", leave=False, desc="Test", disable=not self.is_main)
        loss_sum, n_batches, correct, total = 0.0, 0, 0, 0
        for batch in test_loader:
            data, target = unpack_batch(batch, device=self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                output = self.model(data)
                loss = loss_func(output, target)
            loss_sum += loss.item()
            n_batches += 1
            correct += (torch.argmax(output, dim=1) == target).sum().item()
            total += target.shape[0]
            pbar.update(1)
        pbar.close()
        # aggregate over all ranks when the validation set is sharded
        loss_sum, n_batches, correct, total = self._reduce_sum([loss_sum, n_batches, correct, total])
        self.test_losses.append([loss_sum / max(n_batches, 1)])
        self.accuracies.append(correct / max(total, 1))
        return self.test_losses[-1]

    # ---------------------------------------------------------- one SGD epoch
    def _sgd_step(self, batch, loss_func):
        data, target = unpack_batch(batch, device=self.device)
        self.optim.zero_grad()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            output = self.model(data)
            loss = loss_func(output, target)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optim)
        self.grad_scaler.update()
        return loss

    def _run_sgd_epoch(self, train_loader, loss_func, record_history):
        """Train for one epoch with the base optimizer.

        If ``record_history`` is true, the flattened parameters *before* the epoch are
        appended to ``self.weights`` (the snapshot matrix used by DMD).
        """
        if record_history:
            snapshot = flat_params_as_torch(self.model).reshape((-1, 1))
            self.weights = snapshot if self.weights is None else torch.cat([self.weights, snapshot], dim=1)

        profile_first_batch = (
            (self._flag("count_flops") or self._flag("count_flops_fast"))
            and self.batch_forward_flops is None and self.is_main
        )
        timing = self._flag("profile_timing") and self.is_main
        if timing:
            self._sync()
            epoch_start = time.perf_counter()
            batch_times = []

        self.model.train()
        pbar = tqdm(total=len(train_loader), position=1, unit="batch", leave=False,
                    desc=f"Epoch {self.cur_epoch}", disable=not self.is_main)
        losses = []
        for batch_idx, batch in enumerate(train_loader):
            if profile_first_batch and batch_idx == 0:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             record_shapes=True, with_flops=True, profile_memory=True) as prof:
                    with record_function("complete_iteration"):
                        loss = self._sgd_step(batch, loss_func)
                if self._flag("count_flops"):
                    print(prof.key_averages().table(sort_by="flops", row_limit=-1))
                self.batch_forward_flops = sum(e.flops for e in prof.key_averages())
                if self.use_wandb:
                    import wandb
                    wandb.config.update({"batch_forward_flops": self.batch_forward_flops}, allow_val_change=True)
            else:
                if timing and batch_idx < 100:
                    self._sync()
                    t0 = time.perf_counter()
                loss = self._sgd_step(batch, loss_func)
                if timing and batch_idx < 100:
                    self._sync()
                    batch_times.append((time.perf_counter() - t0) * 1000)
            losses.append(loss.item())
            if batch_idx % self.log_interval == 0 and self.is_main:
                tqdm.write("Loss: {:.4f}".format(loss.item()))
            pbar.update(1)
        pbar.close()
        self.train_losses.append(losses)

        if timing:
            self._sync()
            self._last_sgd_epoch_time = time.perf_counter() - epoch_start
            self._last_sgd_batch_time = float(np.mean(batch_times)) if batch_times else 0.0

    def _train_normal(self, train_loader, loss_func):
        """One epoch of the base optimizer, always recording the weight history."""
        self._run_sgd_epoch(train_loader, loss_func, record_history=True)

    def train_epoch(self, train_loader, loss_func):
        """Baseline: one epoch of the base optimizer, no prediction."""
        if self.log_gradient_metrics:
            weights_before = flat_params_as_torch(self.model).clone()
        self._run_sgd_epoch(train_loader, loss_func, record_history=self.save_weights)
        if self.log_gradient_metrics:
            self._log_gradient_metrics(flat_params_as_torch(self.model) - weights_before)
        self._log_epoch_profiling(len(train_loader), dmd_flops=0)

    # -------------------------------------------------------------- profiling
    def _log_epoch_profiling(self, n_batches, dmd_flops=0, prefix="baseline"):
        """Log FLOPs / timing / memory of the epoch that just finished (if enabled).

        The metrics are sent to wandb (with ``--use_wandb``) and printed on the main process.
        """
        metrics = {}
        if self._flag("count_flops") or self._flag("count_flops_fast"):
            epoch_flops = (self.batch_forward_flops or 0) * n_batches + dmd_flops
            self.epoch_flops.append(epoch_flops)
            self.total_flops += epoch_flops
            metrics.update({"flops/epoch_flops": epoch_flops, "flops/total_flops": self.total_flops,
                            "flops/dmd_flops": dmd_flops})
        if self._flag("profile_timing") and self._last_sgd_epoch_time is not None:
            metrics.update({f"timing_{prefix}/sgd_epoch_time_s": self._last_sgd_epoch_time,
                            f"timing_{prefix}/avg_batch_time_ms": self._last_sgd_batch_time,
                            f"timing_{prefix}/total_batches": n_batches})
        if self._flag("profile_memory") and self.device.type == "cuda":
            metrics.update({
                f"memory_{prefix}/allocated_mb": torch.cuda.memory_allocated(self.device) / 1024 ** 2,
                f"memory_{prefix}/reserved_mb": torch.cuda.memory_reserved(self.device) / 1024 ** 2,
                f"memory_{prefix}/peak_mb": torch.cuda.max_memory_allocated(self.device) / 1024 ** 2,
            })
        if metrics:
            self._log_wandb(metrics)
            if self.is_main:
                print(f"Epoch {self.cur_epoch:04d} | profiling: "
                      + ", ".join(f"{k}={v:.4g}" for k, v in metrics.items()))

    def _log_gradient_metrics(self, weight_differences):
        """Log the norm of the epoch update and its cosine similarity to the previous one."""
        if not self.log_gradient_metrics:
            return
        metrics = {"gradient/norm": torch.norm(weight_differences).item()}
        if self.prev_gradient is not None:
            cos_sim = torch.nn.functional.cosine_similarity(
                weight_differences.view(1, -1), self.prev_gradient.view(1, -1), dim=1
            ).item()
            metrics["gradient/direction_stability"] = cos_sim
            metrics["gradient/direction_angle_deg"] = float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))))
        self._log_wandb(metrics)
        self.prev_gradient = weight_differences.clone().detach()

    def val_score(self):
        return np.mean(self.test_losses[-1])
