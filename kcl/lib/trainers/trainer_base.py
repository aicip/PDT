import os
import pickle
import random
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import wandb
from kcl.utils.misc import (full_seed, pretty_size, query_gpu_memory,
                            unpack_batch)
from kcl.utils.weight_tools import flat_params_as_torch

from torch.profiler import profile, record_function, ProfilerActivity

# try:
#     from thop import profile, clever_format
#     THOP_AVAILABLE = True
# except ImportError:
#     THOP_AVAILABLE = False


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
        save_weights=True,
        log_gpu_memory=True,
        use_tb=False,
        log_gradient_metrics=False,
        args=None,
        **kwargs,
    ):
        """
        :param save_freq: how often to save the model and weight history
        :param train_epochs: how many epochs to train for
        :param random_seed: random seed to use
        :param log_dir: directory to save logs to
        :param device: device to use for training
        :param log_interval: how often to log training progress
        """

        if len(kwargs) > 0:
            print("WARNING: unused kwargs: {}".format(kwargs))
        self.save_freq = save_freq
        self.train_epochs = train_epochs
        self.log_dir = log_dir
        self.device = torch.device(device)
        self.log_interval = 10
        self.use_wandb = use_wandb
        self.save_weights = save_weights
        self.log_gpu_memory = log_gpu_memory
        self.log_gradient_metrics = log_gradient_metrics

        self.options = {
            "save_freq": save_freq,
            "train_epochs": train_epochs,
            "log_dir": log_dir,
            "device": device,
            "log_interval": log_interval,
        }

        self.train_losses = []
        self.test_losses = []
        self.accuracies = []
        self.weights = None
        self.model = None
        self.optim = None
        self.cur_epoch = 0
        self.use_tb = use_tb
        self.args = args
        if self.use_tb:
            self.tb = SummaryWriter(log_dir)
        else:
            self.tb = None

        # For gradient analysis (only used if log_gradient_metrics=True)
        self.prev_gradient = None  # Store previous epoch's gradient for direction stability
        
        if self.args.count_flops:
            self.epoch_flops = []
            self.total_flops = 0
            self.batch_forward_flops = None

        if self.args.count_flops_fast:
            self.epoch_flops = []
            self.total_flops = 0
            self.batch_forward_flops = None  # Will be measured once on first batch

        full_seed(random_seed)

        # Optional training features (backward compatible: enabled only when set in args)
        # 1. Label smoothing loss function
        if hasattr(args, 'label_smoothing') and args.label_smoothing > 0:
            self.loss_func = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            print(f"Using label smoothing: {args.label_smoothing}")
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()

        # 2. Model EMA (Exponential Moving Average) - will be initialized in set_model()
        self.model_ema = None
        self.use_model_ema = hasattr(args, 'use_model_ema') and args.use_model_ema
        if self.use_model_ema:
            self.model_ema_decay = getattr(args, 'model_ema_decay', 0.9999)
            print(f"Model EMA enabled (decay={self.model_ema_decay})")

        # 3. Gradient clipping
        self.clip_grad_norm = getattr(args, 'clip_grad_norm', 0.0)
        if self.clip_grad_norm > 0:
            print(f"Gradient norm clipping: {self.clip_grad_norm}")

        # 4. MixUp/CutMix transform (will be set by train_test.py if needed)
        self.mixup_cutmix = None

        torch.set_float32_matmul_precision('medium')

    def log_gpu_memory_stats(self, stage=""):
        """Log GPU memory statistics to wandb with detailed breakdown"""
        if not torch.cuda.is_available():
            return {}

        stats = {
            f"memory/{stage}/allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
            f"memory/{stage}/reserved_mb": torch.cuda.memory_reserved(self.device) / 1024**2,
            f"memory/{stage}/max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1024**2,
        }

        # Add weight snapshot storage if available
        if self.weights is not None:
            stats[f"memory/{stage}/weight_snapshot_mb"] = (
                self.weights.numel() * self.weights.element_size() / 1024**2
            )

        if self.use_wandb:
            wandb.log(stats, step=self.cur_epoch)

        return stats

    def reset_peak_memory_stats(self):
        """Reset peak memory statistics for more accurate per-epoch tracking"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def _log_gradient_metrics(self, weight_differences):
        """
        Log gradient norm and direction stability metrics.
        Used for LR vs mask ratio analysis.

        Args:
            weight_differences: Current epoch's weight update (gradient)
        """
        if not self.log_gradient_metrics or not self.use_wandb:
            return

        # Calculate gradient norm (L2 norm)
        gradient_norm = torch.norm(weight_differences).item()

        metrics = {
            "gradient/norm": gradient_norm,
        }

        # Calculate gradient direction stability (cosine similarity with previous epoch)
        if self.prev_gradient is not None:
            # Cosine similarity between current and previous gradient
            cos_sim = torch.nn.functional.cosine_similarity(
                weight_differences.view(1, -1),
                self.prev_gradient.view(1, -1),
                dim=1
            ).item()

            metrics["gradient/direction_stability"] = cos_sim

            # Also log the angle in degrees for easier interpretation
            angle_deg = torch.acos(torch.clamp(torch.tensor(cos_sim), -1.0, 1.0)) * 180 / 3.14159
            metrics["gradient/direction_angle_deg"] = angle_deg.item()

        # Log all metrics
        wandb.log(metrics, step=self.cur_epoch)

        # Update previous gradient for next epoch
        self.prev_gradient = weight_differences.clone().detach()

    def _update_model_ema(self):
        """
        Update Model EMA using exponential moving average.
        SOTA feature for better evaluation performance.

        EMA update: ema_param = decay * ema_param + (1 - decay) * model_param
        """
        if self.model_ema is None:
            return

        with torch.no_grad():
            for ema_param, model_param in zip(self.model_ema.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.model_ema_decay).add_(
                    model_param.data, alpha=1.0 - self.model_ema_decay
                )

    def get_lr_scheduler(self, optimizer):
        # Support both old (cosine_lr_sched boolean) and new (lr_scheduler string) parameters
        use_cosine = False
        if hasattr(self.args, 'lr_scheduler') and self.args.lr_scheduler is not None:
            # New SOTA parameter (string): 'cosineannealinglr', 'steplr', etc.
            use_cosine = (self.args.lr_scheduler.lower() == 'cosineannealinglr')
        elif hasattr(self.args, 'cosine_lr_sched') and self.args.cosine_lr_sched:
            # Old parameter (boolean) - backward compatible
            use_cosine = True

        if use_cosine:
            # Get lr_min with backward compatibility
            lr_min = getattr(self.args, 'lr_min', 0.0)
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.train_epochs - getattr(self.args, 'lr_warmup_epochs', 0),
                eta_min=lr_min
            )
        else:
            main_lr_scheduler = None

        # Warmup scheduler (compatible with both old and new parameters)
        if hasattr(self.args, 'lr_warmup_epochs') and self.args.lr_warmup_epochs > 0:
            # Use start_factor from args if available, otherwise default to 0.1
            start_factor = getattr(self.args, 'lr_warmup_start_factor', 0.1)
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                total_iters=self.args.lr_warmup_epochs
            )
        else:
            warmup_lr_scheduler = None

        # Combine warmup + main scheduler
        if warmup_lr_scheduler is not None and main_lr_scheduler is not None:
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                milestones=[self.args.lr_warmup_epochs]
            )
        else:
            lr_scheduler = main_lr_scheduler or warmup_lr_scheduler

        return lr_scheduler
    
    def check_batch_distribution(self, batch_idx, batch, max_batches_to_check=5):
        """check the class distribution in the first few batches of the first epoch"""
        if batch_idx >= max_batches_to_check:
            return
            
        _, target = unpack_batch(batch, device='cpu')  # use CPU to avoid GPU memory overhead
        unique_labels = torch.unique(target)
        
        print(f"\nBatch {batch_idx} distribution:")
        print(f"Number of unique classes in batch: {len(unique_labels)}")
        print(f"Classes present: {unique_labels.numpy()}")
        print(f"Batch size: {len(target)}")
        
        # if more than one class, print counts per class
        if len(unique_labels) > 1:
            for label in unique_labels:
                count = (target == label).sum().item()
                print(f"Class {label}: {count} samples")
        print("-" * 50)

    def train(
        self,
        model,
        train_loader,
        optim,
        loss_func,
        to_loss_val=None,
        test_loader=None
    ):
        self.model = model
        self.grad_scaler = torch.cuda.amp.GradScaler()
        self.model.to(torch.float32)
        self.optim = optim
        self.lr_scheduler = self.get_lr_scheduler(optim)
        self.dmd = None

        # Initialize Model EMA after model is set (SOTA feature)
        if self.use_model_ema:
            from copy import deepcopy
            self.model_ema = deepcopy(self.model).to(self.device)
            self.model_ema.eval()
            for param in self.model_ema.parameters():
                param.requires_grad = False
            print(f"Model EMA initialized with decay={self.model_ema_decay}")

        # if self.args.cosine_lr_sched:
        #     main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #         self.optim, T_max=self.train_epochs - self.args.lr_warmup_epochs, eta_min=self.args.lr_min
        #     )
        # else:
        #     main_lr_scheduler= None

        # if self.args.lr_warmup_epochs > 0:
        #     warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        #         self.optim, start_factor=0.01, total_iters=self.args.lr_warmup_epochs
        #     )
        # else:
        #     warmup_lr_scheduler = None
        
        # if warmup_lr_scheduler is not None and main_lr_scheduler is not None:
        #     lr_scheduler = None
        # elif warmup_lr_scheduler is not None and main_lr_scheduler is None:
        #     lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        #         self.optim, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[self.args.lr_warmup_epochs]
        #     )
        # else:
        #     lr_scheduler = main_lr_scheduler or warmup_lr_scheduler

        n = sum(p.numel() for p in self.model.parameters())
        n2 = self.model.parameters().__next__().element_size()

        print(f"Training model of size {pretty_size(n * n2)}")

        pbar = None
        if to_loss_val is None:
            pbar = tqdm(
                total=self.train_epochs,
                unit="epoch",
                desc="Train",
                position=0
            )
            condition = lambda epoch: (epoch - 1) < self.train_epochs
        else:

            def condition(epoch):
                if epoch == 0:
                    return True
                return np.mean(self.train_losses[-5:]) > to_loss_val

        while condition(self.cur_epoch):
            self.test(test_loader, loss_func)
            if self.use_wandb:
                wandb.log(
                    {
                        "accuracy": self.accuracies[-1],
                        "test_loss": np.mean(self.test_losses[-1])
                    },
                    step=self.cur_epoch
                )
            self.train_epoch(train_loader, loss_func)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.use_wandb:
                wandb.log(
                    {"train_loss": np.mean(self.train_losses[-1]),
                    "learning_rate": self.optim.param_groups[0]['lr']},
                    step=self.cur_epoch,
                )
                # if self.log_gpu_memory:
                # TODO: update to select correct GPU
                #     wandb.log({"gpu_memory": query_gpu_memory()[1]}, step=self.cur_epoch)

            if self.use_tb:
                self.tb.add_scalar(
                    "train_loss", np.mean(self.train_losses[-1]),
                    self.cur_epoch
                )
                self.tb.add_scalar(
                    "test_loss", np.mean(self.test_losses[-1]), self.cur_epoch
                )
                self.tb.add_scalar(
                    "accuracy", self.accuracies[-1], self.cur_epoch
                )

            if pbar:
                pbar.update(1)
                if self.weights is not None:
                    tqdm.write(
                        "size of weights matrix: {}".format(
                            pretty_size(
                                int(self.weights.numel()) *
                                self.weights.element_size()
                            )
                        )
                    )
            else:
                if self.weights is not None:
                    tqdm.write(
                        f"Epoch {self.cur_epoch:04d} | Size of weights matrix: {pretty_size(self.weights.numel() * self.weights.element_size())}"
                    )
                else:
                    tqdm.write(f"Epoch {self.cur_epoch:04d}")

            if (
                self.save_freq is not None
                and self.cur_epoch % self.save_freq == 0
                and self.cur_epoch != 0
            ):
                self.save_all(
                    os.path.join(self.log_dir, f"epoch_{self.cur_epoch:04d}")
                )

            self.cur_epoch += 1

        if pbar:
            pbar.close()

    def save_all(self, save_path):
        tqdm.write("Saving network and weight history")
        os.makedirs(save_path, exist_ok=True)

        torch.save(
            self.model.state_dict(), os.path.join(save_path, "model.pth")
        )
        torch.save(
            self.optim.state_dict(), os.path.join(save_path, "optimizer.pth")
        )
        if isinstance(self.weights, torch.Tensor):
            torch.save(self.weights, os.path.join(save_path, "weights.pth"))
        elif isinstance(self.weights, np.ndarray):
            np.save(os.path.join(save_path, "weights.npy"), self.weights)

        # self.weights = None

        with open(os.path.join(save_path, "meta.pkl"), "wb") as f:
            data = {
                "train_losses": self.train_losses,
                "options": self.options,
                "cur_epoch": self.cur_epoch,
                "test_loss": self.test_losses,
                "accuracies": self.accuracies,
            }
            if self.args.count_flops:
                data.update({
                    "epoch_flops": self.epoch_flops,
                    "total_flops": self.total_flops,
                })
            pickle.dump(data, f)

    @torch.no_grad()
    def test(self, test_loader, loss_func):
        # Use Model EMA for evaluation if enabled (SOTA feature)
        eval_model = self.model_ema if (self.use_model_ema and self.model_ema is not None) else self.model
        eval_model.eval()

        pbar = tqdm(
            total=len(test_loader),
            position=1,
            unit="batch",
            leave=True,
            desc="Test",
            disable=(not self.args.dist_rank_0)
        )

        losses = []
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                data, target = unpack_batch(batch, device=self.device)
                # output = eval_model(data)

                # loss = loss_func(output, target)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    output = eval_model(data)
                    loss = loss_func(output, target)
                losses.append(loss.item())
                predicted = torch.argmax(output, dim=1)
                correct_preds += (predicted == target).sum().item()
                total_preds += target.shape[0]

                pbar.set_postfix({"test_loss": loss.item()})
                pbar.update(1)
        self.test_losses.append(losses)
        self.accuracies.append(correct_preds / total_preds)

        return losses

    def train_epoch(self, train_loader, loss_func):
        """
        Train the model for one epoch
        :param train_loader: the training data loader
        :param loss_func: the loss function
        """
        # Timing profiling: epoch start
        if self.args.profile_timing:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            epoch_start_time = time.perf_counter()
            batch_times = []  # Store batch times for averaging

        # Gradient metrics: capture weights before training (for baseline gradient analysis)
        if self.log_gradient_metrics:
            weights_before = flat_params_as_torch(self.model).clone()

        if self.save_weights:
            if self.weights is None:
                self.weights = flat_params_as_torch(self.model).reshape(
                    (-1, 1)
                )
            else:
                self.weights = torch.cat(
                    [
                        self.weights,
                        flat_params_as_torch(self.model).reshape((-1, 1))
                    ],
                    dim=1
                )
        self.model.train()
        pbar = tqdm(
            total=len(train_loader),
            position=1,
            unit="batch",
            leave=False,
            desc="Epoch {}".format(self.cur_epoch),
        )
        losses = []

        # code for checking batch distribution
        if self.cur_epoch == 0:  # only check in the first epoch
            print("\nChecking batch distribution in first epoch...")

        do_profile = self.args.count_flops and self.batch_forward_flops is None  # only profile on first batch
        do_profile_fast = self.args.count_flops_fast and self.batch_forward_flops is None  # only profile on first batch for fast mode

        for batch_idx, (batch) in enumerate(train_loader):
            if do_profile and batch_idx == 0:
                # only profile the first batch
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_flops=True,
                    profile_memory=True
                ) as prof:
                    with record_function("complete_iteration"):
                        data, target = unpack_batch(batch, device=self.device)

                        # Apply MixUp/CutMix if enabled (SOTA feature)
                        if self.mixup_cutmix is not None:
                            data, target = self.mixup_cutmix(data, target)

                        self.optim.zero_grad()
                        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                            output = self.model(data)
                            # Handle soft labels for MixUp/CutMix
                            if self.mixup_cutmix is not None and target.dim() == 2:
                                loss = -torch.sum(target * torch.nn.functional.log_softmax(output, dim=1), dim=1).mean()
                            else:
                                loss = loss_func(output, target)
                        self.grad_scaler.scale(loss).backward()

                        # Gradient clipping if enabled (SOTA feature)
                        if self.clip_grad_norm > 0:
                            self.grad_scaler.unscale_(self.optim)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                        self.grad_scaler.step(self.optim)
                        self.grad_scaler.update()

                        # Update Model EMA if enabled (SOTA feature)
                        if self.use_model_ema and self.model_ema is not None:
                            self._update_model_ema()
                
                print("\nComplete iteration profiling results:")
                print(prof.key_averages().table(sort_by="flops", row_limit=-1))
                
                # save the FLOPs for the forward pass
                self.batch_forward_flops = sum(e.flops for e in prof.key_averages())
                
                if self.use_wandb:
                    wandb.config.update({
                        "batch_forward_flops": self.batch_forward_flops,
                    }, allow_val_change=True)

            elif do_profile_fast and batch_idx == 0:
                # Profile first batch for count_flops_fast mode
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_flops=True,
                    profile_memory=True
                ) as prof:
                    with record_function("complete_iteration"):
                        data, target = unpack_batch(batch, device=self.device)

                        # Apply MixUp/CutMix if enabled (SOTA feature)
                        if self.mixup_cutmix is not None:
                            data, target = self.mixup_cutmix(data, target)

                        self.optim.zero_grad()
                        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                            output = self.model(data)
                            # Handle soft labels for MixUp/CutMix
                            if self.mixup_cutmix is not None and target.dim() == 2:
                                loss = -torch.sum(target * torch.nn.functional.log_softmax(output, dim=1), dim=1).mean()
                            else:
                                loss = loss_func(output, target)
                        self.grad_scaler.scale(loss).backward()

                        # Gradient clipping if enabled (SOTA feature)
                        if self.clip_grad_norm > 0:
                            self.grad_scaler.unscale_(self.optim)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                        self.grad_scaler.step(self.optim)
                        self.grad_scaler.update()

                        # Update Model EMA if enabled (SOTA feature)
                        if self.use_model_ema and self.model_ema is not None:
                            self._update_model_ema()

                self.batch_forward_flops = sum(e.flops for e in prof.key_averages())

                if self.use_wandb:
                    wandb.config.update({
                        "batch_forward_flops": self.batch_forward_flops,
                    }, allow_val_change=True)

            else:
                # normal training

                # only check batch distribution in first epoch, and only for the first few batches
                if self.cur_epoch == 0 and batch_idx < 5:
                    self.check_batch_distribution(batch_idx, batch)

                # Timing profiling: batch start (measure first 100 batches for averaging)
                if self.args.profile_timing and batch_idx < 100:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    batch_start_time = time.perf_counter()

                data, target = unpack_batch(batch, device=self.device)

                # Apply MixUp/CutMix if enabled (SOTA feature)
                if self.mixup_cutmix is not None:
                    data, target = self.mixup_cutmix(data, target)

                self.optim.zero_grad()

                with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16
                ):
                    output = self.model(data)
                    # For soft labels (MixUp/CutMix), compute soft target loss
                    if self.mixup_cutmix is not None and target.dim() == 2:
                        # Soft target: target is [batch_size, num_classes]
                        loss = -torch.sum(target * torch.nn.functional.log_softmax(output, dim=1), dim=1).mean()
                    else:
                        # Hard target: use standard loss function
                        loss = loss_func(output, target)

                self.grad_scaler.scale(loss).backward()

                # Gradient clipping if enabled (SOTA feature)
                if self.clip_grad_norm > 0:
                    self.grad_scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                self.grad_scaler.step(self.optim)
                self.grad_scaler.update()

                # Update Model EMA if enabled (SOTA feature)
                if self.use_model_ema and self.model_ema is not None:
                    self._update_model_ema()

                # Timing profiling: batch end
                if self.args.profile_timing and batch_idx < 100:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    batch_end_time = time.perf_counter()
                    batch_times.append((batch_end_time - batch_start_time) * 1000)  # Convert to ms
                
                # output = self.model(data)
                # loss = loss_func(output, target)
                # loss.backward()
                # self.optim.step()
            if batch_idx % self.log_interval == 0:
                tqdm.write("Loss: {:.4f}".format(loss.item()))
            losses.append(loss.item())
            pbar.update(1)

        # Timing profiling: epoch end
        if self.args.profile_timing:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            epoch_end_time = time.perf_counter()
            epoch_time_s = epoch_end_time - epoch_start_time
            avg_batch_time_ms = np.mean(batch_times) if len(batch_times) > 0 else 0

            if self.use_wandb:
                wandb.log({
                    "timing_baseline/epoch_time_s": epoch_time_s,
                    "timing_baseline/avg_batch_time_ms": avg_batch_time_ms,
                    "timing_baseline/total_batches": len(train_loader),
                }, step=self.cur_epoch)

        # Memory profiling: log baseline memory usage
        if self.args.profile_memory and torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            mem_peak = torch.cuda.max_memory_allocated(self.device) / 1024**2

            if self.use_wandb:
                wandb.log({
                    "memory_baseline/allocated_mb": mem_allocated,
                    "memory_baseline/reserved_mb": mem_reserved,
                    "memory_baseline/peak_mb": mem_peak,
                }, step=self.cur_epoch)

        if self.args.count_flops:
            epoch_flops = self.batch_forward_flops * len(train_loader)
            self.epoch_flops.append(epoch_flops)
            self.total_flops += epoch_flops

            if self.use_wandb:
                wandb.log({
                    "epoch_flops": epoch_flops,
                    "total_flops": self.total_flops,
                }, step=self.cur_epoch)

        if self.args.count_flops_fast:
            epoch_flops = self.batch_forward_flops * len(train_loader) if self.batch_forward_flops is not None else 0
            self.epoch_flops.append(epoch_flops)
            self.total_flops += epoch_flops

            if self.use_wandb:
                wandb.log({
                    "flops_fast/epoch_flops": epoch_flops,
                    "flops_fast/total_flops": self.total_flops,
                }, step=self.cur_epoch)

        self.train_losses.append(losses)

        # Gradient metrics: log gradient norm and direction stability (for baseline)
        if self.log_gradient_metrics:
            weights_after = flat_params_as_torch(self.model).clone()
            weight_differences = weights_after - weights_before
            self._log_gradient_metrics(weight_differences)

        # print the learning rate
        tqdm.write(f"Learning rate: {self.optim.param_groups[0]['lr']}")
        pbar.close()

    def val_score(self):
        """
        :return: the validation score
        """
        return np.mean(self.test_losses[-1])
