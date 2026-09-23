import os
import pickle
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import wandb
from kcl.utils.misc import (full_seed, pretty_size, query_gpu_memory)
from kcl.utils.weight_tools import flat_params_as_torch

from torch.profiler import profile, record_function, ProfilerActivity


class SSLTrainerBase:
    """
    Base trainer for Self-Supervised Learning methods.
    This class provides common training infrastructure while remaining
    agnostic to specific SSL method implementations.
    """
    
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
        args=None,
        **kwargs,
    ):
        if len(kwargs) > 0:
            print("WARNING: unused kwargs: {}".format(kwargs))
            
        self.save_freq = save_freq
        self.train_epochs = train_epochs
        self.log_dir = log_dir
        self.device = torch.device(device)
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.save_weights = save_weights
        self.log_gpu_memory = log_gpu_memory

        self.options = {
            "save_freq": save_freq,
            "train_epochs": train_epochs,
            "log_dir": log_dir,
            "device": device,
            "log_interval": log_interval,
        }

        # Training metrics - generic for all SSL methods
        self.train_losses = []
        self.ssl_metrics = []  # For method-specific metrics
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
            
        # FLOP counting support
        if self.args and hasattr(self.args, 'count_flops') and self.args.count_flops:
            self.epoch_flops = []
            self.total_flops = 0
            self.batch_forward_flops = None

        full_seed(random_seed)
        torch.set_float32_matmul_precision('medium')

    def get_lr_scheduler(self, optimizer):
        """Create learning rate scheduler - same as supervised version"""
        if hasattr(self.args, 'cosine_lr_sched') and self.args.cosine_lr_sched:
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.train_epochs - getattr(self.args, 'lr_warmup_epochs', 0),
                eta_min=getattr(self.args, 'lr_min', 0.0)
            )
        else:
            main_lr_scheduler = None

        if hasattr(self.args, 'lr_warmup_epochs') and self.args.lr_warmup_epochs > 0:
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.1, 
                total_iters=self.args.lr_warmup_epochs
            )
        else:
            warmup_lr_scheduler = None

        if warmup_lr_scheduler is not None and main_lr_scheduler is not None:
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, 
                schedulers=[warmup_lr_scheduler, main_lr_scheduler], 
                milestones=[self.args.lr_warmup_epochs]
            )
        else:
            lr_scheduler = main_lr_scheduler or warmup_lr_scheduler

        return lr_scheduler

    def train(
        self,
        model,
        train_loader,
        optim,
        loss_func,
        to_loss_val=None,
        eval_loader=None,
        eval_func=None
    ):
        """
        Main training loop - generic SSL training framework
        """
        self.model = model
        self.grad_scaler = torch.cuda.amp.GradScaler()
        self.model.to(torch.float32)
        self.optim = optim
        self.lr_scheduler = self.get_lr_scheduler(optim)
        
        n = sum(p.numel() for p in self.model.parameters())
        n2 = self.model.parameters().__next__().element_size()
        print(f"Training SSL model of size {pretty_size(n * n2)}")

        pbar = None
        if to_loss_val is None:
            pbar = tqdm(
                total=self.train_epochs,
                unit="epoch", 
                desc="SSL Train",
                position=0
            )
            condition = lambda epoch: (epoch - 1) < self.train_epochs
        else:
            def condition(epoch):
                if epoch == 0:
                    return True
                return np.mean(self.train_losses[-5:]) > to_loss_val

        while self.cur_epoch < self.train_epochs:
            # Optional evaluation
            if (eval_loader is not None and eval_func is not None and self.cur_epoch % getattr(self.args, 'eval_freq', 1) == 0):
                self.evaluate(eval_loader, eval_func)
                
            # Train for one epoch
            self.train_epoch(train_loader, loss_func)
            
            # Update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
            # Logging
            self._log_epoch_metrics()
            
            # Progress updates
            self._update_progress(pbar)

            # Save checkpoints
            if (self.save_freq is not None and 
                self.cur_epoch % self.save_freq == 0 and 
                self.cur_epoch != 0):
                self.save_all(os.path.join(self.log_dir, f"epoch_{self.cur_epoch:04d}"))

            self.cur_epoch += 1

        if pbar:
            pbar.close()

    def _log_epoch_metrics(self):
        """Handle logging after each epoch"""
        if self.use_wandb:
            log_dict = {
                "train_loss": np.mean(self.train_losses[-1]),
                "learning_rate": self.optim.param_groups[0]['lr'],
                "epoch": self.cur_epoch
            }
            
            # Add SSL-specific metrics if available
            if len(self.ssl_metrics) > 0 and isinstance(self.ssl_metrics[-1], dict):
                log_dict.update(self.ssl_metrics[-1])
                
            wandb.log(log_dict, step=self.cur_epoch)

        if self.use_tb:
            self.tb.add_scalar("train_loss", np.mean(self.train_losses[-1]), self.cur_epoch)

    def _update_progress(self, pbar):
        """Update progress bar and print status"""
        if pbar:
            pbar.update(1)
            if self.weights is not None:
                tqdm.write(f"Size of weights matrix: {pretty_size(self.weights.numel() * self.weights.element_size())}")
        else:
            if self.weights is not None:
                tqdm.write(f"Epoch {self.cur_epoch:04d} | Size of weights matrix: {pretty_size(self.weights.numel() * self.weights.element_size())}")
            else:
                tqdm.write(f"Epoch {self.cur_epoch:04d}")

    def save_all(self, save_path):
        """Save model, optimizer, and training history"""
        tqdm.write("Saving SSL network and weight history")
        os.makedirs(save_path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(save_path, "model.pth"))
        torch.save(self.optim.state_dict(), os.path.join(save_path, "optimizer.pth"))
        
        if isinstance(self.weights, torch.Tensor):
            torch.save(self.weights, os.path.join(save_path, "weights.pth"))
        elif isinstance(self.weights, np.ndarray):
            np.save(os.path.join(save_path, "weights.npy"), self.weights)

        with open(os.path.join(save_path, "meta.pkl"), "wb") as f:
            data = {
                "train_losses": self.train_losses,
                "ssl_metrics": self.ssl_metrics,
                "options": self.options,
                "cur_epoch": self.cur_epoch,
            }
            if hasattr(self, 'epoch_flops'):
                data.update({
                    "epoch_flops": self.epoch_flops,
                    "total_flops": self.total_flops,
                })
            pickle.dump(data, f)

    def evaluate(self, eval_loader, eval_func):
        """
        Optional evaluation hook - can be overridden by subclasses
        """
        self.model.eval()
        
        with torch.no_grad():
            metric_value = eval_func(self.model, eval_loader, self.device)
            self.ssl_metrics.append({"eval_metric": metric_value})
            
        self.model.train()
        return metric_value

    def train_epoch(self, train_loader, loss_func):
        """
        Train the model for one epoch.
        This method provides the common training loop structure,
        delegating SSL-specific logic to subclasses.
        """
        # Store weights for potential use in prediction-based methods
        if self.save_weights:
            if self.weights is None:
                self.weights = flat_params_as_torch(self.model).reshape((-1, 1))
            else:
                self.weights = torch.cat([
                    self.weights,
                    flat_params_as_torch(self.model).reshape((-1, 1))
                ], dim=1)
                
        self.model.train()
        pbar = tqdm(
            total=len(train_loader),
            position=1,
            unit="batch",
            leave=False,
            desc="Epoch {}".format(self.cur_epoch),
        )
        losses = []

        # FLOP profiling setup
        do_profile = (hasattr(self, 'args') and 
                     hasattr(self.args, 'count_flops') and 
                     self.args.count_flops and 
                     self.batch_forward_flops is None)

        for batch_idx, batch in enumerate(train_loader):
            if do_profile and batch_idx == 0:
                # Profile first batch for FLOP counting
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_flops=True,
                    profile_memory=True
                ) as prof:
                    with record_function("ssl_iteration"):
                        loss = self.ssl_forward_step(batch, loss_func)
                
                print("\nSSL iteration profiling results:")
                print(prof.key_averages().table(sort_by="flops", row_limit=-1))
                
                self.batch_forward_flops = sum(e.flops for e in prof.key_averages())
                
                if self.use_wandb:
                    wandb.config.update({"batch_forward_flops": self.batch_forward_flops}, allow_val_change=True)
            else:
                loss = self.ssl_forward_step(batch, loss_func)

            if batch_idx % self.log_interval == 0:
                tqdm.write("Loss: {:.4f}".format(loss.item()))
            losses.append(loss.item())
            pbar.update(1)

        # Record FLOP usage
        if hasattr(self, 'args') and hasattr(self.args, 'count_flops') and self.args.count_flops:
            epoch_flops = self.batch_forward_flops * len(train_loader) if self.batch_forward_flops else 0
            self.epoch_flops.append(epoch_flops)
            self.total_flops += epoch_flops
            
            if self.use_wandb:
                wandb.log({
                    "epoch_flops": epoch_flops,
                    "total_flops": self.total_flops,
                }, step=self.cur_epoch)

        self.train_losses.append(losses)
        tqdm.write(f"Learning rate: {self.optim.param_groups[0]['lr']}")
        pbar.close()

    def ssl_forward_step(self, batch, loss_func):
        """
        SSL-specific forward step - MUST be implemented by subclasses.
        
        Args:
            batch: Batch data (format depends on SSL method)
            loss_func: Loss function for the SSL method
            
        Returns:
            torch.Tensor: Computed loss
        """
        raise NotImplementedError("Subclasses must implement ssl_forward_step")

    def val_score(self):
        """Return validation score for early stopping, etc."""
        if len(self.ssl_metrics) > 0 and isinstance(self.ssl_metrics[-1], dict):
            return self.ssl_metrics[-1].get("eval_metric", float('inf'))
        else:
            return np.mean(self.train_losses[-1]) if len(self.train_losses) > 0 else float('inf')