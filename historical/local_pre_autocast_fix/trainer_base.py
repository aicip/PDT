import os
import pickle
import random

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
        
        if self.args.count_flops:
            self.epoch_flops = []
            self.total_flops = 0
            self.batch_forward_flops = None

        full_seed(random_seed)

        torch.set_float32_matmul_precision('medium')

    def get_lr_scheduler(self, optimizer):
        if self.args.cosine_lr_sched:
            # main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            #     optimizer, 
            #     T_0=self.train_epochs - self.args.lr_warmup_epochs, 
            #     T_mult=1, 
            #     eta_min=self.args.lr_min
            # )
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.train_epochs - self.args.lr_warmup_epochs,
            eta_min=self.args.lr_min
        )
        else:
            main_lr_scheduler = None

        if self.args.lr_warmup_epochs > 0:
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
        self.model.eval()

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
                # output = self.model(data)

                # loss = loss_func(output, target)
                
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    output = self.model(data)
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
                        self.optim.zero_grad()
                        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                            output = self.model(data)
                            loss = loss_func(output, target)
                        self.grad_scaler.scale(loss).backward()
                        self.grad_scaler.step(self.optim)
                        self.grad_scaler.update()
                
                print("\nComplete iteration profiling results:")
                print(prof.key_averages().table(sort_by="flops", row_limit=-1))
                
                # save the FLOPs for the forward pass
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

                data, target = unpack_batch(batch, device=self.device)

                self.optim.zero_grad()

                with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16
                ):
                    output = self.model(data)
                    loss = loss_func(output, target)

                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optim)
                self.grad_scaler.update()
                
                # output = self.model(data)
                # loss = loss_func(output, target)
                # loss.backward()
                # self.optim.step()
            if batch_idx % self.log_interval == 0:
                tqdm.write("Loss: {:.4f}".format(loss.item()))
            losses.append(loss.item())
            pbar.update(1)

        if self.args.count_flops:
            epoch_flops = self.batch_forward_flops * len(train_loader)
            self.epoch_flops.append(epoch_flops)
            self.total_flops += epoch_flops
            
            if self.use_wandb:
                wandb.log({
                    "epoch_flops": epoch_flops,
                    "total_flops": self.total_flops,
                }, step=self.cur_epoch)
                    
        self.train_losses.append(losses)
        # print the learning rate
        tqdm.write(f"Learning rate: {self.optim.param_groups[0]['lr']}")
        pbar.close()

    def val_score(self):
        """
        :return: the validation score
        """
        return np.mean(self.test_losses[-1])
