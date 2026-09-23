"""Command-line arguments and factories for models, datasets, optimizers and trainers."""

import argparse
import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet18, resnet50, vit_b_16, vit_h_14

from kcl.lib.datasets.ag_news import create_ag_news
from kcl.lib.datasets.cifar import create_cifar10, create_cifar10_single_class
from kcl.lib.datasets.imagenet import create_imagenet
from kcl.lib.models.AlexNet import AlexNet
from kcl.lib.models.fc_networks import FCNet
from kcl.lib.models.fcn import FCN
from kcl.lib.models.text_lstm import TextLSTM
from kcl.lib.trainers.baselines import (NonSelectivePredictionTrainer, RandomAcceleratedTrainer,
                                        RandomMaskTrainer, SwitchByLossTrainer)
from kcl.lib.trainers.pdt_distributed_trainer import PDTDistributedTrainer
from kcl.lib.trainers.pdt_layerwise_trainer import PDTLayerwiseTrainer
from kcl.lib.trainers.pdt_trainer import PDTTrainer
from kcl.lib.trainers.trainer_base import TrainerBase
from kcl.utils.misc import full_seed

MODELS = ["fcnet", "fcn", "alexnet", "resnet18", "resnet50", "vit_base", "vit_huge", "text_lstm"]
DATASETS = ["cifar10", "imagenet", "ag_news"]
NUM_CLASSES = {"cifar10": 10, "imagenet": 1000, "ag_news": 4}
OPTIMIZERS = ["sgd", "sgd_m", "adam", "adamw", "rmsprop", "adadelta", "adagrad", "adamax", "nadam", "shampoo", "lamb"]
TRAINERS = {
    "base": TrainerBase,
    "pdt": PDTTrainer,
    "pdt_distributed": PDTDistributedTrainer,
    "pdt_layerwise": PDTLayerwiseTrainer,
    "nonselective": NonSelectivePredictionTrainer,
    "switch_by_loss": SwitchByLossTrainer,
    "random_accelerated": RandomAcceleratedTrainer,
    "random_mask": RandomMaskTrainer,
}


def get_args(argv=None):
    p = argparse.ArgumentParser(description="Predictive Differential Training (PDT)")

    g = p.add_argument_group("Model and data")
    g.add_argument("--model", type=str, choices=MODELS, default="alexnet")
    g.add_argument("--fc_layers", type=int, default=4, choices=[2, 4, 6],
                   help="depth of the fully connected network used with --model fcnet")
    g.add_argument("--dataset", type=str, choices=DATASETS, default="cifar10")
    g.add_argument("--data_dir", type=str, default="data",
                   help="CIFAR-10: download root; ImageNet: directory with train/ and val/; AG News: unused")
    g.add_argument("--img_size", type=int, default=128, help="CIFAR-10 images are resized to this size")
    g.add_argument("--train_batch_size", type=int, default=256)
    g.add_argument("--test_batch_size", type=int, default=256)
    g.add_argument("--num_workers", type=int, default=4)
    g.add_argument("--single_class_batch", action="store_true",
                   help="non-i.i.d. mini-batches: every training batch contains a single class (CIFAR-10)")
    g.add_argument("--max_seq_len", type=int, default=50, help="AG News: token sequence length")
    g.add_argument("--vocab_size", type=int, default=5000, help="AG News: vocabulary size")

    g = p.add_argument_group("Training")
    g.add_argument("--optimizer", type=str, choices=OPTIMIZERS, default="sgd")
    g.add_argument("--lr", type=float, default=0.05)
    g.add_argument("--weight_decay", type=float, default=0.0, help="used by sgd_m and adamw")
    g.add_argument("--label_smoothing", type=float, default=0.0)
    g.add_argument("--train_epochs", type=int, default=60)
    g.add_argument("--random_seed", type=int, default=0)
    g.add_argument("--cosine_lr_sched", action="store_true", help="cosine annealing to --lr_min")
    g.add_argument("--lr_min", type=float, default=0.0)
    g.add_argument("--lr_warmup_epochs", type=int, default=0, help="linear warm-up epochs before the schedule")
    g.add_argument("--two_stage_training", action="store_true",
                   help="two-stage cosine schedule used for ImageNet (see TrainerBase.get_lr_scheduler)")
    g.add_argument("--first_stage_epochs", type=int, default=200)
    g.add_argument("--distributed", action="store_true",
                   help="DistributedDataParallel; launch with torchrun or srun (one process per GPU)")
    g.add_argument("--w_device", type=str, default=None,
                   help="device holding the weight history for the distributed trainers (default: training device)")

    g = p.add_argument_group("PDT")
    g.add_argument("--trainer", type=str, choices=list(TRAINERS), default="base")
    g.add_argument("--predict_start_epoch", type=int, default=5, help="T_0: first epoch with a prediction")
    g.add_argument("--predict_epoch_interval", type=int, default=1, help="T_i: epochs between predictions")
    g.add_argument("--predicted_num", type=int, default=5, help="tau: number of steps predicted ahead")
    g.add_argument("--n_past_weights", type=int, default=5, help="h: number of past snapshots used by DMD")
    g.add_argument("--svd_rank", type=int, default=0,
                   help="SVD truncation rank for DMD (0 = optimal hard threshold of Gavish and Donoho)")
    g.add_argument("--mask_mode", type=str, default="both", choices=["both", "accel_only", "consistency_only"],
                   help="masking ablation: both criteria (PDT), acceleration only, or consistency only")
    g.add_argument("--save_masks", action="store_true", help="write the boolean mask of every prediction epoch to disk")
    g.add_argument("--no_adaptive_schedule", action="store_true",
                   help="disable the loss-based adjustment of tau / T_i in the distributed trainers")
    g.add_argument("--random_mask_ratio", type=float, default=None,
                   help="fraction of weights selected by the random baselines (defaults differ per baseline)")
    g.add_argument("--random_mask_seed", type=int, default=None, help="seed of the random baselines' mask")

    g = p.add_argument_group("Profiling")
    g.add_argument("--count_flops", action="store_true", help="profiler-based FLOP counting (slow)")
    g.add_argument("--count_flops_fast", action="store_true",
                   help="FLOP counting from one profiled batch plus analytical DMD cost")
    g.add_argument("--profile_memory", action="store_true")
    g.add_argument("--profile_timing", action="store_true")
    g.add_argument("--log_gradient_metrics", action="store_true",
                   help="log norm and direction stability of the epoch update")

    g = p.add_argument_group("Logging")
    g.add_argument("--log_dir", type=str, default="logs/run")
    g.add_argument("--save_freq", type=int, default=0, help="checkpoint every N epochs (0 = final only)")
    g.add_argument("--log_interval", type=int, default=50, help="mini-batches between loss print-outs")
    g.add_argument("--use_wandb", action="store_true")
    g.add_argument("--wandb_project", type=str, default="pdt")
    g.add_argument("--wandb_entity", type=str, default=None)
    g.add_argument("--run_name", type=str, default=None)

    args = p.parse_args(argv)
    if args.count_flops and args.count_flops_fast:
        p.error("--count_flops and --count_flops_fast are mutually exclusive")
    if args.two_stage_training and not args.first_stage_epochs < args.train_epochs:
        p.error("--first_stage_epochs must be smaller than --train_epochs")
    if args.trainer in ("pdt_distributed", "pdt_layerwise") and not args.distributed:
        print("Note: running a distributed trainer in a single process")
    if args.trainer == "pdt" and args.distributed:
        p.error("use --trainer pdt_distributed (or pdt_layerwise) together with --distributed")
    setup_distributed(args)
    return args


def setup_distributed(args):
    """Populate ``dist_rank``, ``ws``, ``local_rank``, ``dist_rank_0`` and ``device``.

    Works with ``torchrun`` (RANK / WORLD_SIZE / LOCAL_RANK) and with SLURM ``srun``
    (SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID). MASTER_ADDR and MASTER_PORT must be set
    in the SLURM case.
    """
    if args.distributed:
        if "RANK" in os.environ:
            args.dist_rank = int(os.environ["RANK"])
            args.ws = int(os.environ["WORLD_SIZE"])
            args.local_rank = int(os.environ.get("LOCAL_RANK", args.dist_rank % max(torch.cuda.device_count(), 1)))
        elif "SLURM_PROCID" in os.environ:
            args.dist_rank = int(os.environ["SLURM_PROCID"])
            args.ws = int(os.environ["SLURM_NTASKS"])
            args.local_rank = int(os.environ.get("SLURM_LOCALID", args.dist_rank % max(torch.cuda.device_count(), 1)))
        else:
            raise RuntimeError("--distributed requires torchrun or SLURM environment variables")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group("nccl", world_size=args.ws, rank=args.dist_rank)
    else:
        args.dist_rank, args.ws, args.local_rank = 0, 1, 0
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.dist_rank_0 = args.dist_rank == 0
    if not args.dist_rank_0:
        args.use_wandb = False
    args.w_device = torch.device(args.w_device) if args.w_device else args.device
    return args


def get_model(args):
    full_seed(args.random_seed)
    n_classes = NUM_CLASSES[args.dataset]
    name = args.model
    if name == "fcnet":
        model = FCNet(num_classes=n_classes, n_layers=args.fc_layers, img_size=args.img_size)
    elif name == "fcn":
        model = FCN(n_classes)
    elif name == "alexnet":
        model = AlexNet(n_classes)
    elif name == "resnet18":
        model = resnet18(weights=None, num_classes=n_classes)
    elif name == "resnet50":
        model = resnet50(weights=None, num_classes=n_classes)
    elif name == "vit_base":
        model = vit_b_16(weights=None, num_classes=n_classes)
    elif name == "vit_huge":
        model = vit_h_14(weights=None, num_classes=n_classes)
    elif name == "text_lstm":
        model = TextLSTM(vocab_size=args.vocab_size, num_classes=n_classes)
    else:
        raise ValueError(name)

    model = model.to(args.device)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank])
    return model


def get_dataset(args):
    """Return ``(train_loader, test_loader)``. For AG News, ``args.vocab_size`` is updated."""
    if args.dataset == "cifar10":
        create = create_cifar10_single_class if args.single_class_batch else create_cifar10
        return create(
            args.data_dir, train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size,
            img_size=args.img_size, num_workers=args.num_workers,
            dist=args.distributed, ws=args.ws, rank=args.dist_rank,
        )
    if args.dataset == "imagenet":
        return create_imagenet(
            args.data_dir, train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size,
            num_workers=args.num_workers, distributed=args.distributed, ws=args.ws, rank=args.dist_rank,
        )
    if args.dataset == "ag_news":
        train, test, vocab_size = create_ag_news(
            train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size,
            max_len=args.max_seq_len, vocab_size=args.vocab_size, num_workers=args.num_workers,
            dist=args.distributed, ws=args.ws, rank=args.dist_rank,
        )
        args.vocab_size = vocab_size
        return train, test
    raise ValueError(args.dataset)


def get_optim(args, model):
    params = model.parameters()
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr)
    if args.optimizer == "sgd_m":
        return torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "rmsprop":
        return torch.optim.RMSprop(params, lr=args.lr)
    if args.optimizer == "adadelta":
        return torch.optim.Adadelta(params, lr=args.lr)
    if args.optimizer == "adagrad":
        return torch.optim.Adagrad(params, lr=args.lr)
    if args.optimizer == "adamax":
        return torch.optim.Adamax(params, lr=args.lr)
    if args.optimizer == "nadam":
        return torch.optim.NAdam(params, lr=args.lr)
    if args.optimizer == "shampoo":
        import torch_optimizer
        return torch_optimizer.Shampoo(params, lr=args.lr, momentum=0.99, weight_decay=0, epsilon=1e-4, update_freq=10)
    if args.optimizer == "lamb":
        import torch_optimizer
        return torch_optimizer.Lamb(params, lr=args.lr)
    raise ValueError(args.optimizer)


def get_trainer(args):
    common = dict(
        save_freq=args.save_freq, train_epochs=args.train_epochs, random_seed=args.random_seed,
        log_dir=args.log_dir, device=args.device, log_interval=args.log_interval,
        use_wandb=args.use_wandb, log_gradient_metrics=args.log_gradient_metrics, args=args,
    )
    pdt = dict(
        svd_rank=args.svd_rank, n_past_weights=args.n_past_weights, predict_start_epoch=args.predict_start_epoch,
        predicted_num=args.predicted_num, predict_epoch_interval=args.predict_epoch_interval,
    )
    name = args.trainer
    if name == "base":
        return TrainerBase(**common)
    if name == "pdt":
        return PDTTrainer(mask_mode=args.mask_mode, save_masks=args.save_masks, **pdt, **common)
    if name in ("pdt_distributed", "pdt_layerwise"):
        cls = TRAINERS[name]
        return cls(mask_mode=args.mask_mode, save_masks=args.save_masks, w_device=args.w_device,
                   adaptive_schedule=not args.no_adaptive_schedule, **pdt, **common)
    if name in ("nonselective", "switch_by_loss"):
        return TRAINERS[name](**pdt, **common)
    if name in ("random_accelerated", "random_mask"):
        extra = {}
        if args.random_mask_ratio is not None:
            extra["random_mask_ratio"] = args.random_mask_ratio
        if args.random_mask_seed is not None:
            extra["random_mask_seed"] = args.random_mask_seed
        return TRAINERS[name](**pdt, **common, **extra)
    raise ValueError(name)
