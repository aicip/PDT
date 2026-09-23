import argparse
import copy
import os

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_optimizer

from kcl.lib.models.simsiam import create_simsiam_model, simsiam_loss
from kcl.lib.datasets.simsiam_datasets import create_simsiam_cifar10_with_probe, create_simsiam_imagenet_with_probe
from kcl.lib.trainers.simsiam_trainer import SimSiamTrainer
from kcl.lib.trainers.simsiam_predicted_trainer import SimSiamPredictedTrainer

from kcl.utils.misc import full_seed, str2bool


def get_ssl_model(args, device=None):
    """
    Create SSL model based on arguments
    
    Args:
        args: Parsed command line arguments
        device: Device to place model on
        
    Returns:
        SSL model instance
    """
    full_seed(args.random_seed)
    
    # Determine number of classes based on dataset
    if args.dataset == 'cifar10':
        n_classes = 10
    elif args.dataset == 'imagenet':
        n_classes = 1000
    else:
        n_classes = None  # For custom datasets
    
    # Create model based on SSL method
    if args.ssl_method.lower() == 'simsiam':
        model = create_simsiam_model(
            dataset=args.dataset,
            backbone=args.backbone,
            dim=args.projection_dim,
            pred_dim=args.prediction_dim,
            num_classes=n_classes
        )
    else:
        raise NotImplementedError(f"SSL method {args.ssl_method} not implemented")
    
    # Move to device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Handle distributed training
    if args.distributed:
        model = model.cuda(args.dist_rank)
        model = DDP(model, device_ids=[args.dist_rank])
    
    return model


def get_ssl_dataset(args):
    """
    Create SSL dataset based on arguments (dispatch only)
    """
    if args.dataset == 'cifar10':
        return create_simsiam_cifar10_with_probe(
            data_dir=args.data_dir,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            add_suffix=args.local_mode,
            img_size=args.img_size,
            dist=args.distributed,
            ws=args.ws,
            rank=args.dist_rank,
            num_workers=args.num_workers
        )
    elif args.dataset == 'imagenet':
        return create_simsiam_imagenet_with_probe(
            data_dir=args.data_dir,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            img_size=args.img_size,
            dist=args.distributed,
            ws=args.ws,
            rank=args.dist_rank,
            num_workers=args.num_workers
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented for SSL")

def get_ssl_optim(args, model):
    """
    Create optimizer for SSL training
    
    Args:
        args: Parsed command line arguments
        model: Model to optimize
        
    Returns:
        Optimizer instance
    """
    if args.optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd_m":
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "rmsprop":
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adadelta":
        optim = torch.optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adagrad":
        optim = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamax":
        optim = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "nadam":
        optim = torch.optim.NAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "shampoo":
        optim = torch_optimizer.Shampoo(
            model.parameters(), lr=args.lr, momentum=0.99, weight_decay=args.weight_decay, 
            epsilon=1e-4, update_freq=10
        )
    elif args.optimizer == "lamb":
        optim = torch_optimizer.Lamb(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")

    return optim


def get_ssl_trainer(args, device=None):
    """
    Create SSL trainer based on arguments
    
    Args:
        args: Parsed command line arguments
        device: Device for training
        
    Returns:
        SSL trainer instance
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Base trainer arguments
    trainer_kwargs = {
        'save_freq': args.save_freq,
        'train_epochs': args.train_epochs,
        'random_seed': args.random_seed,
        'log_dir': args.log_dir,
        'device': device,
        'log_interval': args.log_interval,
        'use_wandb': args.use_wandb and args.dist_rank_0,
        'save_weights': args.save_weights,
        'args': args
    }
    
    # Create trainer based on method and whether PDT is enabled
    if args.ssl_method.lower() == 'simsiam':
        if args.trainer == 'base':
            trainer = SimSiamTrainer(**trainer_kwargs)
        elif args.trainer == 'predicted':
            trainer_kwargs.update({
                'svd_rank': args.svd_rank,
                'n_past_weights': args.n_past_weights,
                'predict_start_epoch': args.predict_start_epoch,
                'predicted_num': args.predicted_num,
                'predict_epoch_interval': args.predict_epoch_interval,
            })
            trainer = SimSiamPredictedTrainer(**trainer_kwargs)
        else:
            raise ValueError(f"Unknown trainer type: {args.trainer}")
    else:
        raise NotImplementedError(f"SSL method {args.ssl_method} not implemented")
    
    # Handle distributed training setup
    if args.distributed:
        rank = args.dist_rank
        ws = args.ws
        torch.distributed.init_process_group('nccl', world_size=ws, rank=rank)
        torch.cuda.set_device(rank)
    
    return trainer


def get_ssl_loss_func(args):
    """
    Create loss function for SSL training
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Loss function
    """
    if args.ssl_method.lower() == 'simsiam':
        return simsiam_loss
    else:
        raise NotImplementedError(f"Loss function for {args.ssl_method} not implemented")


def get_ssl_args():
    """
    Parse command line arguments for SSL training
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Self-Supervised Learning Training')
    
    # Model arguments
    model_p = parser.add_argument_group("Model")
    model_p.add_argument(
        "--ssl_method", type=str, default='simsiam',
        choices=['simsiam'],
        help="Self-supervised learning method"
    )
    model_p.add_argument(
        "--backbone", type=str, default='resnet18',
        choices=['resnet18', 'resnet50'],
        help="Backbone architecture"
    )
    model_p.add_argument(
        "--projection_dim", type=int, default=512,
        help="Projection head output dimension"
    )
    model_p.add_argument(
        "--prediction_dim", type=int, default=128,
        help="Prediction head hidden dimension"
    )
    model_p.add_argument('--random_seed', type=int, default=0)

    # Data arguments
    data_p = parser.add_argument_group("Data")
    data_p.add_argument(
        "--dataset", type=str, default='cifar10',
        choices=['cifar10', 'imagenet'],
        help="Dataset for training"
    )
    data_p.add_argument("--data_dir", type=str, default='~/data')
    data_p.add_argument("--train_batch_size", type=int, default=256)
    data_p.add_argument("--test_batch_size", type=int, default=256)
    data_p.add_argument("--img_size", type=int, default=32)
    data_p.add_argument("--num_workers", type=int, default=4)
    data_p.add_argument(
        '--local_mode', action='store_true', default=False,
        help="Adds suffix to data dir for each dataset to avoid collision"
    )

    # Training arguments
    train_p = parser.add_argument_group("Training")
    train_p.add_argument(
        "--optimizer", type=str, default='sgd_m',
        choices=['adam', 'sgd', 'sgd_m', 'rmsprop', 'adadelta', 'adagrad', 
                'adamax', 'nadam', 'adamw', 'shampoo', 'lamb'],
        help="Optimizer for training"
    )
    train_p.add_argument('--lr', type=float, default=0.03, help="Learning rate")
    train_p.add_argument('--momentum', type=float, default=0.9, help="Momentum for SGD")
    train_p.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    train_p.add_argument(
        '--trainer', type=str, default='base',
        choices=['base', 'predicted'],
        help="Trainer type (base for standard SSL, predicted for SSL+PDT)"
    )
    train_p.add_argument('--train_epochs', type=int, default=100, help="Number of epochs to train")
    train_p.add_argument('--distributed', action='store_true', help="Use distributed training")
    
    # Learning rate scheduler
    train_p.add_argument('--cosine_lr_sched', action='store_true', help="Use cosine annealing LR")
    train_p.add_argument('--lr_min', type=float, default=0.0, help="Minimum LR for cosine annealing")
    train_p.add_argument('--lr_warmup_epochs', type=int, default=0, help="Warmup epochs")
    
    # SSL-specific arguments
    ssl_p = parser.add_argument_group("SSL")
    ssl_p.add_argument('--eval_freq', type=int, default=10, help="Evaluation frequency")
    ssl_p.add_argument('--save_weights', action='store_true', default=True, 
                      help="Save weight history (required for PDT)")
    
    # Logging arguments
    log_p = parser.add_argument_group("Logging")
    log_p.add_argument('--save_freq', type=int, default=10, help="Model save frequency")
    log_p.add_argument('--log_interval', type=int, default=10, help="Log interval")
    log_p.add_argument('--log_dir', type=str, default='logs/ssl')
    log_p.add_argument('--job_name', type=str, default=None, help="Job name for wandb")
    log_p.add_argument('--use_wandb', action='store_true', default=False)
    log_p.add_argument('--save_masks', action='store_true', default=False, help="Write the mask of every prediction epoch to disk")
    log_p.add_argument('--wandb_project_name', type=str, default='ssl_experiments')
    log_p.add_argument('--wandb_entity', type=str, default=None)
    log_p.add_argument('--count_flops', action='store_true', default=False, help="Count FLOPs")
    
    # PDT arguments (for when we implement SimSiamPredictedTrainer)
    pdt_p = parser.add_argument_group("PDT")
    pdt_p.add_argument('--svd_rank', type=int, default=10, help="SVD rank for DMD")
    pdt_p.add_argument('--n_past_weights', type=int, default=5, help="Number of past weights for PDT")
    pdt_p.add_argument('--predict_start_epoch', type=int, default=10, help="Start epoch for prediction")
    pdt_p.add_argument('--predicted_num', type=int, default=5, help="Number of predicted steps")
    pdt_p.add_argument('--predict_epoch_interval', type=int, default=5, help="Prediction interval")
    
    args = parser.parse_args()

    # Handle distributed training
    if args.distributed:
        if "SLURM_PROCID" in os.environ:
            args.dist_rank = int(os.environ["SLURM_PROCID"])
            args.ws = int(os.environ["SLURM_NTASKS"])
        else:
            args.dist_rank = int(os.environ["RANK"])
            args.ws = int(os.environ["WORLD_SIZE"])
        if args.dist_rank != 0:
            args.use_wandb = False
    else:
        args.dist_rank = 0
        args.ws = 1

    args.dist_rank_0 = (args.dist_rank == 0 and args.distributed) or not args.distributed

    # Set default job name
    if args.job_name is None:
        args.job_name = f"{args.ssl_method}_{args.dataset}_{args.backbone}"

    return args


# Example usage and testing
if __name__ == "__main__":
    # Test the configuration functions
    args = get_ssl_args()
    print("SSL Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    print("\nTesting component creation...")
    
    try:
        # Test model creation
        model = get_ssl_model(args)
        print(f"Created {args.ssl_method} model")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test dataset creation
        train_loader, test_loader = get_ssl_dataset(args)
        print(f"Created {args.dataset} dataset")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test optimizer creation
        optimizer = get_ssl_optim(args, model)
        print(f"Created {args.optimizer} optimizer")
        
        # Test trainer creation
        trainer = get_ssl_trainer(args)
        print(f"Created {args.trainer} trainer")
        
        # Test loss function
        loss_func = get_ssl_loss_func(args)
        print(f"Created loss function for {args.ssl_method}")
        
        print("\nAll components created successfully!")
        
    except Exception as e:
        print(f"Error creating components: {e}")