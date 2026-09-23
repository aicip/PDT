import argparse
import copy

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet50, vit_b_16, resnet18
import os
import torch_optimizer

from kcl.lib.datasets.cifar import create_cifar10, create_cifar10_single_class
from kcl.lib.datasets.imagenet import create_imagenet
from kcl.lib.datasets.ag_news_hf import create_ag_news
from kcl.lib.models.adam_mlp import FCN
from kcl.lib.models.AlexNet import AlexNet
from kcl.lib.models.SimpleFCN import SimpleFCN
from kcl.lib.models.lstm import TextLSTM_Deep
#from kcl.lib.trainers.predicted_trainer import PredictedTrainer
from kcl.lib.trainers.predicted_trainer_switch_by_loss import PredictedTrainer
from kcl.lib.trainers.predicted_trainer_conf_post import PredictedTrainer_conf
#from kcl.lib.trainers.predicted_trainer_conf_post_random_mask import PredictedTrainer_conf
from kcl.lib.trainers.predicted_trainer_conf_post_adaptive import PredictedTrainer_conf_adaptive
from kcl.lib.trainers.randomly_accelerated import Random_accelerated
from kcl.lib.trainers.trainer_base import TrainerBase
from kcl.utils.misc import full_seed
from kcl.utils.misc import str2bool
from kcl.utils.options import DEFAULT_TRAINER_OPTIONS


def get_model(args, device=None):
    full_seed(args.random_seed)

    # Determine number of classes based on dataset
    if args.dataset == 'cifar10':
        n_classes = 10
    elif args.dataset == 'imagenet':
        n_classes = 1000
    elif args.dataset == 'ag_news':
        n_classes = 4
    else:
        n_classes = 10  # Default fallback

    if isinstance(args.model, str):
        model_name = args.model.lower()
    else:
        model_name = args.model

    if model_name == 'simplefcn':
        model = SimpleFCN(n_classes)
    elif model_name == 'fcn':
        model = FCN(n_classes)
    elif model_name == 'alexnet':
        model = AlexNet(n_classes)
    elif model_name == 'resnet18':
        model = resnet18(weights=None, num_classes=n_classes)
    elif model_name == 'resnet50':
        model = resnet50(weights=None, num_classes=n_classes)
    elif model_name == 'vit':
        model = vit_b_16(weights=None, num_classes=n_classes)
    elif model_name == 'lstm_deep':
        # 4-layer LSTM text classifier used for the AG News experiment; vocab_size is set by the dataset loader
        vocab_size = getattr(args, 'vocab_size', 5000)
        model = TextLSTM_Deep(vocab_size=vocab_size, num_classes=n_classes)
    else:
        raise NotImplementedError

    if device is None:
        device = torch.device('cuda')
    # if torch.cuda.get_device_capability(device)[0] >= 7:
    #     model = torch.compile(model)
    # else:
    #     print("skipping cuda compile. GPU Too old")
    model = model.to(device)
    if args.distributed:
        model = model.cuda(args.dist_rank)
        model = DDP(model, device_ids=[args.dist_rank])
    return model


def get_dataset(args):
    if args.dataset == 'cifar10':
        if args.single_class_batch:  # add a new parameter to control whether to use single class batch sampler
            train, test = create_cifar10_single_class(
                args.data_dir,
                train_batch_size=args.train_batch_size,
                test_batch_size=args.test_batch_size,
                add_suffix=True if args.local_mode else False,
                img_size=args.img_size,
                dist=args.distributed,
                ws=args.ws,
                rank=args.dist_rank
            )
        else:
            train, test = create_cifar10(
                args.data_dir,
                train_batch_size=args.train_batch_size,
                test_batch_size=args.test_batch_size,
                add_suffix=True if args.local_mode else False,
                img_size=args.img_size,
                dist=args.distributed,
                ws=args.ws,
                rank=args.dist_rank
            )
    elif args.dataset == 'imagenet':
        # Check if SOTA training features are enabled
        if hasattr(args, 'auto_augment') and args.auto_augment is not None:
            # Use SOTA dataloader with advanced augmentations
            from kcl.lib.datasets.imagenet import create_imagenet_sota
            train, test, mixup_cutmix = create_imagenet_sota(
                dataset_dir=args.data_dir,
                train_batch_size=args.train_batch_size,
                test_batch_size=args.test_batch_size,
                distributed=args.distributed,
                rank=args.dist_rank,
                ws=args.ws,
                auto_augment=args.auto_augment,
                ra_magnitude=getattr(args, 'ra_magnitude', 9),
                random_erase_prob=getattr(args, 'random_erase', 0.0),
                mixup_alpha=getattr(args, 'mixup_alpha', 0.0),
                cutmix_alpha=getattr(args, 'cutmix_alpha', 0.0),
                interpolation=getattr(args, 'interpolation', 'bilinear'),
                train_crop_size=getattr(args, 'train_crop_size', 224),
                val_crop_size=getattr(args, 'val_crop_size', 224),
                val_resize_size=getattr(args, 'val_resize_size', 256),
                use_ra_sampler=getattr(args, 'use_ra_sampler', False),
            )
            # Return 3 values for SOTA mode
            return train, test, mixup_cutmix
        else:
            # Use standard dataloader (original behavior for backward compatibility)
            train, test = create_imagenet(args.data_dir,
                                          train_batch_size=args.train_batch_size,
                                          test_batch_size=args.test_batch_size,
                                          distributed=args.distributed,
                                          ws=args.ws,
                                          rank=args.dist_rank)
            # Return 2 values + None for backward compatibility
            return train, test, None
        # train, test = create_resize_dataset(args.data_dir,
        #                                     train_batch_size=args.train_batch_size,
        #                                     test_batch_size=args.test_batch_size)
    elif args.dataset == 'ag_news':
        # NLP dataset for cross-domain generalization experiment
        train, test, vocab_size = create_ag_news(
            args.data_dir,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            max_len=getattr(args, 'max_seq_len', 50),  # Default 50 (200 causes gradient vanishing in LSTM)
            vocab_size=getattr(args, 'vocab_size', 5000),  # Default 5000 (50000 causes training issues)
            dist=args.distributed,
            ws=args.ws,
            rank=args.dist_rank
        )
        # Store vocab_size for model initialization
        args.vocab_size = vocab_size
    else:
        raise NotImplementedError

    # For CIFAR-10 and AG News, return None for mixup_cutmix (not applicable)
    return train, test, None


def get_optim(args, model):
    if args.optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd_m":
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == "rmsprop":
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer == "adadelta":
        optim = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    elif args.optimizer == "adagrad":
        optim = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamax":
        optim = torch.optim.Adamax(model.parameters(), lr=args.lr)
    elif args.optimizer == "nadam":
        optim = torch.optim.NAdam(model.parameters(), lr=args.lr)
    elif args.optimizer == "shampoo":
        #optim = create_optimizer(model, "shampoo", lr=args.lr)
        #optim = Shampoo(model.parameters())
        #optim = torch_optimizer.Shampoo(model.parameters(), lr=args.lr)
        optim = torch_optimizer.Shampoo(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=0, epsilon=1e-4, update_freq=10)
    elif args.optimizer == "lamb":
        #optim = create_optimizer(model, "lamb", lr=args.lr)
        #optim = Lamb(model.parameters())
        optim = torch_optimizer.Lamb(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    return optim


def get_trainer(args, device=None):
    options = copy.deepcopy(DEFAULT_TRAINER_OPTIONS)
    options['svd_rank'] = args.svd_rank
    options['save_freq'] = args.save_freq
    options['train_epochs'] = args.train_epochs
    options['random_seed'] = args.random_seed
    options['log_dir'] = args.log_dir
    options['log_interval'] = args.log_interval
    options['reduced'] = True if args.svd_mode == 'reduced' else False
    options['use_wandb'] = (not args.skip_wandb) and args.dist_rank_0
    options['save_weights'] = False
    options['n_past_weights'] = args.n_past_weights
    options['predict_start_epoch'] = args.predict_start_epoch
    options['predicted_num'] = args.predicted_num
    options['predict_epoch_interval'] = args.predict_epoch_interval
    options['mask_mode'] = args.mask_mode
    options['log_gradient_metrics'] = args.log_gradient_metrics
    options['w_device'] = args.w_device
    options['if_sp'] = args.if_sp
    options['if_LV'] = args.if_LV
    options['args'] = args
    

    if device is None:
        device = torch.device('cuda')
    options['device'] = device

    if args.trainer == 'base':
        trainer = TrainerBase(**options)
    elif args.trainer == 'predicted_conf':
        trainer = PredictedTrainer_conf(**options)
    elif args.trainer == 'predicted_conf_adaptive':
        trainer = PredictedTrainer_conf_adaptive(**options)
    elif args.trainer == 'Random_accelerated':
        trainer = Random_accelerated(**options)
    elif args.trainer == 'predicted':
        trainer = PredictedTrainer(**options)
    else:
        raise ValueError("Unknown trainer type")
    
    if args.distributed:
        rank = args.dist_rank
        ws = args.ws
        torch.distributed.init_process_group('nccl', world_size=ws, rank=rank)
        torch.cuda.set_device(rank)

    return trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=['simplefcn', 'fcn', 'alexnet', 'resnet18', 'resnet50', 'vit', 'lstm_deep'],
        default='simplefcn'
    )
    parser.add_argument('--random_seed', type=int, default=0)

    data_p = parser.add_argument_group("Data")
    data_p.add_argument(
        "--dataset",
        type=str,
        choices=['cifar10', 'imagenet', 'ag_news'],
        default='cifar10'
    )
    data_p.add_argument("--data_dir", type=str, default='~/data')
    data_p.add_argument("--train_batch_size", type=int, default=1024)
    data_p.add_argument("--test_batch_size", type=int, default=128)
    data_p.add_argument(
        '--local_mode',
        action='store_true',
        default=False,
        help="Adds suffix to data dir for each dataset to avoid collision"
    )
    data_p.add_argument("--img_size", type=int, default=128)

    # Single class batch sampling
    data_p.add_argument('--single_class_batch', action='store_true', help='Use single class batch sampling for training')

    train_p = parser.add_argument_group("Training")

    train_p.add_argument(
        "--optimizer",
        type=str,
        choices=[
            'adam', 'sgd', 'sgd_m', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam', 'adamw', 'shampoo', 'lamb'
        ],
        default='adam'
    )
    train_p.add_argument('--lr', type=float, default=1e-3)
    train_p.add_argument(
        '--trainer',
        type=str,
        choices=['base', 'predicted', 'predicted_conf', 'predicted_conf_adaptive', 'Random_accelerated'],
        default='base'
    )
    train_p.add_argument(
        '--train_epochs',
        type=int,
        default=30,
        help="Number of epochs to train"
    )
    train_p.add_argument(
        '--distributed',
        action='store_true',
    )
    train_p.add_argument(
        '--cosine_lr_sched',
        action='store_true',
    )
    train_p.add_argument(
        '--lr_min',
        type=float,
        default=0.0,
        help='Minimum learning rate for cosine annealing (if used)'
    )
    train_p.add_argument(
        '--lr_warmup_epochs',
        type=int,
        default=0,
    )
    train_p.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help="Weight decay for optimizer"
    )
    # Note: --label_smoothing moved to SOTA argument group (line ~496) with better help text
    train_p.add_argument(
        '--count_flops',
        action='store_true',
        help='Whether to count flops during training (uses torch.profiler for all operations including DMD, ~10-20%% overhead)'
    )
    train_p.add_argument(
        '--count_flops_fast',
        action='store_true',
        help='Fast FLOPs counting: profile first batch only, use analytical formulas for DMD (negligible overhead). Note: cannot be used together with --count_flops'
    )
    train_p.add_argument(
        '--profile_memory',
        action='store_true',
        help='Profile memory usage: breakdown of snapshot storage, SVD workspace, peak GPU memory (negligible overhead)'
    )
    train_p.add_argument(
        '--profile_timing',
        action='store_true',
        help='Profile timing breakdown: SVD, prediction, masking computation time (negligible overhead)'
    )

    accel_p = parser.add_argument_group("Acceleration")
    accel_p.add_argument('--svd_rank', type=int, default=0)
    accel_p.add_argument(
        '--svd_mode', type=str, default='full', choices=['reduced', 'full']
    )
    accel_p.add_argument(
        '--w_device',
        type=str,
        default='',
        help="Cuda device to store weight history on"
    )

    log_p = parser.add_argument_group("Logging")
    log_p.add_argument(
        '--save_freq',
        type=int,
        default=5,
        help="Frequency to write weights to disk"
    )
    log_p.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help="Number of minibatches between printing loss"
    )
    log_p.add_argument('--log_dir', type=str, default='logs')
    log_p.add_argument(
        '--job_name', type=str, default=None, help="Name of the job for wandb"
    )
    log_p.add_argument('--skip_wandb', action='store_true', default=False)
    log_p.add_argument('--wandb_project_name', type=str, default='kcl')
    log_p.add_argument('--wandb_entity', type=str, default=None)
    
    log_p.add_argument('--n_past_weights', type=int, default=None,help="Number of past weights to use in DMD")
    log_p.add_argument('--predict_start_epoch', type=int, default=10,help="The start epoch to make prediction")
    log_p.add_argument('--predicted_num', type=int, default=5,help="Number of predicted weights")
    log_p.add_argument('--predict_epoch_interval', type=int, default=5,help="The interval epochs to make prediction")
    log_p.add_argument('--mask_mode', type=str, default='both',
                      choices=['both', 'accel_only', 'consistency_only'],
                      help="Masking strategy: 'both' (Eq6+Eq7), 'accel_only' (only Eq6), 'consistency_only' (only Eq7)")
    log_p.add_argument('--log_gradient_metrics', action='store_true',
                      help="Enable gradient norm and direction stability logging (for LR vs mask ratio analysis)")
    log_p.add_argument('--if_sp', type=str2bool, default=False,help="If use segment DMD prediction")
    log_p.add_argument('--if_LV', type=str2bool, default=False,help="If apply DMD only on largevalues")

    # SOTA Training Augmentations (for ImageNet)
    sota_p = parser.add_argument_group("SOTA Training Features")
    sota_p.add_argument('--auto_augment', type=str, default=None,
                        choices=['ra', 'ta_wide', 'imagenet', None],
                        help='Auto-augmentation policy: ra (RandAugment), ta_wide (TrivialAugmentWide), imagenet (AutoAugment). None disables.')
    sota_p.add_argument('--ra_magnitude', type=int, default=9,
                        help='RandAugment magnitude (default: 9)')
    sota_p.add_argument('--mixup_alpha', type=float, default=0.0,
                        help='MixUp alpha parameter (0 to disable, typical: 0.2)')
    sota_p.add_argument('--cutmix_alpha', type=float, default=0.0,
                        help='CutMix alpha parameter (0 to disable, typical: 1.0)')
    sota_p.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing epsilon (0-1, 0 to disable, typical: 0.1)')
    sota_p.add_argument('--random_erase', type=float, default=0.0,
                        help='Random erasing probability (0-1, 0 to disable, typical: 0.1)')

    # Model EMA
    sota_p.add_argument('--use_model_ema', action='store_true', default=False,
                        help='Use exponential moving average of model weights for evaluation')
    sota_p.add_argument('--model_ema_decay', type=float, default=0.9999,
                        help='Model EMA decay rate (default: 0.9999 for ResNet, 0.99998 for ViT)')

    # Training enhancements
    sota_p.add_argument('--clip_grad_norm', type=float, default=0.0,
                        help='Gradient norm clipping threshold (0 to disable, typical: 1.0)')
    # Note: --lr_warmup_epochs already defined in train_p group (line 375-378)
    sota_p.add_argument('--lr_warmup_decay', type=float, default=0.01,
                        help='Warmup start factor: lr_start = lr * warmup_decay (default: 0.01)')

    # Data loading options
    sota_p.add_argument('--use_ra_sampler', action='store_true', default=False,
                        help='Use Repeated Augmentation Sampler (recommended for ViT training)')
    sota_p.add_argument('--interpolation', type=str, default='bilinear',
                        choices=['bilinear', 'bicubic'],
                        help='Interpolation mode for image resizing (bilinear or bicubic)')
    sota_p.add_argument('--train_crop_size', type=int, default=224,
                        help='Training crop size (default: 224 for ResNet/ViT)')
    sota_p.add_argument('--val_crop_size', type=int, default=224,
                        help='Validation crop size (default: 224)')
    sota_p.add_argument('--val_resize_size', type=int, default=256,
                        help='Validation resize size before center crop (default: 256)')

    args, extra = parser.parse_known_args()
    if len(extra) > 0:
        print("WARNING: Ignoring the following UNKNOWN arguments: ", extra)

    # Check for conflicting FLOPs counting options
    if args.count_flops and args.count_flops_fast:
        raise ValueError("Cannot use both --count_flops and --count_flops_fast at the same time. "
                        "Use --count_flops for profiler-based (slower, matches paper), "
                        "or --count_flops_fast for formula-based (faster, ~same accuracy).")

    if args.distributed:
        if "SLURM_PROCID" in os.environ:
            args.dist_rank = int(os.environ["SLURM_PROCID"])
            args.ws = int(os.environ["SLURM_NTASKS"])
        else:
            args.dist_rank = int(os.environ["RANK"])
            args.ws = int(os.environ["WORLD_SIZE"])
        if args.dist_rank != 0:
            args.skip_wandb = True
    else:
        args.dist_rank = 0
        args.ws = 1

    if args.w_device == "":
        args.w_device = torch.device('cuda')
    else:
        args.w_device = torch.device(args.w_device)

    args.dist_rank_0 = (args.dist_rank == 0 and args.distributed) or not args.distributed

    return args
