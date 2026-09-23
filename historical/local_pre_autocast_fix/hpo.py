import argparse
import copy

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet50, vit_b_16, resnet18, resnet34
import os
# from pytorch_optimizer.optimizer import create_optimizer, load_optimizer
# from pytorch_optimizer.optimizer import Shampoo
# from pytorch_optimizer.optimizer import Lamb
import torch_optimizer

from kcl.lib.datasets.cifar import create_cifar10, create_cifar10_single_class
from kcl.lib.datasets.imagenet import (create_imagenet, create_imagenet_dali,
                                       create_resize_dataset)
from kcl.lib.models.adam_mlp import FCN, SimpleCNN, CIFARMLP
from kcl.lib.models.AlexNet import AlexNet, AlexNet_Small
from kcl.lib.models.resnet8 import resnet8
from kcl.lib.models.SimpleFCN import SimpleFCN
from kcl.lib.trainers.accelerated_trainer_v2 import AcceleratedTrainerV2
from kcl.lib.trainers.accelerated_trainer_v4 import AcceleratedTrainerV4
from kcl.lib.trainers.accelerated_trainer_v5 import AcceleratedTrainerV5
#from kcl.lib.trainers.predicted_trainer import PredictedTrainer
#from kcl.lib.trainers.predicted_trainer_base import PredictedTrainer
from kcl.lib.trainers.predicted_trainer_switch_by_loss import PredictedTrainer
from kcl.lib.trainers.predicted_err_trainer import Predicted_err_Trainer
from kcl.lib.trainers.predicted_err_trainer_anl import Predicted_err_anl_Trainer
from kcl.lib.trainers.faster_convergence_trainer import FconvergenceTrainer
from kcl.lib.trainers.predicted_trainer_ly import PredictedTrainer_ly
from kcl.lib.trainers.predicted_trainer_conf_post import PredictedTrainer_conf
#from kcl.lib.trainers.predicted_trainer_conf_post_random_mask import PredictedTrainer_conf
#from kcl.lib.trainers.predicted_trainer_conf_post_mask3 import PredictedTrainer_conf
from kcl.lib.trainers.predicted_trainer_conf_post_adaptive import PredictedTrainer_conf_adaptive
from kcl.lib.trainers.predicted_trainer_conf_ly import PredictedTrainer_conf_ly
from kcl.lib.trainers.randomly_accelerated import Random_accelerated
from kcl.lib.trainers.trainer_base import TrainerBase
from kcl.utils.misc import full_seed
from kcl.utils.misc import str2bool
from kcl.utils.options import DEFAULT_TRAINER_OPTIONS


def get_model(args, device=None):
    full_seed(args.random_seed)
    n_classes = 10 if args.dataset == 'cifar10' else 1000

    if isinstance(args.model, str):
        model_name = args.model.lower()
    else:
        model_name = args.model

    if model_name == 'simplefcn':
        model = SimpleFCN(n_classes)
    elif model_name == 'resnet50':
        model = resnet50(weights=None, num_classes=n_classes)
    elif model_name == 'resnet18':
        model = resnet18(weights=None, num_classes=n_classes)
    elif model_name == 'resnet34':
        model = resnet34(weights=None, num_classes=n_classes)
        # model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif model_name == 'resnet8':
        model = resnet8(n_classes)
        # model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif model_name == 'alexnet':
        model = AlexNet(n_classes)
    elif model_name == 'alexnet_small':
        model = AlexNet_Small(in_chan=3, n_classes=n_classes)
    elif model_name == 'simplecnn':
        model = SimpleCNN()
    elif model_name == 'fcn':
        model = FCN(n_classes)
    elif model_name == 'vit':
        model = vit_b_16(weights=None, num_classes=n_classes)
    elif model_name == 'mlp':
        model = CIFARMLP(n_classes)
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
        #train, test = create_imagenet_dali(
        #    args.data_dir,
        #    train_size=args.train_batch_size,
        #    test_size=args.test_batch_size,
        #    shard_id=args.dist_rank,
        #    num_shards=args.ws,
        #    device_id=args.dist_rank
        #)
        train, test = create_imagenet(args.data_dir,
                                      train_batch_size=args.train_batch_size,
                                      test_batch_size=args.test_batch_size,
                                      distributed=args.distributed,
                                      ws=args.ws,
                                      rank=args.dist_rank)
        # train, test = create_resize_dataset(args.data_dir,
        #                                     train_batch_size=args.train_batch_size,
        #                                     test_batch_size=args.test_batch_size)
    else:
        raise NotImplementedError

    return train, test


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
    options['n_accel_eigs'] = args.n_accel_eigs
    options['accel_const'] = args.accel_const
    options['save_freq'] = args.save_freq
    options['train_epochs'] = args.train_epochs
    options['random_seed'] = args.random_seed
    options['log_dir'] = args.log_dir
    options['log_interval'] = args.log_interval
    options['acceleration_mode'] = args.acceleration_mode
    options['reduced'] = True if args.svd_mode == 'reduced' else False
    options['use_wandb'] = not args.skip_wandb
    options['use_wandb'] = (not args.skip_wandb) and args.dist_rank_0
    options['save_weights'] = False
    options['projection_delay'] = args.projection_delay
    options['window_size'] = args.window_size
    options['n_past_weights'] = args.n_past_weights
    options['predict_start_epoch'] = args.predict_start_epoch
    options['predicted_num'] = args.predicted_num
    options['predict_epoch_interval'] = args.predict_epoch_interval
    options['w_device'] = args.w_device
    options['if_sp'] = args.if_sp
    options['if_LV'] = args.if_LV
    options['args'] = args
    

    if device is None:
        device = torch.device('cuda')
    options['device'] = device

    if args.trainer == 'base':
        trainer = TrainerBase(**options)
    elif args.trainer == 'accelerated':
        trainer = AcceleratedTrainerV2(**options)
    elif args.trainer == 'acceleratedv5':
        trainer = AcceleratedTrainerV5(**options)
    elif args.trainer == 'prediction':
        options['only_predict'] = True
        trainer = AcceleratedTrainerV4(**options)
    elif args.trainer == 'bootstrap':
        options['only_predict'] = False
        trainer = AcceleratedTrainerV4(**options)
    elif args.trainer == 'predicted':
        trainer = PredictedTrainer(**options)
    elif args.trainer == 'predicted_err':
        trainer = Predicted_err_Trainer(**options)
    elif args.trainer == 'predicted_err_anl':
        trainer = Predicted_err_anl_Trainer(**options)
    elif args.trainer == 'Fast_convergence':
        trainer = FconvergenceTrainer(**options)
    elif args.trainer == 'predicted_ly':
        trainer = PredictedTrainer_ly(**options)
    elif args.trainer == 'predicted_conf':
        trainer = PredictedTrainer_conf(**options)
    elif args.trainer == 'predicted_conf_adaptive':
        trainer = PredictedTrainer_conf_adaptive(**options)
    elif args.trainer == 'predicted_conf_ly':
        trainer = PredictedTrainer_conf_ly(**options)
    elif args.trainer == 'Random_accelerated':
        trainer = Random_accelerated(**options)
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
        choices=[
            'simplefcn', 'resnet50', 'resnet8', 'func_approx', 'alexnet',
            'simplecnn', 'fcn', 'vit', 'mlp', 'resnet18'
        ],
        default='simplefcn'
    )
    parser.add_argument('--random_seed', type=int, default=0)

    data_p = parser.add_argument_group("Data")
    data_p.add_argument(
        "--dataset",
        type=str,
        choices=['cifar10', 'imagenet'],
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
        choices=[
            'base', 'accelerated', 'acceleratedv5', 'prediction', 'bootstrap', 'predicted','predicted_err','predicted_err_anl','Fast_convergence','predicted_ly','predicted_conf','predicted_conf_adaptive','predicted_conf_ly','Random_accelerated'
        ],
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
    train_p.add_argument(
        '--label_smoothing',
        type=float,
        default=0.0,
        help='label_smoothing'
    )
    train_p.add_argument(
        '--count_flops',
        action='store_true',
        help='Whether to count flops during training'
    )

    accel_p = parser.add_argument_group("Acceleration")
    accel_p.add_argument('--svd_rank', type=int, default=0)
    accel_p.add_argument(
        '--svd_mode', type=str, default='full', choices=['reduced', 'full']
    )
    accel_p.add_argument(
        '--n_accel_eigs',
        type=int,
        default=5,
        help="Number of eigenvalues to accelerate"
    )
    accel_p.add_argument(
        '--accel_const',
        type=float,
        default=2.0,
        help="\alpha in the paper. Acceleration constant"
    )
    accel_p.add_argument(
        '--acceleration_mode',
        type=str,
        default='stable',
        choices=['stable', 'unstbale', 'neutral']
    )
    accel_p.add_argument(
        '--projection_delay',
        type=int,
        default=5,
        help="Number of epochs to wait before calculating U in V5"
    )
    accel_p.add_argument('--window_size', type=int, default=5, help="unused")
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
    log_p.add_argument('--if_sp', type=str2bool, default=False,help="If use segment DMD prediction")
    log_p.add_argument('--if_LV', type=str2bool, default=False,help="If apply DMD only on largevalues")
    log_p.add_argument('--accelerated_epoch', type=int, default=10,help="The epoch to start acceleration")
    log_p.add_argument('--accelerated_num', type=int, default=5,help="The number of epochs to accelerate")
    log_p.add_argument('--percentage', type=float, default=0.6,help="The percentage of weights to accelerate")
    
    args, extra = parser.parse_known_args()
    if len(extra) > 0:
        print("WARNING: Ignoring the following UNKNOWN arguments: ", extra)

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
