"""Train a classifier with the base optimizer, with PDT, or with one of the baselines.

Examples (see scripts/cifar10, scripts/imagenet, scripts/nlp for the paper configurations):

    python scripts/train.py --dataset cifar10 --model alexnet --trainer base --lr 0.05 --cosine_lr_sched --lr_min 1e-3
    python scripts/train.py --dataset cifar10 --model alexnet --trainer pdt  --lr 0.05 --cosine_lr_sched --lr_min 1e-3
    torchrun --nproc_per_node=3 scripts/train.py --distributed --dataset imagenet --model resnet50 --trainer pdt_distributed ...
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kcl.utils.factory import get_args, get_dataset, get_model, get_optim, get_trainer  # noqa: E402


def main():
    args = get_args()
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name,
                   config={k: (str(v) if isinstance(v, torch.device) else v) for k, v in vars(args).items()})

    trainer = get_trainer(args)
    train_loader, test_loader = get_dataset(args)  # sets args.vocab_size for AG News
    model = get_model(args)
    optim = get_optim(args, model)
    loss_func = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(args.device)

    trainer.train(model, train_loader, optim, loss_func, test_loader=test_loader)

    if args.dist_rank_0:
        print(f"Final test loss: {np.mean(trainer.test_losses[-1]):.4f}")
        print(f"Final train loss: {np.mean(trainer.train_losses[-1]):.4f}")
        print(f"Final accuracy: {trainer.accuracies[-1]:.4f}")
        print(f"Per-epoch metrics written to {os.path.join(args.log_dir, 'metrics.csv')}")
    if args.use_wandb:
        import wandb
        wandb.finish()
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
