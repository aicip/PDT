"""Self-supervised pre-training (SimSiam) with the base optimizer or with PDT.

Example (see scripts/ssl for the paper configuration):

    python scripts/train_ssl.py --dataset cifar10 --backbone resnet18 --trainer predicted --lr 0.03 --cosine_lr_sched --lr_min 1e-3
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kcl.utils.factory_ssl import (get_ssl_args, get_ssl_dataset, get_ssl_loss_func,  # noqa: E402
                                   get_ssl_model, get_ssl_optim, get_ssl_trainer)


def main():
    args = get_ssl_args()
    if args.use_wandb and args.dist_rank_0:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args), name=args.job_name)

    model = get_ssl_model(args)
    dataset_result = get_ssl_dataset(args)
    if len(dataset_result) == 3:
        train_loader, test_loader, probe_train_loader = dataset_result
    else:
        train_loader, test_loader = dataset_result
        probe_train_loader = None
    optimizer = get_ssl_optim(args, model)
    trainer = get_ssl_trainer(args)
    loss_func = get_ssl_loss_func(args)
    num_classes = 10 if args.dataset == "cifar10" else 1000

    print(f"{args.ssl_method} with {args.backbone} backbone, {sum(p.numel() for p in model.parameters()):,} parameters")

    def eval_func(model, eval_loader, device):
        return trainer.evaluate_representation_quality(eval_loader[0], eval_loader[1], num_classes=num_classes)

    trainer.train(
        model=model, train_loader=train_loader, optim=optimizer, loss_func=loss_func,
        eval_loader=(test_loader, probe_train_loader) if args.eval_freq > 0 else None,
        eval_func=eval_func if args.eval_freq > 0 else None,
    )

    final_accuracy = trainer.evaluate_representation_quality(test_loader, probe_train_loader, num_classes=num_classes)
    final_train_loss = np.mean(trainer.train_losses[-1]) if trainer.train_losses else float("nan")
    print(f"Final linear-probe accuracy: {final_accuracy:.4f}")
    print(f"Final training loss: {final_train_loss:.4f}")
    if args.use_wandb and args.dist_rank_0:
        import wandb
        wandb.log({"final_train_loss": final_train_loss, "final_linear_probe_accuracy": final_accuracy})
        wandb.finish()


if __name__ == "__main__":
    main()
