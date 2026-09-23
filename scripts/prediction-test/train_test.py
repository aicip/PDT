import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from kcl.utils.hpo import (get_args, get_dataset, get_model, get_optim,
                           get_trainer)
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def main():
    args = get_args()
    if not args.skip_wandb:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.job_name
        )
    trainer = get_trainer(args)

    # IMPORTANT: Create dataset BEFORE model for NLP tasks
    # Dataset loader sets args.vocab_size which model needs for embedding layer
    # Handle both 2-value (backward compatible) and 3-value (SOTA) returns from get_dataset
    dataset_output = get_dataset(args)
    if len(dataset_output) == 3:
        train, test, mixup_cutmix = dataset_output
        # Set MixUp/CutMix transform on trainer for SOTA ImageNet training
        if mixup_cutmix is not None:
            trainer.mixup_cutmix = mixup_cutmix
            print(f"MixUp/CutMix transform set on trainer")
    else:
        # Backward compatible: CIFAR-10 and other datasets return (train, test)
        train, test = dataset_output
        mixup_cutmix = None

    # Create model AFTER dataset so vocab_size is set for NLP models
    model = get_model(args)

    optim = get_optim(args, model)
    #scheduler = StepLR(optim, step_size=50, gamma=0.1)
    #scheduler = CosineAnnealingLR(optim, T_max=67, eta_min=1e-6)
    #scheduler = ExponentialLR(optim, gamma=0.95)

    #loss_func = torch.nn.CrossEntropyLoss().cuda()

    # Note: Label smoothing is also handled in trainer_base.py for consistency
    # This external loss function is kept for backward compatibility
    if hasattr(args, 'label_smoothing') and args.label_smoothing > 0:
        loss_func = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing).cuda()
    else:
        loss_func = torch.nn.CrossEntropyLoss().cuda()

    trainer.train(model, train, optim, loss_func, test_loader=test)
    #trainer.train(model, train, optim, scheduler, loss_func, test_loader=test)

    test_loss = np.mean(trainer.test_losses[-1])
    train_loss = np.mean(trainer.train_losses[-1])
    accuracy = trainer.accuracies[-1]

    print(f"Final Test Loss: {test_loss:.3f}")
    print(f"Final Train Loss: {train_loss:.3f}")
    print(f"Final Accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
