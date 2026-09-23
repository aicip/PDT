import numpy as np
import torch
import wandb
from kcl.utils.hpo_ssl import (
    get_ssl_args, get_ssl_model, get_ssl_dataset, 
    get_ssl_optim, get_ssl_trainer, get_ssl_loss_func
)


def main():
    """Main training function"""
    args = get_ssl_args()
    
    # Initialize wandb
    if not args.skip_wandb and args.dist_rank_0:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.job_name
        )
    
    # Create components using hpo_ssl functions
    print("Creating SSL model...")
    model = get_ssl_model(args)
    print(f"Created {args.ssl_method} model with {args.backbone} backbone")
    
    print("Creating SSL dataset...")
    #train_loader, test_loader = get_ssl_dataset(args)
    dataset_result = get_ssl_dataset(args)
    if len(dataset_result) == 3:
        train_loader, test_loader, probe_train_loader = dataset_result
    else:
        train_loader, test_loader = dataset_result
        probe_train_loader = None
    print(f"Created {args.dataset} dataset")
    
    print("Creating optimizer...")
    optimizer = get_ssl_optim(args, model)
    print(f"Created {args.optimizer} optimizer")
    
    print("Creating SSL trainer...")
    trainer = get_ssl_trainer(args)
    print(f"Created {args.trainer} trainer")
    
    # Loss function
    loss_func = get_ssl_loss_func(args)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Print training info
    print(f"Training for {args.train_epochs} epochs")
    print(f"Batch size: {args.train_batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Dataset: {args.dataset}")
    
    # Start training
    print("Starting SSL training...")
    
    def eval_func(model, eval_loader, device):
        """Evaluation function for representation quality"""
        # eval_loader may be a tuple (test_loader, probe_train_loader)
        return trainer.evaluate_representation_quality(
            eval_loader[0],  # test_loader
            eval_loader[1],  # probe_train_loader
            num_classes=10 if args.dataset == 'cifar10' else 1000
        )
    
    # Train the model
    trainer.train(
        model=model,
        train_loader=train_loader,
        optim=optimizer,
        loss_func=loss_func,
        eval_loader=(test_loader, probe_train_loader) if args.eval_freq > 0 else None,
        eval_func=eval_func if args.eval_freq > 0 else None
    )
    
    # Final evaluation
    print("Training completed!")
    final_accuracy = 0  # add a default value
    if test_loader is not None:
        final_accuracy = trainer.evaluate_representation_quality(
            test_loader, 
            probe_train_loader,  # use separate probe train loader if available
            num_classes=10 if args.dataset == 'cifar10' else 1000
        )
        print(f"Final linear probe accuracy: {final_accuracy:.4f}")
    
    # Print training summary
    final_train_loss = np.mean(trainer.train_losses[-1]) if trainer.train_losses else 0
    print(f"Final training loss: {final_train_loss:.4f}")
    print(f"Total epochs trained: {trainer.cur_epoch}")
    
    if wandb.run is not None:
        wandb.log({
            "final_train_loss": final_train_loss,
            "final_linear_probe_accuracy": final_accuracy if test_loader else 0,
            "total_epochs": trainer.cur_epoch
        })
        wandb.finish()


if __name__ == "__main__":
    main()