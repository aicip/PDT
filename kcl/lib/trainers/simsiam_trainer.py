import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from kcl.lib.trainers.ssl_trainer_base import SSLTrainerBase
from kcl.lib.models.simsiam import simsiam_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class SimSiamTrainer(SSLTrainerBase):
    """
    Trainer for SimSiam self-supervised learning
    """
    
    def __init__(self, **kwargs):
        """
        Initialize SimSiam trainer
        
        Args:
            **kwargs: Arguments passed to SSLTrainerBase
        """
        super().__init__(**kwargs)
        
    def ssl_forward_step(self, batch, loss_func):
        """
        SimSiam-specific forward step
        
        Args:
            batch: Batch containing (x1, x2) - two augmented views
            loss_func: Loss function (should be simsiam_loss or compatible)
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Extract two augmented views from batch
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x1, x2 = batch
            x1, x2 = x1.to(self.device), x2.to(self.device)
        else:
            raise ValueError("SimSiam batch should contain two augmented views (x1, x2)")
        
        self.optim.zero_grad()
        
        # Forward pass through SimSiam model
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            # Model should return (p1, p2, z1, z2)
            p1, p2, z1, z2 = self.model(x1, x2, mode='ssl')
            
            # Compute SimSiam loss
            if loss_func == simsiam_loss or loss_func.__name__ == 'simsiam_loss':
                loss = loss_func(p1, p2, z1, z2)
            else:
                # If custom loss function, pass the outputs
                loss = loss_func((p1, p2, z1, z2))
        
        # Backward pass
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optim)
        self.grad_scaler.update()
        
        return loss

    def evaluate_representation_quality(self, test_loader, probe_train_loader=None, num_classes=10):
        """train on probe_train_loader, eval on test_loader using linear probing"""
        self.model.eval()
        
        if probe_train_loader is None:
            print("Warning: No probe_train_loader provided, using test_loader for both train and eval (optimistic)")
            probe_train_loader = test_loader
        
        # Extract training features
        train_features, train_labels = [], []
        with torch.no_grad():
            for data, target in probe_train_loader:
                data = data.to(self.device)
                features = self.model.forward_encoder(data)
                train_features.append(features.cpu())
                train_labels.append(target)
        
        # Extract test features
        test_features, test_labels = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                features = self.model.forward_encoder(data)
                test_features.append(features.cpu())
                test_labels.append(target)

        # Concatenate features and labels
        train_features = torch.cat(train_features, dim=0).numpy()
        train_labels = torch.cat(train_labels, dim=0).numpy()
        test_features = torch.cat(test_features, dim=0).numpy()
        test_labels = torch.cat(test_labels, dim=0).numpy()

        # Train classifier on train features, eval on test features
        classifier = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
        classifier.fit(train_features, train_labels)
        predictions = classifier.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        tqdm.write(f"Linear probe accuracy: {accuracy:.4f}")
        self.model.train()
        return accuracy
    
    def _linear_probe_accuracy(self, features, labels, num_classes, max_iter=2000):
        """
        Train a linear classifier on frozen features and return accuracy
        
        Args:
            features: Extracted features [N, feature_dim]
            labels: Ground truth labels [N]
            num_classes: Number of classes
            max_iter: Maximum iterations for linear classifier training
            
        Returns:
            float: Classification accuracy
        """
        # from sklearn.linear_model import LogisticRegression
        # from sklearn.metrics import accuracy_score
        
        # Convert to numpy
        features_np = features.numpy()
        labels_np = labels.numpy()
        
        # Train logistic regression classifier
        classifier = LogisticRegression(
            random_state=42, 
            max_iter=max_iter,
            solver='lbfgs' if num_classes <= 10 else 'saga'
        )
        classifier.fit(features_np, labels_np)
        
        # Predict and compute accuracy
        predictions = classifier.predict(features_np)
        accuracy = accuracy_score(labels_np, predictions)
        
        return accuracy

    def evaluate(self, eval_loader, eval_func=None):
        """
        Override evaluation to include representation quality assessment
        
        Args:
            eval_loader: DataLoader for evaluation
            eval_func: Optional custom evaluation function
            
        Returns:
            dict: Evaluation metrics
        """
        if eval_func is not None:
            # Use custom evaluation function
            return super().evaluate(eval_loader, eval_func)
        else:
            # need to check if eval_loader is a tuple (test_loader, probe_train_loader)
            if isinstance(eval_loader, tuple) and len(eval_loader) == 2:
                test_loader, probe_train_loader = eval_loader
            else:
                test_loader = eval_loader
                probe_train_loader = None
                
            accuracy = self.evaluate_representation_quality(test_loader, probe_train_loader)
            
            # Store metrics
            metrics = {
                "linear_probe_accuracy": accuracy,
                "eval_metric": accuracy  # For compatibility with base class
            }
            self.ssl_metrics.append(metrics)
            
            if self.use_wandb:
                wandb.log({
                    "linear_probe_accuracy": accuracy,
                }, step=self.cur_epoch)
            
            tqdm.write(f"Linear probe accuracy: {accuracy:.4f}")
            return accuracy

    def compute_feature_statistics(self):
        """
        Compute statistics about learned features (optional analysis)
        
        Returns:
            dict: Feature statistics
        """
        self.model.eval()
        stats = {}
        
        with torch.no_grad():
            # Get a batch to analyze
            dummy_input = torch.randn(8, 3, 32, 32).to(self.device)  # Assume CIFAR-10 size
            
            # Forward through different parts of the network
            backbone_features = self.model.forward_encoder(dummy_input)
            _, projections = self.model.forward_features(dummy_input)
            
            # Compute statistics
            stats['backbone_feature_mean'] = backbone_features.mean().item()
            stats['backbone_feature_std'] = backbone_features.std().item()
            stats['projection_mean'] = projections.mean().item()
            stats['projection_std'] = projections.std().item()
            stats['projection_norm'] = torch.norm(projections, dim=1).mean().item()
        
        self.model.train()
        return stats

    def save_all(self, save_path):
        """
        Save model and training state (override to add SimSiam-specific info)
        """
        super().save_all(save_path)
        
        # Save additional SimSiam-specific information
        import os
        import pickle
        
        simsiam_info = {
            'model_type': 'SimSiam',
            'backbone': getattr(self.model, 'backbone_name', 'unknown'),
            'projection_dim': getattr(self.model, 'dim', None),
            'prediction_dim': getattr(self.model, 'pred_dim', None),
        }
        
        # Add feature statistics if available
        try:
            simsiam_info['feature_stats'] = self.compute_feature_statistics()
        except:
            pass  # Skip if error occurs
        
        with open(os.path.join(save_path, "simsiam_info.pkl"), "wb") as f:
            pickle.dump(simsiam_info, f)

    def train_epoch(self, train_loader, loss_func):
        """
        Override train_epoch to add SimSiam-specific logging
        """
        # Call parent train_epoch
        super().train_epoch(train_loader, loss_func)
        
        # Add SimSiam-specific metrics logging
        if self.use_wandb and self.cur_epoch % 10 == 0:  # Log every 10 epochs
            try:
                feature_stats = self.compute_feature_statistics()
                wandb.log({
                    f"features/{k}": v for k, v in feature_stats.items()
                }, step=self.cur_epoch)
            except:
                pass  # Skip if error occurs


# Convenience function to create SimSiam loss
def create_simsiam_loss():
    """
    Create SimSiam loss function
    
    Returns:
        callable: SimSiam loss function
    """
    return simsiam_loss


# Example usage
if __name__ == "__main__":
    # This would typically be called from a training script
    print("SimSiam trainer implementation ready!")
    
    # Example of how it would be used:
    """
    from kcl.lib.models.simsiam import create_simsiam_model
    from kcl.lib.datasets.simsiam_datasets import create_simsiam_cifar10
    
    # Create model and data
    model = create_simsiam_model('cifar10', backbone='resnet18')
    train_loader, test_loader = create_simsiam_cifar10('./data')
    
    # Create trainer
    trainer = SimSiamTrainer(
        train_epochs=100,
        log_dir='./logs/simsiam',
        use_wandb=True
    )
    
    # Train
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
    loss_func = create_simsiam_loss()
    
    trainer.train(model, train_loader, optimizer, loss_func, eval_loader=test_loader)
    """