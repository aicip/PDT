import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50


class SimSiam(nn.Module):
    """
    SimSiam: Simple Siamese Networks for Representation Learning
    
    Architecture:
    - Backbone encoder (e.g., ResNet)
    - Projection head: 3-layer MLP
    - Prediction head: 2-layer MLP
    """
    
    def __init__(self, backbone='resnet18', dim=2048, pred_dim=512, num_classes=None):
        """
        Args:
            backbone (str): Backbone architecture ('resnet18', 'resnet50', etc.)
            dim (int): Dimension of projection head output
            pred_dim (int): Dimension of prediction head hidden layer
            num_classes (int): Number of classes for classification head (optional)
        """
        super(SimSiam, self).__init__()
        
        self.backbone_name = backbone
        self.dim = dim
        self.pred_dim = pred_dim
        
        # Build backbone encoder
        if backbone == 'resnet18':
            self.encoder = resnet18(weights=None)
            backbone_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()  # Remove classification head
        elif backbone == 'resnet50':
            self.encoder = resnet50(weights=None)
            backbone_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()  # Remove classification head
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Projection head: 3-layer MLP
        # backbone_dim -> dim -> dim -> dim
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False)  # No bias and no learnable affine parameters for final BN
        )
        
        # Prediction head: 2-layer MLP  
        # dim -> pred_dim -> dim
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)  # No BN and ReLU for final layer
        )
        
        # Optional classification head for downstream tasks
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Linear(backbone_dim, num_classes)
        else:
            self.classifier = None
    
    def forward_encoder(self, x):
        """Forward through backbone encoder only"""
        return self.encoder(x)
    
    def forward_features(self, x):
        """Forward through encoder and projector"""
        h = self.forward_encoder(x)  # Backbone features
        z = self.projector(h)        # Projected features
        return h, z
    
    def forward(self, x1, x2=None, mode='ssl'):
        """
        Forward pass for different modes
        
        Args:
            x1: First augmented view (or single input for classification)
            x2: Second augmented view (None for classification mode)
            mode: 'ssl' for self-supervised learning, 'classify' for classification
            
        Returns:
            For SSL mode: (p1, p2, z1, z2) where p=prediction, z=projection
            For classify mode: classification logits
        """
        if mode == 'ssl':
            if x2 is None:
                raise ValueError("x2 is required for SSL mode")
            
            # Forward both views through encoder and projector
            h1, z1 = self.forward_features(x1)
            h2, z2 = self.forward_features(x2)
            
            # Forward through predictor
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)
            
            #return p1, p2, z1.detach(), z2.detach()  # Stop gradient on z
            return p1, p2, z1, z2
            
        elif mode == 'classify':
            if self.classifier is None:
                raise ValueError("Classifier head not initialized. Set num_classes when creating model.")
            
            h = self.forward_encoder(x1)
            return self.classifier(h)
            
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'ssl' or 'classify'")


def simsiam_loss(p1, p2, z1, z2):
    """
    SimSiam loss function: negative cosine similarity
    
    Args:
        p1, p2: Predictions from predictor head
        z1, z2: Projections from projector head (should be detached)
    
    Returns:
        loss: Scalar loss value
    """
    # Normalize to unit vectors
    p1 = F.normalize(p1, dim=1, p=2)
    p2 = F.normalize(p2, dim=1, p=2) 
    z1 = F.normalize(z1, dim=1, p=2)
    z2 = F.normalize(z2, dim=1, p=2)
    
    # Negative cosine similarity
    # loss = -(p1 * z2).sum(dim=1).mean() - (p2 * z1).sum(dim=1).mean()
    loss = -(F.cosine_similarity(p1, z2.detach(), dim=-1).mean() + 
             F.cosine_similarity(p2, z1.detach(), dim=-1).mean())
    return loss * 0.5  # Scale by 0.5 as in the original implementation


class SimSiamCIFAR(SimSiam):
    """SimSiam variant optimized for CIFAR datasets"""
    
    def __init__(self, backbone='resnet18', dim=512, pred_dim=128, num_classes=10):
        """
        CIFAR-optimized SimSiam with smaller dimensions
        
        Args:
            backbone: Backbone architecture
            dim: Projection dimension (smaller for CIFAR)
            pred_dim: Prediction dimension (smaller for CIFAR)  
            num_classes: Number of classes (10 for CIFAR-10)
        """
        super().__init__(backbone=backbone, dim=dim, pred_dim=pred_dim, num_classes=num_classes)
        
        # Replace the first conv layer for CIFAR (32x32 inputs)
        if backbone.startswith('resnet'):
            # CIFAR modification: smaller kernel, stride, no maxpool
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder.maxpool = nn.Identity()


class SimSiamImageNet(SimSiam):
    """SimSiam variant for ImageNet and larger images"""
    
    def __init__(self, backbone='resnet50', dim=2048, pred_dim=512, num_classes=1000):
        """
        ImageNet-optimized SimSiam with standard dimensions
        
        Args:
            backbone: Backbone architecture
            dim: Projection dimension
            pred_dim: Prediction dimension
            num_classes: Number of classes (1000 for ImageNet)
        """
        super().__init__(backbone=backbone, dim=dim, pred_dim=pred_dim, num_classes=num_classes)


def create_simsiam_model(dataset='cifar10', backbone='resnet18', **kwargs):
    """
    Factory function to create appropriate SimSiam model
    
    Args:
        dataset: Target dataset ('cifar10', 'imagenet')
        backbone: Backbone architecture
        **kwargs: Additional arguments passed to model
        
    Returns:
        SimSiam model instance
    """
    if dataset.lower() == 'cifar10':
        return SimSiamCIFAR(backbone=backbone, **kwargs)
    elif dataset.lower() == 'imagenet':
        return SimSiamImageNet(backbone=backbone, **kwargs)
    else:
        # Default to standard SimSiam
        return SimSiam(backbone=backbone, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test CIFAR-10 model
    model_cifar = create_simsiam_model('cifar10', backbone='resnet18')
    x1 = torch.randn(32, 3, 32, 32)  # CIFAR-10 batch
    x2 = torch.randn(32, 3, 32, 32)
    
    # SSL forward pass
    p1, p2, z1, z2 = model_cifar(x1, x2, mode='ssl')
    loss = simsiam_loss(p1, p2, z1, z2)
    
    print(f"CIFAR-10 model:")
    print(f"p1 shape: {p1.shape}, p2 shape: {p2.shape}")
    print(f"z1 shape: {z1.shape}, z2 shape: {z2.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Classification forward pass
    logits = model_cifar(x1, mode='classify')
    print(f"Classification logits shape: {logits.shape}")
    
    # Test ImageNet model
    model_imagenet = create_simsiam_model('imagenet', backbone='resnet50')
    x1_large = torch.randn(16, 3, 224, 224)  # ImageNet batch
    x2_large = torch.randn(16, 3, 224, 224)
    
    p1, p2, z1, z2 = model_imagenet(x1_large, x2_large, mode='ssl')
    loss = simsiam_loss(p1, p2, z1, z2)
    
    print(f"\nImageNet model:")
    print(f"p1 shape: {p1.shape}, p2 shape: {p2.shape}")
    print(f"z1 shape: {z1.shape}, z2 shape: {z2.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Print model size
    total_params = sum(p.numel() for p in model_cifar.parameters())
    print(f"\nCIFAR-10 model parameters: {total_params:,}")
    
    total_params = sum(p.numel() for p in model_imagenet.parameters())
    print(f"ImageNet model parameters: {total_params:,}")