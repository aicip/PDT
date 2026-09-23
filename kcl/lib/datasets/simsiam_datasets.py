import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, ImageNet
import torch.distributed as dist


class SimSiamTransform:
    """
    Data augmentation for SimSiam on CIFAR-10
    Returns two differently augmented versions of the same image
    """
    
    def __init__(self, image_size=32, normalize=None):
        """
        Args:
            image_size (int): Size of the input image 
            normalize (transforms.Normalize): Normalization transform
        """
        # Default CIFAR-10 normalization if none provided
        if normalize is None:
            normalize = transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        
        # SimSiam data augmentation pipeline for CIFAR-10
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=int(0.1 * image_size) * 2 + 1, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            normalize
        ])
        
    def __call__(self, x):
        """
        Apply augmentation twice to get two different views
        Returns: (x1, x2) - two augmented versions of the same image
        """
        return self.transform(x), self.transform(x)


class SimSiamTransformImageNet:
    """
    Data augmentation for SimSiam on ImageNet
    """
    
    def __init__(self, image_size=224, normalize=None):
        # Default ImageNet normalization
        if normalize is None:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            normalize
        ])
        
    def __call__(self, x):
        return self.transform(x), self.transform(x)


class SimSiamCIFAR10(Dataset):
    """
    CIFAR-10 dataset wrapper for SimSiam
    Returns pairs of augmented images instead of (image, label)
    """
    
    def __init__(self, root, train=True, download=True, transform=None):
        self.dataset = CIFAR10(root=root, train=train, download=download, transform=None)
        
        if transform is None:
            self.transform = SimSiamTransform(image_size=32)
        else:
            self.transform = transform
            
    def __getitem__(self, index):
        image, target = self.dataset[index]  # target is not used in SSL
        
        # Apply transform to get two augmented versions
        if self.transform:
            x1, x2 = self.transform(image)
            return x1, x2
        else:
            # If no transform, return the same image twice
            image = transforms.ToTensor()(image)
            return image, image.clone()
    
    def __len__(self):
        return len(self.dataset)


class SimSiamImageNetDataset(Dataset):
    """
    ImageNet dataset wrapper for SimSiam
    """
    
    def __init__(self, root, split='train', transform=None):
        self.dataset = ImageNet(root=root, split=split)
        
        if transform is None:
            self.transform = SimSiamTransformImageNet(image_size=224)
        else:
            self.transform = transform
            
    def __getitem__(self, index):
        image, target = self.dataset[index]
        
        if self.transform:
            x1, x2 = self.transform(image)
            return x1, x2
        else:
            image = transforms.ToTensor()(image)
            return image, image.clone()
    
    def __len__(self):
        return len(self.dataset)


def create_simsiam_cifar10(
    data_dir, 
    train_batch_size=256, 
    test_batch_size=256,
    add_suffix=False,
    img_size=32,
    dist=False,
    ws=None,
    rank=None,
    num_workers=4
):
    """
    Create CIFAR-10 dataloaders for SimSiam training (original function)
    
    Args:
        data_dir: Path to dataset
        train_batch_size: Training batch size
        test_batch_size: Test batch size  
        add_suffix: Add suffix to data directory
        img_size: Input image size
        dist: Use distributed training
        ws: World size for distributed training
        rank: Rank for distributed training
        num_workers: Number of data loading workers
        
    Returns:
        train_loader, test_loader
    """
    if add_suffix:
        data_dir = os.path.join(data_dir, "cifar10")
    
    # Create transforms
    train_transform = SimSiamTransform(image_size=img_size)
    
    # For test/validation, we typically don't need augmentation pairs
    # We use a simple transform for downstream evaluation
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    # Create datasets
    train_dataset = SimSiamCIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # For test set, we usually want single images for evaluation
    # But keep the same interface for consistency
    test_dataset = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create samplers for distributed training
    train_sampler = None
    test_sampler = None
    if dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=ws, rank=rank, shuffle=True
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=ws, rank=rank, shuffle=False
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, test_loader


def create_simsiam_imagenet(
    data_dir,
    train_batch_size=256,
    test_batch_size=256,
    img_size=224,
    dist=False,
    ws=None,
    rank=None,
    num_workers=4
):
    """
    Create ImageNet dataloaders for SimSiam training (original function)
    """
    # Create transforms
    train_transform = SimSiamTransformImageNet(image_size=img_size)
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create datasets  
    train_dataset = SimSiamImageNetDataset(
        root=data_dir,
        split='train',
        transform=train_transform
    )
    
    # For validation, use standard ImageNet dataset
    val_dataset = ImageNet(
        root=data_dir,
        split='val',
        transform=test_transform
    )
    
    # Create samplers
    train_sampler = None
    val_sampler = None
    if dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=ws, rank=rank, shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=ws, rank=rank, shuffle=False
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def create_simsiam_cifar10_with_probe(
    data_dir, 
    train_batch_size=256, 
    test_batch_size=256,
    add_suffix=False,
    img_size=32,
    dist=False,
    ws=None,
    rank=None,
    num_workers=4
):
    """
    Create CIFAR-10 dataloaders for SimSiam training WITH probe dataset for proper evaluation
    
    Args:
        data_dir: Path to dataset
        train_batch_size: Training batch size
        test_batch_size: Test batch size  
        add_suffix: Add suffix to data directory
        img_size: Input image size
        dist: Use distributed training
        ws: World size for distributed training
        rank: Rank for distributed training
        num_workers: Number of data loading workers
        
    Returns:
        train_loader (SSL pairs), test_loader (single images), probe_train_loader (single images)
    """
    # 1. Create original SSL datasets
    train_loader, test_loader = create_simsiam_cifar10(
        data_dir=data_dir,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        add_suffix=add_suffix,
        img_size=img_size,
        dist=dist,
        ws=ws,
        rank=rank,
        num_workers=num_workers
    )
    
    # 2. Create probe training dataset (standard augmentation for linear probe)
    if add_suffix:
        probe_data_dir = os.path.join(data_dir, "cifar10")
    else:
        probe_data_dir = data_dir
    
    probe_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    probe_train_dataset = CIFAR10(
        root=probe_data_dir, 
        train=True, 
        download=True, 
        transform=probe_transform
    )
    
    # 3. Create probe sampler for distributed training
    probe_train_sampler = None
    if dist:
        probe_train_sampler = torch.utils.data.distributed.DistributedSampler(
            probe_train_dataset, 
            num_replicas=ws, 
            rank=rank, 
            shuffle=False  # No need to shuffle for probe training
        )
    
    # 4. Create probe dataloader
    probe_train_loader = DataLoader(
        probe_train_dataset,
        batch_size=test_batch_size,  # Use test batch size for probe
        shuffle=(probe_train_sampler is None),
        sampler=probe_train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, test_loader, probe_train_loader


def create_simsiam_imagenet_with_probe(
    data_dir,
    train_batch_size=256,
    test_batch_size=256,
    img_size=224,
    dist=False,
    ws=None,
    rank=None,
    num_workers=4
):
    """
    Create ImageNet dataloaders for SimSiam training WITH probe dataset for proper evaluation
    
    Returns:
        train_loader (SSL pairs), val_loader (single images), probe_train_loader (single images)
    """
    # 1. Create original SSL datasets
    train_loader, val_loader = create_simsiam_imagenet(
        data_dir=data_dir,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        img_size=img_size,
        dist=dist,
        ws=ws,
        rank=rank,
        num_workers=num_workers
    )
    
    # 2. Create probe training dataset (standard augmentation)
    probe_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # ImageNet training set for probe training
    probe_train_dataset = ImageNet(
        root=data_dir,
        split='train',
        transform=probe_transform
    )
    
    # 3. Create probe sampler
    probe_train_sampler = None
    if dist:
        probe_train_sampler = torch.utils.data.distributed.DistributedSampler(
            probe_train_dataset,
            num_replicas=ws,
            rank=rank,
            shuffle=False
        )
    
    # 4. Create probe dataloader
    probe_train_loader = DataLoader(
        probe_train_dataset,
        batch_size=test_batch_size,
        shuffle=(probe_train_sampler is None),
        sampler=probe_train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, probe_train_loader


# Test the dataset implementation
if __name__ == "__main__":
    print("Testing SimSiam CIFAR-10 dataset...")
    
    # Test original function
    train_loader, test_loader = create_simsiam_cifar10(
        data_dir="./data",
        train_batch_size=32,
        test_batch_size=32
    )
    
    # Check training data (should return pairs)
    train_batch = next(iter(train_loader))
    x1, x2 = train_batch
    print(f"Original - Train batch x1 shape: {x1.shape}, x2 shape: {x2.shape}")
    print(f"Original - x1 dtype: {x1.dtype}, x2 dtype: {x2.dtype}")
    print(f"Original - x1 range: [{x1.min():.3f}, {x1.max():.3f}]")
    
    # Check test data (single images with labels)
    test_batch = next(iter(test_loader))
    if len(test_batch) == 2:  # (data, target) format
        data, target = test_batch
        print(f"Original - Test batch data shape: {data.shape}, target shape: {target.shape}")
    
    # Verify augmentation is working - x1 and x2 should be different
    diff = torch.abs(x1 - x2).mean()
    print(f"Original - Mean difference between x1 and x2: {diff:.6f} (should be > 0)")
    
    print("\n" + "="*60)
    print("Testing SimSiam CIFAR-10 dataset WITH probe...")
    
    # Test new function with probe
    train_loader_probe, test_loader_probe, probe_train_loader = create_simsiam_cifar10_with_probe(
        data_dir="./data",
        train_batch_size=32,
        test_batch_size=32
    )
    
    # Check probe training data
    probe_batch = next(iter(probe_train_loader))
    probe_data, probe_target = probe_batch
    print(f"With probe - Probe train data shape: {probe_data.shape}, target shape: {probe_target.shape}")
    print(f"With probe - Probe data range: [{probe_data.min():.3f}, {probe_data.max():.3f}]")
    
    # Check that we get the same SSL training data
    train_batch_probe = next(iter(train_loader_probe))
    x1_probe, x2_probe = train_batch_probe
    print(f"With probe - Train batch x1 shape: {x1_probe.shape}, x2 shape: {x2_probe.shape}")
    
    print("\nDataset tests completed successfully!")
    print("Key differences:")
    print("- Original function returns: (train_loader, test_loader)")
    print("- New function returns: (train_loader, test_loader, probe_train_loader)")
    print("- probe_train_loader provides single images for proper linear probe evaluation")