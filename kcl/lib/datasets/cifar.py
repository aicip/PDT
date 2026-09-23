import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from kcl.lib.datasets.single_class_sampler import SingleClassBatchSampler


def cifar10_dataset(dataset_dir, add_suffix=True, transforms_=None):
    root = os.path.join(dataset_dir, 'CIFAR10') if add_suffix else dataset_dir
    train_set = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transforms_
    )
    test_set = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transforms_
    )

    return train_set, test_set


def create_cifar10(
    dataset_dir='data',
    train_batch_size=128,
    test_batch_size=128,
    add_suffix=True,
    dist=False,
    img_size=128,
    ws=1,
    rank=0
):
    
    __transform = transforms.Compose(
        [
            transforms.Resize(size=img_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    train_set, test_set = cifar10_dataset(dataset_dir, add_suffix=add_suffix, transforms_=__transform)
    
    if dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=ws, rank=rank)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=ws, rank=rank)
    else:
        train_sampler, test_sampler = None, None

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, shuffle=train_sampler is None, num_workers=4, sampler=train_sampler
    )
    test_loader = DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False, num_workers=2, sampler=test_sampler
    )

    return train_loader, test_loader


def create_cifar10_single_class(
    dataset_dir='data',
    train_batch_size=256,
    test_batch_size=128,
    add_suffix=True,
    img_size=32,
    dist=False,
    ws=1,
    rank=0
):
    __transform = transforms.Compose([
        transforms.Resize(size=img_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    train_set, test_set = cifar10_dataset(dataset_dir, add_suffix=add_suffix, transforms_=__transform)
    
    if dist:
        # distributed training
        train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            sampler=torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=ws, rank=rank),
            num_workers=4
        )
    else:
        # non-distributed training
        train_loader = DataLoader(
            train_set,
            batch_sampler=SingleClassBatchSampler(train_set, batch_size=train_batch_size),
            num_workers=4
        )

    # test set remains unchanged
    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=2,
        sampler=torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=ws, rank=rank) if dist else None
    )

    return train_loader, test_loader