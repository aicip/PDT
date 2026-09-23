import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from kcl.lib.datasets.single_class_sampler import SingleClassBatchSampler


def cifar10_dataset(dataset_dir, img_size):
    """CIFAR-10 train / test sets, downloaded to ``dataset_dir`` if needed."""
    transform = transforms.Compose([
        transforms.Resize(size=img_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    train_set = datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=transform)
    return train_set, test_set


def create_cifar10(dataset_dir="data", train_batch_size=256, test_batch_size=256, img_size=128,
                   num_workers=4, dist=False, ws=1, rank=0):
    train_set, test_set = cifar10_dataset(dataset_dir, img_size)
    if dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=ws, rank=rank)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=ws, rank=rank, shuffle=False)
    else:
        train_sampler, test_sampler = None, None
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=train_sampler is None,
                              num_workers=num_workers, sampler=train_sampler)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False,
                             num_workers=max(num_workers // 2, 1), sampler=test_sampler)
    return train_loader, test_loader


def create_cifar10_single_class(dataset_dir="data", train_batch_size=256, test_batch_size=256, img_size=128,
                                num_workers=4, dist=False, ws=1, rank=0):
    """Non-i.i.d. variant: every training mini-batch contains samples of a single class."""
    if dist:
        raise NotImplementedError("single-class batches are only implemented for single-process training")
    train_set, test_set = cifar10_dataset(dataset_dir, img_size)
    train_loader = DataLoader(train_set, batch_sampler=SingleClassBatchSampler(train_set, batch_size=train_batch_size),
                              num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=max(num_workers // 2, 1))
    return train_loader, test_loader
