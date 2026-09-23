"""ImageNet-1k loaders (ImageFolder layout: ``<dataset_dir>/train/<class>/*``, ``<dataset_dir>/val/<class>/*``)."""

import math
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode


class RASampler(torch.utils.data.Sampler):
    """Distributed sampler with repeated augmentation.

    Each process sees a different part of the data, and every image is repeated three
    times per epoch with different augmentations. Adapted from ``samplers.py`` of
    facebookresearch/deit (Apache License 2.0), see THIRD_PARTY_NOTICES.md.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, num_repeats=3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * self.num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        indices = [ele for ele in indices for _ in range(self.num_repeats)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def imagenet_dataset(dataset_dir):
    interpolation = InterpolationMode.BILINEAR
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=interpolation, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=interpolation, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_set = datasets.ImageFolder(os.path.join(dataset_dir, "train"), transform=train_transform)
    val_set = datasets.ImageFolder(os.path.join(dataset_dir, "val"), transform=val_transform)
    return train_set, val_set


def create_imagenet(dataset_dir, train_batch_size=128, test_batch_size=128, num_workers=16,
                    distributed=False, rank=0, ws=1):
    train_set, test_set = imagenet_dataset(dataset_dir)
    if distributed:
        train_sampler = RASampler(train_set, num_replicas=ws, rank=rank, shuffle=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=ws, rank=rank, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_set)
        test_sampler = torch.utils.data.SequentialSampler(test_set)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, num_workers=num_workers,
                              sampler=train_sampler, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, num_workers=max(num_workers // 2, 1),
                             sampler=test_sampler, pin_memory=True, drop_last=False)
    return train_loader, test_loader
