import os
import pickle

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm, trange
import math

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import pipeline_def
    from nvidia.dali.plugin.pytorch import (DALIClassificationIterator,
                                            LastBatchPolicy)
    from nvidia.dali.auto_aug import rand_augment
    imagenet_loaded = True
except ImportError:
    imagenet_loaded = False
    pipeline_def = lambda x: x


from kcl.utils.logger import get_logger


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.

    This is borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def fast_collate(batch, memory_format):
    """Based on fast_collate from the APEX example
       https://github.com/NVIDIA/apex/blob/5b5d41034b506591a316c308c3d2cd14d5187e23/examples/imagenet/main_amp.py#L265
    """
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
        memory_format=memory_format
    )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


@pipeline_def(enable_conditionals=True)
def create_dali_pipeline(
    data_dir,
    crop,
    size,
    shard_id,
    num_shards,
    dali_cpu=False,
    is_training=True
):
    images, labels = fn.readers.file(
        file_root=data_dir,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=is_training,
        pad_last_batch=True,
        name="Reader"
    )
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(
            images,
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            preallocate_width_hint=preallocate_width_hint,
            preallocate_height_hint=preallocate_height_hint,
            random_aspect_ratio=[0.8, 1.25],
            random_area=[0.1, 1.0],
            num_attempts=100
        )
        images = fn.resize(
            images,
            device=dali_device,
            resize_x=crop,
            resize_y=crop,
            interp_type=types.INTERP_TRIANGULAR
        )
        images = rand_augment.rand_augment(images, shape=[crop,crop], m=9, n=2)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(
            images, device=decoder_device, output_type=types.RGB
        )
        images = fn.resize(
            images,
            device=dali_device,
            size=size,
            mode="not_smaller",
            interp_type=types.INTERP_TRIANGULAR
        )
        mirror = False

    images = fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(crop, crop),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=mirror
    )
    labels = labels.gpu()
    return images, labels


def create_imagenet_dali(dataset_dir: str, train_size, test_size, shard_id=0, num_shards=1, device_id=0):
    if imagenet_loaded is False:
        raise ImportError(
            "Please install DALI from https://www.github.com/NVIDIA/DALI to use ImageNet."
        )
    train_pipe = create_dali_pipeline(
        batch_size=train_size,
        num_threads=16,
        device_id=device_id,
        seed=12,
        data_dir=os.path.join(dataset_dir, 'train'),
        crop=224,
        size=256,
        dali_cpu=False,
        shard_id=shard_id,
        num_shards=num_shards,
        is_training=True
    )
    val_pipe = create_dali_pipeline(
        batch_size=test_size,
        num_threads=8,
        device_id=device_id,
        seed=12,
        data_dir=os.path.join(dataset_dir, 'val'),
        crop=224,
        size=256,
        dali_cpu=False,
        shard_id=shard_id,
        num_shards=num_shards,
        is_training=False
    )
    train_pipe.build()
    train_loader = DALIClassificationIterator(
        train_pipe,
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True
    )
    val_pipe.build()
    val_loader = DALIClassificationIterator(
        val_pipe,
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True
    )

    return train_loader, val_loader


def imagenet_dataset(dataset_dir: str):
    """
    Create ImageNet dataset
    :param dataset_dir: Directory to store ImageNet dataset (if not already downloaded)
    """
    interpolation=InterpolationMode.BILINEAR
    # transform = transforms.Compose(
    #     [
    #         transforms.RandomResizedCrop(224, interpolation=interpolation, antialias=True),
    #         transforms.RandomHorizontalFlip(0.5),
    #         transforms.RandAugment(interpolation=interpolation, magnitude=9),
    #         transforms.PILToTensor(),
    #         transforms.ConvertImageDtype(torch.float),
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #         # transforms.Resize(256, antialias=True),
    #         # transforms.CenterCrop(224),
    #         # transforms.RandomHorizontalFlip(),
    #         # transforms.ToTensor(),
    #         # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #     ]
    # )

    # train_set = datasets.ImageFolder(
    #     os.path.join(dataset_dir, 'train'), transform=transform
    # )
    # val_set = datasets.ImageFolder(
    #     os.path.join(dataset_dir, 'val'), transform=transform
    # )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=interpolation, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=interpolation, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(
        os.path.join(dataset_dir, 'train'), transform=train_transform
    )
    val_set = datasets.ImageFolder(
        os.path.join(dataset_dir, 'val'), transform=val_transform
    )

    return train_set, val_set


def create_imagenet(
    dataset_dir=os.environ.get('IMAGENET_DIR', 'data/imagenet'),
    train_batch_size=128,
    test_batch_size=128,
    distributed=False,
    rank=0,
    ws=1
):
    train_set, test_set = imagenet_dataset(dataset_dir)

    # if distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(
    #         train_set, num_replicas=ws, rank=rank
    #     )
    #     test_sampler = torch.utils.data.distributed.DistributedSampler(
    #         test_set, num_replicas=ws, rank=rank
    #     )
    # else:
    #     train_sampler = None
    #     test_sampler = None

    # train_loader = DataLoader(
    #     train_set, batch_size=train_batch_size, num_workers=16, sampler=train_sampler, pin_memory=True
    # )
    # test_loader = DataLoader(
    #     test_set, batch_size=test_batch_size, num_workers=8, sampler=test_sampler, pin_memory=True
    # )

    if distributed:
        train_sampler = RASampler(train_set, num_replicas=ws, rank=rank, shuffle=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_set, num_replicas=ws, rank=rank, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(train_set)
        test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = DataLoader(
        train_set, 
        batch_size=train_batch_size, 
        num_workers=16, 
        sampler=train_sampler, 
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=test_batch_size, 
        num_workers=8, 
        sampler=test_sampler, 
        pin_memory=True,
        drop_last=False
    )

    return train_loader, test_loader


def create_resize_dataset(
    dataset_dir=os.environ.get('IMAGENET_DIR', 'data/imagenet'),
    train_batch_size=128,
    test_batch_size=128
):
    train_set = ResizeDataset(root=dataset_dir, train=True)
    test_set = ResizeDataset(root=dataset_dir, train=False)
    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


class ResizeDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        self.logger = get_logger()
        self.tf = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                )
            ]
        )

        if self.train:
            self.data = []
            self.labels = []
            self.logger.info('Loading training data...')
            # Loading all training files
            for i in trange(1, 11, desc='Loading training batch'):
                with open(
                    os.path.join(self.root, f'train_data_batch_{i}'), 'rb'
                ) as f:
                    batch_data = pickle.load(f)
                    self.data.append(batch_data['data'])
                    self.labels.extend(batch_data['labels'])

            self.data = np.vstack(self.data)    # Stacking the data vertically
            self.mean = batch_data[
                'mean'
            ]    # Using mean from the last file, assuming it's the same for all files
            self.logger.info("Done loading training data")

        else:
            # Loading the validation file
            self.logger.info("loading validation data...")
            with open(os.path.join(self.root, 'val_data'), 'rb') as f:
                val_data = pickle.load(f)
                self.data = val_data['data']
                self.labels = val_data['labels']
            self.logger.info("Done loading validation data")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Reshape the data into 3x32x32 format
        img = np.reshape(self.data[idx], (3, 64, 64))
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        img = img.resize((256, 256), Image.BILINEAR)

        img = self.tf(img)

        # If it's training data, subtract the mean

        label = self.labels[
            idx] - 1    # Adjust the label since indexing starts from 1

        return img, torch.tensor(label)


# Example of how to use the dataset
if __name__ == "__main__":
    root = os.environ.get('IMAGENET_DIR', 'data/imagenet')
    train, test = create_imagenet_dali(root, 128, 128)
    train_a, test_a = create_imagenet(root, 128, 128)
    for a in train:
        print(type(a))
        break
    for b in train_a:
        print(type(b))
        break
    input()
    # dataset = ResizeDataset(root=root, train=True)
    # for i, (img, label) in enumerate(tqdm(dataset)):
    #     os.makedirs(f'{root}/train/{label}', exist_ok=True)
    #     img = Image.fromarray(np.transpose(img.numpy(), (1, 2, 0)))
    #     img.save(f'{root}/train/{label}/{i:03d}.png')
    # dataset = ResizeDataset(root=root, train=False)
    # for i, (img, label) in enumerate(tqdm(dataset)):
    #     os.makedirs(f'{root}/val/{label}', exist_ok=True)
    #     img = Image.fromarray(np.transpose(img.numpy(), (1, 2, 0)))
    #     img.save(f'{root}/val/{label}/{i:03d}.png')
