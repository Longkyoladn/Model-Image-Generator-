import os
import torch
from torch.utils.data import Dataset, Subset
import torchvision
from torchvision import transforms
from .splits import create_or_load_split

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

INTERPOLATION_MAP = {
    "bilinear": transforms.InterpolationMode.BILINEAR,
    "bicubic": transforms.InterpolationMode.BICUBIC,
    "nearest": transforms.InterpolationMode.NEAREST,
    "box": transforms.InterpolationMode.BOX,
    "hamming": transforms.InterpolationMode.HAMMING,
    "lanczos": transforms.InterpolationMode.LANCZOS,
}

def get_cifar10_transforms(
    split="train",
    image_size=32,
    interpolation="bilinear",
    norm_range="[-1, 1]",
    mean=None,
    std=None,
    hflip=None,
    flip_prob=0.5
):
    transform_list = []
    interp_mode = INTERPOLATION_MAP.get(str(interpolation).lower(), transforms.InterpolationMode.BILINEAR)
    
    if image_size != 32:
        transform_list.append(transforms.Resize((image_size, image_size), interpolation=interp_mode))
        
    should_flip = (split == "train") if hflip is None else hflip
    if should_flip and flip_prob is not None and flip_prob > 0:
        transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))
        
    transform_list.append(transforms.ToTensor())
    
    if norm_range == "[-1, 1]":
        norm_mean = mean if mean is not None else [0.5, 0.5, 0.5]
        norm_std = std if std is not None else [0.5, 0.5, 0.5]
        transform_list.append(transforms.Normalize(mean=norm_mean, std=norm_std))
        
    return transforms.Compose(transform_list)


def load_cifar10(
    root_dir="datasets/raw/cifar10",
    split="train",
    image_size=32,
    interpolation="bilinear",
    norm_range="[-1, 1]",
    mean=None,
    std=None,
    hflip=None,
    flip_prob=0.5,
    val_size=5000,
    seed=42,
    split_file=None
):
    """
    Nạp CIFAR-10 và chia thành train (45k), val (5k), test (10k) theo split metadata cố định.
    """
    actual_root = root_dir
    os.makedirs(actual_root, exist_ok=True)

    train_transform = get_cifar10_transforms(
        split="train",
        image_size=image_size,
        interpolation=interpolation,
        norm_range=norm_range,
        mean=mean,
        std=std,
        hflip=hflip,
        flip_prob=flip_prob
    )
    test_transform = get_cifar10_transforms(
        split="test",
        image_size=image_size,
        interpolation=interpolation,
        norm_range=norm_range,
        mean=mean,
        std=std,
        hflip=False,
        flip_prob=0.0
    )
    train_full = torchvision.datasets.CIFAR10(
        root=actual_root,
        train=True,
        download=True,
        transform=train_transform
    )
    test_full = torchvision.datasets.CIFAR10(
        root=actual_root,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Tạo hoặc nạp split cố định phân tầng (stratified, seed 42)
    split_meta = create_or_load_split(
        "cifar10",
        total_train_size=len(train_full),
        total_test_size=len(test_full),
        val_size=val_size,
        seed=seed,
        labels=train_full.targets,
        split_file=split_file
    )
    
    if split == "train":
        return Subset(train_full, split_meta["train_indices"])
    elif split == "val":
        val_transform = get_cifar10_transforms(
            split="val",
            image_size=image_size,
            interpolation=interpolation,
            norm_range=norm_range,
            mean=mean,
            std=std,
            hflip=False,
            flip_prob=0.0
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root=actual_root,
            train=True,
            download=False,
            transform=val_transform
        )
        return Subset(val_dataset, split_meta["val_indices"])
    elif split == "test":
        return test_full
    else:
        raise ValueError(f"Split không hợp lệ: {split}. Chọn 'train', 'val', hoặc 'test'.")

