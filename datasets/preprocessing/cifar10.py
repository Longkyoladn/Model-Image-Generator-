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

def get_cifar10_transforms(split="train", image_size=32, norm_range="[-1, 1]", hflip=True):
    transform_list = []
    
    if image_size != 32:
        transform_list.append(transforms.Resize((image_size, image_size)))
        
    if split == "train" and hflip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
    transform_list.append(transforms.ToTensor())
    
    if norm_range == "[-1, 1]":
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        
    return transforms.Compose(transform_list)


def load_cifar10(root_dir="datasets/raw/cifar10", split="train", image_size=32, norm_range="[-1, 1]"):
    """
    Nạp CIFAR-10 và chia thành train (40k), val (10k), test (10k) theo split metadata cố định.
    """
    actual_root = root_dir
    os.makedirs(actual_root, exist_ok=True)

    train_full = torchvision.datasets.CIFAR10(
        root=actual_root,
        train=True,
        download=True,
        transform=get_cifar10_transforms(split=split, image_size=image_size, norm_range=norm_range, hflip=(split == "train"))
    )
    test_full = torchvision.datasets.CIFAR10(
        root=actual_root,
        train=False,
        download=True,
        transform=get_cifar10_transforms(split="test", image_size=image_size, norm_range=norm_range, hflip=False)
    )
    
    split_meta = create_or_load_split("cifar10", total_train_size=len(train_full), total_test_size=len(test_full), val_size=10000, seed=42)
    
    if split == "train":
        return Subset(train_full, split_meta["train_indices"])
    elif split == "val":
        val_dataset = torchvision.datasets.CIFAR10(
            root=actual_root,
            train=True,
            download=False,
            transform=get_cifar10_transforms(split="val", image_size=image_size, norm_range=norm_range, hflip=False)
        )
        return Subset(val_dataset, split_meta["val_indices"])
    elif split == "test":
        return test_full
    else:
        raise ValueError(f"Split không hợp lệ: {split}. Chọn 'train', 'val', hoặc 'test'.")

