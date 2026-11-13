import os
import torch
from torch.utils.data import Dataset, Subset
import torchvision
from torchvision import transforms
from .splits import create_or_load_split

FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def get_fashion_mnist_transforms(split="train", image_size=32, norm_range="[-1, 1]", hflip=True):
    transform_list = []
    
    # 1. Resize từ 28x28 -> 32x32 (lũy thừa của 2, tối ưu cho ConvTranspose và UNet)
    if image_size != 28:
        transform_list.append(transforms.Resize((image_size, image_size)))
        
    # 2. Augmentation cho tập train
    if split == "train" and hflip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
    # 3. Chuyển sang Tensor [0, 1]
    transform_list.append(transforms.ToTensor())
    
    # 4. Chuẩn hóa
    if norm_range == "[-1, 1]":
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        
    return transforms.Compose(transform_list)


def load_fashion_mnist(root_dir="datasets/raw/fashion_mnist", split="train", image_size=32, norm_range="[-1, 1]"):
    """
    Nạp Fashion-MNIST và chia thành train (50k), val (10k), test (10k) theo metadata split cố định.
    """
    # Tự động tìm thư mục dữ liệu (hỗ trợ cả data/ ở thư mục gốc nếu đã tải trước)
    possible_paths = [
        root_dir,
        os.path.join("datasets", "raw", "fashion_mnist"),
        os.path.join("..", "data"),
        "data"
    ]
    actual_root = root_dir
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "FashionMNIST")) or os.path.exists(os.path.join(p, "raw")):
            actual_root = p
            break

    # Dataset gốc
    train_full = torchvision.datasets.FashionMNIST(
        root=actual_root,
        train=True,
        download=True,
        transform=get_fashion_mnist_transforms(split=split, image_size=image_size, norm_range=norm_range, hflip=(split == "train"))
    )
    test_full = torchvision.datasets.FashionMNIST(
        root=actual_root,
        train=False,
        download=True,
        transform=get_fashion_mnist_transforms(split="test", image_size=image_size, norm_range=norm_range, hflip=False)
    )
    
    # Tạo hoặc nạp split cố định (seed 42)
    split_meta = create_or_load_split("fashion_mnist", total_train_size=len(train_full), total_test_size=len(test_full), val_size=10000, seed=42)
    
    if split == "train":
        return Subset(train_full, split_meta["train_indices"])
    elif split == "val":
        # Áp dụng transform không augment cho tập val
        val_dataset = torchvision.datasets.FashionMNIST(
            root=actual_root,
            train=True,
            download=False,
            transform=get_fashion_mnist_transforms(split="val", image_size=image_size, norm_range=norm_range, hflip=False)
        )
        return Subset(val_dataset, split_meta["val_indices"])
    elif split == "test":
        return test_full
    else:
        raise ValueError(f"Split không hợp lệ: {split}. Chọn 'train', 'val', hoặc 'test'.")

