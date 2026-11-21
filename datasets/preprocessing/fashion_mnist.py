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

INTERPOLATION_MAP = {
    "bilinear": transforms.InterpolationMode.BILINEAR,
    "bicubic": transforms.InterpolationMode.BICUBIC,
    "nearest": transforms.InterpolationMode.NEAREST,
    "box": transforms.InterpolationMode.BOX,
    "hamming": transforms.InterpolationMode.HAMMING,
    "lanczos": transforms.InterpolationMode.LANCZOS,
}

def get_fashion_mnist_transforms(
    split="train",
    image_size=32,
    original_size=28,
    resize_method="pad_or_resize",
    interpolation="bilinear",
    norm_range="[-1, 1]",
    mean=None,
    std=None,
    hflip=None,
    flip_prob=0.5
):
    transform_list = []
    interp_mode = INTERPOLATION_MAP.get(str(interpolation).lower(), transforms.InterpolationMode.BILINEAR)
    
    # 1. Resize/Padding theo cấu hình YAML
    if resize_method == "pad":
        pad_val = max(0, (image_size - original_size) // 2)
        transform_list.append(transforms.Pad(pad_val, fill=0))
    elif resize_method == "pad_or_resize":
        # Khuyến nghị mục 2.6 của đề tài: pad 2 pixel thành 32x32 để giữ nguyên độ sắc nét không bị nội suy làm mờ
        if image_size == 32 and original_size == 28:
            transform_list.append(transforms.Pad(2, fill=0))
        elif image_size != original_size:
            transform_list.append(transforms.Resize((image_size, image_size), interpolation=interp_mode))
    else: # "resize"
        if image_size != original_size:
            transform_list.append(transforms.Resize((image_size, image_size), interpolation=interp_mode))
        
    # 2. Augmentation theo cấu hình YAML
    should_flip = (split == "train") if hflip is None else hflip
    if should_flip and flip_prob is not None and flip_prob > 0:
        transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))
        
    # 3. Chuyển sang Tensor [0, 1]
    transform_list.append(transforms.ToTensor())
    
    # 4. Chuẩn hóa theo cấu hình YAML
    if norm_range == "[-1, 1]":
        norm_mean = mean if mean is not None else [0.5]
        norm_std = std if std is not None else [0.5]
        transform_list.append(transforms.Normalize(mean=norm_mean, std=norm_std))
        
    return transforms.Compose(transform_list)


def load_fashion_mnist(
    root_dir="datasets/raw/fashion_mnist",
    split="train",
    image_size=32,
    original_size=28,
    resize_method="pad_or_resize",
    interpolation="bilinear",
    norm_range="[-1, 1]",
    mean=None,
    std=None,
    hflip=None,
    flip_prob=0.5,
    val_size=10000,
    seed=42,
    split_file=None
):
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
    train_transform = get_fashion_mnist_transforms(
        split="train",
        image_size=image_size,
        original_size=original_size,
        resize_method=resize_method,
        interpolation=interpolation,
        norm_range=norm_range,
        mean=mean,
        std=std,
        hflip=hflip,
        flip_prob=flip_prob
    )
    test_transform = get_fashion_mnist_transforms(
        split="test",
        image_size=image_size,
        original_size=original_size,
        resize_method=resize_method,
        interpolation=interpolation,
        norm_range=norm_range,
        mean=mean,
        std=std,
        hflip=False,
        flip_prob=0.0
    )
    train_full = torchvision.datasets.FashionMNIST(
        root=actual_root,
        train=True,
        download=True,
        transform=train_transform
    )
    test_full = torchvision.datasets.FashionMNIST(
        root=actual_root,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Tạo hoặc nạp split cố định phân tầng (stratified, seed 42)
    split_meta = create_or_load_split(
        "fashion_mnist",
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
        # Áp dụng transform không augment cho tập val
        val_transform = get_fashion_mnist_transforms(
            split="val",
            image_size=image_size,
            original_size=original_size,
            resize_method=resize_method,
            interpolation=interpolation,
            norm_range=norm_range,
            mean=mean,
            std=std,
            hflip=False,
            flip_prob=0.0
        )
        val_dataset = torchvision.datasets.FashionMNIST(
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

