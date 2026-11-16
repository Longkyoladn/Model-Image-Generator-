import os
import yaml
import torch
from torch.utils.data import DataLoader

from .fashion_mnist import load_fashion_mnist, FASHION_MNIST_CLASSES
from .cifar10 import load_cifar10, CIFAR10_CLASSES
from .celeba import load_celeba, CELEBA_ATTRIBUTES

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_yaml_config(config_path):
    if not os.path.exists(config_path):
        candidate = os.path.join(PROJECT_ROOT, config_path)
        if os.path.exists(candidate):
            config_path = candidate
        else:
            norm_p = config_path.replace("\\", "/")
            if "configs/datasets/" in norm_p:
                alt = norm_p.replace("configs/datasets/", "configs/datasets_config/")
                if os.path.exists(alt):
                    config_path = alt
                elif os.path.exists(os.path.join(PROJECT_ROOT, alt)):
                    config_path = os.path.join(PROJECT_ROOT, alt)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def denormalize(tensor, norm_range="[-1, 1]"):
    """
    Chuyển đổi ảnh tensor từ khoảng [-1, 1] về [0, 1] để lưu file hoặc hiển thị.
    """
    if norm_range == "[-1, 1]":
        tensor = (tensor + 1.0) / 2.0
    return torch.clamp(tensor, 0.0, 1.0)

def build_dataloader(config_input, split="train", batch_size=None, shuffle=None, num_workers=None):
    """
    Hàm Factory chung tạo DataLoader cho cả 3 dataset (CelebA, Fashion-MNIST, CIFAR-10)
    dựa vào file YAML cấu hình.
    """
    if isinstance(config_input, str):
        cfg = load_yaml_config(config_input)
    else:
        cfg = config_input
        
    ds_cfg = cfg.get("datasets_config", cfg.get("dataset_config", cfg.get("dataset", {})))
    img_cfg = cfg.get("image", {})
    norm_cfg = cfg.get("normalization", {})
    dl_cfg = cfg.get("dataloader", {})
    cond_cfg = cfg.get("conditional", {})
    
    dataset_name = ds_cfg.get("name", "").lower()
    raw_dir = ds_cfg.get("raw_dir", "datasets/raw")
    if not os.path.exists(raw_dir) and not os.path.isabs(raw_dir):
        candidate_raw = os.path.join(PROJECT_ROOT, raw_dir)
        if os.path.exists(candidate_raw) or not os.path.exists(raw_dir):
            raw_dir = candidate_raw
    image_size = img_cfg.get("size", 32)
    norm_range = norm_cfg.get("range", "[-1, 1]")
    
    # 1. Khởi tạo Dataset tương ứng
    if "fashion" in dataset_name:
        dataset = load_fashion_mnist(
            root_dir=raw_dir,
            split=split,
            image_size=image_size,
            norm_range=norm_range
        )
    elif "cifar" in dataset_name:
        dataset = load_cifar10(
            root_dir=raw_dir,
            split=split,
            image_size=image_size,
            norm_range=norm_range
        )
    elif "celeb" in dataset_name:
        crop_size = img_cfg.get("crop_size", 140)
        is_cond = cond_cfg.get("enabled", False)
        target_attrs = cond_cfg.get("target_attributes", ["Smiling"])
        target_attr = target_attrs[0] if (target_attrs and is_cond) else "Smiling"
        dataset = load_celeba(
            root_dir=raw_dir,
            split=split,
            image_size=image_size,
            crop_size=crop_size,
            norm_range=norm_range,
            target_attr=target_attr,
            is_conditional=is_cond
        )
    else:
        raise ValueError(f"Không nhận dạng được dataset: {dataset_name}. Hỗ trợ: 'fashion_mnist', 'cifar10', 'celeba'.")
        
    # 2. Cấu hình DataLoader
    b_size = batch_size or dl_cfg.get("batch_size", 64)
    workers = num_workers if num_workers is not None else dl_cfg.get("num_workers", 0)
    
    if shuffle is None:
        do_shuffle = (split == "train")
    else:
        do_shuffle = shuffle
        
    pin_mem = dl_cfg.get("pin_memory", torch.cuda.is_available())
    
    # Sửa lỗi [P2]: drop_last CHỈ áp dụng cho tập train để cố định kích thước batch cho BatchNorm/GAN.
    # Tập validation và test TUYỆT ĐỐI không drop_last để giữ đủ 100% mẫu đánh giá FID/accuracy công bằng.
    if split == "train":
        drop_l = dl_cfg.get("drop_last", True)
    else:
        drop_l = False
    
    loader = DataLoader(
        dataset,
        batch_size=b_size,
        shuffle=do_shuffle,
        num_workers=workers,
        pin_memory=pin_mem,
        drop_last=drop_l
    )
    
    return loader, dataset

