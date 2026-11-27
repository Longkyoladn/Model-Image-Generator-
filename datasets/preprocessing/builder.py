import os
import yaml
import torch
from torch.utils.data import DataLoader

from .fashion_mnist import load_fashion_mnist, FASHION_MNIST_CLASSES
from .cifar10 import load_cifar10, CIFAR10_CLASSES
from .celeba import load_celeba, CELEBA_ATTRIBUTES

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_yaml_config(config_path):
    """Load dataset YAML, including paths used before the directory rename."""
    candidates = [
        config_path,
        os.path.join(PROJECT_ROOT, config_path),
    ]
    normalized_path = config_path.replace("\\", "/")
    config_dir = "configs/datasets_config"
    if "configs/datasets/" in normalized_path:
        renamed_path = normalized_path.replace("configs/datasets/", f"{config_dir}/")
        candidates.extend([renamed_path, os.path.join(PROJECT_ROOT, renamed_path)])
    elif f"{config_dir}/" not in normalized_path:
        candidates.extend([
            os.path.join(config_dir, config_path),
            os.path.join(PROJECT_ROOT, config_dir, config_path),
        ])

    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            with open(c, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
                
    raise FileNotFoundError(f"[builder.py] Không tìm thấy file cấu hình YAML: '{config_path}'")

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
    dựa vào file YAML cấu hình, đảm bảo mọi tham số (augmentation, normalization,
    resize_method, interpolation, split_file, seed) đều thực sự điều khiển pipeline tiền xử lý.
    """
    if isinstance(config_input, str):
        cfg = load_yaml_config(config_input)
    else:
        cfg = config_input
        
    ds_cfg = cfg.get("datasets_config", cfg.get("dataset_config", cfg.get("dataset", {})))
    img_cfg = cfg.get("image", {})
    norm_cfg = cfg.get("normalization", {})
    aug_cfg = cfg.get("augmentation", {})
    splits_cfg = cfg.get("splits", {})
    dl_cfg = cfg.get("dataloader", {})
    cond_cfg = cfg.get("conditional", {})
    
    dataset_name = ds_cfg.get("name", "").lower()
    raw_dir = ds_cfg.get("raw_dir", "datasets/raw")
    if not os.path.exists(raw_dir) and not os.path.isabs(raw_dir):
        candidate_raw = os.path.join(PROJECT_ROOT, raw_dir)
        if os.path.exists(candidate_raw) or not os.path.exists(raw_dir):
            raw_dir = candidate_raw
            
    split_file = ds_cfg.get("split_file")
    
    # 1. Các tham số hình ảnh từ Config
    image_size = img_cfg.get("size", 32)
    original_size = img_cfg.get("original_size", 28)
    resize_method = img_cfg.get("resize_method", "pad_or_resize")
    interpolation = img_cfg.get("interpolation", "bilinear")
    crop_type = img_cfg.get("crop_type", "center_crop")
    crop_size = img_cfg.get("crop_size", 140)
    
    # 2. Các tham số chuẩn hóa từ Config
    norm_range = norm_cfg.get("range", "[-1, 1]")
    norm_mean = norm_cfg.get("mean")
    norm_std = norm_cfg.get("std")
    
    # 3. Các tham số Augmentation theo từng split từ Config
    aug_split = aug_cfg.get(split, {})
    hflip = aug_split.get("horizontal_flip", (split == "train"))
    flip_prob = aug_split.get("flip_probability", 0.5)
    
    # 4. Các tham số Split & Seed từ Config
    seed = splits_cfg.get("random_seed", 42)
    val_samples = splits_cfg.get("val_samples")
    
    # 5. Khởi tạo Dataset tương ứng theo toàn bộ tham số từ YAML
    if "fashion" in dataset_name:
        dataset = load_fashion_mnist(
            root_dir=raw_dir,
            split=split,
            image_size=image_size,
            original_size=original_size,
            resize_method=resize_method,
            interpolation=interpolation,
            norm_range=norm_range,
            mean=norm_mean,
            std=norm_std,
            hflip=hflip,
            flip_prob=flip_prob,
            val_size=val_samples or 10000,
            seed=seed,
            split_file=split_file
        )
    elif "cifar" in dataset_name:
        dataset = load_cifar10(
            root_dir=raw_dir,
            split=split,
            image_size=image_size,
            interpolation=interpolation,
            norm_range=norm_range,
            mean=norm_mean,
            std=norm_std,
            hflip=hflip,
            flip_prob=flip_prob,
            val_size=val_samples or 5000,
            seed=seed,
            split_file=split_file
        )
    elif "celeb" in dataset_name:
        is_cond = cond_cfg.get("enabled", False)
        target_attrs = cond_cfg.get("target_attributes", ["Smiling"])
        target_attr = target_attrs[0] if (target_attrs and is_cond) else "Smiling"
        dataset = load_celeba(
            root_dir=raw_dir,
            split=split,
            image_size=image_size,
            crop_size=crop_size,
            crop_type=crop_type,
            interpolation=interpolation,
            norm_range=norm_range,
            mean=norm_mean,
            std=norm_std,
            hflip=hflip,
            flip_prob=flip_prob,
            target_attr=target_attr,
            is_conditional=is_cond
        )
    else:
        raise ValueError(f"Không nhận dạng được dataset: {dataset_name}. Hỗ trợ: 'fashion_mnist', 'cifar10', 'celeba'.")
        
    # 6. Cấu hình DataLoader
    b_size = batch_size or dl_cfg.get("batch_size", 64)
    workers = num_workers if num_workers is not None else dl_cfg.get("num_workers", 0)
    
    if shuffle is None:
        do_shuffle = (split == "train")
    else:
        do_shuffle = shuffle
        
    pin_mem = dl_cfg.get("pin_memory", torch.cuda.is_available())
    
    # drop_last CHỈ áp dụng cho tập train để cố định kích thước batch cho BatchNorm/GAN.
    # Tập validation và test TUYỆT ĐỐI không drop_last để giữ đủ 100% mẫu đánh giá FID công bằng.
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
