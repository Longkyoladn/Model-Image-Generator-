"""
Package mô hình DCGAN (Deep Convolutional Generative Adversarial Networks) - Task 3.2.
Cung cấp load_model và build_trainer theo chuẩn interface của toàn bộ dự án (models/interface.py).
"""

from __future__ import annotations

import os
from pathlib import Path
import yaml
import torch

from models.interface import GenerativeModel, Trainer
from .generator import Generator, weights_init
from .discriminator import Discriminator
from .dcgan import DCGAN, DCGANTrainer

DATASET_SHAPES = {
    "cifar10": (3, 32, 32),
    "fashion_mnist": (1, 32, 32),
    "celeba": (3, 64, 64),
}


def load_model(checkpoint: str, device: str = "cpu") -> GenerativeModel:
    """
    Nạp mô hình DCGAN từ file checkpoint .pt.
    Tự động khôi phục cấu hình kiến trúc và trọng số đã train.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    # Lấy thông số kiến trúc từ checkpoint
    image_shape = tuple(cfg.get("image_shape", (3, 32, 32)))
    latent_dim = cfg.get("latent_dim", 100)
    g_feat = cfg.get("generator_features", 64)
    d_feat = cfg.get("discriminator_features", 64)
    d_steps = cfg.get("d_steps", 1)

    model = DCGAN(
        image_shape=image_shape,
        latent_dim=latent_dim,
        generator_features=g_feat,
        discriminator_features=d_feat,
        d_steps=d_steps,
    )

    # Nạp state_dict
    if "generator_state_dict" in ckpt:
        model.generator.load_state_dict(ckpt["generator_state_dict"])
    elif "state_dict" in ckpt and "generator" in ckpt["state_dict"]:
        model.generator.load_state_dict(ckpt["state_dict"]["generator"])

    if "discriminator_state_dict" in ckpt:
        model.discriminator.load_state_dict(ckpt["discriminator_state_dict"])
    elif "state_dict" in ckpt and "discriminator" in ckpt["state_dict"]:
        model.discriminator.load_state_dict(ckpt["state_dict"]["discriminator"])

    return model.to(device)


def build_trainer(config: str, device: str = "cpu") -> Trainer:
    """
    Khởi tạo DCGANTrainer từ chuỗi cấu hình hoặc file YAML (dùng cho evaluation/calibrate.py).
    Hỗ trợ cả 2 định dạng:
    1. Chuỗi cấu hình calibrate/test: "dcgan:<batch_size>[:<dataset>]", ví dụ "dcgan:64:cifar10"
    2. Đường dẫn file cấu hình YAML: "configs/dcgan/cifar10/base.yaml"
    """
    # Trường hợp 1: Chuỗi ngắn gọn dạng "<model>:<batch_size>[:<dataset>]"
    if ":" in str(config) and not os.path.exists(str(config)):
        parts = str(config).split(":")
        batch_size = int(parts[1])
        dataset = parts[2] if len(parts) > 2 else "cifar10"
        image_shape = DATASET_SHAPES.get(dataset, (3, 32, 32))

        model = DCGAN(
            image_shape=image_shape,
            latent_dim=100,
            generator_features=64,
            discriminator_features=64,
        ).to(device)

        return DCGANTrainer(
            model=model,
            batch_size=batch_size,
            lr_d=0.0002,
            lr_g=0.0002,
            beta1=0.5,
            beta2=0.999,
            precision="fp32",
        )

    # Trường hợp 2: Đường dẫn file cấu hình YAML
    cfg_path = Path(config)
    if not cfg_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        cand = repo_root / cfg_path
        if cand.exists():
            cfg_path = cand

    if not cfg_path.exists():
        raise FileNotFoundError(f"[models/gan] Không tìm thấy file cấu hình: {config}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Đọc các thông số từ file YAML
    ds = cfg.get("dataset", "cifar10")
    dataset_name = ds.get("name", "cifar10") if isinstance(ds, dict) else str(ds)
    shape_val = cfg.get("image_shape")
    if shape_val is None and isinstance(ds, dict):
        shape_val = ds.get("image_shape")
    image_shape = tuple(shape_val or DATASET_SHAPES.get(dataset_name, (3, 32, 32)))
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    latent_dim = model_cfg.get("latent_dim", 100)
    g_feat = model_cfg.get("generator_features", 64)
    d_feat = model_cfg.get("discriminator_features", 64)
    d_steps = model_cfg.get("d_steps", 1)

    batch_size = train_cfg.get("batch_size", 64)
    lr_d = train_cfg.get("lr_d", train_cfg.get("lr", 0.0002))
    lr_g = train_cfg.get("lr_g", train_cfg.get("lr", 0.0002))
    beta1 = train_cfg.get("beta1", 0.5)
    beta2 = train_cfg.get("beta2", 0.999)
    precision = train_cfg.get("precision", "fp32")
    label_smoothing = train_cfg.get("label_smoothing", 0.9)

    model = DCGAN(
        image_shape=image_shape,
        latent_dim=latent_dim,
        generator_features=g_feat,
        discriminator_features=d_feat,
        d_steps=d_steps,
    ).to(device)

    return DCGANTrainer(
        model=model,
        batch_size=batch_size,
        lr_d=lr_d,
        lr_g=lr_g,
        beta1=beta1,
        beta2=beta2,
        precision=precision,
        label_smoothing=label_smoothing,
    )


__all__ = [
    "DCGAN",
    "DCGANTrainer",
    "Generator",
    "Discriminator",
    "weights_init",
    "load_model",
    "build_trainer",
]
