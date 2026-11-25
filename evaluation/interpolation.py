"""Latent Space Interpolation utilities cho VAE, DCGAN và DDPM.

Triển khai quy chuẩn đặc tả trong evaluation/interpolation_spec.md:
- LERP (Linear Interpolation) và SLERP (Spherical Linear Interpolation).
- Chuỗi chuyển tiếp chuẩn 10 bước.
- Hỗ trợ nội suy prior z và nội suy giữa 2 ảnh test thật qua vector mu (dành cho VAE).
- Xuất lưới ảnh chuẩn bằng save_grid từ scripts/render_samples.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_samples import save_grid


def linear_interpolate(z1: torch.Tensor, z2: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """Nội suy tuyến tính (LERP) giữa 2 vector hoặc 2 batch vector latent.
    
    Args:
        z1: Điểm bắt đầu (D,) hoặc (B, D).
        z2: Điểm kết thúc (D,) hoặc (B, D).
        steps: Số bước nội suy (mặc định 10).
        
    Returns:
        Tensor có shape (steps, D) hoặc (steps, B, D).
    """
    alphas = torch.linspace(0.0, 1.0, steps, device=z1.device, dtype=z1.dtype)
    while alphas.ndim < z1.ndim + 1:
        alphas = alphas.unsqueeze(-1)
    # z1.unsqueeze(0): (1, ...), alphas: (steps, 1, ...)
    return z1.unsqueeze(0) * (1.0 - alphas) + z2.unsqueeze(0) * alphas


def slerp(z1: torch.Tensor, z2: torch.Tensor, steps: int = 10, eps: float = 1e-7) -> torch.Tensor:
    """Nội suy cầu (Spherical Linear Interpolation - SLERP) bảo toàn chuẩn vector.
    
    Args:
        z1: Điểm bắt đầu (D,) hoặc (1, D).
        z2: Điểm kết thúc (D,) hoặc (1, D).
        steps: Số bước nội suy (mặc định 10).
        eps: Ngưỡng số học tránh chia cho 0.
        
    Returns:
        Tensor có shape (steps, D) hoặc (steps, 1, D).
    """
    orig_shape = z1.shape
    v1 = z1.view(-1).float()
    v2 = z2.view(-1).float()

    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)

    u1 = v1 / (norm1 + eps)
    u2 = v2 / (norm2 + eps)

    dot = torch.clamp(torch.dot(u1, u2), -1.0, 1.0)
    omega = torch.acos(dot)

    alphas = torch.linspace(0.0, 1.0, steps, device=z1.device)
    zs = []

    # Nếu góc quá nhỏ (gần song song), lùi về tuyến tính LERP
    if omega.abs() < eps or torch.sin(omega).abs() < eps:
        for a in alphas:
            z = v1 * (1.0 - a) + v2 * a
            zs.append(z.view(orig_shape))
        return torch.stack(zs)

    sin_omega = torch.sin(omega)
    for a in alphas:
        dir_interp = (torch.sin((1.0 - a) * omega) / sin_omega) * u1 + (torch.sin(a * omega) / sin_omega) * u2
        norm_interp = norm1 * (1.0 - a) + norm2 * a
        z = dir_interp * norm_interp
        zs.append(z.to(z1.dtype).view(orig_shape))

    return torch.stack(zs)


def interpolate_latent_sequence(
    model,
    z1: torch.Tensor,
    z2: torch.Tensor,
    steps: int = 10,
    method: Literal["slerp", "lerp"] = "slerp",
) -> torch.Tensor:
    """Nội suy chuỗi z và giải mã thành tensor ảnh (steps, C, H, W) trong [0, 1]."""
    if method == "slerp":
        zs = slerp(z1, z2, steps=steps)
    else:
        zs = linear_interpolate(z1, z2, steps=steps)

    # Đảm bảo shape (steps, latent_dim)
    if zs.ndim == 3 and zs.size(1) == 1:
        zs = zs.squeeze(1)

    with torch.no_grad():
        if hasattr(model, "decoder"):
            recon = model.decoder(zs)
        elif hasattr(model, "decode"):
            recon = model.decode(zs)
        else:
            raise AttributeError("Mô hình không có decoder hoặc decode(z).")

    # Đưa về [0, 1]
    if recon.min() < 0:
        recon = ((recon + 1.0) / 2.0).clamp(0.0, 1.0)
    else:
        recon = recon.clamp(0.0, 1.0)

    return recon


def interpolate_vae_images(
    model,
    img1: torch.Tensor,
    img2: torch.Tensor,
    steps: int = 10,
    method: Literal["slerp", "lerp"] = "slerp",
) -> torch.Tensor:
    """Nội suy giữa 2 ảnh thật thông qua vector trung bình mu của Encoder.
    
    img1, img2: Tensor ảnh đầu vào (1, C, H, W) hoặc (C, H, W).
    """
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    if img2.ndim == 3:
        img2 = img2.unsqueeze(0)

    # Nếu ảnh ở [0, 1], chuyển sang [-1, 1] cho Encoder
    if img1.min() >= 0:
        img1_enc = img1 * 2.0 - 1.0
    else:
        img1_enc = img1

    if img2.min() >= 0:
        img2_enc = img2 * 2.0 - 1.0
    else:
        img2_enc = img2

    with torch.no_grad():
        mu1, _ = model.encode(img1_enc)
        mu2, _ = model.encode(img2_enc)

    return interpolate_latent_sequence(model, mu1, mu2, steps=steps, method=method)


def render_vae_interpolation_grid(
    model,
    output_path: str | Path,
    test_images: tuple[torch.Tensor, torch.Tensor] | None = None,
    latent_dim: int = 128,
    seed1: int = 42,
    seed2: int = 100,
    steps: int = 10,
) -> Path:
    """Tạo lưới ảnh nội suy hoàn chỉnh theo mục 3.1 & 4.2 của TASK_ASSIGNMENT.md.
    
    Gồm:
    - Hàng 1: SLERP prior z1 -> z2 (random Gaussian).
    - Hàng 2: LERP prior z1 -> z2 (random Gaussian).
    - Hàng 3 (tùy chọn nếu có ảnh test): SLERP giữa 2 ảnh test thật.
    - Hàng 4 (tùy chọn nếu có ảnh test): LERP giữa 2 ảnh test thật.
    """
    dev = model.device
    model.eval()

    # Tạo vector z1 và z2 cố định từ 2 seed
    g1 = torch.Generator(device=dev).manual_seed(seed1)
    g2 = torch.Generator(device=dev).manual_seed(seed2)
    z1 = torch.randn(latent_dim, device=dev, generator=g1)
    z2 = torch.randn(latent_dim, device=dev, generator=g2)

    rows = []
    # 1. SLERP prior z
    seq_slerp_z = interpolate_latent_sequence(model, z1, z2, steps=steps, method="slerp")
    rows.append(seq_slerp_z)

    # 2. LERP prior z
    seq_lerp_z = interpolate_latent_sequence(model, z1, z2, steps=steps, method="lerp")
    rows.append(seq_lerp_z)

    # 3. Nếu có ảnh test thật, nội suy giữa mu1 và mu2
    if test_images is not None:
        t1, t2 = test_images
        t1 = t1.to(dev)
        t2 = t2.to(dev)
        seq_slerp_img = interpolate_vae_images(model, t1, t2, steps=steps, method="slerp")
        seq_lerp_img = interpolate_vae_images(model, t1, t2, steps=steps, method="lerp")
        rows.append(seq_slerp_img)
        rows.append(seq_lerp_img)

    all_images = torch.cat(rows, dim=0)  # (N_rows * 10, C, H, W)
    out = save_grid(all_images, output_path, nrow=steps)
    return out
