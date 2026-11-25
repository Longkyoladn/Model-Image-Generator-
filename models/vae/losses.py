"""Hàm mất mát cho Variational Autoencoder (VAE).

Bao gồm:
- Reconstruction Loss (MSE hoặc L1).
- KL Divergence giữa phân phối hậu nghiệm q(z|x) và phân phối tiên nghiệm p(z) ~ N(0, I).
- Tổng mất mát = Reconstruction Loss + beta * KL Divergence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    loss_type: str = "mse",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tính toán hàm mất mát của VAE.
    
    Args:
        recon_x: Ảnh tái tạo từ Decoder, shape (B, C, H, W) trong [-1, 1].
        x: Ảnh gốc đầu vào, shape (B, C, H, W) trong [-1, 1].
        mu: Vector trung bình không gian ẩn, shape (B, latent_dim).
        logvar: Log-variance không gian ẩn, shape (B, latent_dim).
        beta: Trọng số cân bằng KL divergence (mặc định 1.0).
        loss_type: Loại loss tái tạo ("mse" hoặc "l1").
        
    Returns:
        total_loss, recon_loss, kl_loss (tất cả là scalar tensor).
    """
    b = x.size(0)
    if loss_type == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / b
    elif loss_type == "l1":
        recon_loss = F.l1_loss(recon_x, x, reduction="sum") / b
    else:
        raise ValueError(f"loss_type không hợp lệ: {loss_type}")

    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp()) / b

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
