"""
Generator cho kiến trúc DCGAN (Radford et al., 2015).
Biến đổi vector không gian ẩn (latent vector) z ~ N(0, I) thành ảnh sinh.
Hỗ trợ kích thước ảnh 32x32 (Fashion-MNIST, CIFAR-10) và 64x64 (CelebA).
"""

from __future__ import annotations

import torch
from torch import nn


def weights_init(m: nn.Module) -> None:
    """Khởi tạo trọng số chuẩn DCGAN: Normal(0, 0.02) cho Conv, Normal(1, 0.02) cho BatchNorm."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    """
    Generator mạng tích chập sâu (Deep Convolutional Generator).
    """

    def __init__(
        self,
        latent_dim: int = 100,
        out_channels: int = 3,
        image_size: int = 32,
        features: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.image_size = image_size
        self.features = features

        if image_size == 32:
            # Upsampling cho ảnh 32x32: 1x1 -> 4x4 -> 8x8 -> 16x16 -> 32x32
            self.net = nn.Sequential(
                # Input z: (B, latent_dim, 1, 1) -> (B, features * 4, 4, 4)
                nn.ConvTranspose2d(latent_dim, features * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(features * 4),
                nn.ReLU(True),
                # (B, features * 4, 4, 4) -> (B, features * 2, 8, 8)
                nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 2),
                nn.ReLU(True),
                # (B, features * 2, 8, 8) -> (B, features, 16, 16)
                nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(True),
                # (B, features, 16, 16) -> (B, out_channels, 32, 32)
                nn.ConvTranspose2d(features, out_channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        elif image_size == 64:
            # Upsampling cho ảnh 64x64: 1x1 -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
            self.net = nn.Sequential(
                # Input z: (B, latent_dim, 1, 1) -> (B, features * 8, 4, 4)
                nn.ConvTranspose2d(latent_dim, features * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(features * 8),
                nn.ReLU(True),
                # (B, features * 8, 4, 4) -> (B, features * 4, 8, 8)
                nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 4),
                nn.ReLU(True),
                # (B, features * 4, 8, 8) -> (B, features * 2, 16, 16)
                nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 2),
                nn.ReLU(True),
                # (B, features * 2, 16, 16) -> (B, features, 32, 32)
                nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(True),
                # (B, features, 32, 32) -> (B, out_channels, 64, 64)
                nn.ConvTranspose2d(features, out_channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        else:
            raise ValueError(f"image_size chỉ hỗ trợ 32 hoặc 64, nhận được: {image_size}")

        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Nhận z có shape (B, latent_dim) hoặc (B, latent_dim, 1, 1).
        Trả về ảnh có shape (B, out_channels, image_size, image_size) trong khoảng [-1, 1].
        """
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)
