"""
Discriminator cho kiến trúc DCGAN (Radford et al., 2015).
Phân biệt ảnh thật và ảnh do Generator sinh ra bằng mạng tích chập sâu.
Hỗ trợ kích thước ảnh 32x32 (Fashion-MNIST, CIFAR-10) và 64x64 (CelebA).
"""

from __future__ import annotations

import torch
from torch import nn

from .generator import weights_init


class Discriminator(nn.Module):
    """
    Discriminator mạng tích chập sâu (Deep Convolutional Discriminator).
    Đầu ra là 1 logit chưa qua Sigmoid (dùng với BCEWithLogitsLoss).
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        features: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.features = features

        if image_size == 32:
            # Downsampling cho ảnh 32x32: 32x32 -> 16x16 -> 8x8 -> 4x4 -> 1x1
            self.net = nn.Sequential(
                # Lớp đầu không dùng BatchNorm theo chuẩn DCGAN
                # (B, in_channels, 32, 32) -> (B, features, 16, 16)
                nn.Conv2d(in_channels, features, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # (B, features, 16, 16) -> (B, features * 2, 8, 8)
                nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # (B, features * 2, 8, 8) -> (B, features * 4, 4, 4)
                nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # (B, features * 4, 4, 4) -> (B, 1, 1, 1)
                nn.Conv2d(features * 4, 1, 4, 1, 0, bias=False),
            )
        elif image_size == 64:
            # Downsampling cho ảnh 64x64: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 -> 1x1
            self.net = nn.Sequential(
                # Lớp đầu không dùng BatchNorm
                # (B, in_channels, 64, 64) -> (B, features, 32, 32)
                nn.Conv2d(in_channels, features, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # (B, features, 32, 32) -> (B, features * 2, 16, 16)
                nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # (B, features * 2, 16, 16) -> (B, features * 4, 8, 8)
                nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # (B, features * 4, 8, 8) -> (B, features * 8, 4, 4)
                nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # (B, features * 8, 4, 4) -> (B, 1, 1, 1)
                nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
            )
        else:
            raise ValueError(f"image_size chỉ hỗ trợ 32 hoặc 64, nhận được: {image_size}")

        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Nhận ảnh x: (B, in_channels, H, W).
        Trả về logit chưa qua sigmoid: (B, 1).
        """
        out = self.net(x)
        return out.view(x.size(0), 1)
