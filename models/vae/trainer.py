"""Trainer cho mô hình VAE phục vụ đo hiệu chuẩn phần cứng và huấn luyện.

Kế thừa Trainer từ models/interface.py:
- batch_size, image_shape, precision, num_gpus
- train_step(images: torch.Tensor) -> None
"""

from __future__ import annotations

import torch
from torch import optim

from models.interface import Trainer
from models.vae.losses import vae_loss
from models.vae.model import VAE


class VAETrainer(Trainer):
    """Đối tượng huấn luyện VAE tuân thủ giao thức Trainer của hệ thống benchmark."""

    def __init__(
        self,
        model: VAE,
        optimizer: optim.Optimizer | None = None,
        batch_size: int = 64,
        image_shape: tuple[int, int, int] | None = None,
        precision: str = "fp32",
        num_gpus: int = 1,
        learning_rate: float = 5e-4,
        beta: float = 1.0,
        loss_type: str = "mse",
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.batch_size = int(batch_size)
        self.image_shape = tuple(image_shape or model.image_shape)
        self.precision = precision
        self.num_gpus = int(num_gpus)
        self.beta = float(beta)
        self.loss_type = loss_type

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(
                list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
                lr=learning_rate,
                betas=(0.9, 0.999),
            )

        self.use_amp = (self.precision == "fp16" and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    def train_step(self, images: torch.Tensor) -> None:
        """Thực hiện đúng một iteration train hoàn chỉnh (forward, backward, optimizer step)."""
        self.model.train()
        images = images.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        if self.use_amp:
            with torch.amp.autocast("cuda"):
                recon, mu, logvar = self.model(images)
                loss, _, _ = vae_loss(recon, images, mu, logvar, beta=self.beta, loss_type=self.loss_type)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            recon, mu, logvar = self.model(images)
            loss, _, _ = vae_loss(recon, images, mu, logvar, beta=self.beta, loss_type=self.loss_type)
            loss.backward()
            self.optimizer.step()
