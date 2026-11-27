"""
DCGAN kế thừa GenerativeModel và DCGANTrainer kế thừa Trainer
Tuân thủ toàn bộ hợp đồng interface tại models/interface.py.
"""

from __future__ import annotations

import torch
from torch import nn

from models.interface import GenerativeModel, Trainer
from .generator import Generator
from .discriminator import Discriminator


class DCGAN(GenerativeModel):
    """
    Mô hình DCGAN kế thừa GenerativeModel dùng chung cho toàn bộ dự án.
    Hỗ trợ sinh ảnh cho cả 3 bộ dữ liệu: Fashion-MNIST (1, 32, 32), CIFAR-10 (3, 32, 32), CelebA (3, 64, 64).
    """

    name: str = "dcgan"

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (3, 32, 32),
        latent_dim: int = 100,
        generator_features: int = 64,
        discriminator_features: int = 64,
        d_steps: int = 1,
    ) -> None:
        super().__init__()
        self.image_shape = tuple(image_shape)
        c, h, w = self.image_shape
        if h != w or h not in (32, 64):
            raise ValueError(f"DCGAN chỉ hỗ trợ kích thước ảnh 32x32 hoặc 64x64, nhận được: {self.image_shape}")

        if d_steps < 1:
            raise ValueError("DCGAN yêu cầu d_steps >= 1.")

        self.latent_dim = latent_dim
        self.generator_features = generator_features
        self.discriminator_features = discriminator_features
        self.d_steps = d_steps

        # Khởi tạo Generator và Discriminator
        self.generator = Generator(
            latent_dim=latent_dim,
            out_channels=c,
            image_size=h,
            features=generator_features,
        )
        self.discriminator = Discriminator(
            in_channels=c,
            image_size=h,
            features=discriminator_features,
        )

    # 1. Module dùng khi sinh ảnh: chỉ gồm Generator
    def sampling_modules(self) -> dict[str, nn.Module]:
        return {"generator": self.generator}

    # 2. Toàn bộ module tham gia train: Generator và Discriminator
    def training_modules(self) -> dict[str, nn.Module]:
        return {
            "generator": self.generator,
            "discriminator": self.discriminator,
        }

    # 3. Số lần forward pass để sinh 1 ảnh (DCGAN chỉ cần 1 lần forward)
    def forward_passes_per_image(self, **kwargs) -> int:
        return 1

    # 4. Chuyển toàn bộ module (cả generator và discriminator) sang device
    def to(self, device: str | torch.device) -> "DCGAN":
        self.generator.to(device)
        self.discriminator.to(device)
        return self

    # 5. Input mẫu batch = 1 cho từng module để đo FLOPs/MACs (mục 2.5 và 4.1)
    def example_inputs(self) -> dict[str, tuple]:
        d = self.device
        z = torch.zeros(1, self.latent_dim, device=d)
        img = torch.zeros(1, *self.image_shape, device=d)
        return {
            "generator": (z,),
            "discriminator": (img,),
        }

    # 6. Sinh ảnh chuẩn: tensor (n, C, H, W) trong [0, 1] trên đúng device
    @torch.no_grad()
    def sample(self, n: int, seed: int | None = None, **kwargs) -> torch.Tensor:
        seed = kwargs.pop("seed", seed)
        gen = self.make_generator(seed)
        z = torch.randn(n, self.latent_dim, generator=gen, device=self.device)

        was_training = self.generator.training
        self.generator.eval()
        try:
            fake_imgs = self.generator(z)
        finally:
            self.generator.train(was_training)

        # Output của Generator dùng Tanh trong [-1, 1], đưa về [0, 1] float
        imgs_01 = (fake_imgs + 1.0) / 2.0
        return torch.clamp(imgs_01, 0.0, 1.0)


class DCGANTrainer(Trainer):
    """
    Trainer cho DCGAN phục vụ đo throughput (calibrate.py) và huấn luyện.
    Thực hiện 1 iteration đầy đủ: cập nhật Discriminator (k bước) và cập nhật Generator (1 bước).
    """

    def __init__(
        self,
        model: DCGAN,
        batch_size: int = 64,
        lr_d: float = 0.0002,
        lr_g: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        precision: str = "fp32",
        label_smoothing: float = 0.9,
    ) -> None:
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"Precision không hợp lệ cho DCGAN: {precision!r}. Chọn fp32 hoặc fp16.")
        if precision == "fp16" and model.device.type != "cuda":
            raise ValueError("DCGAN precision fp16 yêu cầu CUDA. Chọn fp32 khi chạy trên CPU.")

        self.model = model
        self.batch_size = batch_size
        self.image_shape = model.image_shape
        self.precision = precision
        self.num_gpus = 1
        self.label_smoothing = label_smoothing

        self.opt_d = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=lr_d,
            betas=(beta1, beta2),
        )
        self.opt_g = torch.optim.Adam(
            self.model.generator.parameters(),
            lr=lr_g,
            betas=(beta1, beta2),
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.use_amp = precision == "fp16"
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_step(self, images: torch.Tensor) -> None:
        self.step(images)

    def step(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Cập nhật D và G, trả về loss D, G, D trên ảnh thật và D trên ảnh giả."""
        device = images.device
        batch_size = images.size(0)
        real_label = torch.full((batch_size,), self.label_smoothing, device=device)
        fake_label = torch.zeros(batch_size, device=device)
        gen_label = torch.ones(batch_size, device=device)

        for step in range(self.model.d_steps):
            self.opt_d.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=self.use_amp):
                real_pred = self.model.discriminator(images).view(-1)
                loss_d_real = self.criterion(real_pred, real_label)

                z = torch.randn(batch_size, self.model.latent_dim, device=device)
                fake_images = self.model.generator(z)
                fake_pred = self.model.discriminator(fake_images.detach()).view(-1)
                loss_d_fake = self.criterion(fake_pred, fake_label)
                loss_d = loss_d_real + loss_d_fake

            self.scaler.scale(loss_d).backward()
            self.scaler.step(self.opt_d)
            if step + 1 < self.model.d_steps:
                self.scaler.update()

        self.opt_g.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=self.use_amp):
            fake_pred_for_g = self.model.discriminator(fake_images).view(-1)
            loss_g = self.criterion(fake_pred_for_g, gen_label)

        self.scaler.scale(loss_g).backward()
        self.scaler.step(self.opt_g)
        self.scaler.update()

        return loss_d.detach(), loss_g.detach(), loss_d_real.detach(), loss_d_fake.detach()
