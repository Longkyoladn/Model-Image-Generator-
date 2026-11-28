"""Kiến trúc mạng tích chập Variational Autoencoder (Convolutional VAE).

Tuân thủ nghiêm ngặt GenerativeModel từ models/interface.py:
- name = "vae"
- image_shape = (C, H, W)
- sample(n, seed) trả về tensor float (n, C, H, W) trong [0, 1] trên device của mô hình
- sampling_modules() = {"decoder": self.decoder}
- training_modules() = {"encoder": self.encoder, "decoder": self.decoder}
- example_inputs() với batch = 1 cho encoder và decoder
"""

from __future__ import annotations

import torch
from torch import nn

from models.interface import GenerativeModel


class Encoder(nn.Module):
    """Mạng mã hóa trích xuất tham số phân phối tiềm ẩn (mu, logvar)."""

    def __init__(self, image_shape: tuple[int, int, int], latent_dim: int = 128, base_channels: int = 32):
        super().__init__()
        c, h, w = image_shape
        self.image_shape = image_shape
        self.latent_dim = latent_dim

        layers = []
        in_c = c
        curr_c = base_channels

        # Xác định số bước downsampling (cho đến khi spatial size = 4x4)
        # 32x32 -> 3 bước downsample (16, 8, 4)
        # 64x64 -> 4 bước downsample (32, 16, 8, 4)
        num_downsamples = 3 if h == 32 else 4

        for _ in range(num_downsamples):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_c, curr_c, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(curr_c),
                    nn.LeakyReLU(0.2, inplace=False),
                )
            )
            in_c = curr_c
            curr_c *= 2

        self.conv_net = nn.Sequential(*layers)
        self.final_channels = in_c
        self.flat_dim = self.final_channels * 4 * 4

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv_net(x)
        feat_flat = feat.view(feat.size(0), -1)
        mu = self.fc_mu(feat_flat)
        logvar = torch.clamp(self.fc_logvar(feat_flat), -30.0, 20.0)
        return mu, logvar


class Decoder(nn.Module):
    """Mạng giải mã tái tạo ảnh từ vector ẩn z trong [-1, 1]."""

    def __init__(self, image_shape: tuple[int, int, int], latent_dim: int = 128, base_channels: int = 32):
        super().__init__()
        c, h, w = image_shape
        self.image_shape = image_shape
        self.latent_dim = latent_dim

        num_upsamples = 3 if h == 32 else 4
        # Channel bắt đầu từ bottleneck
        start_c = base_channels * (2 ** (num_upsamples - 1))
        self.start_c = start_c
        self.fc = nn.Linear(latent_dim, start_c * 4 * 4)

        layers = []
        curr_c = start_c
        for i in range(num_upsamples - 1):
            next_c = curr_c // 2
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(curr_c, next_c, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(next_c),
                    nn.ReLU(inplace=True),
                )
            )
            curr_c = next_c

        # Lớp giải mã cuối cùng đưa về số kênh gốc và khoảng [-1, 1] qua Tanh
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(curr_c, c, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            )
        )
        self.deconv_net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        feat = self.fc(z)
        feat = feat.view(feat.size(0), self.start_c, 4, 4)
        recon_x = self.deconv_net(feat)
        return recon_x


class VAE(GenerativeModel):
    """Mô hình Variational Autoencoder hoàn chỉnh tuân thủ GenerativeModel."""

    name = "vae"

    def __init__(self, image_shape: tuple[int, int, int] = (3, 32, 32), latent_dim: int = 128, base_channels: int = 32):
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        self.encoder = Encoder(self.image_shape, latent_dim=latent_dim, base_channels=base_channels)
        self.decoder = Decoder(self.image_shape, latent_dim=latent_dim, base_channels=base_channels)

    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict của VAE phải là dict, nhận được {type(state_dict).__name__}.")

        if "encoder" in state_dict and "decoder" in state_dict:
            extra = set(state_dict.keys()) - {"encoder", "decoder"}
            if extra:
                raise ValueError(f"Schema state_dict của VAE có khóa thừa không hợp lệ: {sorted(extra)}")
            self.encoder.load_state_dict(state_dict["encoder"], strict=True)
            self.decoder.load_state_dict(state_dict["decoder"], strict=True)
            return

        enc_dict = {
            k.removeprefix("encoder."): v
            for k, v in state_dict.items()
            if isinstance(k, str) and k.startswith("encoder.")
        }
        dec_dict = {
            k.removeprefix("decoder."): v
            for k, v in state_dict.items()
            if isinstance(k, str) and k.startswith("decoder.")
        }
        unmatched = [
            k for k in state_dict if not (isinstance(k, str) and (k.startswith("encoder.") or k.startswith("decoder.")))
        ]
        if not enc_dict or not dec_dict or unmatched:
            raise ValueError(
                "Schema state_dict của VAE không hợp lệ: cần bộ khóa ('encoder', 'decoder') "
                f"hoặc tiền tố 'encoder.'/'decoder.'. Không khớp: {sorted(map(str, unmatched))[:10]}"
            )
        self.encoder.load_state_dict(enc_dict, strict=True)
        self.decoder.load_state_dict(dec_dict, strict=True)

    def parameters(self):
        seen = {}
        for m in self.training_modules().values():
            for p in m.parameters():
                seen[id(p)] = p
        return list(seen.values())

    def train(self, mode: bool = True) -> "VAE":
        self.encoder.train(mode)
        self.decoder.train(mode)
        return self

    def eval(self) -> "VAE":
        self.encoder.eval()
        self.decoder.eval()
        return self

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Kỹ thuật tái tham số hóa z = mu + eps * sigma."""
        std = torch.exp(0.5 * logvar)
        if generator is not None:
            eps = torch.randn(std.shape, device=std.device, dtype=std.dtype, generator=generator)
        else:
            eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def sample(self, n: int, seed: int | None = None, **kwargs) -> torch.Tensor:
        """Sinh n ảnh mới từ phân phối tiên nghiệm p(z) ~ N(0, I).
        
        Trả về tensor float (n, C, H, W) trong [0, 1] trên device của mô hình.
        """
        gen = self.make_generator(seed)
        z = torch.randn(n, self.latent_dim, device=self.device, generator=gen)
        # decoder trả về [-1, 1], chuyển đổi chuẩn xác sang [0, 1]
        x_recon = self.decoder(z)
        x_01 = ((x_recon + 1.0) / 2.0).clamp(0.0, 1.0)
        return x_01

    def sampling_modules(self) -> dict[str, nn.Module]:
        return {"decoder": self.decoder}

    def training_modules(self) -> dict[str, nn.Module]:
        return {"encoder": self.encoder, "decoder": self.decoder}

    def example_inputs(self) -> dict[str, tuple]:
        """Input batch = 1 cho từng module phục vụ đo MACs/FLOPs."""
        return {
            "encoder": (torch.zeros(1, *self.image_shape, device=self.device),),
            "decoder": (torch.zeros(1, self.latent_dim, device=self.device),),
        }

    def to(self, device: str | torch.device) -> "VAE":
        super().to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        return self
