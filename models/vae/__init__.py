"""Package VAE cung cấp load_model và build_trainer theo giao thức models/interface.py."""

from __future__ import annotations

from pathlib import Path
import yaml
import torch

from models.interface import GenerativeModel, Trainer
from models.vae.model import VAE, Decoder, Encoder
from models.vae.trainer import VAETrainer

__all__ = ["VAE", "Encoder", "Decoder", "VAETrainer", "load_model", "build_trainer"]

DATASET_SHAPES = {
    "cifar10": (3, 32, 32),
    "fashion_mnist": (1, 32, 32),
    "celeba": (3, 64, 64),
}


def _validate_image_shape(raw_shape) -> tuple[int, int, int]:
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 3:
        raise ValueError(f"image_shape không hợp lệ trong checkpoint VAE: {raw_shape!r}")
    c, h, w = (int(raw_shape[0]), int(raw_shape[1]), int(raw_shape[2]))
    if c not in (1, 3) or h != w or h not in (32, 64):
        raise ValueError(f"VAE chỉ hỗ trợ image_shape (1|3, 32, 32) hoặc (1|3, 64, 64), nhận được {(c, h, w)}")
    return (c, h, w)


def _extract_state_dict(payload: dict, ckpt_path: Path) -> dict:
    if "model_state_dict" in payload:
        sd = payload["model_state_dict"]
    elif "state_dict" in payload:
        sd = payload["state_dict"]
    elif ("encoder" in payload and "decoder" in payload) or (
        any(isinstance(k, str) and k.startswith("encoder.") for k in payload)
        and any(isinstance(k, str) and k.startswith("decoder.") for k in payload)
    ):
        sd = payload
    else:
        raise ValueError(
            f"Checkpoint VAE không đúng schema ({ckpt_path}): thiếu 'model_state_dict', "
            "'state_dict', hoặc cặp khóa 'encoder'/'decoder'."
        )
    if not isinstance(sd, dict) or not sd:
        raise ValueError(f"State dict trong checkpoint VAE không hợp lệ hoặc rỗng ({ckpt_path}).")
    return sd


def load_model(checkpoint: str, device: str | torch.device = "cpu") -> GenerativeModel:
    """Tải checkpoint mô hình VAE tuân thủ interface GenerativeModel.

    Chỉ khi `checkpoint == "auto"` mới khởi tạo mô hình trọng số ngẫu nhiên phục vụ
    kiểm tra contract. Mọi đường dẫn khác phải tồn tại và đúng schema checkpoint VAE.
    """
    dev = torch.device(device)
    if str(checkpoint) == "auto":
        model = VAE(image_shape=(3, 32, 32), latent_dim=128, base_channels=32)
        return model.to(dev).eval()

    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint VAE: {checkpoint}")

    payload = torch.load(ckpt_path, map_location=dev, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint VAE phải là dict, nhận được {type(payload).__name__} ({ckpt_path}).")

    if "image_shape" not in payload:
        raise ValueError(f"Checkpoint VAE thiếu trường bắt buộc 'image_shape' ({ckpt_path}).")
    image_shape = _validate_image_shape(payload["image_shape"])

    if "latent_dim" not in payload or "base_channels" not in payload:
        raise ValueError(f"Checkpoint VAE thiếu 'latent_dim' hoặc 'base_channels' ({ckpt_path}).")
    latent_dim = int(payload["latent_dim"])
    base_channels = int(payload["base_channels"])
    if latent_dim <= 0 or base_channels <= 0:
        raise ValueError(f"latent_dim ({latent_dim}) và base_channels ({base_channels}) phải > 0.")

    sd = _extract_state_dict(payload, ckpt_path)
    model = VAE(image_shape=image_shape, latent_dim=latent_dim, base_channels=base_channels)
    model.load_state_dict(sd)
    return model.to(dev).eval()


def build_trainer(config: str, device: str | torch.device = "cpu") -> Trainer:
    """Khởi tạo VAETrainer từ file cấu hình YAML hoặc spec chuỗi dạng vae:<batch>:<dataset>."""
    dev = torch.device(device)

    # Dạng chuỗi vắn tắt cho smoke test: vae:<batch>:<dataset>
    if str(config).startswith("vae:"):
        parts = str(config).split(":")
        batch_size = int(parts[1]) if len(parts) > 1 else 64
        dataset_name = parts[2] if len(parts) > 2 else "cifar10"
        image_shape = DATASET_SHAPES.get(dataset_name.lower(), (3, 32, 32))

        model = VAE(image_shape=image_shape, latent_dim=128, base_channels=32)
        return VAETrainer(
            model=model,
            batch_size=batch_size,
            image_shape=image_shape,
            precision="fp32",
            device=dev,
        )

    # Đọc từ file YAML
    cfg_path = Path(config)
    if not cfg_path.is_file():
        cfg_path = Path(__file__).resolve().parents[2] / config
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình VAE: {config}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    ds_name = cfg.get("dataset", "cifar10")
    image_shape = tuple(cfg.get("image_shape", DATASET_SHAPES.get(ds_name, (3, 32, 32))))
    latent_dim = int(cfg.get("latent_dim", 128))
    base_channels = int(cfg.get("base_channels", 32))

    train_cfg = cfg.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 64))
    lr = float(train_cfg.get("learning_rate", 5e-4))
    beta = float(train_cfg.get("beta", 1.0))
    precision = train_cfg.get("precision", "fp32")
    loss_type = train_cfg.get("loss_type", "mse")

    model = VAE(image_shape=image_shape, latent_dim=latent_dim, base_channels=base_channels)
    return VAETrainer(
        model=model,
        batch_size=batch_size,
        image_shape=image_shape,
        precision=precision,
        learning_rate=lr,
        beta=beta,
        loss_type=loss_type,
        device=dev,
    )
