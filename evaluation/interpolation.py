"""Latent Space Interpolation utilities cho VAE, DCGAN và DDPM.

Triển khai quy chuẩn đặc tả trong evaluation/interpolation_spec.md (TASK_ASSIGNMENT.md mục 3.2 & 4.2):
- LERP (Linear Interpolation) và SLERP (Spherical Linear Interpolation).
- Chuỗi chuyển tiếp chuẩn 10 bước từ cặp seed cố định (mặc định seed1=42, seed2=100).
- Hỗ trợ giải mã cho cả VAE (decoder / decode) và DCGAN (generator), kèm nội suy giữa μ của 2 ảnh test thật cho VAE.
- Xuất lưới ảnh chuẩn bằng save_grid từ scripts/render_samples.py và ghi rõ metadata chọn hai đầu mút.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Literal

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_samples import save_grid  # noqa: E402


def linear_interpolate(z1: torch.Tensor, z2: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """Nội suy tuyến tính (LERP) giữa 2 vector hoặc 2 batch vector latent.

    Args:
        z1: Điểm bắt đầu (D,) hoặc (B, D) hoặc (C, H, W).
        z2: Điểm kết thúc cùng shape với z1.
        steps: Số bước nội suy (mặc định 10).

    Returns:
        Tensor có shape (steps, *z1.shape).
    """
    alphas = torch.linspace(0.0, 1.0, steps, device=z1.device, dtype=z1.dtype)
    while alphas.ndim < z1.ndim + 1:
        alphas = alphas.unsqueeze(-1)
    return z1.unsqueeze(0) * (1.0 - alphas) + z2.unsqueeze(0) * alphas


def slerp(z1: torch.Tensor, z2: torch.Tensor, steps: int = 10, eps: float = 1e-7) -> torch.Tensor:
    """Nội suy cầu (Spherical Linear Interpolation - SLERP) bảo toàn chuẩn góc trên mặt cầu nhiều chiều.

    Args:
        z1: Điểm bắt đầu (D,), (1, D) hoặc tensor nhiễu DDPM (C, H, W).
        z2: Điểm kết thúc cùng shape với z1.
        steps: Số bước nội suy (mặc định 10).
        eps: Ngưỡng số học tránh chia cho 0 khi 2 vector gần song song.

    Returns:
        Tensor có shape (steps, *z1.shape).
    """
    orig_shape = z1.shape
    v1 = z1.reshape(-1).float()
    v2 = z2.reshape(-1).float()

    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)

    u1 = v1 / (norm1 + eps)
    u2 = v2 / (norm2 + eps)

    dot = torch.clamp(torch.dot(u1, u2), -1.0, 1.0)
    omega = torch.acos(dot)

    alphas = torch.linspace(0.0, 1.0, steps, device=z1.device)
    zs = []

    # Nếu góc quá nhỏ (gần song song hoặc ngược hướng suy biến), lùi về tuyến tính LERP
    if omega.abs() < eps or torch.sin(omega).abs() < eps:
        for a in alphas:
            z = v1 * (1.0 - a) + v2 * a
            zs.append(z.to(z1.dtype).reshape(orig_shape))
        return torch.stack(zs)

    sin_omega = torch.sin(omega)
    for a in alphas:
        dir_interp = (torch.sin((1.0 - a) * omega) / sin_omega) * u1 + (torch.sin(a * omega) / sin_omega) * u2
        norm_interp = norm1 * (1.0 - a) + norm2 * a
        z = dir_interp * norm_interp
        zs.append(z.to(z1.dtype).reshape(orig_shape))

    return torch.stack(zs)


def _decode_latents_to_01(model, zs: torch.Tensor) -> torch.Tensor:
    """Giải mã batch vector ẩn `zs` qua VAE decoder hoặc DCGAN generator về miền [0, 1]."""
    with torch.no_grad():
        if hasattr(model, "decoder"):
            out = model.decoder(zs)
        elif hasattr(model, "decode"):
            out = model.decode(zs)
        elif hasattr(model, "generator"):
            out = model.generator(zs)
        else:
            raise AttributeError("Mô hình không có decoder, decode(z) hoặc generator(z).")

    if out.min() < 0:
        return ((out + 1.0) / 2.0).clamp(0.0, 1.0)
    return out.clamp(0.0, 1.0)


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
    elif method == "lerp":
        zs = linear_interpolate(z1, z2, steps=steps)
    else:
        raise ValueError(f"Phương pháp nội suy không hợp lệ: {method!r} (chọn 'slerp' hoặc 'lerp').")

    if zs.ndim == 3 and zs.size(1) == 1:
        zs = zs.squeeze(1)

    return _decode_latents_to_01(model, zs)


def interpolate_vae_images(
    model,
    img1: torch.Tensor,
    img2: torch.Tensor,
    steps: int = 10,
    method: Literal["slerp", "lerp"] = "slerp",
) -> torch.Tensor:
    """Nội suy giữa 2 ảnh thật thông qua vector trung bình mu của Encoder.

    img1, img2: Tensor ảnh đầu vào (1, C, H, W) hoặc (C, H, W) trong [0, 1] hoặc [-1, 1].
    """
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    if img2.ndim == 3:
        img2 = img2.unsqueeze(0)

    img1_enc = img1 * 2.0 - 1.0 if img1.min() >= 0 else img1
    img2_enc = img2 * 2.0 - 1.0 if img2.min() >= 0 else img2

    with torch.no_grad():
        mu1, _ = model.encode(img1_enc)
        mu2, _ = model.encode(img2_enc)

    return interpolate_latent_sequence(model, mu1, mu2, steps=steps, method=method)


def _write_endpoint_metadata(output_path: Path, metadata: dict) -> Path:
    """Ghi rõ cách chọn hai đầu mút bên cạnh file ảnh nội suy (mục 3.2 TASK_ASSIGNMENT.md)."""
    meta_path = output_path.parent / "interpolation_endpoints.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


def render_prior_interpolation_grid(
    model,
    output_path: str | Path,
    latent_dim: int | None = None,
    seed1: int = 42,
    seed2: int = 100,
    steps: int = 10,
) -> Path:
    """Tạo lưới nội suy prior z (2 hàng: SLERP và LERP, 10 bước) dùng chung cho DCGAN và VAE."""
    dev = model.device
    model.eval()
    dim = int(latent_dim or getattr(model, "latent_dim", 100))

    g1 = torch.Generator(device=dev).manual_seed(seed1)
    g2 = torch.Generator(device=dev).manual_seed(seed2)
    z1 = torch.randn(dim, device=dev, generator=g1)
    z2 = torch.randn(dim, device=dev, generator=g2)

    seq_slerp = interpolate_latent_sequence(model, z1, z2, steps=steps, method="slerp")
    seq_lerp = interpolate_latent_sequence(model, z1, z2, steps=steps, method="lerp")
    all_images = torch.cat([seq_slerp, seq_lerp], dim=0)

    out = save_grid(all_images, output_path, nrow=steps)
    _write_endpoint_metadata(
        out,
        {
            "model": getattr(model, "name", "unknown"),
            "steps": steps,
            "latent_dim": dim,
            "prior_endpoints": {"distribution": "N(0, I)", "seed1": seed1, "seed2": seed2},
            "rows": ["row_1: prior_slerp", "row_2: prior_lerp"],
        },
    )
    return out


def render_vae_interpolation_grid(
    model,
    output_path: str | Path,
    test_images: tuple[torch.Tensor, torch.Tensor] | None = None,
    test_endpoint_names: tuple[str, str] | None = None,
    latent_dim: int = 128,
    seed1: int = 42,
    seed2: int = 100,
    steps: int = 10,
) -> Path:
    """Tạo lưới ảnh nội suy hoàn chỉnh cho VAE theo mục 3.2 & 4.2 của TASK_ASSIGNMENT.md.

    Gồm:
    - Hàng 1: SLERP prior z1 -> z2 (random Gaussian từ seed1, seed2).
    - Hàng 2: LERP prior z1 -> z2 (random Gaussian từ seed1, seed2).
    - Hàng 3 (nếu có ảnh test): SLERP giữa mu(x1) -> mu(x2) của 2 ảnh test thật.
    - Hàng 4 (nếu có ảnh test): LERP giữa mu(x1) -> mu(x2) của 2 ảnh test thật.
    """
    dev = model.device
    model.eval()

    g1 = torch.Generator(device=dev).manual_seed(seed1)
    g2 = torch.Generator(device=dev).manual_seed(seed2)
    z1 = torch.randn(latent_dim, device=dev, generator=g1)
    z2 = torch.randn(latent_dim, device=dev, generator=g2)

    rows = [
        interpolate_latent_sequence(model, z1, z2, steps=steps, method="slerp"),
        interpolate_latent_sequence(model, z1, z2, steps=steps, method="lerp"),
    ]
    row_labels = ["row_1: prior_slerp (z1 -> z2)", "row_2: prior_lerp (z1 -> z2)"]

    if test_images is not None:
        t1, t2 = test_images
        t1 = t1.to(dev)
        t2 = t2.to(dev)
        rows.append(interpolate_vae_images(model, t1, t2, steps=steps, method="slerp"))
        rows.append(interpolate_vae_images(model, t1, t2, steps=steps, method="lerp"))
        row_labels.extend(
            [
                "row_3: test_mu_slerp (Encoder(x1).mu -> Encoder(x2).mu)",
                "row_4: test_mu_lerp (Encoder(x1).mu -> Encoder(x2).mu)",
            ]
        )

    all_images = torch.cat(rows, dim=0)
    out = save_grid(all_images, output_path, nrow=steps)

    endpoint_info: dict = {
        "model": getattr(model, "name", "vae"),
        "steps": steps,
        "latent_dim": latent_dim,
        "prior_endpoints": {"distribution": "N(0, I)", "seed1": seed1, "seed2": seed2},
        "rows": row_labels,
    }
    if test_images is not None:
        endpoint_info["test_reconstruction_endpoints"] = {
            "encoding": "deterministic posterior mean mu = Encoder(x)[0]",
            "files": list(test_endpoint_names) if test_endpoint_names else ["test_img_1", "test_img_2"],
        }
    _write_endpoint_metadata(out, endpoint_info)
    return out
