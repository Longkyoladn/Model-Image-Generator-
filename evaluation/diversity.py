"""Generated-image diversity evaluation utilities.

Cung cấp các hàm và lớp đo độ đa dạng của tập ảnh sinh theo TASK_ASSIGNMENT.md (mục 2.6):
- LPIPS diversity: trung bình khoảng cách tri giác LPIPS (AlexNet) giữa 2000 cặp ảnh ngẫu nhiên.
- PRDC: Precision, Recall, Density, Coverage (k=5) qua thư viện prdc trên không gian đặc trưng Inception.
- Feature Diversity: phương sai trong không gian đặc trưng.
- MS-SSIM: độ tương đồng cấu trúc đa tỉ lệ (càng thấp càng đa dạng).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import lpips
except ImportError:
    lpips = None

try:
    import prdc
except ImportError:
    prdc = None

try:
    from pytorch_msssim import ms_ssim
except ImportError:
    ms_ssim = None


def calculate_feature_diversity(features: torch.Tensor | np.ndarray) -> float:
    """Tính độ phân tán (Variance) của features trên không gian đã chuẩn hóa L2."""
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    features = features.float()
    features = F.normalize(features, p=2, dim=1)
    variance = torch.var(features, dim=0).mean().item()
    return float(variance)


def calculate_prdc(
    real_features: np.ndarray | torch.Tensor,
    fake_features: np.ndarray | torch.Tensor,
    nearest_k: int = 5,
) -> dict[str, float]:
    """Tính Precision, Recall, Density, Coverage theo TASK_ASSIGNMENT mục 2.6 (k=5)."""
    if prdc is None:
        raise ImportError("Chưa cài đặt thư viện prdc. Chạy: pip install prdc")

    if isinstance(real_features, torch.Tensor):
        real_features = real_features.detach().cpu().numpy()
    if isinstance(fake_features, torch.Tensor):
        fake_features = fake_features.detach().cpu().numpy()

    real_features = real_features.astype(np.float32)
    fake_features = fake_features.astype(np.float32)

    res = prdc.compute_prdc(
        real_features=real_features,
        fake_features=fake_features,
        nearest_k=nearest_k,
    )
    return {
        "precision": float(res["precision"]),
        "recall": float(res["recall"]),
        "density": float(res["density"]),
        "coverage": float(res["coverage"]),
    }


def _load_image_tensor(path: Path | str) -> torch.Tensor:
    """Đọc ảnh từ file và trả về tensor float (3, H, W) trong [0, 1]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def calculate_lpips_diversity(
    images: torch.Tensor | list[Path | str] | str | Path,
    num_pairs: int = 2000,
    net: str = "alex",
    device: str | torch.device = "cpu",
    seed: int = 42,
    batch_size: int = 64,
) -> float:
    """Tính trung bình LPIPS giữa `num_pairs` cặp ảnh ngẫu nhiên.
    
    `images` có thể là:
    - Tensor float (N, C, H, W) trong [0, 1] hoặc [-1, 1].
    - Thư mục chứa file ảnh PNG/JPG.
    - Danh sách đường dẫn file ảnh.
    """
    if lpips is None:
        raise ImportError("Chưa cài đặt thư viện lpips. Chạy: pip install lpips")

    rng = random.Random(seed)
    dev = torch.device(device)
    loss_fn = lpips.LPIPS(net=net).to(dev)
    loss_fn.eval()

    # Thu thập ảnh
    if isinstance(images, (str, Path)):
        p = Path(images)
        if p.is_dir():
            files = sorted(list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg")))
        else:
            files = [p]
        n_total = len(files)
        if n_total < 2:
            return 0.0

        num_pairs = min(num_pairs, n_total * 2)
        pairs = [(rng.choice(files), rng.choice(files)) for _ in range(num_pairs)]

        distances = []
        with torch.no_grad():
            for i in range(0, num_pairs, batch_size):
                batch_pairs = pairs[i : i + batch_size]
                t1 = torch.stack([_load_image_tensor(f1) for f1, _ in batch_pairs]).to(dev)
                t2 = torch.stack([_load_image_tensor(f2) for _, f2 in batch_pairs]).to(dev)
                # Đổi về [-1, 1] cho LPIPS
                t1 = t1 * 2.0 - 1.0
                t2 = t2 * 2.0 - 1.0
                d = loss_fn(t1, t2).flatten()
                distances.extend(d.cpu().tolist())
        return float(np.mean(distances)) if distances else 0.0

    elif isinstance(images, torch.Tensor):
        n = images.shape[0]
        if n < 2:
            return 0.0

        # Đảm bảo 3 kênh RGB
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        # Chuyển về [-1, 1]
        if images.min() >= 0:
            images = images * 2.0 - 1.0

        idx1 = [rng.randint(0, n - 1) for _ in range(num_pairs)]
        idx2 = [rng.randint(0, n - 1) for _ in range(num_pairs)]

        distances = []
        with torch.no_grad():
            for i in range(0, num_pairs, batch_size):
                b1 = images[idx1[i : i + batch_size]].to(dev)
                b2 = images[idx2[i : i + batch_size]].to(dev)
                d = loss_fn(b1, b2).flatten()
                distances.extend(d.cpu().tolist())
        return float(np.mean(distances)) if distances else 0.0

    elif isinstance(images, list):
        n_total = len(images)
        if n_total < 2:
            return 0.0
        num_pairs = min(num_pairs, n_total * 2)
        pairs = [(rng.choice(images), rng.choice(images)) for _ in range(num_pairs)]
        distances = []
        with torch.no_grad():
            for i in range(0, num_pairs, batch_size):
                batch_pairs = pairs[i : i + batch_size]
                t1 = torch.stack([_load_image_tensor(f1) for f1, _ in batch_pairs]).to(dev)
                t2 = torch.stack([_load_image_tensor(f2) for _, f2 in batch_pairs]).to(dev)
                t1 = t1 * 2.0 - 1.0
                t2 = t2 * 2.0 - 1.0
                d = loss_fn(t1, t2).flatten()
                distances.extend(d.cpu().tolist())
        return float(np.mean(distances)) if distances else 0.0

    raise TypeError(f"Không hỗ trợ kiểu images: {type(images)}")


def calculate_ms_ssim(images: torch.Tensor, num_pairs: int = 100, seed: int = 42) -> float:
    """Tính MS-SSIM trung bình giữa các cặp ảnh ngẫu nhiên."""
    if ms_ssim is None:
        raise ImportError("Chưa cài pytorch-msssim. Chạy: pip install pytorch-msssim")

    n = images.shape[0]
    if n < 2:
        return 1.0

    num_pairs = min(num_pairs, n // 2)
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=rng)
    img1 = images[indices[:num_pairs]]
    img2 = images[indices[num_pairs : num_pairs * 2]]

    if img1.min() < 0:
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2

    with torch.no_grad():
        if img1.shape[-1] < 256:
            img1 = F.interpolate(img1, size=(256, 256), mode="bilinear", align_corners=False)
            img2 = F.interpolate(img2, size=(256, 256), mode="bilinear", align_corners=False)
        score = ms_ssim(img1, img2, data_range=1.0, size_average=True)

    return float(score.item())


class DiversityCalculator:
    """Lớp tiện ích tổng hợp các metric đa dạng."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def calculate_feature_diversity(self, features: torch.Tensor | np.ndarray) -> float:
        return calculate_feature_diversity(features)

    def calculate_lpips_diversity(
        self, images: torch.Tensor | list[Path | str] | str | Path, num_pairs: int = 2000, net: str = "alex"
    ) -> float:
        return calculate_lpips_diversity(images, num_pairs=num_pairs, net=net, device=self.device)

    def calculate_prdc(
        self, real_features: np.ndarray | torch.Tensor, fake_features: np.ndarray | torch.Tensor, nearest_k: int = 5
    ) -> dict[str, float]:
        return calculate_prdc(real_features, fake_features, nearest_k=nearest_k)

    def calculate_ms_ssim(self, images: torch.Tensor, num_pairs: int = 100) -> float:
        return calculate_ms_ssim(images, num_pairs=num_pairs)
