"""Generated-image diversity evaluation utilities.

Cung cấp các hàm và lớp đo độ đa dạng của tập ảnh sinh theo TASK_ASSIGNMENT.md (mục 2.6):
- LPIPS diversity: trung bình khoảng cách tri giác LPIPS (AlexNet) giữa `num_pairs` (mặc định 2000) cặp ảnh khác nhau.
- PRDC: Precision, Recall, Density, Coverage (k=5) qua thư viện prdc trên không gian đặc trưng Inception.
- Feature Diversity: phương sai trong không gian đặc trưng.
- MS-SSIM: độ tương đồng cấu trúc đa tỉ lệ (càng thấp càng đa dạng).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

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
    with Image.open(path) as im:
        img = im.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _sample_distinct_pairs(n: int, num_pairs: int, rng: random.Random) -> list[tuple[int, int]]:
    """Chọn đúng `num_pairs` cặp chỉ số (i, j) với i != j.

    Nếu tổng số cặp vô hướng phân biệt C(n, 2) >= num_pairs thì lấy mẫu không hoàn lại
    các cặp duy nhất; ngược lại vẫn bảo toàn `num_pairs` cặp nhưng luôn đảm bảo i != j.
    """
    if n < 2 or num_pairs <= 0:
        return []

    total_unique = n * (n - 1) // 2
    if total_unique >= num_pairs:
        if n <= 500 and num_pairs > total_unique // 2:
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            return rng.sample(all_pairs, num_pairs)

        seen: set[tuple[int, int]] = set()
        pairs: list[tuple[int, int]] = []
        while len(pairs) < num_pairs:
            i = rng.randrange(n)
            j = rng.randrange(n - 1)
            if j >= i:
                j += 1
            pair = (i, j) if i < j else (j, i)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
        return pairs

    pairs = []
    for _ in range(num_pairs):
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        pairs.append((i, j))
    return pairs


def _collect_image_files(images: str | Path | list[Path | str]) -> list[Path]:
    if isinstance(images, (str, Path)):
        p = Path(images)
        if p.is_dir():
            return sorted(list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg")))
        return [p]
    if isinstance(images, list):
        return [Path(f) for f in images]
    raise TypeError(f"Không hỗ trợ kiểu images: {type(images)}")


def calculate_lpips_diversity(
    images: torch.Tensor | list[Path | str] | str | Path,
    num_pairs: int = 2000,
    net: str = "alex",
    device: str | torch.device = "cpu",
    seed: int = 42,
    batch_size: int = 64,
    return_details: bool = False,
) -> float | dict[str, Any]:
    """Tính trung bình LPIPS giữa `num_pairs` cặp ảnh khác nhau (i != j).

    `images` có thể là:
    - Tensor float (N, C, H, W) trong [0, 1] hoặc [-1, 1].
    - Thư mục chứa file ảnh PNG/JPG.
    - Danh sách đường dẫn file ảnh.
    """
    if lpips is None:
        raise ImportError("Chưa cài đặt thư viện lpips. Chạy: pip install lpips")
    if num_pairs <= 0:
        raise ValueError(f"num_pairs phải > 0, nhận {num_pairs}")

    rng = random.Random(seed)
    dev = torch.device(device)
    is_tensor = isinstance(images, torch.Tensor)

    if is_tensor:
        tensor_imgs = images
        if tensor_imgs.ndim != 4:
            raise ValueError(f"Tensor ảnh phải có 4 chiều (N, C, H, W), nhận {tuple(tensor_imgs.shape)}")
        n_total = int(tensor_imgs.shape[0])
        if tensor_imgs.shape[1] == 1:
            tensor_imgs = tensor_imgs.repeat(1, 3, 1, 1)
        if tensor_imgs.min() >= 0:
            tensor_imgs = tensor_imgs * 2.0 - 1.0
        files: list[Path] = []
    else:
        files = _collect_image_files(images)
        n_total = len(files)
        tensor_imgs = None

    if n_total < 2:
        details = {
            "lpips_diversity": 0.0,
            "requested_pairs": int(num_pairs),
            "effective_pairs": 0,
            "unique_pairs": 0,
            "num_images": n_total,
        }
        return details if return_details else 0.0

    pairs = _sample_distinct_pairs(n_total, num_pairs, rng)
    loss_fn = lpips.LPIPS(net=net).to(dev)
    loss_fn.eval()

    distances: list[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            if is_tensor and tensor_imgs is not None:
                idx1 = [i for i, _ in batch_pairs]
                idx2 = [j for _, j in batch_pairs]
                b1 = tensor_imgs[idx1].to(dev)
                b2 = tensor_imgs[idx2].to(dev)
            else:
                b1 = torch.stack([_load_image_tensor(files[i]) for i, _ in batch_pairs]).to(dev)
                b2 = torch.stack([_load_image_tensor(files[j]) for _, j in batch_pairs]).to(dev)
                b1 = b1 * 2.0 - 1.0
                b2 = b2 * 2.0 - 1.0
            d = loss_fn(b1, b2).flatten()
            distances.extend(d.cpu().tolist())

    score = float(np.mean(distances)) if distances else 0.0
    if not return_details:
        return score

    unique_count = len({(min(i, j), max(i, j)) for i, j in pairs})
    return {
        "lpips_diversity": score,
        "requested_pairs": int(num_pairs),
        "effective_pairs": len(pairs),
        "unique_pairs": unique_count,
        "num_images": n_total,
    }


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
        self,
        images: torch.Tensor | list[Path | str] | str | Path,
        num_pairs: int = 2000,
        net: str = "alex",
        return_details: bool = False,
    ) -> float | dict[str, Any]:
        return calculate_lpips_diversity(
            images, num_pairs=num_pairs, net=net, device=self.device, return_details=return_details
        )

    def calculate_prdc(
        self, real_features: np.ndarray | torch.Tensor, fake_features: np.ndarray | torch.Tensor, nearest_k: int = 5
    ) -> dict[str, float]:
        return calculate_prdc(real_features, fake_features, nearest_k=nearest_k)

    def calculate_ms_ssim(self, images: torch.Tensor, num_pairs: int = 100) -> float:
        return calculate_ms_ssim(images, num_pairs=num_pairs)
