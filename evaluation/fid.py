"""Fréchet Inception Distance and Kernel Inception Distance evaluation utilities.

Triển khai quy chuẩn đánh giá chất lượng theo TASK_ASSIGNMENT.md (mục 2.6 và 4.2):
- clean-fid (mode clean) cho FID và KID với InceptionV3 pool3 2048 chiều.
- Quản lý cache statistics và feature của tập ảnh thật theo tên <dataset>_<split>_<resolution>.
- Hỗ trợ Tuning mode (KID-5k trên validation).
- Hỗ trợ Final mode: FID_train-50k, FID_test-10k (mean ± std), FID sàn, KID_test, PRDC, LPIPS.
- Xuất JSON chuẩn cho kind = "quality" và kind = "real_baseline".
- Tạo ảnh nearest-neighbor (32 ảnh sinh × 3 ảnh train gần nhất).
"""

from __future__ import annotations

import copy
import hashlib
import json
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy import linalg

# Thêm gốc repo vào sys.path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.common import git_commit  # noqa: E402
from evaluation.diversity import calculate_lpips_diversity, calculate_prdc  # noqa: E402
from scripts.render_samples import save_grid  # noqa: E402

try:
    import cleanfid
    import cleanfid.fid as clean_fid
except ImportError:
    cleanfid = None
    clean_fid = None


def load_config(path: str | Path) -> tuple[dict, str]:
    """Đọc file YAML config và trả về (config_dict, sha256_digest)."""
    p = Path(path)
    content = p.read_bytes()
    digest = hashlib.sha256(content).hexdigest()[:16]
    cfg = yaml.safe_load(content.decode("utf-8")) or {}
    return cfg, digest


def write_json(record: dict, path: str | Path) -> None:
    """Ghi dữ liệu JSON ra file một cách an toàn (qua file tạm)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """Tính khoảng cách Frechet giữa 2 phân phối Gaussian đa chiều N(mu1, sigma1) và N(mu2, sigma2)."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, f"Shape mu không khớp: {mu1.shape} vs {mu2.shape}"
    assert sigma1.shape == sigma2.shape, f"Shape sigma không khớp: {sigma1.shape} vs {sigma2.shape}"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component lớn trong covmean: {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return max(0.0, fid)


def calculate_fid_from_features(feats1: np.ndarray, feats2: np.ndarray) -> float:
    """Tính FID từ 2 ma trận đặc trưng (N1, D) và (N2, D)."""
    mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    return calculate_frechet_distance(mu1, sig1, mu2, sig2)


def calculate_kid_from_features(
    feats1: np.ndarray,
    feats2: np.ndarray,
    num_subsets: int = 100,
    subset_size: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Tính Kernel Inception Distance (KID mean ± std) bằng kernel đa thức k(x,y) = (x^T y / d + 1)^3."""
    rng = np.random.default_rng(seed)
    n = feats1.shape[1]
    m = min(min(feats1.shape[0], feats2.shape[0]), subset_size)
    if m < 2:
        return 0.0, 0.0

    scores = []
    for _ in range(num_subsets):
        idx1 = rng.choice(feats1.shape[0], m, replace=False)
        idx2 = rng.choice(feats2.shape[0], m, replace=False)
        x = feats1[idx1]
        y = feats2[idx2]

        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        val = ((a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m) / m
        scores.append(float(val))

    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    return mean, std


def get_cleanfid_features(
    folder_or_files: str | Path | list[Path | str],
    device: str | torch.device = "cpu",
    batch_size: int = 100,
    mode: str = "clean",
    model_name: str = "inception_v3",
) -> np.ndarray:
    """Trích xuất features 2048 chiều bằng clean-fid InceptionV3."""
    if clean_fid is None:
        raise ImportError("Chưa cài clean-fid. Chạy: pip install clean-fid")

    feat_model = clean_fid.build_feature_extractor(mode=mode, device=device)
    if isinstance(folder_or_files, (str, Path)):
        p = Path(folder_or_files)
        if p.is_dir():
            files = sorted(list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg")))
            feats = clean_fid.get_files_features(
                [str(f) for f in files],
                model=feat_model,
                num_workers=0,
                batch_size=batch_size,
                device=device,
                mode=mode,
                custom_fn_resize=None,
            )
            return feats
        else:
            files = [str(p)]
            return clean_fid.get_files_features(files, model=feat_model, num_workers=0, batch_size=batch_size, device=device)
    elif isinstance(folder_or_files, list):
        files = [str(f) for f in folder_or_files]
        return clean_fid.get_files_features(files, model=feat_model, num_workers=0, batch_size=batch_size, device=device)
    raise TypeError(f"folder_or_files không hợp lệ: {type(folder_or_files)}")


def load_or_compute_reference_features(
    ref_name: str,
    data_dir: str | Path,
    cache_dir: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int = 100,
    mode: str = "clean",
) -> np.ndarray:
    """Tải từ cache hoặc trích xuất đặc trưng của tập ảnh thật tham chiếu theo quy ước tên <dataset>_<split>_<res>."""
    cache_path = Path(cache_dir) / f"{ref_name}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data["features"]

    feats = get_cleanfid_features(data_dir, device=device, batch_size=batch_size, mode=mode)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=feats)
    return feats


def calculate_kid_tuning(
    fake_dir: str | Path,
    val_dir: str | Path,
    cfg: dict | None = None,
    device: str | torch.device = "cpu",
    seed: int = 42,
    cache_dir: str | Path = "outputs/cache/fid",
    dataset: str = "custom",
    resolution: int = 32,
) -> dict[str, float]:
    """Tính KID trên tập validation (5.000 ảnh sinh) dùng cho giai đoạn 1 Tuning theo mục 2.3 & 2.6."""
    batch_size = cfg.get("feature_batch_size", 100) if cfg else 100
    sub_size = cfg.get("kid_subset_size", 1000) if cfg else 1000
    n_sub = cfg.get("kid_num_subsets", 100) if cfg else 100

    ref_name = f"{dataset}_val_{resolution}"
    val_feats = load_or_compute_reference_features(
        ref_name=ref_name, data_dir=val_dir, cache_dir=cache_dir, device=device, batch_size=batch_size
    )
    fake_feats = get_cleanfid_features(fake_dir, device=device, batch_size=batch_size)

    mean, std = calculate_kid_from_features(fake_feats, val_feats, num_subsets=n_sub, subset_size=sub_size, seed=seed)
    return {"kid_val_mean": float(mean), "kid_val_std": float(std)}


def render_nearest_neighbors(
    fake_dir: str | Path,
    train_dir: str | Path,
    output_path: str | Path,
    num_samples: int = 32,
    num_neighbors: int = 3,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> Path:
    """Tạo ảnh kiểm tra học thuộc: 32 ảnh sinh, mỗi ảnh đặt cạnh 3 ảnh train gần nhất trong feature space Inception."""
    rng = np.random.default_rng(seed)
    fake_files = sorted(list(Path(fake_dir).glob("*.png")))
    train_files = sorted(list(Path(train_dir).glob("*.png")))

    if len(fake_files) < num_samples:
        raise ValueError(f"Số ảnh fake ({len(fake_files)}) nhỏ hơn num_samples={num_samples}")

    selected_fake_idx = rng.choice(len(fake_files), num_samples, replace=False)
    selected_fake_files = [fake_files[i] for i in selected_fake_idx]

    # Trích feature Inception
    fake_feats = get_cleanfid_features(selected_fake_files, device=device)
    train_feats = get_cleanfid_features(train_files, device=device)

    # Chuẩn hóa L2 để tính Cosine Similarity
    fake_norm = fake_feats / np.linalg.norm(fake_feats, axis=1, keepdims=True)
    train_norm = train_feats / np.linalg.norm(train_feats, axis=1, keepdims=True)

    sim_matrix = fake_norm @ train_norm.T  # (num_samples, num_train)

    # Đọc ảnh để ghép grid
    grid_images = []
    for row in range(num_samples):
        # Ảnh fake đầu hàng
        img_fake = Image.open(selected_fake_files[row]).convert("RGB")
        t_fake = torch.from_numpy(np.array(img_fake, dtype=np.float32) / 255.0).permute(2, 0, 1)
        grid_images.append(t_fake)

        # 3 ảnh train gần nhất
        top_k_train_idx = np.argsort(sim_matrix[row])[::-1][:num_neighbors]
        for idx in top_k_train_idx:
            img_train = Image.open(train_files[idx]).convert("RGB")
            t_train = torch.from_numpy(np.array(img_train, dtype=np.float32) / 255.0).permute(2, 0, 1)
            grid_images.append(t_train)

    tensor_grid = torch.stack(grid_images)  # (32 * 4, 3, H, W)
    out = save_grid(tensor_grid, output_path, nrow=1 + num_neighbors)
    return out


def evaluate_final_record(
    model: str,
    dataset: str,
    run_name: str,
    label: str,
    fake_dir: str | Path,
    train_dir: str | Path,
    test_dir: str | Path,
    cfg: dict,
    cfg_hash: str,
    resolution: int,
    device: str = "cpu",
    sample_kwargs: dict | None = None,
    seed: int = 42,
    cache_dir: str | Path = "outputs/cache/fid",
    comparison: str | None = None,
) -> tuple[dict, dict]:
    """Tính toán toàn bộ metric chất lượng và baseline thật-với-thật theo mục 2.6 và 4.2."""
    sample_kwargs = sample_kwargs or {}
    batch_size = cfg.get("feature_batch_size", 100)
    mode = cfg.get("mode", "clean")
    lib_ver = pkg_version("clean-fid") if cleanfid else "clean-fid"

    ref_train = f"{dataset}_train_{resolution}"
    ref_test = f"{dataset}_test_{resolution}"

    feats_train = load_or_compute_reference_features(ref_train, train_dir, cache_dir, device, batch_size, mode)
    feats_test = load_or_compute_reference_features(ref_test, test_dir, cache_dir, device, batch_size, mode)
    feats_fake = get_cleanfid_features(fake_dir, device=device, batch_size=batch_size, mode=mode)
    num_gen = len(feats_fake)

    eval_block = {
        "eval_config": "configs/eval/fid.yaml",
        "eval_config_hash": cfg_hash,
        "library": f"clean-fid=={lib_ver}",
        "commit": git_commit() or "unknown",
        "ref_train": ref_train,
        "ref_test": ref_test,
        "num_gen": num_gen,
    }

    # Dòng phụ so sánh sampler DDPM
    if comparison == "sampler":
        fid_train_10k = calculate_fid_from_features(feats_fake, feats_train)
        kid_mean, kid_std = calculate_kid_from_features(
            feats_fake, feats_test, num_subsets=cfg.get("kid_num_subsets", 100), subset_size=cfg.get("kid_subset_size", 1000), seed=seed
        )
        quality = {
            "kind": "quality",
            "model": model,
            "dataset": dataset,
            "run_name": run_name,
            "label": label,
            "sampler": sample_kwargs.get("sampler"),
            "sampling_steps": sample_kwargs.get("steps"),
            "comparison": comparison,
            "eval": eval_block,
            "metrics": {
                "fid_train_10k": float(fid_train_10k),
                "kid_test_mean": float(kid_mean),
                "kid_test_std": float(kid_std),
            },
        }
    else:
        # Bảng chính (50k ảnh)
        splits = cfg.get("fid_test_splits", 5)
        fid_train_50k = calculate_fid_from_features(feats_fake, feats_train)

        # 10k ảnh đầu dùng cho KID test, PRDC và fid_train_10k
        num_first = num_gen // splits if num_gen >= splits else num_gen
        feats_first_10k = feats_fake[:num_first]

        fid_train_10k = calculate_fid_from_features(feats_first_10k, feats_train)

        # FID test trên 5 fold 10k
        test_fids = [calculate_fid_from_features(chunk, feats_test) for chunk in np.array_split(feats_fake, splits)]
        fid_test_10k_mean = float(np.mean(test_fids))
        fid_test_10k_std = float(np.std(test_fids, ddof=1)) if len(test_fids) > 1 else 0.0

        kid_mean, kid_std = calculate_kid_from_features(
            feats_first_10k, feats_test, num_subsets=cfg.get("kid_num_subsets", 100), subset_size=cfg.get("kid_subset_size", 1000), seed=seed
        )

        # PRDC (10k train vs 10k fake đầu)
        prdc_n = min(cfg.get("prdc_num", 10000), len(feats_train), len(feats_first_10k))
        prdc_res = calculate_prdc(feats_train[:prdc_n], feats_first_10k[:prdc_n], nearest_k=cfg.get("prdc_k", 5))

        # LPIPS Diversity
        lpips_div = calculate_lpips_diversity(
            fake_dir, num_pairs=cfg.get("lpips_pairs", 2000), net=cfg.get("lpips_net", "alex"), device=device, seed=seed
        )

        quality = {
            "kind": "quality",
            "model": model,
            "dataset": dataset,
            "run_name": run_name,
            "label": label,
            "sampler": sample_kwargs.get("sampler"),
            "sampling_steps": sample_kwargs.get("steps"),
            "eval": eval_block,
            "metrics": {
                "fid_train_50k": float(fid_train_50k),
                "fid_train_10k": float(fid_train_10k),
                "fid_test_10k_mean": float(fid_test_10k_mean),
                "fid_test_10k_std": float(fid_test_10k_std),
                "kid_test_mean": float(kid_mean),
                "kid_test_std": float(kid_std),
                "precision": float(prdc_res["precision"]),
                "recall": float(prdc_res["recall"]),
                "density": float(prdc_res["density"]),
                "coverage": float(prdc_res["coverage"]),
                "lpips_diversity": float(lpips_div),
            },
        }

    # Baseline thật-với-thật
    rng = np.random.default_rng(seed)
    n_sample_train = min(10000, len(feats_train))
    idx_train = rng.choice(len(feats_train), n_sample_train, replace=False)
    fid_floor = calculate_fid_from_features(feats_train[idx_train], feats_test)
    lpips_real = calculate_lpips_diversity(
        test_dir, num_pairs=cfg.get("lpips_pairs", 2000), net=cfg.get("lpips_net", "alex"), device=device, seed=seed
    )

    baseline = {
        "kind": "real_baseline",
        "dataset": dataset,
        "metrics": {
            "fid_floor": float(fid_floor),
            "lpips_real": float(lpips_real),
        },
    }

    return quality, baseline


class FIDCalculator:
    """Lớp tính FID tương thích ngược."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", cache_dir: str = "outputs/cache/fid"):
        self.device = device
        self.cache_dir = cache_dir

    def calculate_fid(self, real_loader, fake_loader, dataset_name: str, use_cache: bool = True) -> float:
        """Tính FID giữa 2 dataloader."""
        # Giữ tính năng tương thích ngược
        real_imgs = []
        for b in real_loader:
            imgs = b[0] if isinstance(b, (list, tuple)) else b
            real_imgs.append(imgs)
        fake_imgs = []
        for b in fake_loader:
            imgs = b[0] if isinstance(b, (list, tuple)) else b
            fake_imgs.append(imgs)

        t_real = torch.cat(real_imgs, dim=0)
        t_fake = torch.cat(fake_imgs, dim=0)

        # Đảm bảo 3 kênh
        if t_real.shape[1] == 1:
            t_real = t_real.repeat(1, 3, 1, 1)
        if t_fake.shape[1] == 1:
            t_fake = t_fake.repeat(1, 3, 1, 1)

        feat_model = clean_fid.build_feature_extractor(mode="clean", device=self.device)
        f_real = clean_fid.get_batch_features(t_real, feat_model, device=self.device)
        f_fake = clean_fid.get_batch_features(t_fake, feat_model, device=self.device)

        return calculate_fid_from_features(f_real, f_fake)
