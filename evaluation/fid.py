"""Fréchet Inception Distance and Kernel Inception Distance evaluation utilities.

Triển khai quy chuẩn đánh giá chất lượng theo TASK_ASSIGNMENT.md (mục 2.6, 3.2 và 4.2):
- clean-fid (mode clean) cho FID và KID với InceptionV3 pool3 2048 chiều.
- Kiểm tra tính toàn vẹn manifest và cardinality của thư mục ảnh thật/ảnh sinh theo configs/eval/fid.yaml.
- Quản lý cache đặc trưng gắn với hash config, split, file manifest, extractor mode và phiên bản thư viện.
- Hỗ trợ Tuning mode (KID-5k trên validation).
- Hỗ trợ Final mode: FID_train-50k, FID_test-10k (mean ± std), FID sàn, KID_test, PRDC, LPIPS.
- Bổ sung FID dùng feature riêng (128-d penultimate của FashionMNISTCNN) cho Fashion-MNIST kèm cache.
- Xuất JSON chuẩn cho kind = "quality" và kind = "real_baseline".
- Tạo ảnh nearest-neighbor (32 ảnh sinh × 3 ảnh train gần nhất trong feature space Inception, tái sử dụng cache).
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version as pkg_version
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from PIL import Image
from scipy import linalg

# Thêm gốc repo vào sys.path
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

NON_PNG_IMAGE_EXTS = {".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_FMNIST_CNN_CKPT = ROOT / "outputs" / "checkpoints" / "fashion_mnist_cnn.pt"

EXPECTED_SPLIT_COUNTS = {
    ("celeba", "train"): 162770,
    ("celeba", "val"): 19867,
    ("celeba", "test"): 19962,
    ("fashion_mnist", "train"): 50000,
    ("fashion_mnist", "val"): 10000,
    ("fashion_mnist", "test"): 10000,
    ("cifar10", "train"): 45000,
    ("cifar10", "val"): 5000,
    ("cifar10", "test"): 10000,
}


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


def _sha256_file(path: Path) -> str | None:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return None


def _validate_image_folder(
    folder: str | Path,
    expected_count: int | None = None,
    min_count: int = 1,
    allow_partial: bool = False,
    label: str = "folder",
) -> list[Path]:
    """Kiểm tra thư mục ảnh PNG: tồn tại, không lẫn định dạng khác, khớp manifest và đúng cardinality."""
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"{label}: không tìm thấy thư mục '{p}'.")
    if not p.is_dir():
        raise NotADirectoryError(f"{label}: đường dẫn '{p}' không phải thư mục.")

    disallowed = sorted(
        f.name for f in p.iterdir() if f.is_file() and f.suffix.lower() in NON_PNG_IMAGE_EXTS
    )
    if disallowed:
        raise ValueError(
            f"{label}: phát hiện định dạng ảnh không phải PNG trong '{p}' "
            f"(ví dụ: {disallowed[:5]}). Mục 2.6 yêu cầu chỉ dùng PNG uint8."
        )

    png_files = sorted(p.glob("*.png"))
    if len(png_files) < min_count:
        raise ValueError(f"{label}: chỉ có {len(png_files)} ảnh PNG trong '{p}', cần tối thiểu {min_count}.")

    manifest_path = p / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is not True:
            raise ValueError(f"{label}: manifest.json trong '{p}' chưa hoàn tất (complete != True).")
        if not allow_partial and manifest.get("is_partial") is True:
            raise ValueError(f"{label}: dữ liệu trong '{p}' chỉ là bản partial (--max_samples), không hợp lệ.")
        manifest_files = manifest.get("files")
        if manifest_files is not None:
            disk_names = [f.name for f in png_files]
            expected_names = sorted(manifest_files)
            if disk_names != expected_names:
                raise ValueError(
                    f"{label}: danh sách file PNG trong '{p}' ({len(disk_names)}) không khớp "
                    f"manifest.json ({len(expected_names)}). Có thể thư mục bị thiếu hoặc lẫn ảnh cũ."
                )
        expected_manifest_count = manifest.get("num_samples")
        if expected_manifest_count is None:
            expected_manifest_count = manifest.get("total_images")
        if expected_manifest_count is not None and len(png_files) != int(expected_manifest_count):
            raise ValueError(
                f"{label}: số lượng PNG ({len(png_files)}) không khớp manifest ({expected_manifest_count}) trong '{p}'."
            )
    else:
        meta_path = p / "metadata.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not allow_partial and meta.get("is_partial") is True:
                raise ValueError(f"{label}: metadata.json trong '{p}' đánh dấu bản partial.")
            if meta.get("total_images") is not None and len(png_files) != int(meta["total_images"]):
                raise ValueError(
                    f"{label}: số lượng PNG ({len(png_files)}) không khớp metadata.json ({meta['total_images']}) trong '{p}'."
                )

    if expected_count is not None and len(png_files) != int(expected_count):
        raise ValueError(
            f"{label}: yêu cầu đúng {expected_count} ảnh PNG theo cấu hình, nhưng tìm thấy {len(png_files)} ảnh trong '{p}'."
        )

    return png_files


def calculate_frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6
) -> float:
    """Tính khoảng cách Frechet giữa 2 phân phối Gaussian đa chiều N(mu1, sigma1) và N(mu2, sigma2)."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, f"Shape mu không khớp: {mu1.shape} vs {mu2.shape}"
    assert sigma1.shape == sigma2.shape, f"Shape sigma không khớp: {sigma1.shape} vs {sigma2.shape}"

    diff = mu1 - mu2

    covmean = linalg.sqrtm(sigma1.dot(sigma2))
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
            files = sorted(p.glob("*.png"))
            if not files:
                raise ValueError(f"Không tìm thấy ảnh PNG nào trong '{p}'.")
            return clean_fid.get_files_features(
                [str(f) for f in files],
                model=feat_model,
                num_workers=0,
                batch_size=batch_size,
                device=device,
                mode=mode,
                custom_fn_resize=None,
            )
        files = [str(p)]
        return clean_fid.get_files_features(
            files, model=feat_model, num_workers=0, batch_size=batch_size, device=device, mode=mode
        )
    if isinstance(folder_or_files, list):
        files = [str(f) for f in folder_or_files]
        return clean_fid.get_files_features(
            files, model=feat_model, num_workers=0, batch_size=batch_size, device=device, mode=mode
        )
    raise TypeError(f"folder_or_files không hợp lệ: {type(folder_or_files)}")


def _compute_reference_cache_key(
    ref_name: str,
    data_dir: Path,
    png_files: list[Path],
    dataset: str,
    split: str,
    mode: str,
    model_name: str,
    eval_config_hash: str | None,
) -> tuple[str, dict]:
    """Tạo khóa cache gắn chặt với config, split, file manifest, extractor mode và library version."""
    lib_ver = pkg_version("clean-fid") if cleanfid else "unknown"
    resolved_eval_hash = eval_config_hash or _sha256_file(ROOT / "configs" / "eval" / "fid.yaml") or "none"
    dataset_cfg_hash = _sha256_file(ROOT / "configs" / "datasets" / f"{dataset}.yaml") or "none"
    split_file_hash = _sha256_file(ROOT / "datasets" / "splits" / f"{dataset}_split.json") or "none"
    folder_meta_hash = _sha256_file(data_dir / "metadata.json") or "none"
    folder_manifest_hash = _sha256_file(data_dir / "manifest.json") or "none"

    file_entries = [(f.name, f.stat().st_size) for f in png_files]
    files_digest = hashlib.sha256(
        json.dumps(file_entries, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]

    meta = {
        "ref_name": ref_name,
        "dataset": dataset,
        "split": split,
        "num_files": len(png_files),
        "mode": mode,
        "model_name": model_name,
        "library": f"clean-fid=={lib_ver}",
        "eval_config_hash": resolved_eval_hash,
        "dataset_config_hash": dataset_cfg_hash,
        "split_file_hash": split_file_hash,
        "folder_metadata_hash": folder_meta_hash,
        "folder_manifest_hash": folder_manifest_hash,
        "files_digest": files_digest,
    }
    fingerprint = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    cache_key = f"{ref_name}__{fingerprint}"
    meta["cache_key"] = cache_key
    return cache_key, meta


def load_or_compute_reference_features(
    ref_name: str,
    data_dir: str | Path,
    cache_dir: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int = 100,
    mode: str = "clean",
    model_name: str = "inception_v3",
    dataset: str = "custom",
    split: str = "train",
    eval_config_hash: str | None = None,
    png_files: list[Path] | None = None,
    return_cache_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Tải từ cache hoặc trích xuất đặc trưng tập ảnh thật tham chiếu với xác thực đầy đủ."""
    p = Path(data_dir)
    if png_files is None:
        png_files = _validate_image_folder(p, label=f"reference({ref_name})")

    cache_key, meta = _compute_reference_cache_key(
        ref_name=ref_name,
        data_dir=p,
        png_files=png_files,
        dataset=dataset,
        split=split,
        mode=mode,
        model_name=model_name,
        eval_config_hash=eval_config_hash,
    )
    cache_path = Path(cache_dir) / f"{cache_key}.npz"

    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                if "features" in data and "cache_meta" in data:
                    stored_meta = json.loads(str(data["cache_meta"][0]))
                    feats = data["features"]
                    if stored_meta == meta and feats.shape[0] == len(png_files):
                        return (feats, meta) if return_cache_meta else feats
        except Exception:
            pass

    feats = get_cleanfid_features(png_files, device=device, batch_size=batch_size, mode=mode, model_name=model_name)
    if feats.shape[0] != len(png_files):
        raise RuntimeError(
            f"Số feature trích xuất ({feats.shape[0]}) không khớp số file ({len(png_files)}) cho {ref_name}."
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=feats,
        cache_meta=np.array([json.dumps(meta, sort_keys=True)]),
    )
    return (feats, meta) if return_cache_meta else feats


def _load_or_compute_fmnist_cnn_features(
    cnn_model,
    ckpt_hash: str,
    split: str,
    png_files: list[Path],
    cache_dir: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int = 100,
) -> np.ndarray:
    """Trích xuất (hoặc đọc từ cache) đặc trưng 128-d của FashionMNISTCNN cho tập ảnh tham chiếu."""
    from models.classifiers.fashion_mnist_cnn import extract_features_for_folder

    file_entries = [(f.name, f.stat().st_size) for f in png_files]
    files_digest = hashlib.sha256(
        json.dumps(file_entries, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    meta = {
        "dataset": "fashion_mnist",
        "split": split,
        "extractor": "fashion_mnist_cnn_128d",
        "ckpt_hash": ckpt_hash,
        "num_files": len(png_files),
        "files_digest": files_digest,
    }
    fp = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    cache_path = Path(cache_dir) / f"fashion_mnist_{split}_32_custom_cnn__{fp}.npz"

    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                if "features" in data and "cache_meta" in data:
                    stored_meta = json.loads(str(data["cache_meta"][0]))
                    feats = data["features"]
                    if stored_meta == meta and feats.shape[0] == len(png_files):
                        return feats
        except Exception:
            pass

    feats = extract_features_for_folder(cnn_model, png_files, device=device, batch_size=batch_size).numpy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=feats,
        cache_meta=np.array([json.dumps(meta, sort_keys=True)]),
    )
    return feats


def calculate_fashion_mnist_custom_fid(
    fake_files: list[Path],
    train_files: list[Path],
    test_files: list[Path],
    cfg: dict,
    cnn_ckpt_path: str | Path = DEFAULT_FMNIST_CNN_CKPT,
    cache_dir: str | Path = "outputs/cache/fid",
    device: str | torch.device = "cpu",
    comparison: str | None = None,
    strict_counts: bool = True,
    train_floor_idx: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[dict[str, float], float, dict]:
    """Tính FID đặc trưng riêng (128-d FashionMNISTCNN) cho Fashion-MNIST theo mục 2.6 & 3.2."""
    from models.classifiers.fashion_mnist_cnn import ARCHITECTURE_DESC, extract_features_for_folder, load_checkpoint

    ckpt_p = Path(cnn_ckpt_path)
    if not ckpt_p.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint FashionMNISTCNN tại '{ckpt_p}'. "
            "Chạy `python models/classifiers/fashion_mnist_cnn.py` trước."
        )

    cnn_model, payload = load_checkpoint(ckpt_p, device=device)
    ckpt_hash = _sha256_file(ckpt_p) or "unknown"
    batch_size = int(cfg.get("feature_batch_size", 100))

    c_train = _load_or_compute_fmnist_cnn_features(
        cnn_model, ckpt_hash, "train", train_files, cache_dir=cache_dir, device=device, batch_size=batch_size
    )
    c_test = _load_or_compute_fmnist_cnn_features(
        cnn_model, ckpt_hash, "test", test_files, cache_dir=cache_dir, device=device, batch_size=batch_size
    )
    c_fake = extract_features_for_folder(cnn_model, fake_files, device=device, batch_size=batch_size).numpy()

    if comparison == "sampler":
        custom_metrics = {
            "fid_custom_train_10k": float(calculate_fid_from_features(c_fake, c_train)),
        }
    else:
        splits = int(cfg.get("fid_test_splits", 5))
        num_first = int(cfg.get("fid_test_num_gen", 10000)) if strict_counts else (len(c_fake) // splits)
        c_first_10k = c_fake[:num_first]
        c_test_fids = [calculate_fid_from_features(chunk, c_test) for chunk in np.array_split(c_fake, splits)]
        custom_metrics = {
            "fid_custom_train_50k": float(calculate_fid_from_features(c_fake, c_train)),
            "fid_custom_train_10k": float(calculate_fid_from_features(c_first_10k, c_train)),
            "fid_custom_test_10k_mean": float(np.mean(c_test_fids)),
            "fid_custom_test_10k_std": float(np.std(c_test_fids, ddof=1)) if len(c_test_fids) > 1 else 0.0,
        }

    if train_floor_idx is None:
        rng = np.random.default_rng(seed)
        train_floor_idx = rng.choice(len(c_train), min(10000, len(c_train)), replace=False)
    fid_custom_floor = float(calculate_fid_from_features(c_train[train_floor_idx], c_test))

    cnn_meta = {
        "checkpoint": "outputs/checkpoints/fashion_mnist_cnn.pt",
        "checkpoint_hash": ckpt_hash,
        "architecture": payload.get("architecture", ARCHITECTURE_DESC),
        "feature_dim": int(payload.get("feature_dim", 128)),
        "test_acc": payload.get("test_acc", payload.get("val_acc")),
        "val_acc": payload.get("val_acc"),
    }
    return custom_metrics, fid_custom_floor, cnn_meta


def calculate_kid_tuning(
    fake_dir: str | Path,
    val_dir: str | Path,
    cfg: dict | None = None,
    device: str | torch.device = "cpu",
    seed: int = 42,
    cache_dir: str | Path = "outputs/cache/fid",
    dataset: str = "custom",
    resolution: int = 32,
    eval_config_hash: str | None = None,
    strict_counts: bool = True,
) -> dict[str, float]:
    """Tính KID trên tập validation (5.000 ảnh sinh) dùng cho giai đoạn 1 Tuning theo mục 2.3 & 2.6."""
    batch_size = int((cfg or {}).get("feature_batch_size", 100))
    sub_size = int((cfg or {}).get("kid_subset_size", 1000))
    n_sub = int((cfg or {}).get("kid_num_subsets", 100))
    mode = str((cfg or {}).get("mode", "clean"))
    model_name = str((cfg or {}).get("feature_extractor", "inception_v3"))

    expected_fake = int((cfg or {}).get("num_gen_tuning", 5000)) if strict_counts else None
    expected_val = EXPECTED_SPLIT_COUNTS.get((dataset, "val")) if strict_counts else None

    fake_files = _validate_image_folder(fake_dir, expected_count=expected_fake, label="fake_tuning_dir")
    val_files = _validate_image_folder(val_dir, expected_count=expected_val, label="val_dir")

    ref_name = f"{dataset}_val_{resolution}"
    val_feats = load_or_compute_reference_features(
        ref_name=ref_name,
        data_dir=val_dir,
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
        mode=mode,
        model_name=model_name,
        dataset=dataset,
        split="val",
        eval_config_hash=eval_config_hash,
        png_files=val_files,
    )
    fake_feats = get_cleanfid_features(fake_files, device=device, batch_size=batch_size, mode=mode, model_name=model_name)

    mean, std = calculate_kid_from_features(fake_feats, val_feats, num_subsets=n_sub, subset_size=sub_size, seed=seed)
    return {"kid_val_mean": float(mean), "kid_val_std": float(std)}


def _infer_dataset_and_res_from_dir(train_dir: Path, train_files: list[Path]) -> tuple[str, int]:
    """Suy ra tên dataset và độ phân giải từ đường dẫn outputs/reference/<dataset>/train nếu có."""
    dataset = "custom"
    parts = [p.lower() for p in train_dir.parts]
    for cand in ("celeba", "fashion_mnist", "cifar10"):
        if cand in parts:
            dataset = cand
            break
    with Image.open(train_files[0]) as im:
        res = int(im.size[1])
    return dataset, res


def render_nearest_neighbors(
    fake_dir: str | Path,
    train_dir: str | Path,
    output_path: str | Path,
    num_samples: int = 32,
    num_neighbors: int = 3,
    device: str | torch.device = "cpu",
    seed: int = 42,
    cache_dir: str | Path = "outputs/cache/fid",
    dataset: str | None = None,
    resolution: int | None = None,
    eval_config_hash: str | None = None,
    fake_files: list[Path] | None = None,
    train_files: list[Path] | None = None,
    feats_fake: np.ndarray | None = None,
    feats_train: np.ndarray | None = None,
    metric: str = "euclidean",
) -> Path:
    """Tạo ảnh kiểm tra học thuộc: 32 ảnh sinh, mỗi ảnh đặt cạnh 3 ảnh train gần nhất trong feature space Inception.

    Tự động tái sử dụng `feats_fake` / `feats_train` đã trích xuất hoặc cache `.npz` của tập train.
    """
    rng = np.random.default_rng(seed)
    if fake_files is None:
        fake_files = _validate_image_folder(fake_dir, min_count=num_samples, label="fake_dir")
    if train_files is None:
        train_files = _validate_image_folder(train_dir, min_count=num_neighbors, label="train_dir")

    if len(fake_files) < num_samples:
        raise ValueError(f"fake_dir chỉ có {len(fake_files)} ảnh, cần tối thiểu {num_samples}.")

    selected_fake_idx = rng.choice(len(fake_files), num_samples, replace=False)
    selected_fake_files = [fake_files[i] for i in selected_fake_idx]

    if feats_fake is not None and len(feats_fake) == len(fake_files):
        sel_fake_feats = feats_fake[selected_fake_idx]
    else:
        sel_fake_feats = get_cleanfid_features(selected_fake_files, device=device)

    if feats_train is None or len(feats_train) != len(train_files):
        ds_inferred, res_inferred = _infer_dataset_and_res_from_dir(Path(train_dir), train_files)
        ds_use = dataset or ds_inferred
        res_use = resolution or res_inferred
        feats_train = load_or_compute_reference_features(
            ref_name=f"{ds_use}_train_{res_use}",
            data_dir=train_dir,
            cache_dir=cache_dir,
            device=device,
            dataset=ds_use,
            split="train",
            eval_config_hash=eval_config_hash,
            png_files=train_files,
        )

    sel_f = sel_fake_feats.astype(np.float32)
    trn_f = feats_train.astype(np.float32)

    if metric == "cosine":
        f_norm = sel_f / (np.linalg.norm(sel_f, axis=1, keepdims=True) + 1e-12)
        t_norm = trn_f / (np.linalg.norm(trn_f, axis=1, keepdims=True) + 1e-12)
        # Khoảng cách cosine = 1 - cosine_similarity (càng nhỏ càng gần)
        dist_matrix = 1.0 - (f_norm @ t_norm.T)
    else:
        # Khoảng cách bình phương Euclidean trong không gian Inception pool3: ||x - y||^2
        f_sq = np.sum(sel_f * sel_f, axis=1, keepdims=True)
        t_sq = np.sum(trn_f * trn_f, axis=1, keepdims=True).T
        dist_matrix = np.maximum(f_sq + t_sq - 2.0 * (sel_f @ trn_f.T), 0.0)

    grid_images = []
    for row in range(num_samples):
        with Image.open(selected_fake_files[row]) as im:
            img_fake = im.convert("RGB")
            t_fake = torch.from_numpy(np.array(img_fake, dtype=np.float32) / 255.0).permute(2, 0, 1)
        grid_images.append(t_fake)

        if num_neighbors < dist_matrix.shape[1]:
            part = np.argpartition(dist_matrix[row], num_neighbors)[:num_neighbors]
            top_k_train_idx = part[np.argsort(dist_matrix[row, part])]
        else:
            top_k_train_idx = np.argsort(dist_matrix[row])[:num_neighbors]

        for idx in top_k_train_idx:
            with Image.open(train_files[int(idx)]) as im:
                img_train = im.convert("RGB")
                t_train = torch.from_numpy(np.array(img_train, dtype=np.float32) / 255.0).permute(2, 0, 1)
            grid_images.append(t_train)

    tensor_grid = torch.stack(grid_images)
    return save_grid(tensor_grid, output_path, nrow=1 + num_neighbors)


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
    strict_counts: bool = True,
    nearest_neighbor_path: str | Path | None = None,
    fmnist_cnn_ckpt: str | Path | None = DEFAULT_FMNIST_CNN_CKPT,
) -> tuple[dict, dict]:
    """Tính toán toàn bộ metric chất lượng, custom FID (nếu Fashion-MNIST), nearest-neighbor và baseline thật-với-thật."""
    sample_kwargs = sample_kwargs or {}
    batch_size = int(cfg.get("feature_batch_size", 100))
    mode = str(cfg.get("mode", "clean"))
    model_name = str(cfg.get("feature_extractor", "inception_v3"))
    lib_ver = pkg_version("clean-fid") if cleanfid else "clean-fid"

    if comparison == "sampler":
        expected_num_gen = int(cfg.get("sampler_compare_num_gen", 10000))
    else:
        expected_num_gen = int(cfg.get("num_gen_final", 50000))
        splits = int(cfg.get("fid_test_splits", 5))
        split_num_gen = int(cfg.get("fid_test_num_gen", 10000))
        if expected_num_gen != splits * split_num_gen:
            raise ValueError(
                f"Cấu hình fid.yaml không nhất quán: num_gen_final={expected_num_gen} "
                f"!= fid_test_splits ({splits}) * fid_test_num_gen ({split_num_gen})."
            )

    expected_train = EXPECTED_SPLIT_COUNTS.get((dataset, "train")) if strict_counts else None
    expected_test = EXPECTED_SPLIT_COUNTS.get((dataset, "test")) if strict_counts else None

    fake_files = _validate_image_folder(
        fake_dir,
        expected_count=expected_num_gen if strict_counts else None,
        min_count=1 if comparison == "sampler" else int(cfg.get("fid_test_splits", 5)),
        label="fake_dir",
    )
    train_files = _validate_image_folder(train_dir, expected_count=expected_train, label="train_dir")
    test_files = _validate_image_folder(test_dir, expected_count=expected_test, label="test_dir")

    ref_train = f"{dataset}_train_{resolution}"
    ref_test = f"{dataset}_test_{resolution}"

    feats_train, train_cache_meta = load_or_compute_reference_features(
        ref_name=ref_train,
        data_dir=train_dir,
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
        mode=mode,
        model_name=model_name,
        dataset=dataset,
        split="train",
        eval_config_hash=cfg_hash,
        png_files=train_files,
        return_cache_meta=True,
    )
    feats_test, test_cache_meta = load_or_compute_reference_features(
        ref_name=ref_test,
        data_dir=test_dir,
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
        mode=mode,
        model_name=model_name,
        dataset=dataset,
        split="test",
        eval_config_hash=cfg_hash,
        png_files=test_files,
        return_cache_meta=True,
    )
    feats_fake = get_cleanfid_features(fake_files, device=device, batch_size=batch_size, mode=mode, model_name=model_name)
    num_gen = len(feats_fake)
    if strict_counts and num_gen != expected_num_gen:
        raise ValueError(f"Số feature ảnh sinh ({num_gen}) khác kỳ vọng ({expected_num_gen}).")

    eval_block = {
        "eval_config": "configs/eval/fid.yaml",
        "eval_config_hash": cfg_hash,
        "library": f"clean-fid=={lib_ver}",
        "commit": git_commit() or "unknown",
        "ref_train": ref_train,
        "ref_test": ref_test,
        "ref_train_cache_key": train_cache_meta["cache_key"],
        "ref_test_cache_key": test_cache_meta["cache_key"],
        "num_gen": num_gen,
    }

    if comparison == "sampler":
        fid_train_10k = calculate_fid_from_features(feats_fake, feats_train)
        kid_mean, kid_std = calculate_kid_from_features(
            feats_fake,
            feats_test,
            num_subsets=int(cfg.get("kid_num_subsets", 100)),
            subset_size=int(cfg.get("kid_subset_size", 1000)),
            seed=seed,
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
        splits = int(cfg.get("fid_test_splits", 5))
        fid_train_50k = calculate_fid_from_features(feats_fake, feats_train)

        num_first = int(cfg.get("fid_test_num_gen", 10000)) if strict_counts else (num_gen // splits)
        feats_first_10k = feats_fake[:num_first]

        fid_train_10k = calculate_fid_from_features(feats_first_10k, feats_train)

        test_fids = [calculate_fid_from_features(chunk, feats_test) for chunk in np.array_split(feats_fake, splits)]
        fid_test_10k_mean = float(np.mean(test_fids))
        fid_test_10k_std = float(np.std(test_fids, ddof=1)) if len(test_fids) > 1 else 0.0

        kid_mean, kid_std = calculate_kid_from_features(
            feats_first_10k,
            feats_test,
            num_subsets=int(cfg.get("kid_num_subsets", 100)),
            subset_size=int(cfg.get("kid_subset_size", 1000)),
            seed=seed,
        )

        prdc_n = min(int(cfg.get("prdc_num", 10000)), len(feats_train), len(feats_first_10k))
        rng_prdc = np.random.default_rng(seed)
        prdc_train_idx = rng_prdc.choice(len(feats_train), prdc_n, replace=False)
        prdc_res = calculate_prdc(
            feats_train[prdc_train_idx], feats_first_10k[:prdc_n], nearest_k=int(cfg.get("prdc_k", 5))
        )

        lpips_details = calculate_lpips_diversity(
            fake_files,
            num_pairs=int(cfg.get("lpips_pairs", 2000)),
            net=str(cfg.get("lpips_net", "alex")),
            device=device,
            seed=seed,
            return_details=True,
        )
        eval_block["lpips_requested_pairs"] = lpips_details["requested_pairs"]
        eval_block["lpips_effective_pairs"] = lpips_details["effective_pairs"]

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
                "lpips_diversity": float(lpips_details["lpips_diversity"]),
            },
        }

        if nearest_neighbor_path is not None and len(fake_files) >= 32 and len(train_files) >= 3:
            render_nearest_neighbors(
                fake_dir=fake_dir,
                train_dir=train_dir,
                output_path=nearest_neighbor_path,
                num_samples=32,
                num_neighbors=3,
                device=device,
                seed=seed,
                cache_dir=cache_dir,
                dataset=dataset,
                resolution=resolution,
                eval_config_hash=cfg_hash,
                fake_files=fake_files,
                train_files=train_files,
                feats_fake=feats_fake,
                feats_train=feats_train,
            )

    rng = np.random.default_rng(seed)
    n_sample_train = min(10000, len(feats_train))
    idx_train = rng.choice(len(feats_train), n_sample_train, replace=False)
    fid_floor = calculate_fid_from_features(feats_train[idx_train], feats_test)
    lpips_real_details = calculate_lpips_diversity(
        test_files,
        num_pairs=int(cfg.get("lpips_pairs", 2000)),
        net=str(cfg.get("lpips_net", "alex")),
        device=device,
        seed=seed,
        return_details=True,
    )

    baseline: dict = {
        "kind": "real_baseline",
        "dataset": dataset,
        "metrics": {
            "fid_floor": float(fid_floor),
            "lpips_real": float(lpips_real_details["lpips_diversity"]),
        },
        "lpips_details": {
            "requested_pairs": lpips_real_details["requested_pairs"],
            "effective_pairs": lpips_real_details["effective_pairs"],
            "unique_pairs": lpips_real_details["unique_pairs"],
        },
    }

    # Fashion-MNIST: tự động tính bổ sung FID trên đặc trưng riêng 128-d của FashionMNISTCNN (mục 2.6 & 3.2)
    if dataset == "fashion_mnist" and fmnist_cnn_ckpt is not None and Path(fmnist_cnn_ckpt).is_file():
        custom_metrics, fid_custom_floor, cnn_meta = calculate_fashion_mnist_custom_fid(
            fake_files=fake_files,
            train_files=train_files,
            test_files=test_files,
            cfg=cfg,
            cnn_ckpt_path=fmnist_cnn_ckpt,
            cache_dir=cache_dir,
            device=device,
            comparison=comparison,
            strict_counts=strict_counts,
            train_floor_idx=idx_train,
            seed=seed,
        )
        quality["metrics"].update(custom_metrics)
        quality["custom_extractor"] = cnn_meta
        baseline["metrics"]["fid_custom_floor"] = fid_custom_floor
        baseline["custom_extractor"] = cnn_meta

    return quality, baseline


class FIDCalculator:
    """Lớp tính FID tương thích ngược."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", cache_dir: str = "outputs/cache/fid"):
        self.device = device
        self.cache_dir = cache_dir

    def calculate_fid(self, real_loader, fake_loader, dataset_name: str, use_cache: bool = True) -> float:
        """Tính FID giữa 2 dataloader."""
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

        if t_real.shape[1] == 1:
            t_real = t_real.repeat(1, 3, 1, 1)
        if t_fake.shape[1] == 1:
            t_fake = t_fake.repeat(1, 3, 1, 1)

        feat_model = clean_fid.build_feature_extractor(mode="clean", device=self.device)
        f_real = clean_fid.get_batch_features(t_real, feat_model, device=self.device)
        f_fake = clean_fid.get_batch_features(t_fake, feat_model, device=self.device)

        return calculate_fid_from_features(f_real, f_fake)
