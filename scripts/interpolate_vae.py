"""Script CLI thực hiện nội suy không gian ẩn cho mô hình VAE.

Sử dụng:
    python scripts/interpolate_vae.py \
        --checkpoint outputs/checkpoints/cifar10/vae/vae_cifar10_full/best.pt \
        --dataset cifar10 \
        --run-name vae_cifar10_full \
        --output-dir outputs/figures \
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.common import reference_split_dir
from evaluation.interpolation import render_vae_interpolation_grid
from models.interface import load_model
from scripts.render_samples import figure_dir


def _load_image_for_model(path: Path, image_shape: tuple[int, int, int]) -> torch.Tensor:
    """Đọc ảnh PNG và chuẩn hóa về đúng số kênh (1 hoặc 3) và kích thước (H, W) của mô hình."""
    channels, height, width = image_shape
    mode = "L" if channels == 1 else "RGB"
    with Image.open(path) as im:
        img = im.convert(mode)
        if img.size != (width, height):
            raise ValueError(
                f"Kích thước ảnh test {path} là {img.size[::-1]}, nhưng mô hình cần {(height, width)}."
            )
        arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)
    if t.ndim == 2:
        return t.unsqueeze(0)
    return t.permute(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo lưới ảnh nội suy không gian ẩn cho VAE.")
    parser.add_argument("--checkpoint", required=True, help="Đường dẫn file checkpoint .pt")
    parser.add_argument("--dataset", required=True, choices=["cifar10", "fashion_mnist", "celeba"])
    parser.add_argument("--run-name", required=True, help="Tên run tương ứng trong metadata")
    parser.add_argument("--label", default="", help="Nhãn bổ sung nếu có")
    parser.add_argument("--output-dir", default="outputs/figures", help="Thư mục gốc lưu figure")
    parser.add_argument(
        "--test-dir",
        default=None,
        help="Thư mục ảnh test thật tham chiếu (mặc định: outputs/reference/<dataset>/test).",
    )
    parser.add_argument(
        "--test-indices",
        type=int,
        nargs=2,
        default=[0, 1],
        help="Hai chỉ số ảnh trong --test-dir dùng để nội suy qua mu_1, mu_2 (mặc định: 0 1).",
    )
    parser.add_argument(
        "--prior-only",
        action="store_true",
        help="Chỉ nội suy prior z (bỏ qua nội suy giữa 2 ảnh test thật).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed1", type=int, default=42)
    parser.add_argument("--seed2", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    print(f"--- Đang tải mô hình VAE từ {args.checkpoint} trên {args.device} ---")
    model = load_model("vae", args.checkpoint, device=args.device)

    out_folder = figure_dir(args.output_dir, args.dataset, "vae", args.run_name, args.label)
    out_file = out_folder / "interpolation.png"

    test_images = None
    test_names = None
    if not args.prior_only:
        test_dir = Path(args.test_dir) if args.test_dir else reference_split_dir(args.dataset, "test")
        if not test_dir.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục ảnh test thật tham chiếu: {test_dir}. "
                "Hãy chạy `python scripts/export_reference_images.py --dataset "
                f"{args.dataset} --split test`, truyền --test-dir hợp lệ, hoặc dùng --prior-only."
            )
        files = sorted(test_dir.glob("*.png"))
        idx1, idx2 = args.test_indices
        if len(files) < 2 or max(idx1, idx2) >= len(files) or min(idx1, idx2) < 0:
            raise ValueError(
                f"Thư mục {test_dir} có {len(files)} ảnh PNG, không đủ cho chỉ số test {args.test_indices}."
            )
        img1 = _load_image_for_model(files[idx1], model.image_shape)
        img2 = _load_image_for_model(files[idx2], model.image_shape)
        test_images = (img1, img2)
        test_names = (files[idx1].name, files[idx2].name)
        print(f"Sử dụng 2 ảnh test thật từ {test_dir}: {test_names[0]} (idx={idx1}) và {test_names[1]} (idx={idx2})")

    latent_dim = getattr(model, "latent_dim", 128)
    print(f"Đang tạo chuỗi nội suy ({args.steps} bước, LERP & SLERP, seeds: {args.seed1} -> {args.seed2})...")
    path = render_vae_interpolation_grid(
        model=model,
        output_path=out_file,
        test_images=test_images,
        test_endpoint_names=test_names,
        latent_dim=latent_dim,
        seed1=args.seed1,
        seed2=args.seed2,
        steps=args.steps,
    )
    print(f"Hoàn tất! Ảnh nội suy và metadata hai đầu mút đã lưu tại: {path.parent}")


if __name__ == "__main__":
    main()
