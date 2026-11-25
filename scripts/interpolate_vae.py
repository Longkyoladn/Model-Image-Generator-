"""Script CLI thực hiện nội suy không gian ẩn cho mô hình VAE.

Sử dụng:
    python scripts/interpolate_vae.py \
        --checkpoint outputs/checkpoints/cifar10/vae/best.pt \
        --dataset cifar10 \
        --run-name vae_base \
        --output-dir outputs/figures \
        --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.interpolation import render_vae_interpolation_grid
from models.interface import load_model
from scripts.render_samples import figure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo lưới ảnh nội suy không gian ẩn cho VAE.")
    parser.add_argument("--checkpoint", required=True, help="Đường dẫn file checkpoint .pt")
    parser.add_argument("--dataset", required=True, choices=["cifar10", "fashion_mnist", "celeba"])
    parser.add_argument("--run-name", required=True, help="Tên run tương ứng trong metadata")
    parser.add_argument("--label", default="", help="Nhãn bổ sung nếu có")
    parser.add_argument("--output-dir", default="outputs/figures", help="Thư mục gốc lưu figure")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed1", type=int, default=42)
    parser.add_argument("--seed2", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    print(f"--- Đang tải mô hình VAE từ {args.checkpoint} trên {args.device} ---")
    model = load_model("vae", args.checkpoint, device=args.device)

    out_folder = figure_dir(args.output_dir, args.dataset, "vae", args.run_name, args.label)
    out_file = out_folder / "interpolation.png"

    # Thử lấy 2 ảnh test thật nếu có split dữ liệu
    test_images = None
    test_dir = Path("datasets/splits") / args.dataset / "test"
    if test_dir.is_dir():
        files = sorted(list(test_dir.glob("*.png")))
        if len(files) >= 2:
            from PIL import Image
            import numpy as np
            def load_img(p):
                im = Image.open(p)
                arr = np.array(im, dtype=np.float32) / 255.0
                t = torch.from_numpy(arr)
                if t.ndim == 2:
                    t = t.unsqueeze(0)
                else:
                    t = t.permute(2, 0, 1)
                return t
            test_images = (load_img(files[0]), load_img(files[1]))
            print(f"Tìm thấy ảnh test thật: {files[0].name} và {files[1].name}")

    latent_dim = getattr(model, "latent_dim", 128)
    print(f"Đang tạo chuỗi nội suy ({args.steps} bước, LERP & SLERP, seeds: {args.seed1} -> {args.seed2})...")
    path = render_vae_interpolation_grid(
        model=model,
        output_path=out_file,
        test_images=test_images,
        latent_dim=latent_dim,
        seed1=args.seed1,
        seed2=args.seed2,
        steps=args.steps,
    )
    print(f"Hoàn tất! Ảnh nội suy đã được lưu tại: {path}")


if __name__ == "__main__":
    main()
