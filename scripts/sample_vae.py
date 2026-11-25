"""Script sinh tập ảnh từ checkpoint mô hình VAE.

Sử dụng:
    python scripts/sample_vae.py \
        --checkpoint outputs/checkpoints/cifar10/vae/vae_cifar10_full/best.pt \
        --num-samples 50000 \
        --seed 42 \
        --output-dir outputs/samples/cifar10/vae/vae_base
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.interface import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh ảnh từ checkpoint VAE.")
    parser.add_argument("--checkpoint", required=True, help="Đường dẫn file checkpoint .pt")
    parser.add_argument("--num-samples", type=int, default=50000, help="Số lượng ảnh cần sinh")
    parser.add_argument("--batch-size", type=int, default=128, help="Kích thước batch sinh mỗi lượt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed cố định")
    parser.add_argument("--output-dir", required=True, help="Thư mục xuất ảnh PNG")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Đang tải mô hình VAE từ {args.checkpoint} trên {args.device} ---")
    model = load_model("vae", args.checkpoint, device=args.device)

    total_samples = args.num_samples
    batch_size = args.batch_size
    generated = 0

    print(f"Bắt đầu sinh {total_samples} ảnh vào thư mục: {out_dir}")

    with torch.inference_mode():
        while generated < total_samples:
            current_n = min(batch_size, total_samples - generated)
            batch_seed = args.seed + generated
            # sample() trả về tensor (current_n, C, H, W) trong [0, 1]
            samples = model.sample(current_n, seed=batch_seed)

            # Chuyển đổi sang uint8 PNG theo mục 2.6: round(clamp(x) * 255)
            samples_uint8 = (samples.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()

            for i in range(current_n):
                idx = generated + i
                img_t = samples_uint8[i]
                if img_t.shape[0] == 1:
                    # Grayscale
                    arr = img_t.squeeze(0).numpy()
                    img = Image.fromarray(arr, mode="L")
                else:
                    arr = img_t.permute(1, 2, 0).numpy()
                    img = Image.fromarray(arr, mode="RGB")

                file_path = out_dir / f"{idx:06d}.png"
                img.save(file_path, format="PNG")

            generated += current_n
            if generated % 5000 == 0 or generated == total_samples:
                print(f" -> Đã sinh {generated}/{total_samples} ảnh...")

    print(f"Hoàn thành! Toàn bộ {total_samples} ảnh đã lưu tại: {out_dir}")


if __name__ == "__main__":
    main()
