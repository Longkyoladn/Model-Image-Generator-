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
import json
import shutil
import sys
from pathlib import Path

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.interface import load_model


def _prepare_clean_output_dir(out_dir: Path, overwrite: bool) -> None:
    """Đảm bảo thư mục output rỗng để tránh lẫn ảnh sinh cũ."""
    if out_dir.exists():
        if not out_dir.is_dir():
            raise NotADirectoryError(f"Đường dẫn output không phải thư mục: {out_dir}")
        if any(out_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Thư mục output {out_dir} đã tồn tại và không rỗng. "
                    "Chọn thư mục mới hoặc truyền --overwrite để làm sạch trước khi sinh ảnh."
                )
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _write_manifest(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh ảnh từ checkpoint VAE.")
    parser.add_argument("--checkpoint", required=True, help="Đường dẫn file checkpoint .pt")
    parser.add_argument("--num-samples", type=int, default=50000, help="Số lượng ảnh cần sinh")
    parser.add_argument("--batch-size", type=int, default=128, help="Kích thước batch sinh mỗi lượt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed cố định")
    parser.add_argument("--output-dir", required=True, help="Thư mục xuất ảnh PNG")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Xóa sạch nội dung cũ trong --output-dir trước khi sinh ảnh mới.",
    )
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError(f"--num-samples phải > 0, nhận {args.num_samples}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size phải > 0, nhận {args.batch_size}")

    out_dir = Path(args.output_dir)
    _prepare_clean_output_dir(out_dir, overwrite=args.overwrite)
    manifest_path = out_dir / "manifest.json"

    print(f"--- Đang tải mô hình VAE từ {args.checkpoint} trên {args.device} ---")
    model = load_model("vae", args.checkpoint, device=args.device)

    total_samples = args.num_samples
    batch_size = args.batch_size
    generated = 0
    saved_files: list[str] = []

    _write_manifest(
        manifest_path,
        {
            "model": "vae",
            "checkpoint": str(args.checkpoint),
            "num_samples": total_samples,
            "seed": args.seed,
            "image_shape": list(model.image_shape),
            "complete": False,
            "files": [],
        },
    )

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
                    arr = img_t.squeeze(0).numpy()
                    img = Image.fromarray(arr, mode="L")
                else:
                    arr = img_t.permute(1, 2, 0).numpy()
                    img = Image.fromarray(arr, mode="RGB")

                filename = f"{idx:06d}.png"
                img.save(out_dir / filename, format="PNG")
                saved_files.append(filename)

            generated += current_n
            if generated % 5000 == 0 or generated == total_samples:
                print(f" -> Đã sinh {generated}/{total_samples} ảnh...")

    if len(saved_files) != total_samples:
        raise RuntimeError(f"Số ảnh đã lưu ({len(saved_files)}) không khớp --num-samples ({total_samples}).")

    _write_manifest(
        manifest_path,
        {
            "model": "vae",
            "checkpoint": str(args.checkpoint),
            "num_samples": total_samples,
            "seed": args.seed,
            "image_shape": list(model.image_shape),
            "complete": True,
            "files": saved_files,
        },
    )
    print(f"Hoàn thành! Toàn bộ {total_samples} ảnh và manifest đã lưu tại: {out_dir}")


if __name__ == "__main__":
    main()
