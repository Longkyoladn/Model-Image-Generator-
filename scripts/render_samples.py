"""Sinh lưới ảnh mẫu không chọn lọc cho một mô hình.

    python scripts/render_samples.py --model ddpm --checkpoint <ckpt> \
        --dataset cifar10 --run-name ddpm_base --label ddim100 --sampler ddim --steps 100

Mọi mô hình dùng cùng seed (mặc định 42) và cùng số ảnh (64, lưới 8 × 8):
ảnh nào sinh ra thì đưa vào lưới, không chọn lọc. Ảnh được phóng to bằng
nearest-neighbor để không làm mờ.

Ảnh lưu tại `outputs/figures/<dataset>/<model>__<run_name>[__<label>]/samples.png`,
là đường dẫn mà scripts/aggregate_results.py đọc để ghép ảnh so sánh.
Interpolation và nearest-neighbor của từng mô hình được lưu cùng thư mục với
tên `interpolation.png` và `nearest_neighbor.png`; dùng `save_grid` ở đây để
cùng kiểu lưới.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.interface import check_contract, load_model  # noqa: E402

FIGURE_NAMES = ("samples", "interpolation", "nearest_neighbor")


def figure_dir(root: str | Path, dataset: str, model: str, run_name: str, label: str = "") -> Path:
    name = "__".join([model, run_name] + ([label] if label else []))
    return Path(root) / dataset / name


def save_grid(images: torch.Tensor, path: str | Path, nrow: int, min_cell: int = 64, pad: int = 2) -> Path:
    """Lưu tensor (N, C, H, W) trong [0, 1] thành lưới PNG nền trắng."""
    images = images.detach().float().clamp(0, 1).cpu()
    n, c, h, w = images.shape
    if c == 1:
        images = images.repeat(1, 3, 1, 1)
    scale = max(1, -(-min_cell // h))
    ncol = nrow
    nrows = -(-n // ncol)
    cell_h, cell_w = h * scale, w * scale
    canvas = Image.new("RGB", (ncol * (cell_w + pad) + pad, nrows * (cell_h + pad) + pad), "white")
    arr = (images * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    for i in range(n):
        tile = Image.fromarray(arr[i]).resize((cell_w, cell_h), Image.NEAREST)
        r, col = divmod(i, ncol)
        canvas.paste(tile, (pad + col * (cell_w + pad), pad + r * (cell_h + pad)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Sinh lưới ảnh mẫu không chọn lọc.")
    p.add_argument("--model", required=True, choices=["vae", "dcgan", "ddpm"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--sampler")
    p.add_argument("--steps", type=int)
    p.add_argument("--num", type=int, default=64)
    p.add_argument("--nrow", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device")
    p.add_argument("--loader-module", help="Module chứa load_model, chỉ dùng cho test.")
    p.add_argument("--figures-dir", default="outputs/figures")
    args = p.parse_args()

    kwargs = {k: v for k, v in (("sampler", args.sampler), ("steps", args.steps)) if v is not None}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.checkpoint, device, module=args.loader_module)
    check_contract(model, **kwargs)
    with torch.inference_mode():
        images = model.sample(args.num, seed=args.seed, **kwargs)
    out = figure_dir(args.figures_dir, args.dataset, args.model, args.run_name, args.label) / "samples.png"
    save_grid(images, out, args.nrow)
    print(f"Đã ghi {out} ({args.num} ảnh, seed {args.seed})")


if __name__ == "__main__":
    main()
