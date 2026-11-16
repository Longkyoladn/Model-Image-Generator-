"""Hiệu chuẩn throughput train để quy đổi giờ GPU về T4 (mục 2.5).

Chạy cùng một config trên T4 và trên mỗi phần cứng khác đã dùng để train:

    python evaluation/calibrate.py --model ddpm --dataset cifar10 \
        --config configs/ddpm/cifar10/base.yaml

Script chạy 20 step warm-up rồi đo 200 step train, ghi throughput (ảnh/giây)
ra JSON `kind = "calibration"`. scripts/aggregate_results.py ghép các file
này theo (model, dataset, config) và tính:

    k_h        = throughput_h / throughput_T4
    GPU-giờ_T4 = Σ_h (giờ train thực tế trên h × k_h)

Batch dùng để đo là dữ liệu ngẫu nhiên đúng shape, nên thời gian đọc dữ liệu
không được tính; hệ số k_h chỉ phản ánh tốc độ tính toán.

Script in ra "device key" của máy (ví dụ "Tesla T4", "Tesla T4 x2"). Ghi đúng
chuỗi này vào trường `hardware[].device` trong metadata của run.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.interface import load_trainer  # noqa: E402


def device_key(device: torch.device, num_gpus: int = 1) -> str:
    """Tên phần cứng dùng chung giữa calibration và metadata của run."""
    if device.type != "cuda":
        return "cpu"
    name = torch.cuda.get_device_name(device)
    return name if num_gpus <= 1 else f"{name} x{num_gpus}"


def normalize_config(config: str) -> str:
    """Đường dẫn config dạng posix, tương đối với gốc repo nếu có thể."""
    path = Path(config)
    if path.exists():
        try:
            return path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            pass
    return str(config).replace("\\", "/").removeprefix("./")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def calibrate(model: str, config: str, device: str | None = None, warmup: int = 20, steps: int = 200,
              seed: int = 0, module: str | None = None) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    trainer = load_trainer(model, config, dev, module=module)
    batch = torch.rand(trainer.batch_size, *trainer.image_shape, device=dev) * 2 - 1

    for _ in range(warmup):
        trainer.train_step(batch)
    _sync(dev)
    start = time.perf_counter()
    for _ in range(steps):
        trainer.train_step(batch)
    _sync(dev)
    elapsed = time.perf_counter() - start

    num_gpus = int(getattr(trainer, "num_gpus", 1))
    return {
        "kind": "calibration",
        "model": model,
        "config": normalize_config(config),
        "device": device_key(dev, num_gpus),
        "num_gpus": num_gpus,
        "batch_size": trainer.batch_size,
        "precision": getattr(trainer, "precision", "fp32"),
        "warmup_steps": warmup,
        "measured_steps": steps,
        "elapsed_s": elapsed,
        "step_time_ms": elapsed / steps * 1000,
        "throughput_img_s": steps * trainer.batch_size / elapsed,
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "python": platform.python_version(),
        },
        "commit": _git_commit(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Hiệu chuẩn throughput train để quy đổi giờ GPU về T4.")
    p.add_argument("--model", required=True, choices=["vae", "dcgan", "ddpm"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device")
    p.add_argument("--loader-module", help="Module chứa build_trainer, chỉ dùng cho test.")
    p.add_argument("--output-dir", default="outputs/metrics/calibration")
    args = p.parse_args()

    record = calibrate(args.model, args.config, args.device, args.warmup, args.steps, args.seed, args.loader_module)
    record["dataset"] = args.dataset
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = lambda s: "".join(c if c.isalnum() or c in "-_." else "_" for c in s)  # noqa: E731
    path = out_dir / f"{args.dataset}__{args.model}__{safe(Path(record['config']).stem)}__{safe(record['device'])}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Device key: {record['device']}")
    print(f"Throughput: {record['throughput_img_s']:.1f} ảnh/s ({record['step_time_ms']:.2f} ms/step, "
          f"batch {record['batch_size']}, {record['precision']})")
    print(f"Đã ghi {path}")


if __name__ == "__main__":
    main()
