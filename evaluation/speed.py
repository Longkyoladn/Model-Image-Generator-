"""Benchmark tốc độ sinh ảnh cho VAE, DCGAN và DDPM.

Cách dùng:

    # Một mô hình
    python evaluation/speed.py --model ddpm --checkpoint <ckpt> \
        --dataset cifar10 --run-name base --label ddim100 \
        --sampler ddim --steps 100 --batch-sizes 1 64 256

    # Nhiều mô hình, đo xen kẽ trong cùng một phiên (khuyến nghị cho bảng chính)
    python evaluation/speed.py --spec configs/eval/speed_example.yaml

Quy ước đo:
- Thời gian tính từ lúc gọi `model.sample(...)` đến khi có tensor ảnh trên
  device. Không tính load checkpoint, chuyển về CPU, đổi sang uint8, lưu file.
- Đồng bộ GPU trước và sau mỗi lần đo.
- Warm-up trước khi đo; các mô hình được đo xen kẽ theo từng vòng lặp.
- `warmup`, `repeats` đặt chung ở spec; mỗi target có thể ghi đè (DDPM: warmup 1,
  repeats ít hơn vì mỗi lần đo đã dài vài phút).
- Target cùng checkpoint (ví dụ ddim100 và ddpm500) dùng chung một bản mô hình.
- File JSON được ghi lại sau mỗi vòng đo (`complete = false` cho tới khi đo đủ),
  nên phiên Colab/Kaggle bị ngắt vẫn còn số liệu các vòng đã xong.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.interface import check_contract, load_model  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_info(device: torch.device) -> dict:
    info = {
        "device": device.type,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
    }
    return info


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def _timed_sample(model, batch_size, seed, device, precision, sample_kwargs):
    """Trả về (thời gian giây, peak memory MB hoặc None)."""
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
    _sync(device)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=precision == "fp16"
    ):
        images = model.sample(batch_size, seed=seed, **sample_kwargs)
    _sync(device)
    elapsed = time.perf_counter() - start
    memory_mb = None
    if use_cuda:
        # Các mô hình khác cũng đang nằm trên GPU, nên lấy phần tăng thêm trong
        # lúc sinh cộng với trọng số của chính mô hình này.
        peak = torch.cuda.max_memory_allocated(device)
        memory_mb = (peak - base + model.weight_bytes()) / 2**20
    del images
    return elapsed, memory_mb


def _is_oom(err: BaseException) -> bool:
    return "out of memory" in str(err).lower()


def _model_key(raw: dict) -> tuple:
    """Các target cùng (mô hình, checkpoint, loader) dùng chung một bản mô hình trên device."""
    return (raw["model"], str(Path(raw["checkpoint"]).resolve()), raw.get("loader_module"))


def _target_complete(target: dict) -> bool:
    return all(
        b in target["oom"] or len(target["times"][b]) >= target["repeats"] for b in target["times"]
    )


def _speed_record(target: dict, batch_sizes: list[int], precision: str, env: dict, commit: str | None,
                  base_seed: int) -> dict:
    model = target["model"]
    results = []
    for b in batch_sizes:
        times = target["times"][b]
        if b in target["oom"] or not times:
            results.append({"batch_size": b, "oom": b in target["oom"], "n_repeats": 0})
            continue
        mean, std = _mean_std(times)
        mem = target["memory"][b]
        results.append(
            {
                "batch_size": b,
                "oom": False,
                "n_repeats": len(times),
                "time_per_batch_s_mean": mean,
                "time_per_batch_s_std": std,
                "time_per_image_ms": mean / b * 1000,
                "throughput_img_s": b / mean,
                "peak_memory_mb": max(mem) if mem else None,
            }
        )
    return {
        "kind": "speed",
        "model": target["model_name"],
        "dataset": target["dataset"],
        "run_name": target["run_name"],
        "label": target["label"],
        "checkpoint": target["checkpoint"].name,
        "checkpoint_mb": target["checkpoint"].stat().st_size / 2**20,
        "load_time_s": target["load_time_s"],
        "model_shared": target["model_shared"],
        "sample_kwargs": target["sample_kwargs"],
        "forward_passes_per_image": model.forward_passes_per_image(**target["sample_kwargs"]),
        "image_shape": list(model.image_shape),
        "params_m": model.num_parameters() / 1e6,
        "precision": precision,
        "warmup": target["warmup"],
        "repeats_requested": target["repeats"],
        # False khi phiên bị ngắt giữa chừng: số liệu là của các vòng đã đo xong.
        "complete": _target_complete(target),
        "seed": base_seed,
        "env": env,
        "commit": commit,
        "results": results,
    }


def _write_json(path: Path, record: dict) -> None:
    """Ghi qua file tạm rồi đổi tên, để file không bị hỏng nếu phiên bị ngắt lúc đang ghi."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def run_benchmark(spec: dict, output_dir: str | Path, device: str | None = None,
                  on_round=None) -> list[Path]:
    """Chạy benchmark theo spec và ghi một file JSON cho mỗi target.

    - `warmup` và `repeats` đặt chung ở spec, mỗi target có thể ghi đè riêng
      (DDPM chỉ cần warmup 1 vì một lần sinh đã đủ lâu để GPU ổn định).
    - Target cùng checkpoint dùng chung một bản mô hình (`model_shared = true`).
    - File của mỗi target được ghi lại sau mỗi vòng đo, với `complete = false`
      cho tới khi đo đủ; phiên bị ngắt thì vẫn còn số liệu các vòng đã xong.
    - `on_round(r, paths)` (tuỳ chọn) được gọi sau khi ghi file của vòng r.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    precision = spec.get("precision", "fp32")
    if precision not in ("fp32", "fp16"):
        raise ValueError("precision phải là fp32 hoặc fp16")
    if precision == "fp16" and dev.type != "cuda":
        raise ValueError("fp16 chỉ đo được trên GPU")
    batch_sizes = [int(b) for b in spec.get("batch_sizes", [1, 64])]
    default_warmup = int(spec.get("warmup", 5))
    default_repeats = int(spec.get("repeats", 20))
    base_seed = int(spec.get("seed", 0))
    skip_check = bool(spec.get("skip_contract_check", False))

    loaded: dict[tuple, tuple] = {}   # _model_key -> (mô hình, thời gian load)
    targets = []
    for raw in spec["targets"]:
        kwargs = dict(raw.get("sample_kwargs") or {})
        ckpt = Path(raw["checkpoint"])
        key = _model_key(raw)
        shared = key in loaded
        if not shared:
            start = time.perf_counter()
            model = load_model(raw["model"], ckpt, dev, module=raw.get("loader_module"))
            _sync(dev)
            loaded[key] = (model, time.perf_counter() - start)
        model, load_time = loaded[key]
        if not skip_check:
            check_contract(model, **kwargs)
        repeats = int(raw.get("repeats", default_repeats))
        warmup = int(raw.get("warmup", default_warmup))
        if repeats < 1 or warmup < 0:
            raise ValueError(f"{raw['model']} {raw.get('label', '')}: repeats phải ≥ 1, warmup phải ≥ 0")
        targets.append(
            {
                "model_name": raw["model"],
                "dataset": raw.get("dataset", spec.get("dataset")),
                "run_name": raw.get("run_name", ckpt.stem),
                "label": raw.get("label", ""),
                "checkpoint": ckpt,
                "sample_kwargs": kwargs,
                "repeats": repeats,
                "warmup": warmup,
                "model": model,
                "model_shared": shared,
                "load_time_s": load_time,
                "times": {b: [] for b in batch_sizes},
                "memory": {b: [] for b in batch_sizes},
                "oom": set(),
            }
        )

    def measure(target, batch_size, seed, record):
        if batch_size in target["oom"]:
            return
        try:
            elapsed, mem = _timed_sample(
                target["model"], batch_size, seed, dev, precision, target["sample_kwargs"]
            )
        except RuntimeError as err:
            if not _is_oom(err):
                raise
            target["oom"].add(batch_size)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            return
        if record:
            target["times"][batch_size].append(elapsed)
            if mem is not None:
                target["memory"][batch_size].append(mem)

    env = environment_info(dev)
    commit = _git_commit()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        parts = [str(target["dataset"]), target["model_name"], target["run_name"]]
        if target["label"]:
            parts.append(target["label"])
        target["path"] = out_dir / ("__".join(parts) + ".json")

    def save(which) -> list[Path]:
        for target in which:
            _write_json(target["path"], _speed_record(target, batch_sizes, precision, env, commit, base_seed))
        return [t["path"] for t in which]

    for target in targets:
        for b in batch_sizes:
            for i in range(target["warmup"]):
                measure(target, b, base_seed + i, record=False)

    # Vòng lặp ngoài là số lần đo, vòng trong là mô hình: nhiễu của phần cứng
    # (throttle, GPU dùng chung) được chia đều cho các mô hình.
    for r in range(max(t["repeats"] for t in targets)):
        active = [t for t in targets if r < t["repeats"]]
        for target in active:
            for b in batch_sizes:
                measure(target, b, base_seed + r, record=True)
        paths = save(active)
        if on_round is not None:
            on_round(r, paths)

    written = save(targets)
    for target in targets:
        title = f"{target['model_name']} {target['label']}".strip()
        shared = ", dùng chung mô hình với target trước" if target["model_shared"] else ""
        print(f"\n{title} ({target['run_name']}) trên {env['gpu'] or env['device']}, {precision}, "
              f"warmup {target['warmup']}, {target['repeats']} lần đo{shared}")
        for res in _speed_record(target, batch_sizes, precision, env, commit, base_seed)["results"]:
            if res["oom"]:
                print(f"  batch {res['batch_size']:>5}: hết bộ nhớ")
                continue
            mem = res["peak_memory_mb"]
            print(
                f"  batch {res['batch_size']:>5}: "
                f"{res['time_per_batch_s_mean'] * 1000:10.3f} ± "
                f"{res['time_per_batch_s_std'] * 1000:.3f} ms/batch, "
                f"{res['throughput_img_s']:10.1f} ảnh/s"
                + (f", {mem:.0f} MB" if mem is not None else "")
            )
    return written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark tốc độ sinh ảnh.")
    p.add_argument("--spec", help="File YAML liệt kê nhiều target để đo xen kẽ.")
    p.add_argument("--model", choices=["vae", "dcgan", "ddpm"])
    p.add_argument("--checkpoint")
    p.add_argument("--dataset")
    p.add_argument("--run-name")
    p.add_argument("--label", default="", help="Tên biến thể, ví dụ ddim100.")
    p.add_argument("--sampler", help="Chỉ dùng cho DDPM.")
    p.add_argument("--steps", type=int, help="Số sampling step, chỉ dùng cho DDPM.")
    p.add_argument("--batch-sizes", type=int, nargs="+")
    p.add_argument("--warmup", type=int)
    p.add_argument("--repeats", type=int)
    p.add_argument("--precision", choices=["fp32", "fp16"])
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--output-dir", default="outputs/metrics/speed")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.spec:
        import yaml

        spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    else:
        if not (args.model and args.checkpoint and args.dataset):
            sys.exit("Cần --spec, hoặc đủ --model, --checkpoint và --dataset.")
        target = {"model": args.model, "checkpoint": args.checkpoint, "label": args.label}
        if args.run_name:
            target["run_name"] = args.run_name
        kwargs = {}
        if args.sampler:
            kwargs["sampler"] = args.sampler
        if args.steps:
            kwargs["steps"] = args.steps
        target["sample_kwargs"] = kwargs
        spec = {"dataset": args.dataset, "targets": [target]}
    for key in ("batch_sizes", "warmup", "repeats", "precision", "seed"):
        value = getattr(args, key)
        if value is not None:
            spec[key] = value
    for path in run_benchmark(spec, args.output_dir, args.device):
        print(f"Đã ghi {path}")


if __name__ == "__main__":
    main()
