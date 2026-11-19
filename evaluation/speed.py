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
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # gốc repo, xem evaluation/common.py

from evaluation.common import git_commit, sync  # noqa: E402
from models.interface import GenerativeModel, check_contract, load_model  # noqa: E402


def environment_info(device: torch.device) -> dict:
    return {
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


def mean_and_std(values: list[float]) -> tuple[float, float]:
    """Trung bình và độ lệch chuẩn mẫu (chia n - 1); một giá trị thì std = 0."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def is_out_of_memory(err: BaseException) -> bool:
    return "out of memory" in str(err).lower()


def write_json_atomically(path: Path, record: dict) -> None:
    """Ghi qua file tạm rồi đổi tên, để file không bị hỏng nếu phiên bị ngắt lúc đang ghi."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


@dataclass
class BenchmarkSettings:
    """Thiết lập chung cho mọi target của một lần chạy benchmark."""

    device: torch.device
    precision: str
    batch_sizes: list[int]
    warmup: int
    repeats: int
    seed: int
    skip_contract_check: bool

    @classmethod
    def from_spec(cls, spec: dict, device: str | None = None) -> BenchmarkSettings:
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        precision = spec.get("precision", "fp32")
        if precision not in ("fp32", "fp16"):
            raise ValueError("precision phải là fp32 hoặc fp16")
        if precision == "fp16" and dev.type != "cuda":
            raise ValueError("fp16 chỉ đo được trên GPU")
        return cls(
            device=dev,
            precision=precision,
            batch_sizes=[int(b) for b in spec.get("batch_sizes", [1, 64])],
            warmup=int(spec.get("warmup", 5)),
            repeats=int(spec.get("repeats", 20)),
            seed=int(spec.get("seed", 0)),
            skip_contract_check=bool(spec.get("skip_contract_check", False)),
        )


def time_one_batch(model: GenerativeModel, batch_size: int, seed: int, settings: BenchmarkSettings,
                   sample_kwargs: dict) -> tuple[float, float | None]:
    """Đo thời gian sinh một batch. Trả về (số giây, peak memory MB hoặc None trên CPU)."""
    device = settings.device
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
    sync(device)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=settings.precision == "fp16"
    ):
        images = model.sample(batch_size, seed=seed, **sample_kwargs)
    sync(device)
    elapsed = time.perf_counter() - start
    memory_mb = None
    if use_cuda:
        # Các mô hình khác cũng đang nằm trên GPU, nên lấy phần tăng thêm trong
        # lúc sinh cộng với trọng số của chính mô hình này.
        peak = torch.cuda.max_memory_allocated(device)
        memory_mb = (peak - base + model.weight_bytes()) / 2**20
    del images
    return elapsed, memory_mb


@dataclass
class SpeedTarget:
    """Một dòng cần đo (mô hình + biến thể sampler) và các số đo đã có."""

    model_name: str
    dataset: str
    run_name: str
    label: str
    checkpoint: Path
    sample_kwargs: dict
    warmup: int
    repeats: int
    model: GenerativeModel
    model_shared: bool
    load_time_s: float
    times: dict[int, list[float]] = field(default_factory=dict)
    memory: dict[int, list[float]] = field(default_factory=dict)
    oom_batches: set[int] = field(default_factory=set)
    path: Path | None = None

    def output_name(self) -> str:
        """Tên file kết quả: <dataset>__<model>__<run_name>[__<label>].json"""
        parts = [str(self.dataset), self.model_name, self.run_name] + ([self.label] if self.label else [])
        return "__".join(parts) + ".json"

    def measure(self, batch_size: int, seed: int, settings: BenchmarkSettings, record: bool) -> None:
        """Sinh một batch; `record = False` là lượt warm-up, không lưu số đo."""
        if batch_size in self.oom_batches:
            return
        try:
            elapsed, memory_mb = time_one_batch(self.model, batch_size, seed, settings, self.sample_kwargs)
        except RuntimeError as err:
            if not is_out_of_memory(err):
                raise
            self.oom_batches.add(batch_size)
            if settings.device.type == "cuda":
                torch.cuda.empty_cache()
            return
        if record:
            self.times[batch_size].append(elapsed)
            if memory_mb is not None:
                self.memory[batch_size].append(memory_mb)

    def is_complete(self) -> bool:
        """Mọi batch đã đo đủ số lần, hoặc bị hết bộ nhớ."""
        return all(b in self.oom_batches or len(t) >= self.repeats for b, t in self.times.items())

    def batch_results(self, batch_sizes: list[int]) -> list[dict]:
        results = []
        for b in batch_sizes:
            times = self.times[b]
            if b in self.oom_batches or not times:
                results.append({"batch_size": b, "oom": b in self.oom_batches, "n_repeats": 0})
                continue
            mean, std = mean_and_std(times)
            memory = self.memory[b]
            results.append({
                "batch_size": b,
                "oom": False,
                "n_repeats": len(times),
                "time_per_batch_s_mean": mean,
                "time_per_batch_s_std": std,
                "time_per_image_ms": mean / b * 1000,
                "throughput_img_s": b / mean,
                "peak_memory_mb": max(memory) if memory else None,
            })
        return results

    def to_record(self, settings: BenchmarkSettings, env: dict, commit: str | None) -> dict:
        """Nội dung file JSON kind = "speed"."""
        return {
            "kind": "speed",
            "model": self.model_name,
            "dataset": self.dataset,
            "run_name": self.run_name,
            "label": self.label,
            "checkpoint": self.checkpoint.name,
            "checkpoint_mb": self.checkpoint.stat().st_size / 2**20,
            "load_time_s": self.load_time_s,
            "model_shared": self.model_shared,
            "sample_kwargs": self.sample_kwargs,
            "forward_passes_per_image": self.model.forward_passes_per_image(**self.sample_kwargs),
            "image_shape": list(self.model.image_shape),
            "params_m": self.model.num_parameters() / 1e6,
            "precision": settings.precision,
            "warmup": self.warmup,
            "repeats_requested": self.repeats,
            # False khi phiên bị ngắt giữa chừng: số liệu là của các vòng đã đo xong.
            "complete": self.is_complete(),
            "seed": settings.seed,
            "env": env,
            "commit": commit,
            "results": self.batch_results(settings.batch_sizes),
        }


def shared_model_key(raw: dict) -> tuple:
    """Các target cùng (mô hình, checkpoint, loader) dùng chung một bản mô hình trên device."""
    return (raw["model"], str(Path(raw["checkpoint"]).resolve()), raw.get("loader_module"))


def load_targets(spec: dict, settings: BenchmarkSettings) -> list[SpeedTarget]:
    """Load mô hình cho từng target của spec; target cùng checkpoint dùng chung một bản."""
    loaded: dict[tuple, tuple[GenerativeModel, float]] = {}  # shared_model_key -> (mô hình, thời gian load)
    targets = []
    for raw in spec["targets"]:
        kwargs = dict(raw.get("sample_kwargs") or {})
        checkpoint = Path(raw["checkpoint"])
        key = shared_model_key(raw)
        shared = key in loaded
        if not shared:
            start = time.perf_counter()
            model = load_model(raw["model"], checkpoint, settings.device, module=raw.get("loader_module"))
            sync(settings.device)
            loaded[key] = (model, time.perf_counter() - start)
        model, load_time = loaded[key]
        if not settings.skip_contract_check:
            check_contract(model, **kwargs)
        repeats = int(raw.get("repeats", settings.repeats))
        warmup = int(raw.get("warmup", settings.warmup))
        if repeats < 1 or warmup < 0:
            raise ValueError(f"{raw['model']} {raw.get('label', '')}: repeats phải ≥ 1, warmup phải ≥ 0")
        targets.append(SpeedTarget(
            model_name=raw["model"],
            dataset=raw.get("dataset", spec.get("dataset")),
            run_name=raw.get("run_name", checkpoint.stem),
            label=raw.get("label", ""),
            checkpoint=checkpoint,
            sample_kwargs=kwargs,
            warmup=warmup,
            repeats=repeats,
            model=model,
            model_shared=shared,
            load_time_s=load_time,
            times={b: [] for b in settings.batch_sizes},
            memory={b: [] for b in settings.batch_sizes},
        ))
    return targets


def print_target_summary(target: SpeedTarget, settings: BenchmarkSettings, env: dict, commit: str | None) -> None:
    title = f"{target.model_name} {target.label}".strip()
    shared = ", dùng chung mô hình với target trước" if target.model_shared else ""
    print(f"\n{title} ({target.run_name}) trên {env['gpu'] or env['device']}, {settings.precision}, "
          f"warmup {target.warmup}, {target.repeats} lần đo{shared}")
    for res in target.to_record(settings, env, commit)["results"]:
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
    settings = BenchmarkSettings.from_spec(spec, device)
    targets = load_targets(spec, settings)
    env = environment_info(settings.device)
    commit = git_commit()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        target.path = out_dir / target.output_name()

    def save(which: list[SpeedTarget]) -> list[Path]:
        for target in which:
            write_json_atomically(target.path, target.to_record(settings, env, commit))
        return [t.path for t in which]

    for target in targets:
        for b in settings.batch_sizes:
            for i in range(target.warmup):
                target.measure(b, settings.seed + i, settings, record=False)

    # Vòng lặp ngoài là số lần đo, vòng trong là mô hình: nhiễu của phần cứng
    # (throttle, GPU dùng chung) được chia đều cho các mô hình.
    for r in range(max(t.repeats for t in targets)):
        active = [t for t in targets if r < t.repeats]
        for target in active:
            for b in settings.batch_sizes:
                target.measure(b, settings.seed + r, settings, record=True)
        paths = save(active)
        if on_round is not None:
            on_round(r, paths)

    written = save(targets)
    for target in targets:
        print_target_summary(target, settings, env, commit)
    return written


def parse_args() -> argparse.Namespace:
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


def spec_from_args(args: argparse.Namespace) -> dict:
    """Spec từ file YAML (--spec) hoặc từ các cờ của một mô hình; cờ dòng lệnh ghi đè spec."""
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
    return spec


def main() -> None:
    args = parse_args()
    for path in run_benchmark(spec_from_args(args), args.output_dir, args.device):
        print(f"Đã ghi {path}")


if __name__ == "__main__":
    main()
