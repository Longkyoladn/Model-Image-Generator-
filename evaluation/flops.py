"""Đo số tham số, MACs/FLOPs mỗi ảnh sinh và FLOPs train mỗi ảnh (mục 2.5).

    python evaluation/flops.py --model ddpm --checkpoint <ckpt> --dataset cifar10 \
        --run-name ddpm_base --label ddim100 --sampler ddim --steps 100 --tool fvcore

Quy ước:
- MACs đo trên input batch = 1 do `example_inputs()` của mô hình cung cấp.
- Công cụ: fvcore hoặc thop (theo kế hoạch). Cả hai đều trả về MACs
  (một phép nhân-cộng = 1). FLOPs = 2 × MACs.
- Chi phí sinh một ảnh = MACs của module sinh × forward_passes_per_image.
- FLOPs train mỗi ảnh theo công thức mục 2.5 (backward ≈ 2 × forward):
    VAE:   3 × (F_encoder + F_decoder)
    DCGAN: k × (F_G + 6 F_D) + 3 F_G + 3 F_D, với k = số bước D mỗi bước G
           (k = 1 cho ra 4 F_G + 9 F_D)
    DDPM:  3 × F_unet
  Tổng FLOPs train = số step × batch size × FLOPs train mỗi ảnh; phần nhân
  này do scripts/aggregate_results.py làm từ metadata của run.

Kết quả ghi ra JSON `kind = "compute"`.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.interface import SAMPLING_ROLE, check_roles, load_model  # noqa: E402


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _tool_version(tool: str) -> str:
    try:
        return version(tool)
    except PackageNotFoundError as err:
        raise SystemExit(f"Chưa cài {tool}. Chạy: pip install {tool}") from err


def count_macs(module: torch.nn.Module, inputs: tuple, tool: str) -> tuple[int, list[str]]:
    """Trả về (MACs, danh sách op không được đếm)."""
    module = copy.deepcopy(module).eval()
    with torch.no_grad():
        if tool == "fvcore":
            from fvcore.nn import FlopCountAnalysis

            analysis = FlopCountAnalysis(module, inputs)
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)
            total = int(analysis.total())
            unsupported = sorted(analysis.unsupported_ops().keys())
            return total, unsupported
        if tool == "thop":
            from thop import profile

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                macs, _ = profile(module, inputs=inputs, verbose=False)
            return int(macs), []
    raise ValueError(f"Công cụ không hợp lệ: {tool}")


def train_macs_per_image(model_name: str, macs: dict[str, int], d_steps: int) -> tuple[int, str]:
    if model_name == "vae":
        return 3 * (macs["encoder"] + macs["decoder"]), "3 × (F_encoder + F_decoder)"
    if model_name == "dcgan":
        g, d = macs["generator"], macs["discriminator"]
        return d_steps * (g + 6 * d) + 3 * g + 3 * d, f"{d_steps} × (F_G + 6 F_D) + 3 F_G + 3 F_D"
    if model_name == "ddpm":
        return 3 * macs["unet"], "3 × F_unet"
    raise ValueError(model_name)


def measure(model, tool: str, sample_kwargs: dict, d_steps: int = 1) -> dict:
    check_roles(model)
    inputs = model.example_inputs()
    modules = model.training_modules()
    missing = sorted(set(modules) - set(inputs))
    if missing:
        raise ValueError(f"example_inputs() thiếu input cho {missing}")

    macs, unsupported = {}, {}
    for name, module in modules.items():
        macs[name], ops = count_macs(module, tuple(inputs[name]), tool)
        if ops:
            unsupported[name] = ops

    role = SAMPLING_ROLE[model.name]
    passes = model.forward_passes_per_image(**sample_kwargs)
    train_macs, formula = train_macs_per_image(model.name, macs, d_steps)
    return {
        "params_m": model.num_parameters() / 1e6,
        "params_train_m": model.num_training_parameters() / 1e6,
        "gmacs_by_module": {k: v / 1e9 for k, v in macs.items()},
        "gmacs_per_forward": macs[role] / 1e9,
        "forward_passes_per_image": passes,
        "gmacs_per_image": macs[role] * passes / 1e9,
        "gflops_per_image": 2 * macs[role] * passes / 1e9,
        "train_flops_per_image": 2 * train_macs,
        "train_formula": formula,
        "d_steps": d_steps if model.name == "dcgan" else None,
        "unsupported_ops": unsupported,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Đo tham số và MACs/FLOPs.")
    p.add_argument("--model", required=True, choices=["vae", "dcgan", "ddpm"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--label", default="", help="Biến thể sampler, ví dụ ddim100. Trùng label với file speed/quality.")
    p.add_argument("--sampler")
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float, help="Guidance scale nếu dùng CFG (Conditional DDPM).")
    p.add_argument("--d-steps", type=int, default=1, help="Số bước D mỗi bước G (DCGAN).")
    p.add_argument("--tool", choices=["fvcore", "thop"], default="fvcore")
    p.add_argument("--loader-module", help="Module chứa load_model, chỉ dùng cho test.")
    p.add_argument("--output-dir", default="outputs/metrics/compute")
    args = p.parse_args()

    kwargs = {k: v for k, v in (("sampler", args.sampler), ("steps", args.steps), ("guidance", args.guidance)) if v is not None}
    tool_version = _tool_version(args.tool)
    model = load_model(args.model, args.checkpoint, "cpu", module=args.loader_module)
    result = measure(model, args.tool, kwargs, args.d_steps)
    record = {
        "kind": "compute",
        "model": args.model,
        "dataset": args.dataset,
        "run_name": args.run_name,
        "label": args.label,
        "checkpoint": Path(args.checkpoint).name,
        "sample_kwargs": kwargs,
        "tool": args.tool,
        "tool_version": tool_version,
        "macs_convention": "1 MAC = 1 phép nhân-cộng; FLOPs = 2 × MACs",
        "commit": _git_commit(),
        **result,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = [args.dataset, args.model, args.run_name] + ([args.label] if args.label else [])
    path = out_dir / ("__".join(parts) + ".json")
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{args.model} ({args.run_name}{', ' + args.label if args.label else ''}), đo bằng {args.tool} {tool_version}")
    print(f"  Tham số khi sinh:  {result['params_m']:.3f} M; khi train: {result['params_train_m']:.3f} M")
    for name, g in result["gmacs_by_module"].items():
        print(f"  {name:<14} {g:.4f} GMACs/forward")
    print(f"  Sinh một ảnh:      {result['gmacs_per_image']:.4f} GMACs = {result['gflops_per_image']:.4f} GFLOPs "
          f"({result['forward_passes_per_image']} forward)")
    print(f"  Train một ảnh:     {result['train_flops_per_image']:.3e} FLOPs ({result['train_formula']})")
    if result["unsupported_ops"]:
        print(f"  Op không được đếm: {result['unsupported_ops']}")
    print(f"Đã ghi {path}")


if __name__ == "__main__":
    main()
