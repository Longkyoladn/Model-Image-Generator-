"""Run final quality + real-baseline evaluation and write canonical JSON records."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.fid import evaluate_final_record, load_config, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo quality và real_baseline JSON theo TASK_ASSIGNMENT mục 4.2")
    parser.add_argument("--model", required=True, choices=["vae", "dcgan", "ddpm"])
    parser.add_argument("--dataset", required=True, choices=["celeba", "fashion_mnist", "cifar10"])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--fake", required=True, help="Thư mục PNG ảnh sinh")
    parser.add_argument("--train", required=True, help="Thư mục PNG train đã preprocess")
    parser.add_argument("--test", required=True, help="Thư mục PNG test đã preprocess")
    parser.add_argument("--resolution", required=True, type=int)
    parser.add_argument("--sampler")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--comparison", choices=["sampler"])
    parser.add_argument("--config", default="configs/eval/fid.yaml")
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="outputs/cache/fid")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    args = parser.parse_args()

    cfg, digest = load_config(args.config)
    kwargs = {k: v for k, v in (("sampler", args.sampler), ("steps", args.steps)) if v is not None}
    quality, baseline = evaluate_final_record(
        model=args.model, dataset=args.dataset, run_name=args.run_name, label=args.label,
        fake_dir=args.fake, train_dir=args.train, test_dir=args.test, cfg=cfg,
        cfg_hash=digest, resolution=args.resolution, device=args.device,
        sample_kwargs=kwargs, seed=args.seed, cache_dir=args.cache_dir,
        comparison=args.comparison,
    )
    stem = "__".join([args.dataset, args.model, args.run_name] + ([args.label] if args.label else []))
    quality_path = Path(args.metrics_dir) / "quality" / f"{stem}.json"
    baseline_path = Path(args.metrics_dir) / "real_baseline" / f"{args.dataset}.json"
    write_json(quality, quality_path)
    # Baseline chỉ cần tính/ghi một lần; chạy lại vẫn deterministic và overwrite cùng file.
    write_json(baseline, baseline_path)
    print(f"Quality: {quality_path}")
    print(f"Real baseline: {baseline_path}")


if __name__ == "__main__":
    main()
