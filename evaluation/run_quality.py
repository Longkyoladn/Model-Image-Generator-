"""Run final quality + real-baseline evaluation, custom Fashion-MNIST FID, and nearest-neighbor figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.common import reference_split_dir  # noqa: E402
from evaluation.fid import DEFAULT_FMNIST_CNN_CKPT, evaluate_final_record, load_config, write_json  # noqa: E402
from scripts.render_samples import figure_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tạo quality JSON, real_baseline JSON và ảnh nearest_neighbor.png theo TASK_ASSIGNMENT mục 2.6 & 4.2"
    )
    parser.add_argument("--model", required=True, choices=["vae", "dcgan", "ddpm"])
    parser.add_argument("--dataset", required=True, choices=["celeba", "fashion_mnist", "cifar10"])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--fake", required=True, help="Thư mục PNG ảnh sinh")
    parser.add_argument(
        "--train",
        default=None,
        help="Thư mục PNG train đã preprocess (mặc định: outputs/reference/<dataset>/train)",
    )
    parser.add_argument(
        "--test",
        default=None,
        help="Thư mục PNG test đã preprocess (mặc định: outputs/reference/<dataset>/test)",
    )
    parser.add_argument("--resolution", required=True, type=int)
    parser.add_argument("--sampler")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--comparison", choices=["sampler"])
    parser.add_argument("--config", default="configs/eval/fid.yaml")
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="outputs/cache/fid")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument(
        "--no-nearest-neighbor",
        action="store_true",
        help="Không xuất ảnh kiểm tra học thuộc nearest_neighbor.png",
    )
    parser.add_argument(
        "--fmnist-cnn-ckpt",
        default=str(DEFAULT_FMNIST_CNN_CKPT),
        help="Checkpoint CNN đặc trưng riêng cho Fashion-MNIST (mặc định: outputs/checkpoints/fashion_mnist_cnn.pt)",
    )
    args = parser.parse_args()

    train_dir = Path(args.train) if args.train else reference_split_dir(args.dataset, "train")
    test_dir = Path(args.test) if args.test else reference_split_dir(args.dataset, "test")

    nn_path: Path | None = None
    if not args.no_nearest_neighbor and args.comparison is None:
        nn_path = figure_dir(args.figures_dir, args.dataset, args.model, args.run_name, args.label) / "nearest_neighbor.png"

    cfg, digest = load_config(args.config)
    kwargs = {k: v for k, v in (("sampler", args.sampler), ("steps", args.steps)) if v is not None}
    quality, baseline = evaluate_final_record(
        model=args.model,
        dataset=args.dataset,
        run_name=args.run_name,
        label=args.label,
        fake_dir=args.fake,
        train_dir=train_dir,
        test_dir=test_dir,
        cfg=cfg,
        cfg_hash=digest,
        resolution=args.resolution,
        device=args.device,
        sample_kwargs=kwargs,
        seed=args.seed,
        cache_dir=args.cache_dir,
        comparison=args.comparison,
        nearest_neighbor_path=nn_path,
        fmnist_cnn_ckpt=args.fmnist_cnn_ckpt,
    )
    stem = "__".join([args.dataset, args.model, args.run_name] + ([args.label] if args.label else []))
    quality_path = Path(args.metrics_dir) / "quality" / f"{stem}.json"
    baseline_path = Path(args.metrics_dir) / "real_baseline" / f"{args.dataset}.json"
    write_json(quality, quality_path)
    write_json(baseline, baseline_path)
    print(f"Quality: {quality_path}")
    print(f"Real baseline: {baseline_path}")
    if nn_path is not None and nn_path.is_file():
        print(f"Nearest-neighbor figure: {nn_path}")


if __name__ == "__main__":
    main()
