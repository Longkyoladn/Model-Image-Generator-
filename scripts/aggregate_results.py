"""Tổng hợp kết quả thành bảng, biểu đồ và ảnh so sánh cho từng dataset.

Đọc mọi file JSON có trường "kind" trong các thư mục đầu vào (mục 4.2 của
TASK_ASSIGNMENT.md):

- kind = "run":           metadata của một lần train (mục 4.3).
- kind = "quality":       kết quả FID/KID/diversity của một bộ ảnh sinh.
- kind = "speed":         kết quả của evaluation/speed.py.
- kind = "compute":       kết quả của evaluation/flops.py (tham số, MACs, FLOPs).
- kind = "calibration":   kết quả của evaluation/calibrate.py (throughput train).
- kind = "real_baseline": số liệu thật-với-thật của dataset (FID sàn, LPIPS ảnh thật).

Các file quality/speed/compute được ghép theo (dataset, model, run_name, label)
và nối với metadata run theo (dataset, model, run_name).

File quality có "comparison": "sampler" (chỉ DDPM) là dòng phụ: không vào bảng
chính và không bị so num_gen với bảng chính, mà vào bảng "Đánh đổi sampler của
DDPM" cùng dòng DDPM chính của cùng run_name, so trên 10k ảnh đầu.

Mọi số liệu phải do script sinh ra; script tổng hợp tự tính và không nhận số
điền tay (metadata run có params_m, gmacs_per_forward, train_flops hoặc
gpu_hours_t4 thì báo lỗi):
- Tổng FLOPs train = train_steps × batch_size × train_flops_per_image.
- Thời gian - T4 = Σ_h (giờ train trên h × k_h), với k_h = throughput_h / throughput_T4
  lấy từ các file calibration cùng (dataset, model, config).

Đầu ra trong --output-dir:
- summary.md, <dataset>_quality.csv, <dataset>_cost.csv
- <dataset>_sampler.csv nếu có dòng so sánh sampler của DDPM ("comparison": "sampler")
- <dataset>_kid_vs_gpu_hours.png, <dataset>_kid_vs_train_flops.png
- <dataset>_samples.png, <dataset>_interpolation.png, <dataset>_nearest_neighbor.png

    python scripts/aggregate_results.py --inputs outputs/metrics \
        --figures-dir outputs/figures --output-dir outputs/metrics/summary

Code nằm trong package scripts/aggregate/ (xem scripts/aggregate/__init__.py).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # gốc repo, xem configs/common.py

from scripts.aggregate.fairness_checks import check_comparison_flags, run_fairness_checks  # noqa: E402
from scripts.aggregate.gpu_hours import GpuHoursConverter, build_calibration_index  # noqa: E402
from scripts.aggregate.issues import IssueLog  # noqa: E402
from scripts.aggregate.records import ResultRow, build_rows, load_records  # noqa: E402
from scripts.aggregate.report import dataset_section, summary_text  # noqa: E402


@dataclass
class PreparedDataset:
    """Các dòng của một dataset đã chia theo bảng và đã kiểm tra công bằng."""

    name: str
    main_rows: list[ResultRow]
    secondary: list[ResultRow]
    tuning: list[ResultRow]
    real_baseline: dict | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tổng hợp kết quả thành bảng, biểu đồ và ảnh so sánh.")
    p.add_argument("--inputs", nargs="+", default=["outputs/metrics"])
    p.add_argument("--figures-dir", default="outputs/figures")
    p.add_argument("--output-dir", default="outputs/metrics/summary")
    p.add_argument("--reference-device", default="T4",
                   help="Phần cứng tham chiếu cho tốc độ sinh và quy đổi thời gian train (mục 2.4, 2.5).")
    p.add_argument(
        "--stage",
        choices=["full", "all"],
        default="full",
        help="full: bảng chính chỉ gồm run có stage = full. all: đưa mọi run vào bảng chính (để chạy thử).",
    )
    p.add_argument(
        "--allow-inconsistent",
        action="store_true",
        help="Vẫn ghi bảng khi phát hiện lỗi về điều kiện đánh giá.",
    )
    return p.parse_args()


def split_by_stage(rows: list[ResultRow], stage: str, issues: IssueLog) -> tuple[list[ResultRow], list[ResultRow]]:
    """(dòng bảng chính, dòng bảng "Các config đã thử"); dòng không có metadata bị loại khỏi bảng chính."""
    if stage == "all":
        return rows, []
    main_rows = [r for r in rows if r.stage == "full"]
    tuning = [r for r in rows if r.stage in ("overfit", "tuning")]
    for r in rows:
        if r.stage is None:
            issues.warning(f"{r.who}: không có metadata nên bị loại khỏi bảng chính.")
    return main_rows, tuning


def prepare_dataset(dataset: str, records: dict[str, list[dict]], stage: str, converter: GpuHoursConverter,
                    issues: IssueLog) -> PreparedDataset:
    rows = build_rows(records, dataset, issues)
    main_rows, tuning = split_by_stage(rows, stage, issues)
    check_comparison_flags(main_rows, issues)
    secondary = [r for r in main_rows if r.is_sampler_comparison and r.model == "ddpm"]
    main_rows = [r for r in main_rows if r not in secondary]
    run_fairness_checks(main_rows, secondary, converter, issues)
    real_baseline = next((r for r in records["real_baseline"] if r.get("dataset") == dataset), None)
    return PreparedDataset(dataset, main_rows, secondary, tuning, real_baseline)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    inputs = [Path(i) for i in args.inputs]
    records, problems = load_records([i for i in inputs if i.resolve() != out_dir.resolve()])
    calibrations, calibration_problems = build_calibration_index(records["calibration"])
    issues = IssueLog()
    for message in problems + calibration_problems:
        issues.error(message)
    converter = GpuHoursConverter(calibrations, args.reference_device)

    datasets = sorted(
        {r.get("dataset") for kind in ("quality", "speed", "compute") for r in records[kind] if r.get("dataset")}
    )
    if not datasets:
        print("Không tìm thấy file quality, speed hoặc compute nào.", file=sys.stderr)
        return 1

    prepared = [prepare_dataset(d, records, args.stage, converter, issues.for_dataset(d)) for d in datasets]

    for m in issues.errors:
        print(f"LỖI: {m}", file=sys.stderr)
    if issues.errors and not args.allow_inconsistent:
        for m in issues.warnings:
            print(f"Cảnh báo: {m}", file=sys.stderr)
        print("Không ghi bảng vì có lỗi. Dùng --allow-inconsistent để vẫn ghi.", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    sections = [
        dataset_section(p.name, p.main_rows, p.secondary, p.tuning, p.real_baseline, converter, out_dir, figures_dir,
                        issues.for_dataset(p.name))
        for p in prepared
    ]
    for m in issues.warnings:
        print(f"Cảnh báo: {m}", file=sys.stderr)

    summary = out_dir / "summary.md"
    summary.write_text(summary_text(sections, issues), encoding="utf-8")
    print(f"Đã ghi {summary}")
    return 1 if issues.errors else 0


if __name__ == "__main__":
    sys.exit(main())
