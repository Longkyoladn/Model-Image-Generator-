"""Bảng markdown, file CSV và nội dung summary.md."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.aggregate.charts import plot_kid_curves
from scripts.aggregate.constants import COST_COLUMNS, FIGURE_TITLES, QUALITY_COLUMNS, SAMPLER_COMPARE_NUM_GEN
from scripts.aggregate.figure_grids import combine_model_figures
from scripts.aggregate.gpu_hours import GpuHoursConverter
from scripts.aggregate.issues import IssueLog
from scripts.aggregate.records import ResultRow
from scripts.aggregate.table_values import cost_values, format_value, quality_values, rows_to_bold, times_1000


def markdown_table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def write_csv(path: Path, header: list[str], body: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(body)


def value_with_std(value, std, digits) -> str:
    text = format_value(value, digits)
    if value is not None and std is not None:
        text += f" ± {format_value(std, digits)}"
    return text


def quality_section(dataset: str, rows: list[ResultRow], real_baseline: dict | None, out_dir: Path) -> list[str]:
    values = [quality_values(r) for r in rows]
    bold = {key: rows_to_bold(values, key, d) for key, _, _, d in QUALITY_COLUMNS if d}
    body, raw = [], []
    for i, (row, v) in enumerate(zip(rows, values)):
        cells = [row.name, row.run_name or ""]
        for key, _, digits, direction in QUALITY_COLUMNS:
            text = value_with_std(v[key], v.get(key + "_std"), digits)
            if direction and i in bold[key]:
                text = f"**{text}**"
            cells.append(text)
        body.append(cells)
        raw.append([row.name, row.run_name or ""] + list(v.values()))
    header = ["Mô hình", "Run"] + [title for _, title, _, _ in QUALITY_COLUMNS]
    parts = [
        "### Chất lượng",
        markdown_table(header, body),
        "In đậm: giá trị tốt nhất. Với các cột có ± std, những mô hình chênh giá trị tốt nhất dưới 2 × std được coi là ngang nhau."
        " LPIPS div. không in đậm: giá trị càng gần LPIPS của ảnh thật càng tốt.",
    ]
    if real_baseline:
        text = ", ".join(f"{k} = {format_value(v, 3)}" for k, v in (real_baseline.get("metrics") or {}).items())
        parts.append(f"Mức tham chiếu thật-với-thật: {text}.")
    if values:
        write_csv(out_dir / f"{dataset}_quality.csv", ["model", "run_name"] + list(values[0].keys()), raw)
    return parts


def cost_section(dataset: str, rows: list[ResultRow], converter: GpuHoursConverter, out_dir: Path) -> list[str]:
    reference = converter.reference
    values = [cost_values(r, converter) for r in rows]
    body = [[r.name, r.run_name or ""] + [format_value(v[key], digits) for key, _, digits in COST_COLUMNS]
            for r, v in zip(rows, values)]
    header = ["Mô hình", "Run"] + [
        f"Thời gian - {reference} (giờ)" if key == "gpu_hours_t4" else title for key, title, _ in COST_COLUMNS
    ]
    notes = [f"Thời gian - {reference}: giờ train trên mọi phần cứng được quy đổi về {reference} theo hệ số "
             f"k_h = throughput_h / throughput_{reference} từ file calibration (mục 2.5)."]
    speed = next((r.speed for r in rows if r.speed), None)
    if speed:
        env = speed.get("env") or {}
        notes.append(f"Tốc độ sinh đo trên {env.get('gpu') or env.get('device')}, {speed.get('precision')}, PyTorch {env.get('torch')}.")
    compute = next((r.compute for r in rows if r.compute), None)
    if compute:
        notes.append(f"MACs đo bằng {compute.get('tool')} {compute.get('tool_version')}, batch 1; FLOPs = 2 × MACs.")
    if values:
        write_csv(
            out_dir / f"{dataset}_cost.csv",
            ["model", "run_name"] + [key for key, _, _ in COST_COLUMNS],
            [[r.name, r.run_name or ""] + [v[key] for key, _, _ in COST_COLUMNS] for r, v in zip(rows, values)],
        )
    return ["### Chi phí", markdown_table(header, body), " ".join(notes)]


def sampler_section(dataset: str, rows: list[ResultRow], secondary: list[ResultRow], converter: GpuHoursConverter,
                    out_dir: Path) -> list[str]:
    """Bảng "Đánh đổi sampler của DDPM": dòng DDPM chính (10k ảnh đầu) và các dòng phụ."""
    run_names = {r.run_name for r in secondary}
    group = [r for r in rows if r.model == "ddpm" and r.run_name in run_names] + secondary
    header = ["Mô hình", "Run", "Ảnh dùng để so", "FID_train-10k", "KID_test ×10³", "GFLOPs/ảnh",
              "Latency (ms, batch 1)", "Throughput (ảnh/s)"]
    body, raw = [], []
    for r in group:
        m = r.metrics
        cost = cost_values(r, converter)
        kid, kid_std = m.get("kid_test_mean"), m.get("kid_test_std")
        source = (f"{SAMPLER_COMPARE_NUM_GEN:,} ảnh riêng" if r.is_sampler_comparison
                  else f"{SAMPLER_COMPARE_NUM_GEN:,} ảnh đầu của bộ chính")
        values = [m.get("fid_train_10k"), kid, cost["gflops_per_image"], cost["latency_ms"], cost["throughput_img_s"]]
        body.append([r.name, r.run_name or "", source, format_value(values[0], 2),
                     value_with_std(times_1000(kid), times_1000(kid_std), 2),
                     format_value(values[2], 3), format_value(values[3], 3), format_value(values[4], 1)])
        raw.append([r.name, r.run_name, source] + values)
    write_csv(out_dir / f"{dataset}_sampler.csv",
              ["model", "run_name", "source", "fid_train_10k", "kid_test", "gflops_per_image", "latency_ms",
               "throughput_img_s"], raw)
    return [
        "### Đánh đổi sampler của DDPM",
        markdown_table(header, body),
        f"Cùng một checkpoint, cùng {SAMPLER_COMPARE_NUM_GEN:,} seed đầu: chênh lệch giữa các dòng chỉ do sampler."
        " Không so số ở bảng này với FID_train-50k của bảng chính (FID trên ít ảnh hơn luôn cao hơn).",
    ]


def tuning_section(tuning: list[ResultRow], converter: GpuHoursConverter) -> list[str]:
    """Bảng "Các config đã thử": run overfit và tuning."""
    body = []
    for r in tuning:
        run = r.run or {}
        body.append([
            r.name,
            r.run_name or "",
            r.stage or "—",
            format_value(times_1000(r.metrics.get("kid_val_mean")), 2),
            format_value(converter.run_reference_hours(run), 2),
            str(run.get("config") or "—"),
        ])
    header = ["Mô hình", "Run", "Giai đoạn", "KID_val ×10³", f"Thời gian - {converter.reference} (giờ)", "Config"]
    return ["### Các config đã thử", markdown_table(header, body)]


def dataset_section(dataset: str, rows: list[ResultRow], secondary: list[ResultRow], tuning: list[ResultRow],
                    real_baseline: dict | None, converter: GpuHoursConverter, out_dir: Path, figures_dir: Path,
                    issues: IssueLog) -> str:
    """Toàn bộ phần của một dataset trong summary.md; ghi CSV, biểu đồ và ảnh ghép vào `out_dir`."""
    parts = [f"## {dataset}"]
    parts += quality_section(dataset, rows, real_baseline, out_dir)
    parts += cost_section(dataset, rows, converter, out_dir)
    if secondary:
        parts += sampler_section(dataset, rows, secondary, converter, out_dir)

    charts = plot_kid_curves(dataset, rows, converter, out_dir, issues)
    if charts:
        parts.append("### Chất lượng theo chi phí train")
        parts += [f"![{p.stem}]({p.name})" for p in charts]

    figures = combine_model_figures(dataset, rows, figures_dir, out_dir, issues)
    for name, path in figures.items():
        parts += [f"### {FIGURE_TITLES[name]}", f"![{path.stem}]({path.name})"]
        if name == "samples":
            parts.append("Ảnh mẫu không chọn lọc: cùng seed và cùng số ảnh cho mọi mô hình (scripts/render_samples.py).")

    if tuning:
        parts += tuning_section(tuning, converter)
    return "\n\n".join(parts)


def summary_text(sections: list[str], issues: IssueLog) -> str:
    """Nội dung summary.md: các phần theo dataset rồi mục "Kiểm tra"."""
    report = ["# Bảng kết quả tổng hợp", "File này do `scripts/aggregate_results.py` sinh ra, không sửa bằng tay."]
    report += sections
    report.append("## Kiểm tra")
    if issues.errors:
        report.append("**Lỗi (kết quả chưa so sánh được):**\n\n" + "\n".join(f"- {m}" for m in issues.errors))
    if issues.warnings:
        report.append("**Cảnh báo:**\n\n" + "\n".join(f"- {m}" for m in issues.warnings))
    if not issues.items:
        report.append("Không phát hiện vấn đề.")
    return "\n\n".join(report) + "\n"
