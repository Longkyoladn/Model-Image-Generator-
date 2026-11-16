"""Tổng hợp kết quả thành bảng so sánh cho từng dataset.

Đọc mọi file JSON có trường "kind" trong các thư mục đầu vào:

- kind = "run":       metadata của một lần train (mục 4 của TASK_ASSIGNMENT.md).
- kind = "quality":   kết quả FID/KID/diversity của một bộ ảnh sinh.
- kind = "speed":     kết quả của evaluation/speed.py.
- kind = "real_baseline": số liệu thật-với-thật của một dataset (FID sàn...).

Các file được ghép theo (dataset, model, run_name, label). Script kiểm tra các
run trong cùng dataset có được đánh giá cùng điều kiện không, rồi ghi bảng
Markdown và CSV. Không cần chép số bằng tay.

    python scripts/aggregate_results.py --inputs outputs/metrics --output-dir outputs/metrics/summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

MODELS = ["vae", "dcgan", "ddpm"]
MODEL_DISPLAY = {"vae": "VAE", "dcgan": "DCGAN", "ddpm": "DDPM"}

# Các trường phải giống nhau giữa mọi mô hình trong cùng một dataset.
EVAL_KEYS = ["eval_config", "eval_config_hash", "library", "commit", "ref_train", "ref_test", "num_gen"]
SPEED_ENV_KEYS = ["gpu", "torch"]
RUN_REQUIRED = ["config", "seed", "commit", "train_steps"]

# (khóa, tiêu đề, số chữ số thập phân, hướng tốt: "low" | "high" | None)
QUALITY_COLUMNS = [
    ("fid_train_50k", "FID_train-50k", 2, "low"),
    ("fid_test_10k", "FID_test-10k", 2, "low"),
    ("kid_test", "KID_test -10³", 2, "low"),
    ("precision", "Precision", 3, "high"),
    ("recall", "Recall", 3, "high"),
    ("density", "Density", 3, "high"),
    ("coverage", "Coverage", 3, "high"),
    ("lpips_diversity", "LPIPS div.", 3, None),
]
COST_COLUMNS = [
    ("params_m", "Tham số (M)", 2),
    ("gmacs_per_image", "GMACs/ảnh", 3),
    ("gflops_per_image", "GFLOPs/ảnh", 3),
    ("train_flops", "Train FLOPs", "sci"),
    ("gpu_hours_t4", "GPU-giờ T4", 2),
    ("train_steps", "Train step", 0),
    ("latency_ms", "Latency (ms, batch 1)", 3),
    ("throughput_img_s", "Throughput (ảnh/s)", 1),
    ("throughput_batch", "ở batch", 0),
    ("peak_memory_mb", "Peak mem (MB)", 0),
    ("checkpoint_mb", "Checkpoint (MB)", 1),
]


def load_records(inputs: list[Path]) -> tuple[dict[str, list[dict]], list[str]]:
    records: dict[str, list[dict]] = {"run": [], "quality": [], "speed": [], "reference": []}
    problems = []
    for root in inputs:
        for path in sorted(root.rglob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as err:
                problems.append(f"Không đọc được {path}: {err}")
                continue
            if not isinstance(data, dict) or data.get("kind") not in records:
                continue
            data["_path"] = str(path)
            records[data["kind"]].append(data)
    return records, problems


def fmt(value, digits) -> str:
    if value is None:
        return "—"
    if digits == "sci":
        return f"{value:.2e}"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}"
    return str(value)


def sampler_text(row: dict) -> str:
    kwargs = (row.get("speed") or {}).get("sample_kwargs") or {}
    quality = row.get("quality") or {}
    sampler = kwargs.get("sampler") or quality.get("sampler")
    steps = kwargs.get("steps") or quality.get("sampling_steps")
    if sampler and steps:
        return f"{sampler}, {steps} step"
    return ""


def build_rows(records: dict[str, list[dict]], dataset: str, issues: list[tuple[str, str]]) -> list[dict]:
    runs = {}
    for rec in records["run"]:
        if rec.get("dataset") == dataset:
            runs[(rec.get("model"), rec.get("run_name"))] = rec
    rows: dict[tuple, dict] = {}
    for kind in ("quality", "speed"):
        for rec in records[kind]:
            if rec.get("dataset") != dataset:
                continue
            key = (rec.get("model"), rec.get("run_name"), rec.get("label") or "")
            row = rows.setdefault(
                key, {"model": key[0], "run_name": key[1], "label": key[2], "dataset": dataset}
            )
            if kind in row:
                issues.append(
                    ("error", f"[{dataset}] Trùng file {kind} cho {key}: {row[kind]['_path']} và {rec['_path']}")
                )
            row[kind] = rec
    for row in rows.values():
        row["run"] = runs.get((row["model"], row["run_name"]))
        row["stage"] = (row["run"] or {}).get("stage")
    order = {m: i for i, m in enumerate(MODELS)}
    return sorted(rows.values(), key=lambda r: (order.get(r["model"], 99), r["run_name"] or "", r["label"]))


def row_name(row: dict) -> str:
    name = MODEL_DISPLAY.get(row["model"], str(row["model"]))
    detail = sampler_text(row) or row["label"]
    return f"{name} ({detail})" if detail else name


def check_fairness(dataset: str, rows: list[dict], issues: list[tuple[str, str]]) -> None:
    def add(level, msg):
        issues.append((level, f"[{dataset}] {msg}"))

    present = {r["model"] for r in rows}
    for model in MODELS:
        if model not in present:
            add("warning", f"Chưa có kết quả cho {MODEL_DISPLAY[model]}.")

    for row in rows:
        who = f"{row_name(row)} / {row['run_name']}"
        if row.get("quality") is None:
            add("warning", f"{who}: thiếu file quality.")
        if row.get("speed") is None:
            add("warning", f"{who}: thiếu file speed.")
        if row.get("run") is None:
            add("warning", f"{who}: thiếu metadata của run.")
        else:
            missing = [k for k in RUN_REQUIRED if row["run"].get(k) is None]
            if missing:
                add("warning", f"{who}: metadata thiếu {', '.join(missing)}.")
        if row["model"] == "ddpm" and not sampler_text(row):
            add("error", f"{who}: DDPM phải ghi sampler và số step.")

    # Cùng cấu hình đánh giá chất lượng.
    quality_rows = [r for r in rows if r.get("quality")]
    for key in EVAL_KEYS:
        values = {}
        for row in quality_rows:
            value = (row["quality"].get("eval") or {}).get(key)
            if value is None:
                add("error", f"{row_name(row)} / {row['run_name']}: file quality thiếu eval.{key}.")
            else:
                values.setdefault(json.dumps(value, sort_keys=True), []).append(row_name(row))
        if len(values) > 1:
            detail = "; ".join(f"{v} → {', '.join(names)}" for v, names in values.items())
            add("error", f"eval.{key} không giống nhau giữa các mô hình: {detail}")

    # Cùng điều kiện đo tốc độ.
    speed_rows = [r for r in rows if r.get("speed")]
    checks = {key: (lambda s, k=key: (s.get("env") or {}).get(k)) for key in SPEED_ENV_KEYS}
    checks["precision"] = lambda s: s.get("precision")
    checks["batch_sizes"] = lambda s: sorted(r["batch_size"] for r in s.get("results", []))
    for key, getter in checks.items():
        values = {}
        for row in speed_rows:
            values.setdefault(json.dumps(getter(row["speed"])), []).append(row_name(row))
        if len(values) > 1:
            detail = "; ".join(f"{v} → {', '.join(names)}" for v, names in values.items())
            add("error", f"Điều kiện đo tốc độ khác nhau ({key}): {detail}")


def quality_values(row: dict) -> dict:
    m = (row.get("quality") or {}).get("metrics") or {}
    kid = m.get("kid_test_mean")
    return {
        "fid_train_50k": m.get("fid_train_50k"),
        "fid_test_10k": m.get("fid_test_10k_mean"),
        "fid_test_10k_std": m.get("fid_test_10k_std"),
        "kid_test": kid * 1000 if kid is not None else None,
        "kid_test_std": m["kid_test_std"] * 1000 if m.get("kid_test_std") is not None else None,
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "density": m.get("density"),
        "coverage": m.get("coverage"),
        "lpips_diversity": m.get("lpips_diversity"),
    }


def cost_values(row: dict) -> dict:
    run = row.get("run") or {}
    speed = row.get("speed") or {}
    results = [r for r in speed.get("results", []) if not r.get("oom")]
    latency = next((r["time_per_image_ms"] for r in results if r["batch_size"] == 1), None)
    best = max(results, key=lambda r: r["throughput_img_s"], default=None)
    passes = speed.get("forward_passes_per_image")
    gmacs = run.get("gmacs_per_image")
    if gmacs is None and run.get("gmacs_per_forward") is not None and passes:
        gmacs = run["gmacs_per_forward"] * passes
    return {
        "params_m": speed.get("params_m", run.get("params_m")),
        "gmacs_per_image": gmacs,
        "gflops_per_image": 2 * gmacs if gmacs is not None else None,
        "train_flops": run.get("train_flops"),
        "gpu_hours_t4": run.get("gpu_hours_t4"),
        "train_steps": run.get("train_steps"),
        "latency_ms": latency,
        "throughput_img_s": best["throughput_img_s"] if best else None,
        "throughput_batch": best["batch_size"] if best else None,
        "peak_memory_mb": best.get("peak_memory_mb") if best else None,
        "checkpoint_mb": speed.get("checkpoint_mb"),
    }


def best_rows(values: list[dict], key: str, direction: str) -> set[int]:
    """Chỉ số các hàng được in đậm. FID_test dùng quy tắc 2 × std."""
    scored = [(i, v[key]) for i, v in enumerate(values) if v.get(key) is not None]
    if len(scored) < 2:
        return set()
    pick = min if direction == "low" else max
    best_i, best_v = pick(scored, key=lambda t: t[1])
    winners = {i for i, v in scored if v == best_v}
    std_key = key + "_std"
    if values[best_i].get(std_key) is not None:
        for i, v in scored:
            std = max(values[i].get(std_key) or 0.0, values[best_i][std_key])
            if abs(v - best_v) < 2 * std:
                winners.add(i)
    return winners


def markdown_table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def write_csv(path: Path, header: list[str], body: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(body)


def dataset_section(dataset: str, rows: list[dict], tuning: list[dict], reference: dict | None, out_dir: Path) -> str:
    parts = [f"## {dataset}"]
    names = [row_name(r) for r in rows]
    runs = [r["run_name"] or "" for r in rows]

    # Bảng chất lượng.
    q = [quality_values(r) for r in rows]
    bold = {key: best_rows(q, key, d) for key, _, _, d in QUALITY_COLUMNS if d}
    body, raw = [], []
    for i, v in enumerate(q):
        cells = [names[i], runs[i]]
        for key, _, digits, direction in QUALITY_COLUMNS:
            text = fmt(v[key], digits)
            std = v.get(key + "_std")
            if v[key] is not None and std is not None:
                text += f" ± {fmt(std, digits)}"
            if direction and i in bold[key]:
                text = f"**{text}**"
            cells.append(text)
        body.append(cells)
        raw.append([names[i], runs[i]] + [v[k] for k in v])
    header = ["Mô hình", "Run"] + [title for _, title, _, _ in QUALITY_COLUMNS]
    parts += ["### Chất lượng", markdown_table(header, body)]
    parts.append(
        "In đậm: giá trị tốt nhất. Với các cột có ± std, những mô hình chênh giá trị tốt nhất dưới 2 × std được coi là ngang nhau."
    )
    if reference:
        ref = reference.get("metrics") or {}
        text = ", ".join(f"{k} = {fmt(v, 3)}" for k, v in ref.items())
        parts.append(f"Mức tham chiếu thật-với-thật: {text}.")
    write_csv(out_dir / f"{dataset}_quality.csv", ["model", "run_name"] + list(q[0].keys()) if q else [], raw)

    # Bảng chi phí.
    c = [cost_values(r) for r in rows]
    body = [[names[i], runs[i]] + [fmt(v[key], digits) for key, _, digits in COST_COLUMNS] for i, v in enumerate(c)]
    header = ["Mô hình", "Run"] + [title for _, title, _ in COST_COLUMNS]
    parts += ["### Chi phí", markdown_table(header, body)]
    speed = next((r["speed"] for r in rows if r.get("speed")), None)
    if speed:
        env = speed.get("env") or {}
        parts.append(
            f"Tốc độ đo trên {env.get('gpu') or env.get('device')}, {speed.get('precision')}, PyTorch {env.get('torch')}."
        )
    write_csv(
        out_dir / f"{dataset}_cost.csv",
        ["model", "run_name"] + [key for key, _, _ in COST_COLUMNS],
        [[names[i], runs[i]] + [v[key] for key, _, _ in COST_COLUMNS] for i, v in enumerate(c)],
    )

    # Bảng các config đã thử.
    if tuning:
        body = []
        for r in tuning:
            m = (r.get("quality") or {}).get("metrics") or {}
            run = r.get("run") or {}
            kid = m.get("kid_val_mean")
            body.append(
                [
                    row_name(r),
                    r["run_name"] or "",
                    r.get("stage") or "—",
                    fmt(kid * 1000 if kid is not None else None, 2),
                    fmt(run.get("gpu_hours_t4"), 2),
                    str(run.get("config") or "—"),
                ]
            )
        parts += [
            "### Các config đã thử",
            markdown_table(["Mô hình", "Run", "Giai đoạn", "KID_val ×10³", "GPU-giờ T4", "Config"], body),
        ]
    return "\n\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description="Tổng hợp kết quả thành bảng so sánh.")
    p.add_argument("--inputs", nargs="+", default=["outputs/metrics"])
    p.add_argument("--output-dir", default="outputs/metrics/summary")
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
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    inputs = [Path(i) for i in args.inputs]
    records, problems = load_records([i for i in inputs if i.resolve() != out_dir.resolve()])
    issues: list[tuple[str, str]] = [("error", msg) for msg in problems]

    datasets = sorted(
        {r.get("dataset") for kind in ("quality", "speed") for r in records[kind] if r.get("dataset")}
    )
    if not datasets:
        print("Không tìm thấy file quality hoặc speed nào.", file=sys.stderr)
        return 1

    prepared = []
    for dataset in datasets:
        rows = build_rows(records, dataset, issues)
        if args.stage == "all":
            main_rows, tuning = rows, []
        else:
            main_rows = [r for r in rows if r["stage"] == "full"]
            tuning = [r for r in rows if r["stage"] in ("overfit", "tuning")]
            for r in rows:
                if r["stage"] is None:
                    issues.append(
                        ("warning", f"[{dataset}] {row_name(r)} / {r['run_name']}: không có metadata nên bị loại khỏi bảng chính.")
                    )
        check_fairness(dataset, main_rows, issues)
        reference = next((r for r in records["reference"] if r.get("dataset") == dataset), None)
        prepared.append((dataset, main_rows, tuning, reference))

    errors = [m for level, m in issues if level == "error"]
    warnings = [m for level, m in issues if level == "warning"]
    for m in errors:
        print(f"LỖI: {m}", file=sys.stderr)
    for m in warnings:
        print(f"Cảnh báo: {m}", file=sys.stderr)
    if errors and not args.allow_inconsistent:
        print("Không ghi bảng vì có lỗi. Dùng --allow-inconsistent để vẫn ghi.", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    sections = [dataset_section(*item, out_dir) for item in prepared]
    report = ["# Bảng kết quả tổng hợp", "File này do `scripts/aggregate_results.py` sinh ra, không sửa bằng tay."]
    report += sections
    report.append("## Kiểm tra")
    if errors:
        report.append("**Lỗi (kết quả chưa so sánh được):**\n\n" + "\n".join(f"- {m}" for m in errors))
    if warnings:
        report.append("**Cảnh báo:**\n\n" + "\n".join(f"- {m}" for m in warnings))
    if not issues:
        report.append("Không phát hiện vấn đề.")

    summary = out_dir / "summary.md"
    summary.write_text("\n\n".join(report) + "\n", encoding="utf-8")
    print(f"Đã ghi {summary}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
