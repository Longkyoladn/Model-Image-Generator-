"""Tổng hợp kết quả thành bảng, biểu đồ và ảnh so sánh cho từng dataset.

Đọc mọi file JSON có trường "kind" trong các thư mục đầu vào (mục 4.2 của
TASK_ASSIGNMENT.md):

- kind = "run":         metadata của một lần train (mục 4.3).
- kind = "quality":     kết quả FID/KID/diversity của một bộ ảnh sinh.
- kind = "speed":       kết quả của evaluation/speed.py.
- kind = "compute":     kết quả của evaluation/flops.py (tham số, MACs, FLOPs).
- kind = "calibration": kết quả của evaluation/calibrate.py (throughput train).
- kind = "real_baseline": số liệu thật-với-thật của dataset (FID sàn, LPIPS ảnh thật).
                          Chấp nhận cả tên cũ "reference".

Các file quality/speed/compute được ghép theo (dataset, model, run_name, label)
và nối với metadata run theo (dataset, model, run_name).

Script tự tính (không tin số điền tay):
- Tổng FLOPs train = train_steps × batch_size × train_flops_per_image.
- Thời gian - T4 = Σ_h (giờ train trên h × k_h), với k_h = throughput_h / throughput_T4
  lấy từ các file calibration cùng (model, config).

Đầu ra trong --output-dir:
- summary.md, <dataset>_quality.csv, <dataset>_cost.csv
- <dataset>_kid_vs_gpu_hours.png, <dataset>_kid_vs_train_flops.png
- <dataset>_samples.png, <dataset>_interpolation.png, <dataset>_nearest_neighbor.png

    python scripts/aggregate_results.py --inputs outputs/metrics \
        --figures-dir outputs/figures --output-dir outputs/metrics/summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

MODELS = ["vae", "dcgan", "ddpm"]
MODEL_DISPLAY = {"vae": "VAE", "dcgan": "DCGAN", "ddpm": "DDPM"}
KINDS = ["run", "quality", "speed", "compute", "calibration", "real_baseline"]
KIND_ALIASES = {"reference": "real_baseline"}

# Các trường phải giống nhau giữa mọi mô hình trong cùng một dataset.
EVAL_KEYS = ["eval_config", "eval_config_hash", "library", "commit", "ref_train", "ref_test", "num_gen"]
SPEED_ENV_KEYS = ["gpu", "torch"]
RUN_REQUIRED = ["config", "seed", "commit", "train_steps", "batch_size", "hardware"]
FIGURE_NAMES = ("samples", "interpolation", "nearest_neighbor")
FIGURE_TITLES = {"samples": "Ảnh mẫu", "interpolation": "Latent interpolation", "nearest_neighbor": "Nearest-neighbor"}

# (khóa, tiêu đề, số chữ số thập phân, hướng tốt: "low" | "high" | None)
QUALITY_COLUMNS = [
    ("fid_train_50k", "FID_train-50k", 2, "low"),
    ("fid_test_10k", "FID_test-10k", 2, "low"),
    ("kid_test", "KID_test ×10³", 2, "low"),
    ("precision", "Precision", 3, "high"),
    ("recall", "Recall", 3, "high"),
    ("density", "Density", 3, "high"),
    ("coverage", "Coverage", 3, "high"),
    ("lpips_diversity", "LPIPS div.", 3, None),
]
COST_COLUMNS = [
    ("params_m", "Tham số sinh (M)", 2),
    ("params_train_m", "Tham số train (M)", 2),
    ("gmacs_per_image", "GMACs/ảnh", 3),
    ("gflops_per_image", "GFLOPs/ảnh", 3),
    ("train_flops", "Train FLOPs", "sci"),
    ("train_hours", "Giờ train thực tế", 2),
    ("gpu_hours_t4", "Thời gian - T4 (giờ)", 2),
    ("hardware", "Phần cứng train", None),
    ("train_steps", "Train step", 0),
    ("latency_ms", "Latency (ms, batch 1)", 3),
    ("throughput_img_s", "Throughput (ảnh/s)", 1),
    ("throughput_batch", "ở batch", 0),
    ("peak_memory_mb", "Peak mem (MB)", 0),
    ("checkpoint_mb", "Checkpoint (MB)", 1),
]

# Màu theo mô hình (cố định, không đổi theo thứ hạng): 3 slot đầu của bảng màu
# tham chiếu, đã kiểm tra phân biệt được với người mù màu khi cả 3 cùng xuất hiện.
MODEL_COLORS = {"vae": "#2a78d6", "dcgan": "#eb6834", "ddpm": "#1baf7a"}
MODEL_MARKERS = {"vae": "o", "dcgan": "s", "ddpm": "^"}
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


# --------------------------------------------------------------------------
# Đọc dữ liệu
# --------------------------------------------------------------------------

def load_records(inputs: list[Path]) -> tuple[dict[str, list[dict]], list[str]]:
    records: dict[str, list[dict]] = {k: [] for k in KINDS}
    problems = []
    for root in inputs:
        for path in sorted(root.rglob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as err:
                problems.append(f"Không đọc được {path}: {err}")
                continue
            if not isinstance(data, dict):
                continue
            kind = KIND_ALIASES.get(data.get("kind"), data.get("kind"))
            if kind not in records:
                continue
            data["_path"] = str(path)
            records[kind].append(data)
    return records, problems


def norm_config(path: str | None) -> str | None:
    if path is None:
        return None
    p = str(path).replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def fmt(value, digits) -> str:
    if value is None:
        return "—"
    if digits == "sci":
        return f"{value:.2e}"
    if digits is None or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:,.{digits}f}"


def sampler_text(row: dict) -> str:
    sources = [
        (row.get("speed") or {}).get("sample_kwargs") or {},
        (row.get("compute") or {}).get("sample_kwargs") or {},
        {"sampler": (row.get("quality") or {}).get("sampler"), "steps": (row.get("quality") or {}).get("sampling_steps")},
    ]
    sampler = next((s.get("sampler") for s in sources if s.get("sampler")), None)
    steps = next((s.get("steps") for s in sources if s.get("steps")), None)
    if sampler and steps:
        return f"{sampler}, {steps} step"
    return ""


def build_rows(records: dict[str, list[dict]], dataset: str, issues: list[tuple[str, str]]) -> list[dict]:
    runs = {}
    for rec in records["run"]:
        if rec.get("dataset") == dataset:
            runs[(rec.get("model"), rec.get("run_name"))] = rec
    rows: dict[tuple, dict] = {}
    for kind in ("quality", "speed", "compute"):
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


# --------------------------------------------------------------------------
# Quy đổi giờ GPU về phần cứng tham chiếu
# --------------------------------------------------------------------------

def is_reference_device(device: str, reference: str) -> bool:
    """T4 đơn là tham chiếu; "Tesla T4 x2" (nhiều GPU) thì không."""
    d = (device or "").lower()
    return reference.lower() in d and " x" not in d


def build_calibration_index(records: list[dict]) -> dict[tuple, dict[str, dict]]:
    index: dict[tuple, dict[str, dict]] = {}
    for rec in records:
        key = (rec.get("model"), norm_config(rec.get("config")))
        index.setdefault(key, {})[rec.get("device")] = rec
    return index


class Converter:
    """Tính hệ số k_h và thời gian train quy đổi về phần cứng tham chiếu."""

    def __init__(self, calibrations: dict[tuple, dict[str, dict]], reference: str) -> None:
        self.calibrations = calibrations
        self.reference = reference

    def factor(self, run: dict, device: str) -> tuple[float | None, list[str]]:
        if is_reference_device(device, self.reference):
            return 1.0, []
        cal = self.calibrations.get((run.get("model"), norm_config(run.get("config"))), {})
        ref = next((rec for dev, rec in cal.items() if is_reference_device(dev, self.reference)), None)
        here = cal.get(device)
        missing = []
        if ref is None:
            missing.append(f"chưa có file calibration trên {self.reference}")
        if here is None:
            missing.append(f"chưa có file calibration trên {device!r}")
        if missing:
            return None, [f"không quy đổi được giờ train trên {device!r} của config {run.get('config')}: " + ", ".join(missing)]
        notes = []
        for name, rec in ((self.reference, ref), (device, here)):
            for key in ("batch_size", "precision"):
                if run.get(key) is not None and rec.get(key) != run.get(key):
                    notes.append(f"calibration trên {name} dùng {key} = {rec.get(key)}, run dùng {run.get(key)}")
        return here["throughput_img_s"] / ref["throughput_img_s"], notes

    def convert(self, run: dict, hours_by_device: dict[str, float]) -> tuple[float | None, list[str]]:
        total, notes = 0.0, []
        for device, hours in hours_by_device.items():
            k, msgs = self.factor(run, device)
            notes += msgs
            if k is None:
                return None, notes
            total += hours * k
        return total, notes


def hardware_hours(run: dict) -> dict[str, float]:
    hours: dict[str, float] = {}
    for item in run.get("hardware") or []:
        hours[item.get("device")] = hours.get(item.get("device"), 0.0) + float(item.get("train_hours") or 0.0)
    return hours


# --------------------------------------------------------------------------
# Kiểm tra công bằng
# --------------------------------------------------------------------------

def check_fairness(dataset: str, rows: list[dict], converter: Converter, issues: list[tuple[str, str]]) -> None:
    def add(level, msg):
        issues.append((level, f"[{dataset}] {msg}"))

    present = {r["model"] for r in rows}
    for model in MODELS:
        if model not in present:
            add("warning", f"Chưa có kết quả cho {MODEL_DISPLAY[model]}.")

    for row in rows:
        who = f"{row_name(row)} / {row['run_name']}"
        for kind in ("quality", "speed", "compute"):
            if row.get(kind) is None:
                add("warning", f"{who}: thiếu file {kind}.")
        run = row.get("run")
        if run is None:
            add("warning", f"{who}: thiếu metadata của run.")
        else:
            missing = [k for k in RUN_REQUIRED if run.get(k) is None]
            if missing:
                add("warning", f"{who}: metadata thiếu {', '.join(missing)}.")
            if run.get("hardware"):
                value, notes = converter.convert(run, hardware_hours(run))
                for n in notes:
                    add("error" if value is None else "warning", f"{who}: {n}")
                manual = run.get("gpu_hours_t4")
                if value is not None and manual is not None and abs(manual - value) > 0.02 * max(value, 1e-9):
                    add("warning", f"{who}: gpu_hours_t4 điền tay ({manual:.3f}) khác giá trị tính được ({value:.3f}); bảng dùng giá trị tính được.")
            compute = row.get("compute")
            if compute and run.get("train_flops") is not None and run.get("train_steps") and run.get("batch_size"):
                computed = run["train_steps"] * run["batch_size"] * compute["train_flops_per_image"]
                if abs(run["train_flops"] - computed) > 0.05 * computed:
                    add("warning", f"{who}: train_flops điền tay ({run['train_flops']:.2e}) khác giá trị tính từ flops.py ({computed:.2e}); bảng dùng giá trị tính được.")
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

    # Cùng điều kiện đo tốc độ, và phải đo trên phần cứng tham chiếu.
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
    for row in speed_rows:
        env = row["speed"].get("env") or {}
        device = env.get("gpu") or env.get("device") or ""
        if not is_reference_device(device, converter.reference):
            add("error", f"{row_name(row)} / {row['run_name']}: tốc độ sinh đo trên {device!r}, phải đo trên {converter.reference} (mục 2.4).")

    # Cùng công cụ đo MACs.
    values = {}
    for row in rows:
        if row.get("compute"):
            c = row["compute"]
            values.setdefault(f"{c.get('tool')} {c.get('tool_version')}", []).append(row_name(row))
    if len(values) > 1:
        detail = "; ".join(f"{v} → {', '.join(names)}" for v, names in values.items())
        add("error", f"MACs được đo bằng công cụ khác nhau: {detail}")


# --------------------------------------------------------------------------
# Giá trị cho bảng
# --------------------------------------------------------------------------

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


def total_train_flops(run: dict, compute: dict | None, steps: int | None = None) -> float | None:
    steps = run.get("train_steps") if steps is None else steps
    if compute and steps is not None and run.get("batch_size"):
        return steps * run["batch_size"] * compute["train_flops_per_image"]
    return run.get("train_flops") if steps == run.get("train_steps") else None


def cost_values(row: dict, converter: Converter) -> dict:
    run = row.get("run") or {}
    speed = row.get("speed") or {}
    compute = row.get("compute")
    results = [r for r in speed.get("results", []) if not r.get("oom")]
    latency = next((r["time_per_image_ms"] for r in results if r["batch_size"] == 1), None)
    best = max(results, key=lambda r: r["throughput_img_s"], default=None)
    if compute:
        params, params_train = compute.get("params_m"), compute.get("params_train_m")
        gmacs = compute.get("gmacs_per_image")
    else:
        params, params_train = speed.get("params_m", run.get("params_m")), None
        passes = speed.get("forward_passes_per_image")
        gmacs = run["gmacs_per_forward"] * passes if run.get("gmacs_per_forward") is not None and passes else None
    hours = hardware_hours(run)
    gpu_t4, _ = converter.convert(run, hours) if hours else (None, [])
    return {
        "params_m": params,
        "params_train_m": params_train,
        "gmacs_per_image": gmacs,
        "gflops_per_image": 2 * gmacs if gmacs is not None else None,
        "train_flops": total_train_flops(run, compute),
        "train_hours": sum(hours.values()) if hours else None,
        "gpu_hours_t4": gpu_t4,
        "hardware": ", ".join(f"{d} ({h:.2f} h)" for d, h in hours.items()) if hours else None,
        "train_steps": run.get("train_steps"),
        "latency_ms": latency,
        "throughput_img_s": best["throughput_img_s"] if best else None,
        "throughput_batch": best["batch_size"] if best else None,
        "peak_memory_mb": best.get("peak_memory_mb") if best else None,
        "checkpoint_mb": speed.get("checkpoint_mb"),
    }


def best_rows(values: list[dict], key: str, direction: str) -> set[int]:
    """Chỉ số các hàng được in đậm. Cột có std dùng quy tắc 2 × std."""
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
    # Mọi mô hình ngang nhau thì không in đậm ai.
    return set() if len(winners) == len(scored) else winners


def markdown_table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def write_csv(path: Path, header: list[str], body: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(body)


# --------------------------------------------------------------------------
# Biểu đồ
# --------------------------------------------------------------------------

def curve_points(row: dict, converter: Converter) -> list[dict]:
    run = row.get("run") or {}
    points = []
    for p in run.get("curve") or []:
        if p.get("kid_val_mean") is None or p.get("step") is None:
            continue
        gpu_t4, _ = converter.convert(run, p.get("hardware_hours") or {})
        points.append({
            "step": p["step"],
            "kid": p["kid_val_mean"] * 1000,
            "gpu_hours_t4": gpu_t4,
            "train_flops": total_train_flops(run, row.get("compute"), steps=p["step"]),
        })
    return sorted(points, key=lambda p: p["step"])


def plot_curves(dataset: str, rows: list[dict], converter: Converter, out_dir: Path,
                issues: list[tuple[str, str]]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Mỗi run một đường (các biến thể sampler của DDPM dùng chung một lần train).
    series, seen = [], set()
    for row in rows:
        key = (row["model"], row["run_name"])
        if key in seen:
            continue
        seen.add(key)
        points = curve_points(row, converter)
        if not points:
            issues.append(("warning", f"[{dataset}] {MODEL_DISPLAY.get(row['model'], row['model'])} / {row['run_name']}: metadata không có curve nên không vẽ được đường KID."))
            continue
        series.append((row, points))
    if not series:
        return []

    plt.rcParams.update({
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
    })
    all_kid = [p["kid"] for _, pts in series for p in pts if p["kid"] > 0]
    log_y = bool(all_kid) and max(all_kid) / min(all_kid) > 20

    written = []
    for x_key, x_label, log_x, suffix in (
        ("gpu_hours_t4", f"Thời gian - {converter.reference} (giờ)", False, "gpu_hours"),
        ("train_flops", "FLOPs train (thang log)", True, "train_flops"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        drawn = 0
        for row, points in series:
            xs = [p[x_key] for p in points if p[x_key] is not None]
            ys = [p["kid"] for p in points if p[x_key] is not None]
            if not xs:
                issues.append(("warning", f"[{dataset}] {MODEL_DISPLAY.get(row['model'])} / {row['run_name']}: không tính được {x_key} cho curve."))
                continue
            color = MODEL_COLORS.get(row["model"], TEXT_SECONDARY)
            label = MODEL_DISPLAY.get(row["model"], row["model"])
            ax.plot(xs, ys, color=color, linewidth=2, marker=MODEL_MARKERS.get(row["model"], "o"),
                    markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
            ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT_SECONDARY)
            drawn += 1
        if not drawn:
            plt.close(fig)
            continue
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("KID validation ×10³" + (" (thang log)" if log_y else ""))
        ax.set_title(f"{dataset}: KID validation theo {'thời gian - ' + converter.reference if suffix == 'gpu_hours' else 'FLOPs train'}",
                     loc="left", fontsize=11, color=TEXT_PRIMARY)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        path = out_dir / f"{dataset}_kid_vs_{suffix}.png"
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
        written.append(path)
    return written


# --------------------------------------------------------------------------
# Ghép ảnh mẫu, interpolation, nearest-neighbor
# --------------------------------------------------------------------------

def find_figure(figures_dir: Path, row: dict, name: str) -> Path | None:
    base = figures_dir / row["dataset"]
    candidates = []
    if row["label"]:
        candidates.append(base / f"{row['model']}__{row['run_name']}__{row['label']}" / f"{name}.png")
    candidates.append(base / f"{row['model']}__{row['run_name']}" / f"{name}.png")
    return next((c for c in candidates if c.exists()), None)


def compose_figures(dataset: str, rows: list[dict], figures_dir: Path, out_dir: Path,
                    issues: list[tuple[str, str]]) -> dict[str, Path]:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()

    written = {}
    for name in FIGURE_NAMES:
        panels, seen = [], set()
        for row in rows:
            path = find_figure(figures_dir, row, name)
            if path is None:
                issues.append(("warning", f"[{dataset}] {row_name(row)} / {row['run_name']}: thiếu {name}.png."))
                continue
            if path in seen:
                continue
            seen.add(path)
            title = row_name(row) if path.parent.name.count("__") == 2 else MODEL_DISPLAY.get(row["model"], row["model"])
            panels.append((title, Image.open(path).convert("RGB")))
        if not panels:
            continue
        # Cùng layout cho mọi mô hình: ảnh giữ nguyên độ phân giải, tiêu đề phía trên.
        # interpolation là dải ngang nên xếp dọc; hai loại còn lại xếp ngang.
        title_h, gap = 30, 16
        vertical = name == "interpolation"
        if vertical:
            width = max(img.width for _, img in panels)
            height = sum(title_h + img.height for _, img in panels) + gap * (len(panels) - 1)
        else:
            width = sum(img.width for _, img in panels) + gap * (len(panels) - 1)
            height = title_h + max(img.height for _, img in panels)
        canvas = Image.new("RGB", (width, height), SURFACE)
        draw = ImageDraw.Draw(canvas)
        x = y = 0
        for title, img in panels:
            draw.text((x + 2, y + 4), title, fill=TEXT_PRIMARY, font=font)
            canvas.paste(img, (x, y + title_h))
            if vertical:
                y += title_h + img.height + gap
            else:
                x += img.width + gap
        path = out_dir / f"{dataset}_{name}.png"
        canvas.save(path)
        written[name] = path
    return written


# --------------------------------------------------------------------------
# Báo cáo
# --------------------------------------------------------------------------

def dataset_section(dataset: str, rows: list[dict], tuning: list[dict], real_baseline: dict | None,
                    converter: Converter, out_dir: Path, figures_dir: Path,
                    issues: list[tuple[str, str]]) -> str:
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
        raw.append([names[i], runs[i]] + list(v.values()))
    header = ["Mô hình", "Run"] + [title for _, title, _, _ in QUALITY_COLUMNS]
    parts += ["### Chất lượng", markdown_table(header, body)]
    parts.append(
        "In đậm: giá trị tốt nhất. Với các cột có ± std, những mô hình chênh giá trị tốt nhất dưới 2 × std được coi là ngang nhau."
        " LPIPS div. không in đậm: giá trị càng gần LPIPS của ảnh thật càng tốt."
    )
    if real_baseline:
        ref = real_baseline.get("metrics") or {}
        text = ", ".join(f"{k} = {fmt(v, 3)}" for k, v in ref.items())
        parts.append(f"Mức tham chiếu thật-với-thật: {text}.")
    if q:
        write_csv(out_dir / f"{dataset}_quality.csv", ["model", "run_name"] + list(q[0].keys()), raw)

    # Bảng chi phí.
    c = [cost_values(r, converter) for r in rows]
    body = [[names[i], runs[i]] + [fmt(v[key], digits) for key, _, digits in COST_COLUMNS] for i, v in enumerate(c)]
    header = ["Mô hình", "Run"] + [
        f"Thời gian - {converter.reference} (giờ)" if key == "gpu_hours_t4" else title for key, title, _ in COST_COLUMNS
    ]
    parts += ["### Chi phí", markdown_table(header, body)]
    notes = [f"Thời gian - {converter.reference}: giờ train trên mọi phần cứng được quy đổi về {converter.reference} theo hệ số k_h = throughput_h / throughput_{converter.reference} từ file calibration (mục 2.5)."]
    speed = next((r["speed"] for r in rows if r.get("speed")), None)
    if speed:
        env = speed.get("env") or {}
        notes.append(f"Tốc độ sinh đo trên {env.get('gpu') or env.get('device')}, {speed.get('precision')}, PyTorch {env.get('torch')}.")
    compute = next((r["compute"] for r in rows if r.get("compute")), None)
    if compute:
        notes.append(f"MACs đo bằng {compute.get('tool')} {compute.get('tool_version')}, batch 1; FLOPs = 2 × MACs.")
    parts.append(" ".join(notes))
    if c:
        write_csv(
            out_dir / f"{dataset}_cost.csv",
            ["model", "run_name"] + [key for key, _, _ in COST_COLUMNS],
            [[names[i], runs[i]] + [v[key] for key, _, _ in COST_COLUMNS] for i, v in enumerate(c)],
        )

    # Biểu đồ.
    charts = plot_curves(dataset, rows, converter, out_dir, issues)
    if charts:
        parts.append("### Chất lượng theo chi phí train")
        parts += [f"![{p.stem}]({p.name})" for p in charts]

    # Ảnh mẫu, interpolation, nearest-neighbor.
    figures = compose_figures(dataset, rows, figures_dir, out_dir, issues)
    for name, path in figures.items():
        parts += [f"### {FIGURE_TITLES[name]}", f"![{path.stem}]({path.name})"]
        if name == "samples":
            parts.append("Ảnh mẫu không chọn lọc: cùng seed và cùng số ảnh cho mọi mô hình (scripts/render_samples.py).")

    # Bảng các config đã thử.
    if tuning:
        body = []
        for r in tuning:
            m = (r.get("quality") or {}).get("metrics") or {}
            run = r.get("run") or {}
            kid = m.get("kid_val_mean")
            hours = hardware_hours(run)
            gpu_t4, _ = converter.convert(run, hours) if hours else (None, [])
            body.append([
                row_name(r),
                r["run_name"] or "",
                r.get("stage") or "—",
                fmt(kid * 1000 if kid is not None else None, 2),
                fmt(gpu_t4, 2),
                str(run.get("config") or "—"),
            ])
        parts += [
            "### Các config đã thử",
            markdown_table(["Mô hình", "Run", "Giai đoạn", "KID_val ×10³", f"Thời gian - {converter.reference} (giờ)", "Config"], body),
        ]
    return "\n\n".join(parts)


def main() -> int:
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
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    inputs = [Path(i) for i in args.inputs]
    records, problems = load_records([i for i in inputs if i.resolve() != out_dir.resolve()])
    issues: list[tuple[str, str]] = [("error", msg) for msg in problems]
    converter = Converter(build_calibration_index(records["calibration"]), args.reference_device)

    datasets = sorted(
        {r.get("dataset") for kind in ("quality", "speed", "compute") for r in records[kind] if r.get("dataset")}
    )
    if not datasets:
        print("Không tìm thấy file quality, speed hoặc compute nào.", file=sys.stderr)
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
        check_fairness(dataset, main_rows, converter, issues)
        real_baseline = next((r for r in records["real_baseline"] if r.get("dataset") == dataset), None)
        prepared.append((dataset, main_rows, tuning, real_baseline))

    errors = [m for level, m in issues if level == "error"]
    for m in errors:
        print(f"LỖI: {m}", file=sys.stderr)
    if errors and not args.allow_inconsistent:
        for level, m in issues:
            if level == "warning":
                print(f"Cảnh báo: {m}", file=sys.stderr)
        print("Không ghi bảng vì có lỗi. Dùng --allow-inconsistent để vẫn ghi.", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    sections = [
        dataset_section(d, rows, tuning, baseline, converter, out_dir, figures_dir, issues)
        for d, rows, tuning, baseline in prepared
    ]
    warnings_ = [m for level, m in issues if level == "warning"]
    for m in warnings_:
        print(f"Cảnh báo: {m}", file=sys.stderr)

    report = ["# Bảng kết quả tổng hợp", "File này do `scripts/aggregate_results.py` sinh ra, không sửa bằng tay."]
    report += sections
    report.append("## Kiểm tra")
    if errors:
        report.append("**Lỗi (kết quả chưa so sánh được):**\n\n" + "\n".join(f"- {m}" for m in errors))
    if warnings_:
        report.append("**Cảnh báo:**\n\n" + "\n".join(f"- {m}" for m in warnings_))
    if not issues:
        report.append("Không phát hiện vấn đề.")

    summary = out_dir / "summary.md"
    summary.write_text("\n\n".join(report) + "\n", encoding="utf-8")
    print(f"Đã ghi {summary}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
