"""Biểu đồ KID validation theo Thời gian - T4 và theo FLOPs train."""

from __future__ import annotations

from pathlib import Path

from scripts.aggregate.constants import GRID, MODEL_COLORS, MODEL_DISPLAY, MODEL_MARKERS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY
from scripts.aggregate.gpu_hours import GpuHoursConverter
from scripts.aggregate.issues import IssueLog
from scripts.aggregate.records import ResultRow
from scripts.aggregate.table_values import total_train_flops


def curve_points(row: ResultRow, converter: GpuHoursConverter) -> list[dict]:
    """Các điểm (step, KID ×10³, Thời gian - T4, FLOPs train) từ trường `curve` của metadata run."""
    run = row.run or {}
    points = []
    for p in run.get("curve") or []:
        if p.get("kid_val_mean") is None or p.get("step") is None:
            continue
        gpu_t4, _ = converter.to_reference_hours(run, p.get("hardware_hours") or {})
        points.append({
            "step": p["step"],
            "kid": p["kid_val_mean"] * 1000,
            "gpu_hours_t4": gpu_t4,
            "train_flops": total_train_flops(run, row.compute, steps=p["step"]),
        })
    return sorted(points, key=lambda p: p["step"])


def plot_kid_curves(dataset: str, rows: list[ResultRow], converter: GpuHoursConverter, out_dir: Path,
                    issues: IssueLog) -> list[Path]:
    """Vẽ hai biểu đồ, mỗi run một đường; trả về đường dẫn các file PNG đã ghi."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Mỗi run một đường (các biến thể sampler của DDPM dùng chung một lần train).
    series, seen = [], set()
    for row in rows:
        key = (row.model, row.run_name)
        if key in seen:
            continue
        seen.add(key)
        points = curve_points(row, converter)
        if not points:
            issues.warning(f"{row.run_who}: metadata không có curve nên không vẽ được đường KID.")
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
    for x_key, x_label, log_x, suffix, title_x in (
        ("gpu_hours_t4", f"Thời gian - {converter.reference} (giờ)", False, "gpu_hours", f"thời gian - {converter.reference}"),
        ("train_flops", "FLOPs train (thang log)", True, "train_flops", "FLOPs train"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        drawn = 0
        for row, points in series:
            xs = [p[x_key] for p in points if p[x_key] is not None]
            ys = [p["kid"] for p in points if p[x_key] is not None]
            if not xs:
                issues.warning(f"{row.run_who}: không tính được {x_key} cho curve.")
                continue
            color = MODEL_COLORS.get(row.model, TEXT_SECONDARY)
            label = MODEL_DISPLAY.get(row.model, row.model)
            ax.plot(xs, ys, color=color, linewidth=2, marker=MODEL_MARKERS.get(row.model, "o"),
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
        ax.set_title(f"{dataset}: KID validation theo {title_x}", loc="left", fontsize=11, color=TEXT_PRIMARY)
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
