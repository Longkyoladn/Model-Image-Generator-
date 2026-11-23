"""Giá trị từng ô của bảng chất lượng và bảng chi phí."""

from __future__ import annotations

from scripts.aggregate.gpu_hours import GpuHoursConverter, train_hours_by_device
from scripts.aggregate.records import ResultRow


def format_value(value, digits) -> str:
    """Chuỗi hiển thị trong bảng: "—" nếu thiếu, "sci" là dạng 1.23e+15."""
    if value is None:
        return "—"
    if digits == "sci":
        return f"{value:.2e}"
    if digits is None or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:,.{digits}f}"


def times_1000(value: float | None) -> float | None:
    """KID hiển thị dạng ×10³."""
    return value * 1000 if value is not None else None


def quality_values(row: ResultRow) -> dict:
    m = row.metrics
    return {
        "fid_train_50k": m.get("fid_train_50k"),
        "fid_test_10k": m.get("fid_test_10k_mean"),
        "fid_test_10k_std": m.get("fid_test_10k_std"),
        "kid_test": times_1000(m.get("kid_test_mean")),
        "kid_test_std": times_1000(m.get("kid_test_std")),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "density": m.get("density"),
        "coverage": m.get("coverage"),
        "lpips_diversity": m.get("lpips_diversity"),
    }


def total_train_flops(run: dict, compute: dict | None, steps: int | None = None) -> float | None:
    """FLOPs train = số step × batch size × FLOPs train mỗi ảnh (từ flops.py); None nếu thiếu dữ liệu."""
    steps = run.get("train_steps") if steps is None else steps
    if compute and steps is not None and run.get("batch_size"):
        return steps * run["batch_size"] * compute["train_flops_per_image"]
    return None


def cost_values(row: ResultRow, converter: GpuHoursConverter) -> dict:
    """Giá trị bảng chi phí. Thiếu file compute thì các cột tham số và MACs/FLOPs để trống."""
    run = row.run or {}
    speed = row.speed or {}
    compute = row.compute or {}
    # Bỏ batch bị OOM và batch chưa đo tới (file ghi dở khi phiên bị ngắt).
    results = [r for r in speed.get("results", []) if not r.get("oom") and "throughput_img_s" in r]
    latency = next((r["time_per_image_ms"] for r in results if r["batch_size"] == 1), None)
    best = max(results, key=lambda r: r["throughput_img_s"], default=None)
    gmacs = compute.get("gmacs_per_image")
    hours = train_hours_by_device(run)
    return {
        "params_m": compute.get("params_m"),
        "params_train_m": compute.get("params_train_m"),
        "gmacs_per_image": gmacs,
        "gflops_per_image": 2 * gmacs if gmacs is not None else None,
        "train_flops": total_train_flops(run, row.compute),
        "train_hours": sum(hours.values()) if hours else None,
        "gpu_hours_t4": converter.run_reference_hours(run),
        "hardware": ", ".join(f"{d} ({h:.2f} h)" for d, h in hours.items()) if hours else None,
        "train_steps": run.get("train_steps"),
        "latency_ms": latency,
        "throughput_img_s": best["throughput_img_s"] if best else None,
        "throughput_batch": best["batch_size"] if best else None,
        "peak_memory_mb": best.get("peak_memory_mb") if best else None,
        "checkpoint_mb": speed.get("checkpoint_mb"),
    }


def rows_to_bold(values: list[dict], key: str, direction: str) -> set[int]:
    """Chỉ số các hàng được in đậm ở cột `key`. Cột có std dùng quy tắc 2 × std."""
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
