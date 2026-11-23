"""Kiểm tra các mô hình được so sánh công bằng (mục 2.4, 4.2, 4.3).

Mỗi hàm `check_*` kiểm tra một quy tắc và ghi lỗi/cảnh báo vào `IssueLog`.
`run_fairness_checks` gọi chúng theo thứ tự cố định.
"""

from __future__ import annotations

import json
from typing import Callable

from scripts.aggregate.constants import (
    COMPARISON_SAMPLER,
    EVAL_KEYS,
    MODEL_DISPLAY,
    MODELS,
    RUN_REQUIRED,
    SAMPLER_COMPARE_NUM_GEN,
    SAMPLER_METRICS,
    SCRIPT_COMPUTED_RUN_FIELDS,
    SPEED_ENV_KEYS,
)
from scripts.aggregate.gpu_hours import GpuHoursConverter, is_reference_device, train_hours_by_device
from scripts.aggregate.issues import IssueLog
from scripts.aggregate.records import ResultRow, one_row_per_run


def group_names_by_value(rows: list[ResultRow], value_of: Callable[[ResultRow], str]) -> dict[str, list[str]]:
    """Gom tên các dòng theo giá trị (đã đổi sang chuỗi); hơn một nhóm nghĩa là các dòng không giống nhau."""
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(value_of(row), []).append(row.name)
    return groups


def describe_groups(groups: dict[str, list[str]]) -> str:
    return "; ".join(f"{value} → {', '.join(names)}" for value, names in groups.items())


def as_json(value) -> str:
    return json.dumps(value, sort_keys=True)


def check_comparison_flags(rows: list[ResultRow], issues: IssueLog) -> None:
    """Trường `comparison` chỉ nhận "sampler" và chỉ dùng cho DDPM."""
    for row in rows:
        value = (row.quality or {}).get("comparison")
        if value is None:
            continue
        if value != COMPARISON_SAMPLER:
            issues.error(f"{row.who}: comparison = {value!r} không hợp lệ, chỉ nhận {COMPARISON_SAMPLER!r}.")
        elif row.model != "ddpm":
            issues.error(f"{row.who}: comparison = {COMPARISON_SAMPLER!r} chỉ dùng cho DDPM.")


def check_models_present(rows: list[ResultRow], issues: IssueLog) -> None:
    present = {r.model for r in rows}
    for model in MODELS:
        if model not in present:
            issues.warning(f"Chưa có kết quả cho {MODEL_DISPLAY[model]}.")


def check_row_files(row: ResultRow, issues: IssueLog) -> None:
    """Mỗi dòng cần đủ file quality/speed/compute; dòng DDPM phải ghi sampler và số step."""
    for kind in ("quality", "speed", "compute"):
        if getattr(row, kind) is None:
            issues.warning(f"{row.who}: thiếu file {kind}.")
    if row.model == "ddpm" and not row.sampler_text:
        issues.error(f"{row.who}: DDPM phải ghi sampler và số step.")


def check_run_metadata(row: ResultRow, converter: GpuHoursConverter, issues: IssueLog) -> None:
    """Metadata run: đủ trường, không có số điền tay, quy đổi được giờ train về T4."""
    run = row.run
    if run is None:
        issues.warning(f"{row.run_who}: thiếu metadata của run.")
        return
    missing = [k for k in RUN_REQUIRED if run.get(k) is None]
    if missing:
        issues.warning(f"{row.run_who}: metadata thiếu {', '.join(missing)}.")
    manual = [k for k in SCRIPT_COMPUTED_RUN_FIELDS if k in run]
    if manual:
        issues.error(f"{row.run_who}: metadata run có {', '.join(manual)}, các trường này do script tổng hợp "
                     "tự tính nên không được điền tay; xóa khỏi metadata (mục 4.3).")
    if run.get("hardware"):
        value, notes = converter.to_reference_hours(run, train_hours_by_device(run))
        for note in notes:
            (issues.error if value is None else issues.warning)(f"{row.run_who}: {note}")


def check_eval_consistency(rows: list[ResultRow], issues: IssueLog) -> dict[str, str]:
    """Các dòng chính phải cùng cấu hình đánh giá chất lượng.

    Trả về khối eval chung (khóa -> giá trị dạng JSON) của những trường mà mọi
    dòng giống nhau, để so với dòng phụ.
    """
    quality_rows = [r for r in rows if r.quality]
    shared: dict[str, str] = {}
    for key in EVAL_KEYS:
        for row in quality_rows:
            if row.eval.get(key) is None:
                issues.error(f"{row.who}: file quality thiếu eval.{key}.")
        groups = group_names_by_value([r for r in quality_rows if r.eval.get(key) is not None],
                                      lambda r: as_json(r.eval.get(key)))
        if len(groups) > 1:
            issues.error(f"eval.{key} không giống nhau giữa các mô hình: {describe_groups(groups)}")
        found = {as_json(r.eval.get(key)) for r in quality_rows}
        if len(found) == 1:
            shared[key] = found.pop()
    return shared


def check_sampler_rows(rows: list[ResultRow], secondary: list[ResultRow], main_eval: dict[str, str],
                       issues: IssueLog) -> None:
    """Dòng phụ so sánh sampler: cùng eval với bảng chính trừ num_gen, đủ metric, có dòng DDPM chính."""
    references = {r.run_name: r for r in rows if r.model == "ddpm" and r.quality}
    for row in secondary:
        ev = row.eval
        if ev.get("num_gen") != SAMPLER_COMPARE_NUM_GEN:
            issues.error(f"{row.who}: dòng so sánh sampler phải có eval.num_gen = {SAMPLER_COMPARE_NUM_GEN}, "
                         f"nhận {ev.get('num_gen')}.")
        for key in EVAL_KEYS:
            if key == "num_gen" or key not in main_eval:
                continue
            if as_json(ev.get(key)) != main_eval[key]:
                issues.error(f"{row.who}: eval.{key} = {ev.get(key)!r} khác bảng chính ({main_eval[key]}).")
        missing = [k for k in SAMPLER_METRICS if row.metrics.get(k) is None]
        if missing:
            issues.error(f"{row.who}: file quality so sánh sampler thiếu metrics {', '.join(missing)}.")
        ref = references.get(row.run_name)
        if ref is None:
            issues.warning(f"{row.who}: không có dòng DDPM chính cùng run_name để so sánh sampler.")
        elif ref.metrics.get("fid_train_10k") is None:
            issues.warning(f"{ref.who}: thiếu metrics.fid_train_10k (FID trên 10k ảnh đầu) nên bảng so sánh "
                           "sampler không có FID.")


def check_speed_conditions(rows: list[ResultRow], reference: str, issues: IssueLog) -> None:
    """Mọi file speed (kể cả dòng phụ) cùng GPU, PyTorch, precision, batch size; đo trên T4 và đo xong."""
    speed_rows = [r for r in rows if r.speed]
    readers = {key: (lambda s, k=key: (s.get("env") or {}).get(k)) for key in SPEED_ENV_KEYS}
    readers["precision"] = lambda s: s.get("precision")
    readers["batch_sizes"] = lambda s: sorted(r["batch_size"] for r in s.get("results", []))
    for key, read in readers.items():
        groups = group_names_by_value(speed_rows, lambda r: json.dumps(read(r.speed)))
        if len(groups) > 1:
            issues.error(f"Điều kiện đo tốc độ khác nhau ({key}): {describe_groups(groups)}")
    for row in speed_rows:
        env = row.speed.get("env") or {}
        device = env.get("gpu") or env.get("device") or ""
        if not is_reference_device(device, reference):
            issues.error(f"{row.who}: tốc độ sinh đo trên {device!r}, phải đo trên {reference} (mục 2.4).")
        complete = row.speed.get("complete")
        if complete is False:
            done = min((r.get("n_repeats", 0) for r in row.speed.get("results", []) if not r.get("oom")), default=0)
            issues.warning(f"{row.who}: speed.py chưa đo xong ({done}/{row.speed.get('repeats_requested')} lần đo), "
                           "có thể phiên bị ngắt; nên chạy lại.")
        elif complete is not True:
            issues.warning(f"{row.who}: file speed thiếu trường complete nên coi như speed.py chưa đo xong; "
                           "chạy lại evaluation/speed.py.")


def check_macs_tool(rows: list[ResultRow], issues: IssueLog) -> None:
    """MACs của mọi dòng đo bằng cùng công cụ và cùng phiên bản."""
    compute_rows = [r for r in rows if r.compute]
    groups = group_names_by_value(compute_rows, lambda r: f"{r.compute.get('tool')} {r.compute.get('tool_version')}")
    if len(groups) > 1:
        issues.error(f"MACs được đo bằng công cụ khác nhau: {describe_groups(groups)}")


def run_fairness_checks(rows: list[ResultRow], secondary: list[ResultRow], converter: GpuHoursConverter,
                        issues: IssueLog) -> None:
    """`rows`: dòng bảng chính; `secondary`: dòng phụ so sánh sampler của DDPM."""
    all_rows = rows + secondary
    check_models_present(rows, issues)
    for row in all_rows:
        check_row_files(row, issues)
    for row in one_row_per_run(all_rows):
        check_run_metadata(row, converter, issues)
    main_eval = check_eval_consistency(rows, issues)
    check_sampler_rows(rows, secondary, main_eval, issues)
    check_speed_conditions(all_rows, converter.reference, issues)
    check_macs_tool(all_rows, issues)
