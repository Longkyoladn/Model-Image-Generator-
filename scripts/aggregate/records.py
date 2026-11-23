"""Đọc file JSON kết quả và ghép chúng thành các dòng của bảng."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from scripts.aggregate.constants import COMPARISON_SAMPLER, KINDS, MODEL_DISPLAY, MODELS
from scripts.aggregate.issues import IssueLog


def load_records(inputs: list[Path]) -> tuple[dict[str, list[dict]], list[str]]:
    """Mọi file JSON có `kind` hợp lệ, gom theo kind; kèm danh sách file không đọc được.

    Mỗi record được thêm `_path` để thông báo lỗi chỉ ra đúng file.
    """
    records: dict[str, list[dict]] = {k: [] for k in KINDS}
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


@dataclass
class ResultRow:
    """Một dòng của bảng: các file quality/speed/compute cùng (model, run_name, label) và metadata run."""

    model: str
    run_name: str | None
    label: str
    dataset: str
    quality: dict | None = None
    speed: dict | None = None
    compute: dict | None = None
    run: dict | None = None

    @property
    def stage(self) -> str | None:
        return (self.run or {}).get("stage")

    @property
    def metrics(self) -> dict:
        return (self.quality or {}).get("metrics") or {}

    @property
    def eval(self) -> dict:
        return (self.quality or {}).get("eval") or {}

    @property
    def is_sampler_comparison(self) -> bool:
        return (self.quality or {}).get("comparison") == COMPARISON_SAMPLER

    @property
    def display_model(self) -> str:
        return MODEL_DISPLAY.get(self.model, str(self.model))

    @property
    def sampler_text(self) -> str:
        """Ví dụ "ddim, 100 step"; rỗng nếu không file nào ghi sampler và số step."""
        sources = [
            (self.speed or {}).get("sample_kwargs") or {},
            (self.compute or {}).get("sample_kwargs") or {},
            {"sampler": (self.quality or {}).get("sampler"), "steps": (self.quality or {}).get("sampling_steps")},
        ]
        sampler = next((s.get("sampler") for s in sources if s.get("sampler")), None)
        steps = next((s.get("steps") for s in sources if s.get("steps")), None)
        if sampler and steps:
            return f"{sampler}, {steps} step"
        return ""

    @property
    def name(self) -> str:
        """Tên hiển thị trong bảng, ví dụ "DDPM (ddim, 100 step)"."""
        detail = self.sampler_text or self.label
        return f"{self.display_model} ({detail})" if detail else self.display_model

    @property
    def who(self) -> str:
        """Tên dòng trong thông báo lỗi: "<tên hiển thị> / <run_name>"."""
        return f"{self.name} / {self.run_name}"

    @property
    def run_who(self) -> str:
        """Tên run trong thông báo lỗi (không kèm sampler): "<mô hình> / <run_name>"."""
        return f"{self.display_model} / {self.run_name}"


def build_rows(records: dict[str, list[dict]], dataset: str, issues: IssueLog) -> list[ResultRow]:
    """Ghép quality/speed/compute theo (model, run_name, label) rồi nối metadata run theo (model, run_name)."""
    runs: dict[tuple, dict] = {}
    for rec in records["run"]:
        if rec.get("dataset") != dataset:
            continue
        key = (rec.get("model"), rec.get("run_name"))
        if key in runs:
            issues.error(f"Trùng metadata run cho {key}: {runs[key]['_path']} và {rec['_path']}")
        runs[key] = rec
    rows: dict[tuple, ResultRow] = {}
    for kind in ("quality", "speed", "compute"):
        for rec in records[kind]:
            if rec.get("dataset") != dataset:
                continue
            key = (rec.get("model"), rec.get("run_name"), rec.get("label") or "")
            row = rows.setdefault(key, ResultRow(model=key[0], run_name=key[1], label=key[2], dataset=dataset))
            existing = getattr(row, kind)
            if existing is not None:
                issues.error(f"Trùng file {kind} cho {key}: {existing['_path']} và {rec['_path']}")
            setattr(row, kind, rec)
    for row in rows.values():
        row.run = runs.get((row.model, row.run_name))
    order = {m: i for i, m in enumerate(MODELS)}
    return sorted(rows.values(), key=lambda r: (order.get(r.model, 99), r.run_name or "", r.label))


def one_row_per_run(rows: list[ResultRow]) -> list[ResultRow]:
    """Một dòng đại diện cho mỗi (model, run_name), ưu tiên dòng có file compute.

    Các biến thể sampler (ddim100, ddpm500) dùng chung một lần train.
    """
    chosen: dict[tuple, ResultRow] = {}
    for row in rows:
        key = (row.model, row.run_name)
        if key not in chosen or (row.compute and not chosen[key].compute):
            chosen[key] = row
    return list(chosen.values())
