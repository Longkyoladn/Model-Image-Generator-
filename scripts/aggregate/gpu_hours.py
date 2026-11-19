"""Quy đổi giờ train trên mọi phần cứng về phần cứng tham chiếu (T4), mục 2.5."""

from __future__ import annotations

from evaluation.common import normalize_config


def is_reference_device(device: str, reference: str) -> bool:
    """T4 đơn là tham chiếu; "Tesla T4 x2" (nhiều GPU) thì không."""
    d = (device or "").lower()
    return reference.lower() in d and " x" not in d


def calibration_key(rec: dict) -> tuple:
    """Khóa ghép calibration với run: cùng config nhưng khác dataset là hai phép đo khác nhau."""
    return (rec.get("dataset"), rec.get("model"), normalize_config(rec.get("config")))


def build_calibration_index(records: list[dict]) -> tuple[dict[tuple, dict[str, dict]], list[str]]:
    """calibration_key -> {device -> file calibration}; kèm lỗi thiếu dataset hoặc trùng file."""
    index: dict[tuple, dict[str, dict]] = {}
    problems = []
    for rec in records:
        if not rec.get("dataset"):
            problems.append(f"File calibration {rec['_path']} thiếu trường dataset.")
            continue
        by_device = index.setdefault(calibration_key(rec), {})
        device = rec.get("device")
        if device in by_device:
            problems.append(
                f"Trùng file calibration cho {calibration_key(rec)} trên {device!r}: "
                f"{by_device[device]['_path']} và {rec['_path']}"
            )
        by_device[device] = rec
    return index, problems


def train_hours_by_device(run: dict) -> dict[str, float]:
    """Giờ train thực tế cộng dồn theo từng phần cứng, từ trường `hardware` của metadata run."""
    hours: dict[str, float] = {}
    for item in run.get("hardware") or []:
        hours[item.get("device")] = hours.get(item.get("device"), 0.0) + float(item.get("train_hours") or 0.0)
    return hours


class GpuHoursConverter:
    """Tính hệ số k_h và thời gian train quy đổi về phần cứng tham chiếu."""

    def __init__(self, calibrations: dict[tuple, dict[str, dict]], reference: str) -> None:
        self.calibrations = calibrations
        self.reference = reference

    def speed_ratio(self, run: dict, device: str) -> tuple[float | None, list[str]]:
        """k_h = throughput trên `device` / throughput trên máy tham chiếu, cho config của run.

        Trả về (None, [lý do]) nếu thiếu file calibration; kèm ghi chú khi
        calibration dùng batch size hoặc precision khác run.
        """
        if is_reference_device(device, self.reference):
            return 1.0, []
        by_device = self.calibrations.get(calibration_key(run), {})
        ref = next((rec for dev, rec in by_device.items() if is_reference_device(dev, self.reference)), None)
        here = by_device.get(device)
        missing = []
        if ref is None:
            missing.append(f"chưa có file calibration trên {self.reference}")
        if here is None:
            missing.append(f"chưa có file calibration trên {device!r}")
        if missing:
            return None, [f"không quy đổi được giờ train trên {device!r} của config {run.get('config')} "
                          f"(dataset {run.get('dataset')}): " + ", ".join(missing)]
        notes = []
        for name, rec in ((self.reference, ref), (device, here)):
            for key in ("batch_size", "precision"):
                if run.get(key) is not None and rec.get(key) != run.get(key):
                    notes.append(f"calibration trên {name} dùng {key} = {rec.get(key)}, run dùng {run.get(key)}")
        return here["throughput_img_s"] / ref["throughput_img_s"], notes

    def to_reference_hours(self, run: dict, hours_by_device: dict[str, float]) -> tuple[float | None, list[str]]:
        """Σ_h (giờ trên h × k_h); None nếu một phần cứng không quy đổi được."""
        total, notes = 0.0, []
        for device, hours in hours_by_device.items():
            k, messages = self.speed_ratio(run, device)
            notes += messages
            if k is None:
                return None, notes
            total += hours * k
        return total, notes

    def run_reference_hours(self, run: dict) -> float | None:
        """Thời gian - T4 của cả lần train; None nếu run không ghi `hardware` hoặc không quy đổi được."""
        hours = train_hours_by_device(run)
        return self.to_reference_hours(run, hours)[0] if hours else None
