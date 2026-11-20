"""Ghép ảnh mẫu, interpolation, nearest-neighbor của các mô hình thành một ảnh so sánh."""

from __future__ import annotations

from pathlib import Path

from scripts.aggregate.constants import FIGURE_NAMES, SURFACE, TEXT_PRIMARY
from scripts.aggregate.issues import IssueLog
from scripts.aggregate.records import ResultRow


def find_figure(figures_dir: Path, row: ResultRow, name: str) -> Path | None:
    """Ảnh `<name>.png` của dòng: ưu tiên thư mục có label (biến thể sampler), sau đó thư mục chung của run.

    VAE/DCGAN không có label nên chỉ có thư mục chung; interpolation và
    nearest-neighbor của DDPM cũng thường lưu ở thư mục chung của run.
    """
    base = figures_dir / row.dataset
    candidates = []
    if row.label:
        candidates.append(base / f"{row.model}__{row.run_name}__{row.label}" / f"{name}.png")
    candidates.append(base / f"{row.model}__{row.run_name}" / f"{name}.png")
    return next((c for c in candidates if c.exists()), None)


def combine_model_figures(dataset: str, rows: list[ResultRow], figures_dir: Path, out_dir: Path,
                          issues: IssueLog) -> dict[str, Path]:
    """Mỗi loại ảnh một file `<dataset>_<name>.png`, mỗi mô hình một ô có tiêu đề."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=18)
    written = {}
    for name in FIGURE_NAMES:
        panels, seen = [], set()
        for row in rows:
            path = find_figure(figures_dir, row, name)
            if path is None:
                issues.warning(f"{row.who}: thiếu {name}.png.")
                continue
            if path in seen:
                continue
            seen.add(path)
            # Ảnh trong thư mục có label thì ghi cả sampler; ảnh chung của run chỉ ghi tên mô hình.
            title = row.name if path.parent.name.count("__") == 2 else row.display_model
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
