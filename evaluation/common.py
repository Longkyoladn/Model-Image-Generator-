"""Hàm dùng chung cho các script đánh giá và script tổng hợp.

Mỗi script chạy trực tiếp (`python evaluation/speed.py`) vẫn phải tự thêm gốc
repo vào `sys.path` bằng một dòng trước khi import module này, vì khi chạy như
vậy Python chỉ thêm thư mục chứa script (`evaluation/`, `scripts/`) chứ không
thêm gốc repo:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # gốc repo, xem evaluation/common.py

Module này không import torch ở đầu file để script tổng hợp (không cần torch)
vẫn dùng được `normalize_config`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str | None:
    """Mã commit hiện tại của repo, ghi vào file JSON kết quả; None nếu không lấy được."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sync(device) -> None:
    """Đợi GPU chạy xong mọi lệnh đang xếp hàng, để bấm giờ đúng. Không làm gì trên CPU."""
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


def normalize_config(config: str | None) -> str | None:
    """Đưa đường dẫn config về dạng `configs/...` để mọi máy ghi ra cùng một chuỗi.

    Chỉ xử lý chuỗi, không đọc ổ đĩa: đổi `\\` thành `/`, rồi bỏ mọi thứ đứng
    trước thư mục `configs/` cuối cùng. Nhờ vậy run train trên Kaggle
    (`/kaggle/working/<repo>/configs/ddpm/cifar10/base.yaml`) và file
    calibration đo trên laptop (`configs/ddpm/cifar10/base.yaml`) ghép được với
    nhau, kể cả khi máy chạy script tổng hợp không có thư mục `/kaggle/...`.

    Chuỗi không chứa thư mục `configs` thì chỉ đổi dấu `\\` và bỏ tiền tố `./`.
    """
    if config is None:
        return None
    parts = str(config).replace("\\", "/").split("/")
    if "configs" in parts[:-1]:
        last = len(parts) - 1 - parts[:-1][::-1].index("configs") - 1
        return "/".join(parts[last:])
    path = "/".join(parts)
    while path.startswith("./"):
        path = path[2:]
    return path
