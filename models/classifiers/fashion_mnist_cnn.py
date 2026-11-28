"""CNN phân loại Fashion-MNIST để trích xuất đặc trưng penultimate 128-d.

Dùng cho FID feature riêng của Fashion-MNIST (mục 2.6 & 3.2 TASK_ASSIGNMENT.md),
vì InceptionV3 được train trên ảnh màu tự nhiên nên kém ý nghĩa với ảnh
quần áo grayscale kích thước nhỏ.

Kiến trúc (Input 1x32x32 trong [-1, 1]):
  Conv(1->32, 3x3, pad=1) -> ReLU -> MaxPool(2x2)   [16x16]
  Conv(32->64, 3x3, pad=1) -> ReLU -> MaxPool(2x2)  [8x8]
  Flatten -> Linear(64*8*8, 128) -> ReLU            <- penultimate feature 128-d
  Dropout(0.5) -> Linear(128, 10)                   <- logits 10 lớp
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.preprocessing.builder import build_dataloader, load_yaml_config

FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
NUM_CLASSES = 10
FEATURE_DIM = 128
ARCHITECTURE_DESC = (
    "Conv2d(1->32,3x3)+ReLU+MaxPool2d(2) -> Conv2d(32->64,3x3)+ReLU+MaxPool2d(2) "
    "-> Linear(4096->128)+ReLU [penultimate 128-d] -> Dropout(0.5) -> Linear(128->10)"
)


class FashionMNISTCNN(nn.Module):
    """CNN nhỏ phân loại 10 lớp Fashion-MNIST, trích xuất feature 128-d ở lớp áp chót."""

    def __init__(self, num_classes: int = NUM_CLASSES, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, feature_dim)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về logits (N, num_classes)."""
        h = self.conv(x)
        h = self.flat(h)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        return self.fc2(h)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất vector đặc trưng penultimate 128-d, shape (N, feature_dim)."""
        self.eval()
        with torch.no_grad():
            h = self.conv(x)
            h = self.flat(h)
            h = self.fc1(h)
            h = self.relu(h)
        return h


def build_model(device: str | torch.device = "cpu") -> FashionMNISTCNN:
    return FashionMNISTCNN().to(device)


def train_model(
    model: FashionMNISTCNN,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    device: str | torch.device = "cpu",
    epochs: int = 10,
    lr: float = 1e-3,
) -> dict:
    """Huấn luyện CNN trên Fashion-MNIST, giữ lại trọng số có val_acc cao nhất."""
    dev = torch.device(device)
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_acc": []}
    best_acc = -1.0
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for images, labels in train_loader:
            images = images.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        if val_loader is not None:
            acc = evaluate(model, val_loader, dev)
            history["val_acc"].append(acc)
            if acc >= best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"Epoch {epoch}/{epochs} - loss={avg_loss:.4f} - val_acc={acc:.4f}")
        else:
            print(f"Epoch {epoch}/{epochs} - loss={avg_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(dev)

    return history


def evaluate(model: FashionMNISTCNN, loader: DataLoader, device: str | torch.device = "cpu") -> float:
    """Độ chính xác phân loại trên tập đánh giá."""
    dev = torch.device(device)
    model.to(dev)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(dev)
            labels = labels.to(dev)
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum())
            total += int(labels.size(0))
    return correct / max(total, 1)


def _load_grayscale_32_tensor(path: Path) -> torch.Tensor:
    """Đọc 1 file PNG/JPG bất kỳ thành tensor grayscale (1, 32, 32) trong [-1, 1]."""
    with Image.open(path) as im:
        gray = im.convert("L")
        if gray.size != (32, 32):
            gray = gray.resize((32, 32), Image.BILINEAR)
        arr = np.asarray(gray, dtype=np.float32) / 255.0
    tensor_01 = torch.from_numpy(arr).unsqueeze(0)
    return tensor_01 * 2.0 - 1.0


def extract_features_for_folder(
    model: FashionMNISTCNN,
    image_folder: str | Path | list[Path | str],
    device: str | torch.device = "cpu",
    batch_size: int = 100,
) -> torch.Tensor:
    """Trích xuất feature 128-d cho thư mục phẳng chứa ảnh PNG (hoặc danh sách đường dẫn ảnh)."""
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    if isinstance(image_folder, (str, Path)):
        p = Path(image_folder)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh Fashion-MNIST: {p}")
        if p.is_dir():
            files = sorted(p.glob("*.png"))
            if not files:
                # Hỗ trợ trường hợp thư mục có subfolder (ví dụ cấu trúc ImageFolder cũ)
                files = sorted(p.rglob("*.png"))
        else:
            files = [p]
    elif isinstance(image_folder, list):
        files = [Path(f) for f in image_folder]
    else:
        raise TypeError(f"image_folder không hợp lệ: {type(image_folder)}")

    if not files:
        return torch.empty(0, model.feature_dim)

    features: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(files), batch_size):
            batch_files = files[start : start + batch_size]
            batch_imgs = torch.stack([_load_grayscale_32_tensor(f) for f in batch_files]).to(dev)
            feats = model.extract_features(batch_imgs)
            features.append(feats.cpu())

    return torch.cat(features, dim=0)


def save_checkpoint(
    model: FashionMNISTCNN,
    path: str | Path,
    val_acc: float | None = None,
    test_acc: float | None = None,
    class_to_idx: dict | None = None,
) -> None:
    """Lưu checkpoint CNN kèm kiến trúc và độ chính xác theo mục 2.6."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    num_params = sum(param.numel() for param in model.parameters())
    payload = {
        "state_dict": model.state_dict(),
        "architecture": ARCHITECTURE_DESC,
        "num_parameters": int(num_params),
        "num_classes": model.num_classes,
        "feature_dim": model.feature_dim,
        "classes": FASHION_MNIST_CLASSES,
        "split_file": "datasets/splits/fashion_mnist_split.json",
        "val_acc": val_acc,
        "test_acc": test_acc if test_acc is not None else val_acc,
        "class_to_idx": class_to_idx,
    }
    torch.save(payload, p)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[FashionMNISTCNN, dict]:
    """Tải checkpoint, trả về (model, metadata)."""
    p = Path(path)
    payload = torch.load(p, map_location=device, weights_only=False)
    model = FashionMNISTCNN(
        num_classes=int(payload.get("num_classes", NUM_CLASSES)),
        feature_dim=int(payload.get("feature_dim", FEATURE_DIM)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def build_dataloaders(
    root_dir: str | Path | None = None,
    batch_size: int = 128,
    num_workers: int = 0,
    val_size: int = 10000,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Tạo DataLoader train/val/test đúng chuẩn split phân tầng chung của dự án (50k/10k/10k)."""
    cfg = load_yaml_config("configs/datasets/fashion_mnist.yaml")
    if root_dir is not None:
        cfg.setdefault("datasets_config", {})["raw_dir"] = str(root_dir)
    cfg.setdefault("splits", {})["val_samples"] = int(val_size)
    cfg["splits"]["random_seed"] = int(seed)

    train_loader, _ = build_dataloader(
        cfg, split="train", batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    val_loader, _ = build_dataloader(
        cfg, split="val", batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    test_loader, _ = build_dataloader(
        cfg, split="test", batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    return train_loader, val_loader, test_loader


def train_and_save(
    checkpoint_path: str | Path = "outputs/checkpoints/fashion_mnist_cnn.pt",
    epochs: int = 10,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    seed: int = 42,
    info_json_path: str | Path | None = "outputs/metrics/fashion_mnist_cnn_info.json",
) -> dict:
    """Huấn luyện trên split chính thức 50k/10k/10k và lưu checkpoint + file thông tin CNN."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader, val_loader, test_loader = build_dataloaders(seed=seed)
    model = build_model(device)
    history = train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr)
    best_val_acc = max(history["val_acc"]) if history["val_acc"] else None
    test_acc = evaluate(model, test_loader, device)
    save_checkpoint(model, checkpoint_path, val_acc=best_val_acc, test_acc=test_acc)

    info = {
        "checkpoint": str(checkpoint_path),
        "architecture": ARCHITECTURE_DESC,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "feature_dim": FEATURE_DIM,
        "num_classes": NUM_CLASSES,
        "split_file": "datasets/splits/fashion_mnist_split.json",
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "best_val_acc": float(best_val_acc) if best_val_acc is not None else None,
        "test_acc": float(test_acc),
        "train_loss": [round(x, 6) for x in history["train_loss"]],
        "val_acc": [round(x, 6) for x in history["val_acc"]],
        "classes": FASHION_MNIST_CLASSES,
    }
    if info_json_path is not None:
        p_info = Path(info_json_path)
        p_info.parent.mkdir(parents=True, exist_ok=True)
        p_info.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Huấn luyện CNN phân loại Fashion-MNIST (trích xuất feature 128-d).")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/fashion_mnist_cnn.pt")
    parser.add_argument("--info-json", default="outputs/metrics/fashion_mnist_cnn_info.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    result = train_and_save(
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        info_json_path=args.info_json,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
