"""CNN phân loại Fashion-MNIST để trích xuất đặc trưng penultimate 128-d.

Dùng cho FID feature riêng của Fashion-MNIST (mục 2.6 TASK_ASSIGNMENT.md),
vì InceptionV3 được train trên ảnh màu tự nhiên nên kém ý nghĩa với ảnh
quần áo grayscale kích thước nhỏ.

Kiến trúc:
  Conv(1->32, 3x3) -> ReLU -> MaxPool
  Conv(32->64, 3x3) -> ReLU -> MaxPool
  Flatten -> Linear(64*6*6, 128)  <- penultimate feature 128-d
  ReLU -> Dropout -> Linear(128, 10)

Input: tensor (N, 1, 32, 32) trong [-1, 1] (đã qua preprocessing dataset).
Output logits: (N, 10).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
NUM_CLASSES = 10
FEATURE_DIM = 128


class FashionMNISTCNN(nn.Module):
    """CNN nhỏ phân loại 10 lớp Fashion-MNIST, trích xuất feature 128-d."""

    def __init__(self, num_classes: int = NUM_CLASSES, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, feature_dim)
        self.relu = nn.ReLU(inplace=True)
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
    """Huấn luyện CNN trên Fashion-MNIST. Trả về dict history (train_loss, val_acc)."""
    dev = torch.device(device)
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": []}
    best_acc = 0.0

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

            total_loss += float(loss)
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        if val_loader is not None:
            acc = evaluate(model, val_loader, dev)
            history["val_acc"].append(acc)
            if acc > best_acc:
                best_acc = acc
            print(f"Epoch {epoch}/{epochs} - loss={avg_loss:.4f} - val_acc={acc:.4f}")
        else:
            print(f"Epoch {epoch}/{epochs} - loss={avg_loss:.4f}")

    return history


def evaluate(model: FashionMNISTCNN, loader: DataLoader, device: str | torch.device = "cpu") -> float:
    """Độ chính xác trên tập đánh giá."""
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


def extract_features_for_folder(
    model: FashionMNISTCNN,
    image_folder: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int = 100,
) -> torch.Tensor:
    """Trích xuất feature 128-d cho mọi ảnh PNG trong thư mục, trả về tensor (N, 128)."""
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = datasets.ImageFolder(root=str(image_folder), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    features = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(dev)
            feats = model.extract_features(images)
            features.append(feats.cpu())
    return torch.cat(features, dim=0) if features else torch.empty(0, model.feature_dim)


def save_checkpoint(
    model: FashionMNISTCNN,
    path: str | Path,
    val_acc: float | None = None,
    class_to_idx: dict | None = None,
) -> None:
    """Lưu checkpoint (state_dict + metadata)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "num_classes": model.num_classes,
        "feature_dim": model.feature_dim,
        "classes": FASHION_MNIST_CLASSES,
        "val_acc": val_acc,
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
    root_dir: str | Path = "datasets/raw/fashion_mnist",
    batch_size: int = 128,
    num_workers: int = 0,
    val_size: int = 10000,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Tạo DataLoader train/val/test cho Fashion-MNIST (32x32, [-1, 1])."""
    transform = transforms.Compose([
        transforms.Pad(2, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_full = datasets.FashionMNIST(root=str(root_dir), train=True, download=True, transform=transform)
    test_full = datasets.FashionMNIST(root=str(root_dir), train=False, download=True, transform=transform)

    # Split stratified đơn giản theo seed cố định (dùng cho train classifier).
    generator = torch.Generator().manual_seed(seed)
    train_indices = torch.randperm(len(train_full), generator=generator).tolist()
    val_indices = train_indices[:val_size]
    train_indices = train_indices[val_size:]

    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(train_full, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(Subset(train_full, val_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_full, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def train_and_save(
    checkpoint_path: str | Path = "outputs/checkpoints/fashion_mnist_cnn.pt",
    epochs: int = 10,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
) -> dict:
    """Huấn luyện và lưu checkpoint CNN Fashion-MNIST. Trả về metadata."""
    train_loader, val_loader, test_loader = build_dataloaders()
    model = build_model(device)
    history = train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr)
    test_acc = evaluate(model, test_loader, device)
    save_checkpoint(model, checkpoint_path, val_acc=test_acc)
    return {
        "checkpoint": str(checkpoint_path),
        "test_acc": float(test_acc),
        "train_loss": history["train_loss"],
        "val_acc": history["val_acc"],
        "num_classes": NUM_CLASSES,
        "feature_dim": FEATURE_DIM,
        "classes": FASHION_MNIST_CLASSES,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Huấn luyện CNN phân loại Fashion-MNIST (trích xuất feature 128-d).")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/fashion_mnist_cnn.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    result = train_and_save(
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
