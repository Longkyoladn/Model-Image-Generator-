import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

# Đặt project root lên đầu sys.path để ưu tiên package local thay vì HuggingFace datasets
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
import yaml

from datasets.preprocessing.builder import build_dataloader, denormalize
from datasets.preprocessing.fashion_mnist import FASHION_MNIST_CLASSES
from datasets.preprocessing.cifar10 import CIFAR10_CLASSES

def analyze_and_visualize(dataset_name="fashion_mnist", output_dir="outputs/data_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    config_file = f"configs/datasets/{dataset_name}.yaml"
    
    print("\n" + "=" * 70)
    print(f"📊 PHÂN TÍCH VÀ TRỰC QUAN HÓA DỮ LIỆU: {dataset_name.upper()}")
    print("=" * 70)
    
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    # Tạo Dataloader cho train và test
    train_loader, train_ds = build_dataloader(cfg, split="train", batch_size=64, shuffle=True)
    val_loader, val_ds = build_dataloader(cfg, split="val", batch_size=64, shuffle=False)
    test_loader, test_ds = build_dataloader(cfg, split="test", batch_size=64, shuffle=False)
    
    print(f" - Số mẫu tập Train:      {len(train_ds):,}")
    print(f" - Số mẫu tập Validation: {len(val_ds):,}")
    print(f" - Số mẫu tập Test:       {len(test_ds):,}")
    print(f" - Tổng cộng số mẫu:      {len(train_ds) + len(val_ds) + len(test_ds):,}")
    
    # Lấy 1 batch để kiểm tra
    batch_imgs, batch_labels = next(iter(train_loader))
    print(f" - Kích thước Tensor batch: {list(batch_imgs.shape)}")
    print(f" - Giá trị Min thực tế:      {batch_imgs.min().item():.4f}")
    print(f" - Giá trị Max thực tế:      {batch_imgs.max().item():.4f}")
    print(f" - Giá trị Mean:             {batch_imgs.mean().item():.4f}")
    print(f" - Độ lệch chuẩn Std:        {batch_imgs.std().item():.4f}")
    
    # 1. Vẽ Grid ảnh mẫu (16 ảnh)
    imgs_16 = batch_imgs[:16]
    imgs_display = denormalize(imgs_16, norm_range="[-1, 1]")
    grid = vutils.make_grid(imgs_display, nrow=4, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    if grid_np.shape[2] == 1:
        plt.imshow(grid_np.squeeze(2), cmap="gray")
    else:
        plt.imshow(grid_np)
    plt.title(f"Mẫu ảnh {dataset_name} sau Preprocessing & Normalize [-1, 1]", fontsize=12, fontweight="bold")
    plt.axis("off")
    
    sample_img_path = os.path.join(output_dir, f"{dataset_name}_batch_samples.png")
    plt.savefig(sample_img_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f" -> Đã lưu ảnh mẫu tại: {sample_img_path}")
    
    # 2. Vẽ biểu đồ phân phối nhãn nếu có class names
    class_names = None
    if "fashion" in dataset_name:
        class_names = FASHION_MNIST_CLASSES
    elif "cifar" in dataset_name:
        class_names = CIFAR10_CLASSES
    elif "celeb" in dataset_name:
        class_names = ["Non-smiling (0)", "Smiling (1)"]
        
    if class_names:
        # Thống kê phân phối trong 1000 mẫu đầu tiên
        labels_sample = []
        for i in range(min(len(train_ds), 2000)):
            _, l = train_ds[i]
            labels_sample.append(l)
            
        counts = np.bincount(labels_sample, minlength=len(class_names))
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(class_names)), counts, color="#2b5c8f", edgecolor="black", alpha=0.8)
        plt.xticks(range(len(class_names)), class_names, rotation=35, ha="right", fontsize=9)
        plt.ylabel("Số lượng mẫu")
        plt.title(f"Phân phối nhãn ({dataset_name}) - Kiểm tra cân bằng lớp", fontsize=12, fontweight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        dist_img_path = os.path.join(output_dir, f"{dataset_name}_class_distribution.png")
        plt.savefig(dist_img_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f" -> Đã lưu biểu đồ phân phối nhãn tại: {dist_img_path}")
        
    print(f"✅ Hoàn tất phân tích {dataset_name}!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fashion_mnist", choices=["fashion_mnist", "cifar10", "celeba"])
    args = parser.parse_args()
    
    analyze_and_visualize(args.dataset)
