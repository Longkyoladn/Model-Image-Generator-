import os
import json
import numpy as np
import torch

def create_or_load_split(dataset_name, total_train_size, total_test_size, val_size=10000, seed=42, output_dir="datasets/splits"):
    """
    Tạo hoặc đọc metadata split train/val/test cố định bằng random seed.
    Giúp đảm bảo cả ba thành viên (Lâm, Quân, Long) và cả ba mô hình (GAN, VAE, DDPM)
    đều sử dụng CHÍNH XÁC cùng một tập ảnh train, val và test.
    """
    os.makedirs(output_dir, exist_ok=True)
    split_path = os.path.join(output_dir, f"{dataset_name}_split.json")
    
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    np.random.seed(seed)
    
    # Chia tập train gốc thành (train_new, val)
    train_indices = np.arange(total_train_size)
    np.random.shuffle(train_indices)
    
    actual_val_size = min(val_size, int(total_train_size * 0.2))
    val_idx = train_indices[:actual_val_size].tolist()
    train_idx = train_indices[actual_val_size:].tolist()
    test_idx = list(range(total_test_size))
    
    split_meta = {
        "dataset": dataset_name,
        "seed": seed,
        "total_samples": len(train_idx) + len(val_idx) + len(test_idx),
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "test_count": len(test_idx),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx
    }
    
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)
        
    print(f"Created reproducible split file at: {split_path}")
    print(f"  -> Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")
    return split_meta

