import os
import json
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPLITS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "splits")

def create_or_load_split(dataset_name, total_train_size, total_test_size, val_size=10000, seed=42, output_dir=None, labels=None, force_recreate=False):
    """
    Tạo hoặc đọc metadata split train/val/test cố định bằng random seed.
    Hỗ trợ Stratified Split khi cung cấp `labels` để đảm bảo tỷ lệ các lớp trong tập val
    luôn phân bổ đồng đều tuyệt đối theo đúng yêu cầu phân tầng của đề tài.
    """
    if output_dir is None:
        output_dir = DEFAULT_SPLITS_DIR
    elif not os.path.isabs(output_dir) and not os.path.exists(output_dir):
        output_dir = DEFAULT_SPLITS_DIR
        
    os.makedirs(output_dir, exist_ok=True)
    split_path = os.path.join(output_dir, f"{dataset_name}_split.json")
    
    if not force_recreate and os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Nếu yêu cầu phân tầng (labels is not None) nhưng file cũ chưa phải là stratified_random, tự động tạo lại
            if labels is not None and data.get("split_type") != "stratified_random":
                pass
            else:
                return data
            
    rng = np.random.RandomState(seed)
    
    # 1. Stratified Split theo nhãn lớp nếu có danh sách nhãn
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels_arr = labels.cpu().numpy()
        elif isinstance(labels, (list, tuple)):
            labels_arr = np.array(labels)
        else:
            labels_arr = np.asarray(labels)
            
        unique_classes = np.unique(labels_arr)
        val_idx = []
        train_idx = []
        
        for c in unique_classes:
            c_indices = np.where(labels_arr == c)[0].copy()
            rng.shuffle(c_indices)
            
            # Tính số lượng mẫu validation cho lớp c theo đúng tỷ lệ
            n_val_c = int(round(len(c_indices) / len(labels_arr) * val_size))
            val_idx.extend(c_indices[:n_val_c].tolist())
            train_idx.extend(c_indices[n_val_c:].tolist())
            
        # Đảm bảo tổng số lượng val_idx chính xác tuyệt đối bằng val_size
        if len(val_idx) != val_size:
            diff = val_size - len(val_idx)
            if diff > 0:
                val_idx.extend(train_idx[:diff])
                train_idx = train_idx[diff:]
            else:
                train_idx.extend(val_idx[diff:])
                val_idx = val_idx[:val_size]
                
        # Trộn ngẫu nhiên lại các index đã chọn để các batch nạp không bị gom cụm theo lớp
        rng.shuffle(val_idx)
        rng.shuffle(train_idx)
        split_type = "stratified_random"
        
    else:
        # Fallback chia ngẫu nhiên thuần túy nếu không có nhãn
        train_indices = np.arange(total_train_size)
        rng.shuffle(train_indices)
        
        actual_val_size = min(val_size, int(total_train_size * 0.2))
        val_idx = train_indices[:actual_val_size].tolist()
        train_idx = train_indices[actual_val_size:].tolist()
        split_type = "random"
        
    test_idx = list(range(total_test_size))
    
    split_meta = {
        "dataset": dataset_name,
        "split_type": split_type,
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
        
    print(f"Created reproducible {split_type} split file at: {split_path}")
    print(f"  -> Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")
    return split_meta
