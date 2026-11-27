"""
Script xuất thư mục ảnh thật tham chiếu (Reference Real Images)
Phục vụ mục 2.6 và nhiệm vụ 3.1 trong TASK_ASSIGNMENT.md:
- Trích xuất ảnh thật từ DataLoader (qua đúng preprocessing: crop, pad/resize, normalize).
- TUYỆT ĐỐI KHÔNG augmentation (horizontal flip bị tắt hoàn toàn).
- Định dạng chuẩn: PNG uint8 [0, 255] (công thức round(clamp(x) * 255)).
- Với Fashion-MNIST: lặp 1 kênh grayscale thành 3 kênh RGB để tương thích với InceptionV3.
- Dùng để Quân tính FID sàn (real-to-real baseline) và lưu cache Inception statistics dùng chung.
"""

import os
import sys
import time
import json
import copy
import argparse
from pathlib import Path

# Đặt project root lên đầu sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torchvision.io as tv_io
from PIL import Image

from datasets.preprocessing.builder import build_dataloader, load_yaml_config

SUPPORTED_DATASETS = ["fashion_mnist", "cifar10", "celeba"]
SUPPORTED_SPLITS = ["train", "val", "test"]


def export_reference_split(
    dataset_name: str,
    split: str,
    output_base_dir: str = "outputs/reference",
    batch_size: int = 128,
    num_workers: int = 0,
    max_samples: int = None
):
    """
    Xuất ảnh thật tham chiếu cho 1 split của 1 dataset cụ thể.
    """
    if not os.path.isabs(output_base_dir):
        output_base_dir = os.path.join(PROJECT_ROOT, output_base_dir)

    split_out_dir = os.path.join(output_base_dir, dataset_name, split)
    os.makedirs(split_out_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print(f"📦 XUẤT ẢNH THẬT THAM CHIẾU: {dataset_name.upper()} | SPLIT: {split.upper()}")
    print("=" * 75)

    # 1. Nạp file cấu hình dataset tương ứng
    config_path = f"configs/datasets_config/{dataset_name}.yaml"
    try:
        cfg = load_yaml_config(config_path)
    except FileNotFoundError as e:
        print(f"❌ [LỖI]: {e}")
        return False

    # 2. Ép tắt hoàn toàn Augmentation cho toàn bộ các split (theo Mục 2.6)
    cfg_no_aug = copy.deepcopy(cfg)
    if "augmentation" in cfg_no_aug:
        for s in ["train", "val", "test"]:
            if s in cfg_no_aug["augmentation"]:
                cfg_no_aug["augmentation"][s]["horizontal_flip"] = False
                cfg_no_aug["augmentation"][s]["flip_probability"] = 0.0

    # 3. Tạo DataLoader
    print(f" - Khởi tạo DataLoader không Augmentation (batch_size={batch_size})...")
    try:
        loader, ds = build_dataloader(
            cfg_no_aug,
            split=split,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    except Exception as e:
        print(f"⚠️ Không thể nạp dataset '{dataset_name}' split '{split}': {e}")
        print("   (Nếu chưa có dữ liệu thô, hãy tải dữ liệu hoặc chạy trên Kaggle/Colab)")
        return False

    total_available = len(ds)
    target_count = min(total_available, max_samples) if max_samples else total_available
    print(f" - Tổng số ảnh trong dataset: {total_available:,}")
    print(f" - Số lượng ảnh sẽ xuất:     {target_count:,}")
    print(f" - Thư mục đích:             {split_out_dir}")

    # Lấy thông tin chuẩn hóa từ cấu hình
    norm_cfg = cfg.get("normalization", {})
    norm_range = norm_cfg.get("range", "[-1, 1]")

    saved_count = 0
    labels_list = []
    start_time = time.time()
    img_shape = None

    for batch_idx, (batch_imgs, batch_targets) in enumerate(loader):
        if saved_count >= target_count:
            break

        # Denormalize về khoảng [0, 1]
        if norm_range == "[-1, 1]":
            imgs_01 = (batch_imgs + 1.0) / 2.0
        else:
            imgs_01 = batch_imgs
        imgs_01 = torch.clamp(imgs_01, 0.0, 1.0)

        # Grayscale (Fashion-MNIST): lặp 1 kênh thành 3 kênh RGB theo đúng mục 2.6
        if imgs_01.shape[1] == 1:
            imgs_01 = imgs_01.repeat(1, 3, 1, 1)

        # Chuyển về uint8 [0, 255] theo công thức: round(clamp(x) * 255)
        imgs_uint8 = torch.clamp(torch.round(imgs_01 * 255.0), 0, 255).to(torch.uint8)

        # Lưu từng ảnh thành file .png
        batch_n = imgs_uint8.shape[0]
        for i in range(batch_n):
            if saved_count >= target_count:
                break

            img_chw = imgs_uint8[i]
            if img_shape is None:
                img_shape = [int(img_chw.shape[1]), int(img_chw.shape[2])]

            img_path = os.path.join(split_out_dir, f"{saved_count:06d}.png")

            # Ghi file PNG uint8 không nén mất mát
            try:
                tv_io.write_png(img_chw.cpu(), img_path)
            except Exception:
                # Fallback với PIL
                img_hwc = img_chw.permute(1, 2, 0).cpu().numpy()
                Image.fromarray(img_hwc).save(img_path, format="PNG")

            # Lưu nhãn đi kèm
            target_val = batch_targets[i]
            if isinstance(target_val, torch.Tensor):
                if target_val.numel() == 1:
                    labels_list.append(target_val.item())
                else:
                    labels_list.append(target_val.tolist())
            else:
                labels_list.append(target_val)

            saved_count += 1

        # Tiến trình
        if (saved_count % 1000 == 0) or (saved_count == target_count):
            elapsed = time.time() - start_time
            speed = saved_count / max(elapsed, 1e-4)
            print(f"   --> Đã xuất: {saved_count:,}/{target_count:,} ảnh ({speed:.1f} ảnh/s)")

    # 4. Ghi file metadata và labels
    metadata = {
        "dataset": dataset_name,
        "split": split,
        "total_images": saved_count,
        "resolution": img_shape or [32, 32],
        "channels": 3,
        "format": "PNG uint8 [0, 255]",
        "no_augmentation": True,
        "norm_range_source": norm_range,
        "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    meta_file = os.path.join(split_out_dir, "metadata.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    labels_file = os.path.join(split_out_dir, "labels.json")
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(labels_list, f)

    elapsed_total = time.time() - start_time
    print(f"✅ Hoàn tất xuất {saved_count:,} ảnh cho '{dataset_name} ({split})' trong {elapsed_total:.2f}s!")
    print(f"   Metadata: {meta_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Xuất thư mục ảnh thật tham chiếu (PNG uint8, không augmentation)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["fashion_mnist", "cifar10", "celeba", "all"],
        help="Tên dataset cần xuất (hoặc 'all' cho cả 3)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Split cần xuất ('train', 'val', 'test', hoặc 'all')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/reference",
        help="Thư mục gốc lưu ảnh tham chiếu (mặc định: outputs/reference)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size khi đọc DataLoader (mặc định: 128)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Số luồng đọc dữ liệu (mặc định: 0)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Giới hạn số mẫu xuất tối đa (dùng để test thử hoặc lấy mẫu nhỏ)"
    )

    args = parser.parse_args()

    datasets_to_run = SUPPORTED_DATASETS if args.dataset == "all" else [args.dataset]
    splits_to_run = SUPPORTED_SPLITS if args.split == "all" else [args.split]

    print("\n" + "=" * 75)
    print("🚀 BẮT ĐẦU PIPELINE XUẤT ẢNH THẬT THAM CHIẾU (TASK 3.1 & MỤC 2.6)")
    print("=" * 75)
    print(f" - Datasets:  {datasets_to_run}")
    print(f" - Splits:    {splits_to_run}")
    print(f" - Thư mục:   {args.output_dir}")
    print(f" - Batch size:{args.batch_size}")
    if args.max_samples:
        print(f" - Giới hạn:  {args.max_samples} ảnh/split")

    success_count = 0
    for ds_name in datasets_to_run:
        for sp in splits_to_run:
            ok = export_reference_split(
                dataset_name=ds_name,
                split=sp,
                output_base_dir=args.output_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples=args.max_samples
            )
            if ok:
                success_count += 1

    print("\n" + "=" * 75)
    print(f"🎉 HOÀN TẤT TỔNG CỘNG: {success_count} splits đã được xử lý thành công!")
    print("=" * 75)


if __name__ == "__main__":
    main()
