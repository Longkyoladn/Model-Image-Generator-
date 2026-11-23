# Hướng dẫn sử dụng Pipeline Đánh giá (Evaluation Pipeline)

Tài liệu hướng dẫn chạy các công cụ đánh giá chất lượng ảnh, độ đa dạng và so sánh mô hình theo quy chuẩn `TASK_ASSIGNMENT.md` (mục 2.6 và 4.2).

---

## 1. Cấu hình đánh giá chung (`configs/eval/fid.yaml`)

Cả ba mô hình (VAE, DCGAN, DDPM) đều dùng chung cấu hình tại `configs/eval/fid.yaml`:
- **Thư viện**: `clean-fid` (mode `clean`), mạng InceptionV3 2048-d pool3.
- **Batch size trích đặc trưng**: 100.
- **KID**: `subset_size = 1000`, `num_subsets = 100`.
- **PRDC**: $k = 5$ trên 10.000 ảnh.
- **LPIPS**: Backbone AlexNet trên 2.000 cặp ảnh ngẫu nhiên.
- **Format ảnh**: PNG uint8 (`round(clamp(x) * 255)`). Không lưu JPEG.

---

## 2. Dùng trong Giai đoạn 1 — Tuning ngắn (KID-5k trên validation)

Khi thử nghiệm các siêu tham số, mỗi mô hình sinh **5.000 ảnh** và đánh giá KID trên tập **validation** để chọn ra cấu hình tốt nhất:

```python
from evaluation.fid import calculate_kid_tuning

# fake_dir: Thư mục chứa 5000 ảnh PNG sinh ra từ checkpoint tuning
# val_dir: Thư mục chứa ảnh validation thật đã preprocess (không augmentation)
results = calculate_kid_tuning(
    fake_dir="outputs/samples/cifar10/vae/tuning_run_1",
    val_dir="datasets/splits/cifar10/val",
    dataset="cifar10",
    resolution=32,
    device="cuda", # hoặc "cpu"
)

print(f"KID Val Mean: {results['kid_val_mean']:.6f} ± {results['kid_val_std']:.6f}")
```

Kết quả `kid_val_mean` được ghi vào `metrics.kid_val_mean` của file run metadata giai đoạn tuning.

---

## 3. Dùng trong Giai đoạn 2 — Đánh giá cuối (Bảng chính & baseline)

Mỗi mô hình sinh **50.000 ảnh** (với seed thống nhất), lưu vào một thư mục PNG.
Chạy script `evaluation/run_quality.py` để tự động tính toàn bộ các metric và sinh 2 file JSON chuẩn:

### 3.1. Chạy cho dòng chính (VAE, DCGAN, DDPM DDIM 100)
```bash
python evaluation/run_quality.py \
  --model vae \
  --dataset cifar10 \
  --run-name vae_base \
  --fake outputs/samples/cifar10/vae/vae_base \
  --train datasets/splits/cifar10/train \
  --test datasets/splits/cifar10/test \
  --resolution 32 \
  --device cuda
```
Đối với DDPM (DDIM 100 step):
```bash
python evaluation/run_quality.py \
  --model ddpm \
  --dataset cifar10 \
  --run-name ddpm_base \
  --label ddim100 \
  --sampler ddim \
  --steps 100 \
  --fake outputs/samples/cifar10/ddpm/ddpm_base__ddim100 \
  --train datasets/splits/cifar10/train \
  --test datasets/splits/cifar10/test \
  --resolution 32 \
  --device cuda
```

### 3.2. Chạy cho dòng phụ DDPM (DDPM 500 step - so sánh sampler)
```bash
python evaluation/run_quality.py \
  --model ddpm \
  --dataset cifar10 \
  --run-name ddpm_base \
  --label ddpm500 \
  --sampler ddpm \
  --steps 500 \
  --comparison sampler \
  --fake outputs/samples/cifar10/ddpm/ddpm_base__ddpm500 \
  --train datasets/splits/cifar10/train \
  --test datasets/splits/cifar10/test \
  --resolution 32 \
  --device cuda
```

Output tự động xuất ra:
- `outputs/metrics/quality/<dataset>__<model>__<run_name>[__<label>].json`
- `outputs/metrics/real_baseline/<dataset>.json`

---

## 4. Kiểm tra học thuộc — Nearest Neighbors (`nearest_neighbor.png`)

Sinh lưới 32 hàng × 4 cột (mỗi hàng gồm 1 ảnh sinh + 3 ảnh train gần nhất):

```python
from evaluation.fid import render_nearest_neighbors

render_nearest_neighbors(
    fake_dir="outputs/samples/cifar10/vae/vae_base",
    train_dir="datasets/splits/cifar10/train",
    output_path="outputs/figures/cifar10/vae__vae_base/nearest_neighbor.png",
    num_samples=32,
    num_neighbors=3,
    device="cuda",
)
```

---

## 5. Đo độ đa dạng riêng lẻ (`evaluation/diversity.py`)

```python
from evaluation.diversity import calculate_lpips_diversity, calculate_prdc

# LPIPS giữa 2000 cặp ảnh sinh
lpips_score = calculate_lpips_diversity("outputs/samples/cifar10/vae/vae_base", num_pairs=2000, device="cuda")

# PRDC (Precision, Recall, Density, Coverage)
prdc_scores = calculate_prdc(real_features, fake_features, nearest_k=5)
```
