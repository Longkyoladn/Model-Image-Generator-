# Hướng dẫn sử dụng Pipeline Đánh giá (Evaluation Pipeline)

Tài liệu hướng dẫn chạy các công cụ đánh giá chất lượng ảnh, độ đa dạng, kiểm tra học thuộc và nội suy không gian ẩn theo quy chuẩn `TASK_ASSIGNMENT.md` (mục 2.6, 3.2 và 4.2).

---

## 1. Cấu hình đánh giá chung (`configs/eval/fid.yaml`)

Cả ba mô hình (**VAE**, **DCGAN**, **DDPM**) dùng chung cấu hình tại `configs/eval/fid.yaml`:
- **Thư viện**: `clean-fid` (mode `clean`), mạng InceptionV3 pool3 2048-d.
- **Batch size trích đặc trưng**: `100`.
- **KID**: `subset_size = 1000`, `num_subsets = 100`.
- **PRDC**: $k = 5$ trên `10.000` ảnh sinh đầu tiên so với `10.000` ảnh train chọn ngẫu nhiên theo seed cố định.
- **LPIPS diversity**: Backbone AlexNet trên `2.000` cặp ảnh phân biệt ($i \neq j$).
- **Format ảnh bắt buộc**: PNG uint8 (`round(clamp(x) * 255)`). Tuyệt đối không lưu JPEG.
- **Ảnh thật tham chiếu**: Xuất bằng `scripts/export_reference_images.py` ra `outputs/reference/<dataset>/<split>/`.
- **Cache đặc trưng**: Tự động lưu tại `outputs/cache/fid/<dataset>_<split>_<res>__<hash>.npz` (bị chặn khỏi Git qua `.gitignore`).

---

## 2. Giai đoạn 1 — Tuning ngắn (KID-5k trên validation)

Khi chọn siêu tham số (Giai đoạn 1), mỗi ứng viên sinh đúng **5.000 ảnh** PNG vào thư mục sạch và đo KID trên tập **validation** (`outputs/reference/<dataset>/val`):

```python
from evaluation.fid import calculate_kid_tuning, load_config

cfg, cfg_hash = load_config("configs/eval/fid.yaml")
results = calculate_kid_tuning(
    fake_dir="outputs/samples/cifar10/dcgan/tuning_run_1",
    val_dir="outputs/reference/cifar10/val",
    cfg=cfg,
    dataset="cifar10",
    resolution=32,
    device="cuda",
    seed=42,
    eval_config_hash=cfg_hash,
)
print(f"KID Val: {results['kid_val_mean']:.6f} ± {results['kid_val_std']:.6f}")
```

> Lưu ý: Với `scripts/train_vae.py --stage tuning`, script đã tự động sinh 5.000 ảnh tạm, tính KID validation và xuất file `outputs/metrics/quality/<dataset>__vae__<run_name>.json` cho bảng *"Các config đã thử"* của `aggregate_results.py`.

---

## 3. Giai đoạn 2 — Đánh giá cuối (Bảng chính, FID riêng Fashion-MNIST & Nearest-Neighbor)

Mỗi mô hình sinh đúng **50.000 ảnh** với `seed = 42` vào thư mục rỗng (có `manifest.json` đánh dấu `complete = true`). Chạy `evaluation/run_quality.py`:

### 3.1. Chạy cho dòng chính (VAE, DCGAN, DDPM DDIM 100)
```bash
python evaluation/run_quality.py \
  --model vae \
  --dataset cifar10 \
  --run-name vae_cifar10_full \
  --fake outputs/samples/cifar10/vae/vae_cifar10_full \
  --resolution 32 \
  --device cuda
```

Đối với DDPM bảng chính (DDIM 100 step):
```bash
python evaluation/run_quality.py \
  --model ddpm \
  --dataset cifar10 \
  --run-name ddpm_base \
  --label ddim100 \
  --sampler ddim \
  --steps 100 \
  --fake outputs/samples/cifar10/ddpm/ddpm_base__ddim100 \
  --resolution 32 \
  --device cuda
```

### 3.2. Chạy cho dòng phụ DDPM (DDPM 500 step — 10.000 ảnh, so sánh sampler)
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
  --resolution 32 \
  --device cuda
```

### 3.3. Các file đầu ra tự động của `run_quality.py`
1. `outputs/metrics/quality/<dataset>__<model>__<run_name>[__<label>].json` (`kind = "quality"`).
2. `outputs/metrics/real_baseline/<dataset>.json` (`kind = "real_baseline"`, gồm `fid_floor` và `lpips_real`).
3. `outputs/figures/<dataset>/<model>__<run_name>[__<label>]/nearest_neighbor.png` (lưới 32 ảnh sinh × 3 ảnh train gần nhất trong không gian đặc trưng Inception, tận dụng thẳng feature đã tính nên không tốn thời gian trích lại; tắt bằng `--no-nearest-neighbor`).
4. **Riêng với `fashion_mnist`**: Nếu đã có checkpoint `outputs/checkpoints/fashion_mnist_cnn.pt`, script tự động tính thêm bộ chỉ số FID trên đặc trưng 128-d của CNN chuyên biệt (`fid_custom_train_50k`, `fid_custom_train_10k`, `fid_custom_test_10k_mean`, `fid_custom_test_10k_std` và `fid_custom_floor` trong `real_baseline`) kèm thông tin kiến trúc và `test_acc` của CNN.

---

## 4. Huấn luyện CNN đặc trưng riêng cho Fashion-MNIST

Trước khi chạy `run_quality.py --dataset fashion_mnist`, đảm bảo đã huấn luyện bộ phân loại CNN 10 lớp trên đúng split chuẩn `50k/10k/10k` (`datasets/splits/fashion_mnist_split.json`):

```bash
python models/classifiers/fashion_mnist_cnn.py --epochs 10 --device cuda
```

Lệnh trên lưu:
- Trọng số & metadata kiến trúc tại `outputs/checkpoints/fashion_mnist_cnn.pt`.
- Báo cáo độ chính xác (`best_val_acc`, `test_acc`) tại `outputs/metrics/fashion_mnist_cnn_info.json`.

---

## 5. Nội suy Không gian ẩn (`evaluation/interpolation.py`)

Đặc tả chi tiết tại [`evaluation/interpolation_spec.md`](interpolation_spec.md) (10 bước, cặp seed cố định `42 -> 100`, cả SLERP và LERP, xuất bằng `save_grid(..., nrow=10)` kèm file `interpolation_endpoints.json`):

- **Cho VAE** (4 hàng: SLERP/LERP trên prior $z$ + SLERP/LERP giữa $\mu_1, \mu_2$ của 2 ảnh test thật):
  ```bash
  python scripts/interpolate_vae.py \
    --checkpoint outputs/checkpoints/cifar10/vae/vae_cifar10_full/best.pt \
    --dataset cifar10 \
    --run-name vae_cifar10_full \
    --device cuda
  ```
- **Cho DCGAN** (có thể gọi trực tiếp hàm dùng chung `render_prior_interpolation_grid` trong `evaluation/interpolation.py`):
  ```python
  from evaluation.interpolation import render_prior_interpolation_grid
  from models.interface import load_model
  from scripts.render_samples import figure_dir

  model = load_model("dcgan", "outputs/checkpoints/cifar10/dcgan/base/best.pt", device="cuda")
  out = figure_dir("outputs/figures", "cifar10", "dcgan", "dcgan_base") / "interpolation.png"
  render_prior_interpolation_grid(model, out, seed1=42, seed2=100, steps=10)
  ```
- **Cho DDPM**: Dùng hàm `slerp(x_T1, x_T2, steps=10)` trong `evaluation/interpolation.py` rồi khử nhiễu bằng DDIM ($\eta = 0$).
