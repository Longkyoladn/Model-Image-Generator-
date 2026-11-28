# Mô hình Variational Autoencoder (VAE) — Hướng dẫn & Cấu hình Thực nghiệm

Triển khai Convolutional VAE tuân thủ `models/interface.py` và quy trình thực nghiệm 3 giai đoạn tại `TASK_ASSIGNMENT.md` (mục 2.3, 2.5, 3.2, 4.1–4.4).

---

## 1. Kiến trúc & Hàm mất mát

- **Encoder (`models/vae/model.py`)**: Chuỗi khối `Conv2d(4x4, stride=2, pad=1) -> BatchNorm2d -> LeakyReLU(0.2)` giảm kích thước không gian về `4x4` (3 bước với ảnh `32x32`, 4 bước với ảnh `64x64`), sau đó chiếu tuyến tính ra hai vector $\mu \in \mathbb{R}^d$ và $\log \sigma^2 \in \mathbb{R}^d$ (clamp trong $[-30, 20]$ để ổn định số học).
- **Reparameterization Trick**: $z = \mu + \epsilon \odot \exp(0.5 \log \sigma^2)$ với $\epsilon \sim \mathcal{N}(0, I)$.
- **Decoder (`models/vae/model.py`)**: Chiếu tuyến tính $z \to (C_{\text{start}} \times 4 \times 4)$, sau đó qua các khối `ConvTranspose2d(4x4, stride=2, pad=1) -> BatchNorm2d -> ReLU`, tầng cuối dùng `Tanh()` xuất ảnh trong $[-1, 1]$ (hàm `sample()` tự quy đổi về $[0, 1]$).
- **Hàm mất mát (`models/vae/losses.py`)**:
  $$\mathcal{L}_{\text{VAE}}(x, \hat{x}) = \mathcal{L}_{\text{recon}}(x, \hat{x}) + \beta \cdot D_{\text{KL}}\big(q_\phi(z|x) \,\|\, \mathcal{N}(0, I)\big)$$
  Trong đó $\mathcal{L}_{\text{recon}}$ là tổng sai số trên từng pixel trung bình theo batch (`mse` hoặc `l1`), và $D_{\text{KL}} = -\frac{1}{2B}\sum (1 + \log\sigma^2 - \mu^2 - \sigma^2)$. Cả ba đại lượng `loss`, `recon_loss`, `kl_loss` cùng hệ số `beta`, `loss_type` đều được ghi tách biệt trong `loss_history.json` và metadata `run`.

---

## 2. Danh mục Cấu hình VAE (`configs/vae/<dataset>/`)

| Dataset | Config | `image_shape` | `latent_dim` | `base_channels` | `loss_type` | `beta` | `lr` | Vai trò |
|---|---|---|---:|---:|---|---:|---:|---|
| CIFAR-10 | `configs/vae/cifar10/base.yaml` | `[3, 32, 32]` | 128 | 32 | `mse` | 1.0 | 5e-4 | Cấu hình chuẩn ELBO ($\beta = 1$) |
| CIFAR-10 | `configs/vae/cifar10/beta05.yaml` | `[3, 32, 32]` | 128 | 64 | `mse` | 0.5 | 5e-4 | Ứng viên tuning giảm nhòe MSE, tăng kênh |
| CIFAR-10 | `configs/vae/cifar10/l1_beta05.yaml` | `[3, 32, 32]` | 128 | 64 | `l1` | 0.5 | 5e-4 | Ứng viên tuning L1 giữ biên sắc nét |
| Fashion-MNIST | `configs/vae/fashion_mnist/base.yaml` | `[1, 32, 32]` | 64 | 32 | `mse` | 1.0 | 5e-4 | Cấu hình chuẩn ELBO ($\beta = 1$) |
| Fashion-MNIST | `configs/vae/fashion_mnist/beta05.yaml` | `[1, 32, 32]` | 64 | 32 | `mse` | 0.5 | 5e-4 | Ứng viên tuning giảm trọng số KL |
| Fashion-MNIST | `configs/vae/fashion_mnist/l1_beta05.yaml` | `[1, 32, 32]` | 64 | 32 | `l1` | 0.5 | 5e-4 | Ứng viên tuning L1 cho ảnh quần áo biên rõ |
| CelebA | `configs/vae/celeba/base.yaml` | `[3, 64, 64]` | 256 | 32 | `mse` | 1.0 | 2e-4 | Cấu hình chuẩn ELBO ($\beta = 1$) |
| CelebA | `configs/vae/celeba/beta05.yaml` | `[3, 64, 64]` | 256 | 64 | `mse` | 0.5 | 2e-4 | Ứng viên tuning tăng capacity cho 64×64 |
| CelebA | `configs/vae/celeba/l1_beta05.yaml` | `[3, 64, 64]` | 256 | 64 | `l1` | 0.5 | 2e-4 | Ứng viên tuning L1 cho chi tiết khuôn mặt |

---

## 3. Quy trình Chạy Thực nghiệm Chuẩn

### Bước 0 — Kiểm tra Smoke Test bắt buộc
```bash
python scripts/smoke_test_benchmark.py --spec configs/eval/speed_vae.yaml
```

### Giai đoạn 0 — Kiểm tra Overfit tập nhỏ (256 ảnh)
```bash
python scripts/train_vae.py \
  --config configs/vae/cifar10/base.yaml \
  --stage overfit \
  --overfit-samples 256 \
  --epochs 50 \
  --run-name vae_cifar10_overfit \
  --device cuda
```
Kiểm tra ảnh đối chiếu `outputs/figures/cifar10/vae__vae_cifar10_overfit/reconstruction.png` để xác nhận hàng dưới tái tạo sát hàng ảnh thật phía trên.

### Giai đoạn 1 — Tuning ngắn (chọn config theo KID-5k trên validation)
```bash
python scripts/train_vae.py --config configs/vae/cifar10/base.yaml --stage tuning --epochs 15 --run-name vae_cifar10_tun_base --device cuda
python scripts/train_vae.py --config configs/vae/cifar10/beta05.yaml --stage tuning --epochs 15 --run-name vae_cifar10_tun_beta05 --device cuda
python scripts/train_vae.py --config configs/vae/cifar10/l1_beta05.yaml --stage tuning --epochs 15 --run-name vae_cifar10_tun_l1b05 --device cuda
```
Mỗi lệnh tự xuất cả `outputs/metrics/runs/...` và `outputs/metrics/quality/...` (`metrics.kid_val_mean`), chọn cấu hình có **KID validation thấp nhất** sang Giai đoạn 2.

### Giai đoạn 2 — Huấn luyện đầy đủ 1 lần & Resume khi ngắt phiên
```bash
python scripts/train_vae.py \
  --config configs/vae/cifar10/beta05.yaml \
  --stage full \
  --run-name vae_cifar10_full \
  --device cuda

# Nếu Colab/Kaggle ngắt giữa chừng, resume từ last.pt:
python scripts/train_vae.py \
  --config configs/vae/cifar10/beta05.yaml \
  --stage full \
  --run-name vae_cifar10_full \
  --resume outputs/checkpoints/cifar10/vae/vae_cifar10_full/last.pt \
  --device cuda
```
Lưu ý: `best.pt` chỉ lưu trọng số mô hình (không kèm trạng thái Adam optimizer) để đo đúng dung lượng checkpoint sinh ảnh theo mục 2.5, còn `last.pt` / `checkpoint_ep*.pt` lưu đầy đủ trạng thái optimizer & scaler phục vụ `--resume`.

### Sinh 50.000 ảnh, Vẽ Figure & Đánh giá cuối
```bash
# 1. Sinh 50.000 ảnh PNG chuẩn
python scripts/sample_vae.py \
  --checkpoint outputs/checkpoints/cifar10/vae/vae_cifar10_full/best.pt \
  --num-samples 50000 \
  --seed 42 \
  --output-dir outputs/samples/cifar10/vae/vae_cifar10_full \
  --overwrite

# 2. Xuất lưới 64 ảnh mẫu không chọn lọc (samples.png) và nội suy 10 bước (interpolation.png)
python scripts/render_samples.py \
  --model vae \
  --checkpoint outputs/checkpoints/cifar10/vae/vae_cifar10_full/best.pt \
  --dataset cifar10 \
  --run-name vae_cifar10_full

python scripts/interpolate_vae.py \
  --checkpoint outputs/checkpoints/cifar10/vae/vae_cifar10_full/best.pt \
  --dataset cifar10 \
  --run-name vae_cifar10_full

# 3. Tính FID/KID/PRDC/LPIPS + xuất nearest_neighbor.png
python evaluation/run_quality.py \
  --model vae \
  --dataset cifar10 \
  --run-name vae_cifar10_full \
  --fake outputs/samples/cifar10/vae/vae_cifar10_full \
  --resolution 32 \
  --device cuda
```
