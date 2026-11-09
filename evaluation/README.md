## 1. Cài đặt môi trường

Đảm bảo đã cài đặt các thư viện sau:
pip install torch torchvision scipy lpips pytorch-msssim imageio

## 2

### 2.1. fid.py - Đánh giá FID

Chức năng: Tính Frechet Inception Distance giữa tập ảnh thật và tập ảnh sinh.

Đặc điểm: Tự động xử lý ảnh 1 kênh (Fashion-MNIST) bằng cách nhân bản thành 3 kênh. Có cơ chế cache thống kê (mean, covariance) của ảnh thật để không phải tính lại nhiều lần.

Cách dùng:

from evaluation.fid import FIDCalculator

fid_calc = FIDCalculator(device='cuda', cache_dir='outputs/metrics/cache')
real_loader và fake_loader là torch.utils.data.DataLoader
fid_score = fid_calc.calculate_fid(real_loader, fake_loader, dataset_name="cifar10")
print(f"FID Score: {fid_score}")

Tính FID theo từng class (Conditional)
class_fids, mean_fid = fid_calc.calculate_fid_per_class(
    real_loader, fake_loader, class_indices=[0, 1, 2], use_cache=True
)
print(f"Mean FID across classes: {mean_fid}")

### 2.2. diversity.py - Đánh giá độ đa dạng

Chức năng: Đo độ đa dạng của tập ảnh sinh ra.

Phương pháp:

calculate_feature_diversity: Đo phương sai trong không gian đặc trưng (Feature Space Variance).

calculate_lpips_diversity: Đo khoảng cách tri giác giữa các cặp ảnh (LPIPS).

calculate_ms_ssim: Đo độ tương đồng cấu trúc đa tỉ lệ (MS-SSIM). Giá trị càng thấp thì ảnh càng đa dạng.

Cách dùng:

from evaluation.diversity import DiversityCalculator

div_calc = DiversityCalculator(device='cuda')
fake_images là tensor (N, C, H, W)
diversity_score = div_calc.calculate_lpips_diversity(fake_images)
ms_ssim_score = div_calc.calculate_ms_ssim(fake_images)

print(f"LPIPS Diversity: {diversity_score}")
print(f"MS-SSIM (lower is more diverse): {ms_ssim_score}")

###  2.3. interpolation.py - Nội suy Latent Space

Chức năng: Tạo hiệu ứng chuyển cảnh mượt mà giữa 2 vector latent (z1, z2) để kiểm tra tính liên tục của không gian ẩn.

Phương pháp: Hỗ trợ nội suy tuyến tính (Linear) và nội suy cầu (Slerp). Hỗ trợ cả Conditional (cùng lớp hoặc khác lớp).

Cách dùng:

python
from evaluation.interpolation import make_conditional_interpolation_grid

z1, z2 là các vector latent ngẫu nhiên
mode="same_class" để giữ nguyên nhãn, mode="diff_class" để chuyển đổi nhãn
grid = make_conditional_interpolation_grid(
    model, z1, z2, label1=3, label2=5, steps=8, method="slerp", mode="diff_class"
)

Lưu grid ảnh (grid là tensor)
import torchvision.utils as vutils
vutils.save_image(grid, "outputs/samples/interpolation_grid.png")

## 3. Quy trình chạy Benchmark tổng thể

Sử dụng script scripts/run_benchmark.py để chạy đánh giá và xuất kết quả ra file JSON.

Kết quả đầu ra (outputs/metrics/benchmark_results.json) sẽ có cấu trúc:

json
```text
{
    "dataset": "cifar10",
    "model": "dcgan",
    "checkpoint": "path/to/ckpt",
    "seed": 42,
    "fid": 25.3,
    "diversity_lpips": 0.45,
    "diversity_ms_ssim": 0.62,
    "timestamp": "2023-10-27T10:00:00"
}
```