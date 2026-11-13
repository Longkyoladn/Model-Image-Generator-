# Datasets

Dự án huấn luyện và đánh giá VAE, DCGAN và DDPM trên cả ba bộ dữ liệu: **CelebA**, **Fashion-MNIST** và **CIFAR-10**.

Dataset không được commit vào repository. Dữ liệu được truy cập từ Google Drive, Kaggle Dataset hoặc tự động tải trong runtime Colab/Kaggle.

---

## 1. Thông tin ba Dataset (Đã hoàn thiện bởi Lâm)

| Tiêu chí | CelebA | Fashion-MNIST | CIFAR-10 |
|---|:---:|:---:|:---:|
| **Kích thước đầu vào cuối** | **64×64** | **32×32** *(pad/resize từ 28)* | **32×32** |
| **Số kênh màu** | 3 (RGB) | 1 (Grayscale) | 3 (RGB) |
| **Tập Train** | 162,770 ảnh | 50,000 ảnh | 40,000 ảnh |
| **Tập Validation** | 19,867 ảnh | 10,000 ảnh | 10,000 ảnh |
| **Tập Test** | 19,962 ảnh | 10,000 ảnh | 10,000 ảnh |
| **Tổng số mẫu** | 202,599 ảnh | 70,000 ảnh | 60,000 ảnh |
| **Random Seed Split** | 42 (file txt chuẩn) | 42 (`fashion_mnist_split.json`) | 42 (`cifar10_split.json`) |
| **Miền chuẩn hóa** | `[-1, 1]` | `[-1, 1]` | `[-1, 1]` |
| **Augmentation (Train)** | RandomHorizontalFlip(p=0.5) | RandomHorizontalFlip(p=0.5) | RandomHorizontalFlip(p=0.5) |
| **Điều kiện (Conditional)** | Thuộc tính: "Smiling" (2 lớp: 0/1) | 10 lớp trang phục (0-9) | 10 lớp đối tượng (0-9) |

---

## 2. Chi tiết từng tập dữ liệu & Pipeline Preprocessing

### 2.1. CelebA (CelebFaces Attributes Dataset)
- **Nguồn**: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) hoặc Kaggle Dataset `jessicali9530/celeba-dataset`.
- **Đặc điểm ảnh gốc**: 178×218 pixel, ảnh chân dung khuôn mặt người nổi tiếng.
- **Quy trình tiền xử lý**:
  1. `transforms.CenterCrop(140)`: Cắt trung tâm khuôn mặt để loại bỏ nền viền.
  2. `transforms.Resize((64, 64))`: Đưa về kích thước chuẩn 64×64.
  3. `transforms.RandomHorizontalFlip(p=0.5)` (chỉ áp dụng tập train).
  4. `transforms.ToTensor()`: Ánh xạ pixel về $[0.0, 1.0]$.
  5. `transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])`: Chuẩn hóa về `[-1, 1]`.
- **Nhãn Conditional**: Thuộc tính `Smiling` (index 31) được nhóm chọn làm nhãn điều kiện chính:
  - `0`: Không cười (Non-smiling)
  - `1`: Đang cười (Smiling)
- **Cấu hình**: [`configs/datasets/celeba.yaml`](../configs/datasets/celeba.yaml)

### 2.2. Fashion-MNIST
- **Nguồn**: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando Research).
- **Đặc điểm ảnh gốc**: 28×28 pixel, 1 kênh đen trắng.
- **Quy trình tiền xử lý**:
  1. `transforms.Resize((32, 32))`: Đưa lên 32×32 (lũy thừa của 2: $32 \to 16 \to 8 \to 4$) để các tầng `ConvTranspose2d` giải nén không bị lệch padding.
  2. `transforms.RandomHorizontalFlip(p=0.5)` (train).
  3. `transforms.ToTensor()` + `transforms.Normalize(mean=[0.5], std=[0.5])` về `[-1, 1]`.
- **10 lớp trang phục**:
  `0: T-shirt/top`, `1: Trouser`, `2: Pullover`, `3: Dress`, `4: Coat`, `5: Sandal`, `6: Shirt`, `7: Sneaker`, `8: Bag`, `9: Ankle boot`.
- **Cấu hình**: [`configs/datasets/fashion_mnist.yaml`](../configs/datasets/fashion_mnist.yaml)

### 2.3. CIFAR-10
- **Nguồn**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (Alex Krizhevsky).
- **Đặc điểm ảnh gốc**: 32×32 pixel, 3 kênh màu RGB.
- **Quy trình tiền xử lý**:
  1. `transforms.RandomHorizontalFlip(p=0.5)` (train).
  2. `transforms.ToTensor()` + `transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])` về `[-1, 1]`.
- **10 lớp đối tượng**:
  `0: airplane`, `1: automobile`, `2: bird`, `3: cat`, `4: deer`, `5: dog`, `6: frog`, `7: horse`, `8: ship`, `9: truck`.
- **Cấu hình**: [`configs/datasets/cifar10.yaml`](../configs/datasets/cifar10.yaml)

---

## 3. Quản lý Train / Validation / Test Split cố định

Để đảm bảo kết quả so sánh giữa 3 thành viên (**Lâm - DCGAN**, **Quân - VAE**, **Long - DDPM**) là hoàn toàn công bằng (fair comparison), metadata split được sinh cố định với **Seed 42** và lưu tại:
- `datasets/splits/fashion_mnist_split.json`
- `datasets/splits/cifar10_split.json`
- `datasets/splits/celeba_split.json` (dùng mapping `list_eval_partition.txt`)

Khi gọi `build_dataloader(config_path, split="train"|"val"|"test")`, module sẽ tự động nạp chính xác các chỉ số này.

---

## 4. Hướng dẫn nạp dữ liệu trên Google Colab / Kaggle

### Trên Google Colab:
```python
from google.colab import drive
drive.mount('/content/drive')

# Trỏ biến đường dẫn dữ liệu hoặc cập nhật raw_dir trong config
# Ví dụ CelebA lưu tại Google Drive: /content/drive/MyDrive/BTL_DL_Data/celeba
```

### Trên Kaggle:
```python
# Đính kèm Kaggle Dataset (ví dụ: celeba-dataset)
# Thư mục dữ liệu sẽ nằm tại: /kaggle/input/celeba-dataset/
```

### Sử dụng Dataloader trong Code:
```python
from datasets.preprocessing.builder import build_dataloader, denormalize

# Nạp dữ liệu bất kỳ chỉ với 1 dòng lệnh:
train_loader, train_dataset = build_dataloader("configs/datasets/fashion_mnist.yaml", split="train", batch_size=64)

for images, labels in train_loader:
    # images có shape (B, C, H, W) trong khoảng [-1, 1]
    # labels có shape (B,)
    break
```
