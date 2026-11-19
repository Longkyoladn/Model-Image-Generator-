# Datasets

Dự án huấn luyện và đánh giá VAE, DCGAN và DDPM trên cả ba bộ dữ liệu: CelebA, Fashion-MNIST và CIFAR-10.

Dataset không được commit vào repository. Dữ liệu nên được truy cập từ Google Drive, Kaggle Dataset hoặc tự động tải trong runtime Colab/Kaggle khi nguồn chính thức cho phép.

## CelebA

- Tên đầy đủ: CelebFaces Attributes Dataset.
- Nguồn: [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
- Nội dung: hơn 200.000 ảnh khuôn mặt người nổi tiếng, kèm landmark và các thuộc tính khuôn mặt.
- Ảnh gốc là ảnh màu RGB với kích thước không phải 64×64.
- Preprocessing khởi điểm: center crop khuôn mặt, resize về 64×64, chuyển tensor và normalize theo config.
- Unconditional generation dùng toàn bộ ảnh trong split huấn luyện.
- Split: dùng split chính thức 162.770 / 19.867 / 19.962 (train/validation/test).
- Conditional generation (chỉ Conditional DDPM, phần mở rộng) cần chốt trước một số facial attribute.
- Phải dùng cùng split và cùng preprocessing cho VAE, DCGAN và DDPM.

## Fashion-MNIST

- Nguồn: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).
- Nội dung: 70.000 ảnh grayscale thuộc 10 lớp trang phục, gồm 60.000 ảnh train và 10.000 ảnh test.
- Kích thước gốc: 28×28, một kênh.
- Preprocessing khởi điểm: pad 2 pixel thành 32×32 (khuyến nghị), ghi rõ trong config.
- Split: giữ 10.000 ảnh test chính thức; tách stratified 10.000 ảnh validation từ 60.000 ảnh train → 50.000 / 10.000 / 10.000.
- Conditional generation (chỉ Conditional DDPM) sử dụng 10 nhãn lớp có sẵn.
- Khi tính FID, ảnh grayscale được lặp thành 3 kênh cho cả ảnh thật và ảnh sinh (xem mục 2.6 của `TASK_ASSIGNMENT.md`).

Mười lớp:

```text
0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot
```

## CIFAR-10

- Nguồn: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
- Nội dung: 60.000 ảnh màu thuộc 10 lớp, gồm 50.000 ảnh train và 10.000 ảnh test.
- Kích thước gốc: 32×32, ba kênh RGB.
- Preprocessing khởi điểm: giữ 32×32, chuyển tensor và normalize theo config.
- Split: giữ 10.000 ảnh test chính thức; tách stratified 5.000 ảnh validation từ 50.000 ảnh train → 45.000 / 5.000 / 10.000.
- Conditional generation (chỉ Conditional DDPM) sử dụng 10 nhãn lớp có sẵn.
- Augmentation phải được ghi trong config và không được làm thay đổi ý nghĩa của nhãn.

Mười lớp:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

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
- **Cấu hình**: [`configs/datasets_config/celeba.yaml`](../configs/datasets_config/celeba.yaml)

### 2.2. Fashion-MNIST
- **Nguồn**: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando Research).
- **Đặc điểm ảnh gốc**: 28×28 pixel, 1 kênh đen trắng.
- **Quy trình tiền xử lý**:
  1. `transforms.Resize((32, 32))`: Đưa lên 32×32 (lũy thừa của 2: $32 \to 16 \to 8 \to 4$) để các tầng `ConvTranspose2d` giải nén không bị lệch padding.
  2. `transforms.RandomHorizontalFlip(p=0.5)` (train).
  3. `transforms.ToTensor()` + `transforms.Normalize(mean=[0.5], std=[0.5])` về `[-1, 1]`.
- **10 lớp trang phục**:
  `0: T-shirt/top`, `1: Trouser`, `2: Pullover`, `3: Dress`, `4: Coat`, `5: Sandal`, `6: Shirt`, `7: Sneaker`, `8: Bag`, `9: Ankle boot`.
- **Cấu hình**: [`configs/datasets_config/fashion_mnist.yaml`](../configs/datasets_config/fashion_mnist.yaml)

### 2.3. CIFAR-10
- **Nguồn**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (Alex Krizhevsky).
- **Đặc điểm ảnh gốc**: 32×32 pixel, 3 kênh màu RGB.
- **Quy trình tiền xử lý**:
  1. `transforms.RandomHorizontalFlip(p=0.5)` (train).
  2. `transforms.ToTensor()` + `transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])` về `[-1, 1]`.
- **10 lớp đối tượng**:
  `0: airplane`, `1: automobile`, `2: bird`, `3: cat`, `4: deer`, `5: dog`, `6: frog`, `7: horse`, `8: ship`, `9: truck`.
- **Cấu hình**: [`configs/datasets_config/cifar10.yaml`](../configs/datasets_config/cifar10.yaml)

---

## 3. Quản lý Train / Validation / Test Split cố định

Để đảm bảo kết quả so sánh giữa 3 thành viên (**Lâm - DCGAN**, **Quân - VAE**, **Long - DDPM**) là hoàn toàn công bằng (fair comparison), metadata split được sinh cố định với **Seed 42** và lưu tại:
- `datasets/splits/fashion_mnist_split.json`
- `datasets/splits/cifar10_split.json`
- `datasets/splits/celeba_split.json` (dùng mapping `list_eval_partition.txt`)

Khi gọi `build_dataloader(config_path, split="train"|"val"|"test")`, module sẽ tự động nạp chính xác các chỉ số này.

---

## 4. Hướng dẫn nạp dữ liệu trên Google Colab / Kaggle

### Thư mục Google Drive chung của nhóm (Được tạo bởi Lâm):
- **Link thư mục Google Drive:** [BTL_DL_Data (CelebA)](https://drive.google.com/drive/folders/1yXu8NLNHrcflUo7lwAjjSvzFua72vlzx)
*(Chứa file nén CelebA và 2 file metadata: `list_eval_partition.txt`, `list_attr_celeba.txt`)*

### Khi chạy trên Google Colab (Quân - DDPM, Long - VAE):
```python
# Cách lấy dữ liệu 
# với minst và cifar thì gọi code như bình thường :

gõ code sau:
from datasets.preprocessing.builder import build_dataloader

# A. Lấy tập Test của Fashion-MNIST (Đúng đủ 10.000 ảnh)
test_loader, test_ds = build_dataloader("configs/datasets_config/fashion_mnist.yaml", split="test", batch_size=64)

# B. Lấy tập Test của CIFAR-10 (Đúng đủ 10.000 ảnh)
test_loader, test_ds = build_dataloader("configs/datasets_config/cifar10.yaml", split="test", batch_size=64)

# C. Lấy tập Test của CelebA (Đúng đủ 19.962 ảnh)
test_loader, test_ds = build_dataloader("configs/datasets_config/celeba.yaml", split="test", batch_size=64)

Tương tự với mẫu fashion_minst. 



# với riêng celebA làm theo các bước sau:

Bước 1: Thêm lối tắt vào Google Drive (Chỉ làm 1 lần duy nhất, tốn 5 giây)
click vào link drive file zip của celebA: https://drive.google.com/drive/folders/1yXu8NLNHrcf1Uo7lwAjjSvzEuaZ2vLzx
→ Chọn "Thêm lối tắt vào Drive" (Add shortcut to Drive).

Bước 2: mở Notebook Colab lên để Train

Ở ô đầu tiên của sổ tay Colab, chỉ cần copy đúng đoạn code chuẩn bị này (đã được ghi sẵn trong datasets/README.md):

python
# 1. Kết nối Google Drive vào Colab
from google.colab import drive
drive.mount('/content/drive')

# 2. Giải nén sang ổ SSD tạm thời của Colab để train siêu tốc (mất ~15 giây)
hãy tạo thêm 1 cell và nhập code sau:

## 1. Tạo thư mục đích trên Colab
!mkdir -p /content/celeba
## 2. Giải nén đúng file archive (2).zip từ thư mục Celebs_Zip_Data của bạn
!unzip -q "/content/drive/MyDrive/Celebs_Zip_Data/archive (2).zip" -d "/content/celeba"
## 3. Liệt kê kiểm tra các file đã được giải nén ra
!ls -la /content/celeba

Sau 2 bước trên, nạp dữ liệu vào mô hình thế nào?

--> Chỉ cần đúng 2 dòng code:

python
from datasets.preprocessing.builder import build_dataloader
# Nạp dữ liệu CelebA
train_loader, _ = build_dataloader("configs/datasets_config/celeba.yaml", split="train", batch_size=64)
# Vòng lặp huấn luyện VAE của Long
for images, labels in train_loader:
    # images shape: [64, 3, 64, 64]
    loss = vae_model(images)
    ...

### Khi chạy trên Kaggle:
```python
# Đính kèm trực tiếp Kaggle Dataset (jessicali9530/celeba-dataset)
# Thư mục dữ liệu tự động nằm tại: /kaggle/input/celeba-dataset/
```

### Sử dụng Dataloader trong Code:
```python
from datasets.preprocessing.builder import build_dataloader, denormalize

# Nạp dữ liệu bất kỳ chỉ với 1 dòng lệnh:
train_loader, train_dataset = build_dataloader("configs/datasets_config/fashion_mnist.yaml", split="train", batch_size=64)

for images, labels in train_loader:
    # images có shape (B, C, H, W) trong khoảng [-1, 1]
    # labels có shape (B,)
    break
```
