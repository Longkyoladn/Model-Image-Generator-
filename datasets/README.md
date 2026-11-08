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
- Conditional generation cần chốt một hoặc một số facial attribute chung cho cả ba mô hình.
- Phải dùng cùng split và cùng preprocessing cho VAE, DCGAN và DDPM.

## Fashion-MNIST

- Nguồn: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).
- Nội dung: 70.000 ảnh grayscale thuộc 10 lớp trang phục, gồm 60.000 ảnh train và 10.000 ảnh test.
- Kích thước gốc: 28×28, một kênh.
- Preprocessing khởi điểm: giữ 28×28 hoặc padding/resize thành 32×32 theo config đã thống nhất.
- Conditional generation sử dụng 10 nhãn lớp có sẵn.
- Khi tính FID, phải dùng cùng quy tắc chuyển ảnh grayscale sang định dạng mà feature extractor yêu cầu cho cả ba mô hình.

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
- Conditional generation sử dụng 10 nhãn lớp có sẵn.
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

## Cấu trúc lưu trữ đề xuất

Cấu trúc bên dưới chỉ dùng trên Drive/Kaggle/runtime và không được commit:

```text
datasets/
  raw/
    celeba/
    fashion_mnist/
    cifar10/
  processed/
    celeba/
    fashion_mnist/
    cifar10/
  splits/
    celeba/
    fashion_mnist/
    cifar10/
```

## Quy trình chung

1. Mount Google Drive hoặc attach Kaggle Dataset.
2. Truyền root path qua config hoặc biến môi trường; không hard-code đường dẫn cá nhân.
3. Chạy pipeline trong `datasets/preprocessing/` với config dataset tương ứng.
4. Kiểm tra visualization và thống kê split.
5. Lưu metadata split để cả ba mô hình dùng cùng tập dữ liệu.

Mỗi dataset có config riêng trong `configs/datasets/`. Thay đổi kích thước, normalization, split hoặc augmentation phải được ghi trong config; không sửa ngầm trong notebook.

## Thông tin cần ghi sau khi hoàn thiện pipeline

Với từng dataset, Lâm cần bổ sung:

- Số lượng ảnh thực tế được sử dụng.
- Số lớp và phân phối nhãn/thuộc tính nếu dùng conditional.
- Cấu trúc thư mục trên Drive/Kaggle.
- Kích thước đầu vào cuối cùng.
- Quy trình crop, resize, augmentation và normalization.
- Tỷ lệ và số lượng mẫu trong train/validation/test.
- Random seed dùng để tạo split.
- Cách mount, attach hoặc tự động tải dataset trong Colab/Kaggle.
