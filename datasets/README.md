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
