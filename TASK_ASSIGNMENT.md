# Phân công công việc — So sánh VAE, DCGAN và DDPM

## 1. Mục tiêu

Cài đặt và so sánh ba nhóm mô hình sinh ảnh:

- VAE / Conditional VAE.
- DCGAN / Conditional GAN.
- DDPM / Conditional DDPM.

Cả ba mô hình phải được huấn luyện và đánh giá trên cả ba dataset:

- CelebA.
- Fashion-MNIST.
- CIFAR-10.

Nội dung so sánh gồm chất lượng ảnh, độ đa dạng, tốc độ sinh, thời gian huấn luyện, số tham số, dung lượng checkpoint và chi phí tính toán.

## 2. Quy ước chung

### 2.1. Ba dataset và ba bộ cấu hình riêng

Không dùng một cấu hình dữ liệu duy nhất cho cả ba dataset. Mỗi dataset phải có cấu hình riêng vì khác nhau về kích thước, số kênh, nhãn và độ phức tạp.

Cấu trúc config đề xuất:

```text
configs/
  datasets/
    celeba.yaml
    fashion_mnist.yaml
    cifar10.yaml
  gan/
  vae/
  ddpm/
```

Từng config dataset tối thiểu phải có:

- Tên dataset và đường dẫn dữ liệu.
- Kích thước ảnh đầu ra.
- Số kênh.
- Cách crop/resize.
- Miền giá trị pixel và normalization.
- Augmentation.
- Train/validation/test split.
- Batch size mặc định.
- Số worker.
- Random seed.
- Số lớp và cách mã hóa nhãn khi train conditional.

Cấu hình khởi điểm đề xuất:

| Dataset | Kích thước | Kênh | Nhãn conditional | Lưu ý |
|---|---:|---:|---|---|
| CelebA | 64×64 | 3 | Thuộc tính khuôn mặt được nhóm chọn | Center crop trước khi resize |
| Fashion-MNIST | 28×28 hoặc 32×32 | 1 | 10 lớp trang phục | Ghi rõ nếu padding/resize lên 32×32 |
| CIFAR-10 | 32×32 | 3 | 10 lớp đối tượng | Tránh augmentation làm thay đổi ý nghĩa lớp |

Kích thước trên là mốc khởi điểm. Nếu thay đổi, nhóm phải ghi lại trong config và dùng cùng cấu hình dữ liệu cho ba mô hình trong phép so sánh tương ứng.

### 2.2. Lưu trữ và truy cập dữ liệu

- Không commit dataset lên Git.
- Dữ liệu được lưu trên Google Drive, Kaggle Dataset hoặc kho lưu trữ nhóm thống nhất.
- Huấn luyện trên Google Colab hoặc Kaggle, không bắt buộc kéo dữ liệu về máy cá nhân.
- Không hard-code đường dẫn Drive cá nhân; đường dẫn phải đọc từ config hoặc biến môi trường.
- Không commit token, API key, `kaggle.json` hoặc thông tin tài khoản.
- Hướng dẫn và nguồn của cả ba dataset được ghi trong `datasets/README.md`.

### 2.3. Thử nghiệm nhiều config và chọn config tốt nhất

Mỗi thành viên có thể và nên huấn luyện nhiều config cho mỗi cặp `mô hình × dataset`. Có thể thay đổi:

- Learning rate và optimizer.
- Batch size.
- Latent dimension hoặc embedding dimension.
- Số layer, channel width và model capacity.
- Loss weight.
- Epoch/step.
- Noise schedule và sampling step đối với DDPM.
- Các siêu tham số riêng của từng mô hình.

Yêu cầu khi chọn config tốt nhất:

- Không chọn chỉ bằng quan sát một vài ảnh mẫu.
- Dùng validation metric đã thống nhất, ưu tiên FID; có thể kết hợp diversity và tính ổn định.
- Không dùng test set để tuning siêu tham số.
- Lưu config, seed, metric, thời gian train, phần cứng và checkpoint của từng run.
- Chỉ dùng config tốt nhất đã chọn trên validation set cho bảng so sánh cuối cùng.
- Ghi rõ tiêu chí chọn và bảng tóm tắt các config đã thử.
- Khi so sánh ba mô hình, phải công bố cả chất lượng lẫn chi phí; không che giấu việc config tốt hơn sử dụng nhiều tài nguyên hơn.

Quy ước tên config đề xuất:

```text
configs/<model>/<dataset>/<experiment_name>.yaml
```

Mỗi run nên sinh một file metadata chứa config đã resolve, seed và Git commit hash để có thể tái lập.

### 2.4. Đánh giá công bằng

Trong từng dataset, ba mô hình phải được đánh giá với cùng:

- Tập ảnh thật tham chiếu và split.
- Kích thước, số kênh và preprocessing đánh giá.
- Số lượng ảnh sinh.
- Danh sách seed đã thống nhất.
- Phiên bản code đánh giá.
- Batch size và thiết bị khi benchmark tốc độ, nếu có thể.

Không so sánh trực tiếp giá trị FID giữa các dataset khác nhau. Kết luận chính phải được thực hiện theo từng dataset.

Mỗi kết quả phải kèm:

- Dataset và config dữ liệu.
- Config mô hình.
- Epoch hoặc training step.
- GPU/CPU.
- Thời gian huấn luyện.
- Số tham số và dung lượng checkpoint.
- Số ảnh dùng để tính metric.
- Seed và Git commit hash.

## 3. Phân công

### 3.1. Lâm — DCGAN / Conditional GAN, dữ liệu và visualization

#### Pipeline dữ liệu

- Phân tích CelebA, Fashion-MNIST và CIFAR-10: số lượng ảnh, kích thước, số kênh, phân phối nhãn và ảnh lỗi/trùng nếu có.
- Xây dựng preprocessing và dataloader riêng cho từng dataset.
- Tạo train/validation/test split cố định và lưu metadata split để cả nhóm dùng chung.
- Tạo ba config dataset trong `configs/datasets/`.
- Trực quan hóa ảnh trước/sau preprocessing, phân phối lớp và batch mẫu.
- Chuẩn bị notebook Colab/Kaggle có thể mount Drive hoặc attach Kaggle Dataset.
- Hoàn thiện `datasets/README.md`; không đưa dataset vào Git.

#### DCGAN / Conditional GAN

- Cài Generator, Discriminator, training loop và checkpoint/resume.
- Lưu loss của Generator/Discriminator và ảnh mẫu định kỳ bằng fixed noise.
- Theo dõi mode collapse, mất cân bằng Generator/Discriminator và độ đa dạng.
- Train trên cả ba dataset, có thể thử nhiều config và chọn config tốt nhất cho từng dataset.
- Viết script sinh đúng số lượng ảnh theo format chung.
- Nếu làm conditional, hỗ trợ chọn class/attribute khi sinh ảnh.

#### Deliverables

- Pipeline trong `datasets/preprocessing/`.
- Ba config dataset và notebook visualization.
- Source code, config, notebook/script train và sample DCGAN.
- Bảng các config GAN đã thử và config tốt nhất cho từng dataset.
- Ảnh mẫu, log, metric và liên kết checkpoint lưu ngoài Git.

### 3.2. Quân — VAE / Conditional VAE, FID, diversity và latent interpolation

#### VAE / Conditional VAE

- Cài Encoder, Decoder, reparameterization trick và checkpoint/resume.
- Ghi rõ reconstruction loss, KL divergence và trọng số của từng thành phần.
- Lưu reconstruction, random generation và loss định kỳ.
- Train trên cả ba dataset; thử nhiều config và chọn config tốt nhất riêng cho mỗi dataset.
- Nếu làm conditional, hỗ trợ chọn class/attribute khi sinh.

#### FID và diversity

- Xây dựng pipeline FID dùng chung cho ba mô hình và ba dataset.
- Quy định tập ảnh thật, số ảnh, resize, chuyển số kênh, feature extractor và batch size.
- Cache statistics của ảnh thật theo từng dataset/config.
- Với Fashion-MNIST, quy định một cách chuyển ảnh grayscale sang định dạng mạng FID và áp dụng giống nhau cho cả ba mô hình.
- Chọn và mô tả rõ diversity metric, ưu tiên LPIPS, MS-SSIM hoặc độ phân tán trong feature space.
- Hỗ trợ metric theo class nếu làm conditional.
- Xuất kết quả ra JSON/CSV với metadata tái lập.

#### Latent interpolation

- Nội suy latent tuyến tính; có thể thêm spherical interpolation.
- So sánh nội suy trong cùng lớp và khác lớp khi làm conditional.
- Lưu chuỗi ảnh và ghi rõ cách chọn hai đầu mút.

#### Deliverables

- Source code, config, notebook/script train và sample VAE.
- `evaluation/fid.py` và `evaluation/diversity.py` hoàn chỉnh.
- Script/notebook latent interpolation.
- Bảng config VAE đã thử và config tốt nhất cho từng dataset.
- Tài liệu hướng dẫn thành viên khác chạy cùng pipeline đánh giá.

### 3.3. Long — DDPM / Conditional DDPM, benchmark và tích hợp

#### DDPM / Conditional DDPM

- Cài forward diffusion, reverse diffusion, noise schedule và U-Net nhỏ.
- Cài training loop, checkpoint/resume và lưu ảnh quá trình denoising.
- Sampling script phải hỗ trợ seed, batch, số ảnh và số sampling step.
- Train trên cả ba dataset; thử nhiều config/noise schedule và chọn config tốt nhất cho từng dataset.
- Nếu làm conditional, dùng class/attribute embedding; nếu có classifier-free guidance phải ghi rõ guidance scale.

#### Benchmark tốc độ

- Hoàn thiện `evaluation/speed.py` cho cả ba mô hình.
- Đo thời gian một ảnh, một batch, throughput và peak GPU memory nếu hỗ trợ.
- Warm-up trước khi đo; đồng bộ GPU; chạy nhiều lần và báo mean, standard deviation.
- Không tính thời gian load checkpoint vào sampling time, trừ khi báo cáo thành chỉ số riêng.
- Ghi rõ số sampling step của DDPM.

#### Tích hợp

- Tổng hợp FID, diversity, tốc độ sinh, thời gian train, số tham số, dung lượng checkpoint và memory.
- Lập bảng riêng cho CelebA, Fashion-MNIST và CIFAR-10.
- Dùng config tốt nhất đã chọn của mỗi mô hình trong bảng chính.
- Bổ sung bảng ablation/tuning tóm tắt các config đã thử.
- Trình bày ảnh mẫu cùng layout và phân tích ưu/nhược điểm.

#### Deliverables

- Source code, config, notebook/script train và sample DDPM.
- `evaluation/speed.py` hoàn chỉnh.
- Bảng config DDPM đã thử và config tốt nhất cho từng dataset.
- Ba bảng kết quả chính, biểu đồ, ảnh mẫu và báo cáo tổng hợp.

## 4. Chuẩn giao tiếp giữa các module

Mỗi mô hình cần cung cấp lệnh tương đương:

```bash
python scripts/train_<model>.py --config configs/<model>/<dataset>/<experiment>.yaml
python scripts/sample_<model>.py \
  --checkpoint <checkpoint-path> \
  --num-samples 1000 \
  --seed 42 \
  --output-dir outputs/samples/<dataset>/<model>/<run-name>
```

Mỗi run phải xuất metadata tối thiểu:

```json
{
  "model": "vae|dcgan|ddpm",
  "dataset": "celeba|fashion_mnist|cifar10",
  "run_name": "example_run",
  "config": "path/to/config.yaml",
  "image_size": 64,
  "num_samples": 1000,
  "seed": 42,
  "checkpoint": "checkpoint_name",
  "device": "gpu_name",
  "commit": "git_commit_hash"
}
```

## 5. Quản lý Git và artifact

### Không commit

- Dataset, file nén dữ liệu và cache.
- Checkpoint lớn (`*.pt`, `*.pth`, `*.ckpt`).
- Toàn bộ ảnh sinh trong các run thử nghiệm.
- Token, API key, `.env`, `kaggle.json` và đường dẫn chứa thông tin cá nhân.
- Log hoặc notebook output có dung lượng lớn.

### Có thể commit

- Source code, config và notebook đã dọn output.
- Một số ảnh mẫu đại diện có dung lượng hợp lý.
- Metric JSON/CSV, biểu đồ và bảng kết quả.
- README và hướng dẫn tái lập.

Branch đề xuất:

- `feature/gan-data-pipeline`.
- `feature/vae-evaluation`.
- `feature/ddpm-benchmark`.

Mỗi pull request phải ghi nội dung, cách chạy, dataset/config, kết quả kiểm tra, nơi lưu artifact bên ngoài và hạn chế còn lại.

## 6. Trình tự thực hiện

1. Lâm hoàn thành ba pipeline/config dữ liệu và cố định split.
2. Cả nhóm kiểm tra cùng đọc được dữ liệu từ Colab/Kaggle.
3. Mỗi thành viên train baseline nhỏ trên từng dataset để kiểm tra end-to-end.
4. Thử nhiều config, đánh giá trên validation set và chọn config tốt nhất.
5. Sinh bộ ảnh chính thức bằng seed và số lượng đã thống nhất.
6. Quân chạy FID/diversity; Long chạy benchmark tốc độ và tổng hợp.
7. Báo cáo kết quả riêng cho từng dataset và kết luận chung.

## 7. Tiêu chí hoàn thành tối thiểu

- VAE, DCGAN và DDPM đều train và sinh được ảnh trên cả ba dataset.
- Mỗi cặp `mô hình × dataset` có config, log, checkpoint và hướng dẫn tái lập.
- Có bảng các config đã thử và lý do chọn config tốt nhất.
- Có FID, ít nhất một diversity metric và benchmark tốc độ cho cả ba mô hình trên từng dataset.
- Có thời gian huấn luyện, số tham số và dung lượng checkpoint.
- Có bảng so sánh và ảnh mẫu riêng cho CelebA, Fashion-MNIST và CIFAR-10.
- Code chạy lại được trên Colab hoặc Kaggle.
- Dataset và checkpoint lớn không nằm trong Git.

## 8. Mở rộng

Chỉ thực hiện sau khi hoàn thành yêu cầu tối thiểu:

- Conditional VAE, Conditional GAN và Conditional DDPM.
- Metric theo từng class/attribute.
- Latent interpolation nâng cao.
- Classifier-free guidance và so sánh noise schedule.
- Fine-tune Stable Diffusion bằng LoRA theo phong cách riêng.
- Demo sinh ảnh.
