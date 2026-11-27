# Phân công công việc — So sánh VAE, DCGAN và DDPM

## 1. Mục tiêu

Cài đặt và so sánh ba mô hình sinh ảnh unconditional:

- VAE.
- DCGAN.
- DDPM.

Conditional DDPM là phần mở rộng (mục 8). VAE và DCGAN không làm bản conditional.

Cả ba mô hình phải được huấn luyện và đánh giá trên cả ba dataset:

- CelebA.
- Fashion-MNIST.
- CIFAR-10.

Nội dung so sánh gồm chất lượng ảnh, độ đa dạng, tốc độ sinh, thời gian huấn luyện, số tham số, dung lượng checkpoint và chi phí tính toán (mục 2.5).

## 2. Quy ước chung

### 2.1. Ba dataset và ba bộ cấu hình riêng

Không dùng một cấu hình dữ liệu duy nhất cho cả ba dataset. Mỗi dataset phải có cấu hình riêng vì khác nhau về kích thước, số kênh, nhãn và độ phức tạp.

Cấu trúc config đề xuất:

```text
configs/
  datasets_config/
    celeba.yaml
    fashion_mnist.yaml
    cifar10.yaml
  gan/
  vae/
  ddpm/
  eval/
    fid.yaml
```

Từng config dataset tối thiểu phải có:

- Tên dataset và đường dẫn dữ liệu.
- Kích thước ảnh đầu ra.
- Số kênh.
- Cách crop/resize/padding.
- Miền giá trị pixel và normalization.
- Augmentation.
- Train/validation/test split và đường dẫn file split.
- Batch size mặc định.
- Số worker.
- Random seed.
- Số lớp và cách mã hóa nhãn (chỉ dùng cho Conditional DDPM).

Cấu hình khởi điểm:

| Dataset | Kích thước | Kênh | Lưu ý |
|---|---:|---:|---|
| CelebA | 64×64 | 3 | Center crop trước khi resize |
| Fashion-MNIST | 32×32 (khuyến nghị) | 1 | Pad 2 pixel từ 28×28 để U-Net/DCGAN downsample chẵn; ghi rõ trong config |
| CIFAR-10 | 32×32 | 3 | Chỉ dùng horizontal flip nếu có augmentation |

Nếu thay đổi kích thước, cần phải ghi lại trong config và dùng cùng cấu hình dữ liệu cho cả ba mô hình. Ảnh thật dùng làm tham chiếu khi đánh giá cũng phải đi qua đúng preprocessing này (mục 2.6).

#### Split

| Dataset | Cách chia | Train | Validation | Test |
|---|---|---:|---:|---:|
| CelebA | Split chính thức | 162.770 | 19.867 | 19.962 |
| Fashion-MNIST | Giữ test chính thức; tách validation từ train chính thức | 50.000 | 10.000 | 10.000 |
| CIFAR-10 | Giữ test chính thức; tách validation từ train chính thức | 45.000 | 5.000 | 10.000 |

- Giữ test chính thức để kết quả đối chiếu được với các paper, dùng làm sanity check cho pipeline.
- Validation được tách stratified theo lớp từ train chính thức, dùng seed cố định và lưu danh sách index vào `datasets/splits/<dataset>/`.
- Cả ba mô hình dùng chung một file split.
- Validation chỉ dùng để chọn config; test chỉ dùng cho bảng kết quả cuối.

### 2.2. Lưu trữ và truy cập dữ liệu

- Không commit dataset lên Git.
- Dữ liệu được lưu trên Google Drive, Kaggle Dataset
- Huấn luyện trên Google Colab hoặc Kaggle(nên sử dụng mô hình T4 để train -> có thể tính thời gian so sánh 3 mô hình đúng nhất), không bắt buộc kéo dữ liệu về máy cá nhân.
- Không hard-code đường dẫn Drive cá nhân; đường dẫn phải đọc từ config hoặc biến môi trường.
- Hướng dẫn và nguồn của cả ba dataset được ghi trong `datasets/README.md`.

### 2.3. Quy trình chọn config

Áp dụng giống nhau cho cả ba mô hình. Mỗi cặp `mô hình × dataset` chỉ train đầy đủ **một lần** với config đã chọn.

**Giai đoạn 0 — Kiểm tra overfit (pass/fail).**

- Train trên một tập rất nhỏ, khoảng 64–512 ảnh từ train split.
- Mô hình phải học thuộc được tập này: loss giảm gần mức tối thiểu và ảnh sinh/reconstruction tái tạo được ảnh train.
- Giai đoạn này chỉ chứng minh code đúng và mô hình đủ capacity. Một config không overfit được tập nhỏ thì loại; config overfit được **chưa chắc là config tốt** trên dữ liệu đầy đủ, nên không dùng làm tiêu chí xếp hạng.

**Giai đoạn 1 — Tuning ngắn.**

- Thử các config đã qua giai đoạn 0, mỗi config chạy với ngân sách ngắn (cùng số step hoặc cùng giờ GPU quy đổi cho mọi config của cùng mô hình × dataset).
- Chọn config bằng **KID trên validation** theo cấu hình ở mục 2.6. Không chọn chỉ bằng quan sát vài ảnh mẫu.
- Có thể dùng thêm độ ổn định khi train làm tiêu chí phụ, nhưng phải ghi rõ.

**Giai đoạn 2 — Train đầy đủ một lần.**

- Train config đã chọn trên toàn bộ train split.
- Lưu checkpoint định kỳ và tính KID validation tại các checkpoint để vẽ **đường chất lượng theo chi phí** (KID theo giờ GPU quy đổi và theo training FLOPs).
- Tính các metric cuối theo mục 2.6 (FID-50k so với train, FID-10k so với test).

Ngân sách:

- Mỗi cặp `mô hình × dataset` có giới hạn tổng giờ GPU quy đổi (mục 2.5)
- Bảng kết quả phải ghi chi phí thực tế đã dùng.

Siêu tham số có thể thay đổi:

- Learning rate và optimizer.
- Batch size.
- Latent dimension.
- Số layer, channel width và model capacity.
- Loss weight.
- Epoch/step.
- Noise schedule và sampling step đối với DDPM.
- Các siêu tham số riêng của từng mô hình.

Yêu cầu ghi lại:

- Config, seed, metric, thời gian train, phần cứng và checkpoint của từng run, kể cả run bị loại.
- Bảng tóm tắt các config đã thử và tiêu chí chọn.
- Khi so sánh ba mô hình, công bố cả chất lượng lẫn chi phí; không che giấu việc config tốt hơn dùng nhiều tài nguyên hơn.

Quy ước tên config:

```text
configs/<model>/<dataset>/<experiment_name>.yaml
```

Mỗi run sinh một file metadata chứa config đã resolve, seed và Git commit hash (mục 4).

### 2.4. Đánh giá công bằng

Trong từng dataset, ba mô hình phải được đánh giá với cùng:

- Tập ảnh thật tham chiếu và split.
- Kích thước, số kênh và preprocessing đánh giá.
- Số lượng ảnh sinh.
- Danh sách seed đã thống nhất.
- Thư viện, phiên bản và cấu hình đánh giá (`configs/eval/fid.yaml`).

Benchmark tốc độ sinh **bắt buộc** chạy lại cho cả ba mô hình trên cùng một GPU tham chiếu (T4), cùng batch size và cùng chế độ precision.

Không so sánh trực tiếp giá trị FID giữa các dataset khác nhau. Kết luận chính phải được thực hiện theo từng dataset.

Mỗi kết quả phải kèm:

- Dataset và config dữ liệu.
- Config mô hình.
- Epoch hoặc training step.
- Phần cứng đã dùng.
- Thời gian huấn luyện (thực tế và quy đổi, mục 2.5).
- Số tham số, MACs/FLOPs và dung lượng checkpoint.
- Số ảnh sinh và số ảnh thật dùng để tính metric.
- Sampler và số sampling step (DDPM).
- Seed và Git commit hash.

### 2.5. Chi phí tính toán và quy đổi phần cứng

#### Các chỉ số chi phí

| Chỉ số | Đơn vị | Cách đo |
|---|---|---|
| Số tham số | triệu (M) | Tổng `numel()` của các module dùng khi sinh; báo riêng tổng tham số khi train (gồm Encoder/Discriminator) |
| MACs mỗi ảnh sinh | GMAC | Đo bằng `fvcore` hoặc `thop`, ghi rõ công cụ và phiên bản |
| FLOPs mỗi ảnh sinh | GFLOPs | FLOPs = 2 × MACs |
| Training compute | FLOPs | Công thức bên dưới |
| Giờ GPU | giờ | Thực tế và quy đổi T4 |
| Dung lượng checkpoint | MB | File dùng để sinh ảnh, ghi rõ có/không gồm optimizer state, EMA |

Lưu ý: FLOPs là **số phép tính**, FLOPS là **tốc độ** (phép tính/giây). Báo cáo dùng FLOPs. Nhiều công cụ trả về MACs nhưng gọi là "flops", nên phải ghi rõ đơn vị.

#### Chi phí sinh một ảnh

Gọi F_X là FLOPs forward của module X cho một ảnh:

| Mô hình | FLOPs sinh một ảnh |
|---|---|
| VAE | F_Decoder |
| DCGAN | F_Generator |
| DDPM | K × F_UNet, với K là số sampling step |
| Conditional DDPM + CFG | 2 × K × F_UNet |

#### Training compute (không phụ thuộc phần cứng)

Ước lượng backward ≈ 2 × forward, nên một lần forward + backward ≈ 3 × forward. Với S là số step và B là batch size:

| Mô hình | Training FLOPs |
|---|---|
| VAE | S × B × 3 × (F_Encoder + F_Decoder) |
| DCGAN (1 bước D + 1 bước G mỗi iteration) | S × B × (4 × F_G + 9 × F_D) |
| DDPM | S × B × 3 × F_UNet |

Với DCGAN: bước D gồm sinh ảnh giả (F_G) và forward/backward D trên ảnh thật và giả (6 × F_D); bước G gồm forward/backward qua G và D (3 × F_G + 3 × F_D). Nếu D được update k lần mỗi bước G thì điều chỉnh công thức tương ứng.

Training FLOPs là chỉ số chi phí chính khi so sánh vì không phụ thuộc phần cứng.

#### Quy đổi giờ GPU về T4

Không quy đổi bằng TFLOPS trên spec của GPU, vì tốc độ thực tế phụ thuộc mô hình (giới hạn bộ nhớ, tensor core, mixed precision). Thay vào đó, hiệu chuẩn theo từng config:

1. Với mỗi phần cứng h đã dùng để train (P100, T4×2, A100...), chạy 20 step warm-up rồi đo throughput train (ảnh/giây) trong 200 step với đúng config, batch size và precision đó. Làm tương tự trên T4.
2. Hệ số quy đổi: `k_h = throughput_h / throughput_T4`.
3. Thời gian quy đổi: `Thời gian - T4 = Σ_h (giờ train thực tế trên h × k_h)`. Trong bảng và biểu đồ, chỉ số này có nhãn **Thời gian - T4**.

Ví dụ: P100 nhanh hơn T4 1,3 lần với config này thì 2 giờ trên P100 tính là 2,6 giờ T4.

- Nếu train trên nhiều GPU cùng lúc (Kaggle T4×2), hiệu chuẩn cả cụm như một phần cứng h.
- "Giờ train thực tế" chỉ tính thời gian các training step (log cộng dồn trong training loop), không tính evaluation, lưu checkpoint hay thời gian Colab bị ngắt. Wall-clock tổng được báo riêng.
- Đo throughput bằng `evaluation/calibrate.py`; mỗi lần đo ghi một file `kind = "calibration"` (log hiệu chuẩn). Metadata của run ghi giờ train theo từng phần cứng (`hardware`, mục 4.3). `scripts/aggregate_results.py` tự tính `k_h` và Thời gian - T4 từ hai nguồn này; metadata run có ghi sẵn `gpu_hours_t4` thì bị báo lỗi (mục 4.3). Đường dẫn config được đưa về dạng `configs/...` trước khi ghép, nên run train trên Kaggle/Colab (đường dẫn tuyệt đối) vẫn ghép được với file calibration đo ở máy khác.

Tốc độ sinh không dùng công thức quy đổi; đo trực tiếp trên T4 như mục 2.4.

### 2.6. Cấu hình đánh giá: FID, KID, diversity

Toàn bộ cấu hình dưới đây được ghi trong `configs/eval/fid.yaml` và dùng chung cho mọi mô hình, dataset.

#### Thư viện và feature extractor

- Dùng `clean-fid` (mode `clean`) cho FID và KID, ghim phiên bản trong `requirements.txt`.
- Feature extractor: InceptionV3, pool3, 2048 chiều (mặc định của thư viện).
- Batch size khi trích feature không ảnh hưởng kết quả; đặt cố định 100 và ghi vào metadata.

#### Định dạng ảnh đầu vào

- **Ảnh sinh:** clamp về [0, 1], lưu PNG uint8 (`round(x × 255)`) đúng độ phân giải của dataset config. Không lưu JPEG.
- **Ảnh thật tham chiếu:** lấy từ split tương ứng, đi qua **đúng preprocessing khi train** (crop, resize, pad), **không augmentation**, rồi lưu PNG uint8. Với CelebA, ảnh tham chiếu là ảnh 64×64 đã crop, không phải ảnh gốc.
- **Grayscale (Fashion-MNIST):** lặp 1 kênh thành 3 kênh cho cả ảnh thật và ảnh sinh. Resize lên 299×299 do thư viện thực hiện.
- Statistics của ảnh thật được cache một lần theo tên `<dataset>_<split>_<resolution>` và cả nhóm dùng chung.

#### Khi tuning (giai đoạn 1)

| Thiết lập | Giá trị |
|---|---|
| Metric | KID |
| Ảnh sinh | 5.000, cố định cho mọi config của cùng dataset |
| Ảnh thật | Toàn bộ validation split |
| KID subset | `subset_size = 1000`, `num_subsets = 100` |
| Sampler DDPM | Cùng sampler và số step dự định dùng ở bảng chính |

#### Bảng kết quả cuối (giai đoạn 2)

Mỗi mô hình sinh **một bộ 50.000 ảnh** với seed cố định (cùng danh sách seed cho cả ba mô hình). Mọi metric dưới đây tính trên bộ này:

| Metric | Ảnh sinh | Ảnh thật | Vai trò |
|---|---:|---|---|
| **FID_train-50k** (chính) | 50.000 | Toàn bộ train split (162.770 / 50.000 / 45.000) | Số chính để so sánh; theo protocol của paper nên đối chiếu được |
| **FID_test-10k** mean ± std | 5 phần × 10.000 (chia từ bộ 50k) | Toàn bộ test split (19.962 / 10.000 / 10.000) | Kiểm tra tổng quát hóa/học thuộc; std dùng để đánh giá độ tin cậy của chênh lệch |
| FID sàn (thật với thật) | — | 10.000 ảnh train ngẫu nhiên so với toàn bộ test split | Mức tốt nhất có thể đạt với tập tham chiếu cỡ này, giúp hiểu ý nghĩa con số |
| KID_test mean ± std | 10.000 (phần đầu của bộ 50k) | Toàn bộ test split | `subset_size = 1000`, `num_subsets = 100` |
| Precision, Recall, Density, Coverage | 10.000 (phần đầu của bộ 50k) | 10.000 ảnh từ train split | Thư viện `prdc`, k = 5, dùng cùng feature Inception |
| LPIPS diversity | 2.000 cặp ngẫu nhiên | 2.000 cặp từ test split để làm mức tham chiếu | LPIPS-AlexNet, trung bình khoảng cách theo cặp |

- FID_test dùng 10k ảnh sinh vì test split chỉ có khoảng 10k ảnh. Với CelebA vẫn dùng 10k ảnh sinh để thống nhất giữa các dataset.
- Nếu FID_test tệ hơn FID_train rõ rệt (so với chênh lệch tương ứng của mức sàn), mô hình có dấu hiệu học thuộc. Kiểm tra thêm bằng hình nearest-neighbor.
- Luôn báo cáo số ảnh sinh và số ảnh thật bên cạnh mỗi giá trị FID/KID.

#### Quy tắc kết luận

- Chỉ so sánh trong cùng một dataset.
- Nếu hai mô hình chênh nhau trên FID_test-10k dưới 2 × std thì coi là ngang nhau, không kết luận mô hình nào tốt hơn.
- Luôn ghi sampler và số step của DDPM bên cạnh kết quả.

#### Sampler của DDPM

Sinh 50k ảnh bằng DDPM 1000 step rất tốn. Ước lượng sơ bộ: U-Net cỡ 35M tham số trên CIFAR-10 cần khoảng chục giờ T4; CelebA 64×64 gấp khoảng 4 lần. Tốc độ sinh chỉ phụ thuộc kiến trúc và số step, không phụ thuộc trọng số, nên Long đo thời gian sinh 100 ảnh bằng `speed.py` ngay khi có U-Net (chưa cần train) rồi ngoại suy trước khi chốt. Đã chốt:

- **Bảng chính: DDIM 100 step, η = 0**, dùng cho bộ 50k ảnh (FID_train-50k, FID_test-10k mean ± std, KID, PRDC, LPIPS) và cho KID khi tuning. Trước giai đoạn 1, đo nhanh KID-5k validation của DDIM 100 so với DDIM 200 ở một checkpoint; nếu 100 kém rõ (chênh lớn hơn std của KID) thì xem lại.
- **Dòng phụ: DDPM 500 step** (respace lịch nhiễu từ T = 1000 theo Improved DDPM), sinh **10k ảnh** với 10k seed đầu của danh sách seed chung. Dòng phụ **không** vào bảng chính: file `quality` ghi `"comparison": "sampler"` và chỉ so với 10k ảnh đầu của bộ DDIM 100 trong bảng "Đánh đổi sampler của DDPM" (FID_train-10k, KID_test, GFLOPs/ảnh, latency, throughput). Không so FID-10k với FID-50k.
- Dòng DDPM chính phải ghi thêm `metrics.fid_train_10k` (FID trên 10k ảnh đầu) để có số so với dòng phụ.
- Ghi rõ sampler và số step ở mọi bảng có DDPM.

#### Kiểm tra học thuộc

Vì FID_train (số chính) có thể bị "thưởng" khi mô hình học thuộc dữ liệu train, ngoài việc so với FID_test, mỗi mô hình phải kèm hình nearest-neighbor: 32 ảnh sinh, mỗi ảnh đặt cạnh 3 ảnh train gần nhất trong feature space Inception.

#### Lưu ý riêng cho Fashion-MNIST

InceptionV3 được train trên ảnh màu tự nhiên nên feature cho ảnh quần áo grayscale kích thước nhỏ kém ý nghĩa.

- FID Inception vẫn là số chính để thống nhất với hai dataset còn lại; báo cáo phải nêu hạn chế này.
- Bổ sung **FID dùng feature riêng**: train một CNN nhỏ phân loại 10 lớp trên train split Fashion-MNIST, lấy feature ở lớp áp chót, tính FID theo cùng các thiết lập ở bảng trên. Ghi kiến trúc và accuracy test của CNN này.
- Nếu hai loại FID cho cùng thứ hạng giữa ba mô hình thì kết luận chắc chắn hơn; nếu khác thì phân tích trong báo cáo.

## 3. Phân công

### 3.1. Lâm — DCGAN, dữ liệu và visualization

#### Pipeline dữ liệu

- Phân tích CelebA, Fashion-MNIST và CIFAR-10: số lượng ảnh, kích thước, số kênh, phân phối nhãn và ảnh lỗi/trùng nếu có.
- Xây dựng preprocessing và dataloader riêng cho từng dataset.
- Tạo split theo mục 2.1: giữ split chính thức của CelebA; với Fashion-MNIST và CIFAR-10 giữ test chính thức, tách validation stratified từ train chính thức. Lưu file index split.
- Xuất thư mục ảnh thật tham chiếu (PNG uint8, đúng preprocessing, không augmentation) cho validation, test và train để dùng cho mục 2.6.
- Tạo ba config dataset trong `configs/datasets_config/`.
- Trực quan hóa ảnh trước/sau preprocessing, phân phối lớp và batch mẫu.
- Chuẩn bị notebook Colab/Kaggle có thể mount Drive hoặc attach Kaggle Dataset.
- Hoàn thiện `datasets/README.md`; không đưa dataset vào Git.

#### DCGAN

- Cài Generator, Discriminator, training loop và checkpoint/resume.
- Lưu loss của Generator/Discriminator và ảnh mẫu định kỳ bằng fixed noise.
- Theo dõi mode collapse, mất cân bằng Generator/Discriminator và độ đa dạng.
- Chọn config theo quy trình mục 2.3 cho từng dataset.
- Viết script sinh đúng số lượng ảnh theo format ở mục 2.6.
- Làm latent interpolation cho DCGAN (nội suy z) theo đặc tả của Quân.

#### Deliverables

- Pipeline trong `datasets/preprocessing/`, file split và thư mục ảnh tham chiếu.
- Ba config dataset và notebook visualization.
- Source code, config, notebook/script train và sample DCGAN.
- Bảng các config GAN đã thử và config tốt nhất cho từng dataset.
- Ảnh mẫu, interpolation, log, metric và liên kết checkpoint lưu ngoài Git.

### 3.2. Quân — VAE, pipeline đánh giá và latent interpolation

#### VAE

- Cài Encoder, Decoder, reparameterization trick và checkpoint/resume.
- Ghi rõ reconstruction loss, KL divergence và trọng số của từng thành phần.
- Lưu reconstruction, random generation và loss định kỳ.
- Chọn config theo quy trình mục 2.3 cho từng dataset.

#### Pipeline đánh giá

- Cài đặt toàn bộ cấu hình ở mục 2.6 trong `evaluation/fid.py` và `evaluation/diversity.py`, cấu hình đọc từ `configs/eval/fid.yaml`.
- Cache statistics của ảnh thật theo từng dataset/split.
- Hỗ trợ cả chế độ tuning (KID-5k trên validation) và chế độ cuối (FID_train-50k, FID_test-10k mean ± std, FID sàn, KID_test).
- Cài Precision/Recall/Density/Coverage và LPIPS diversity.
- Train CNN phân loại Fashion-MNIST và cài FID dùng feature riêng cho Fashion-MNIST.
- Cài hình nearest-neighbor để kiểm tra học thuộc.
- Xuất kết quả ra JSON/CSV với metadata tái lập.
- Viết hướng dẫn để thành viên khác chạy cùng pipeline. Cần có sớm, vì giai đoạn 1 của mọi mô hình đều dùng KID.

#### Latent interpolation

- Viết đặc tả chung cho cả ba mô hình:
  - Hai đầu mút lấy từ các cặp seed cố định trong danh sách seed chung.
  - Mỗi chuỗi 10 bước.
  - Dùng cả nội suy tuyến tính (lerp) và cầu (slerp).
- Cài interpolation cho VAE: nội suy z ngẫu nhiên và nội suy giữa μ của hai ảnh test thật.
- Lâm và Long làm interpolation cho DCGAN và DDPM theo đặc tả này.
- Lưu chuỗi ảnh và ghi rõ cách chọn hai đầu mút.

#### Deliverables

- Source code, config, notebook/script train và sample VAE.
- `evaluation/fid.py`, `evaluation/diversity.py` và `configs/eval/fid.yaml` hoàn chỉnh.
- Đặc tả và script latent interpolation.
- Bảng config VAE đã thử và config tốt nhất cho từng dataset.
- Tài liệu hướng dẫn chạy pipeline đánh giá.

### 3.3. Long — DDPM, chi phí tính toán, benchmark và tích hợp

#### DDPM

- Cài forward diffusion, reverse diffusion, noise schedule và U-Net nhỏ.
- Cài training loop, checkpoint/resume và lưu ảnh quá trình denoising.
- Cài sampler DDPM và DDIM. Sampling script hỗ trợ seed, batch, số ảnh, loại sampler và số sampling step.
- Chọn config/noise schedule theo quy trình mục 2.3 cho từng dataset.
- Làm latent interpolation cho DDPM: nội suy noise ban đầu x_T bằng slerp, sinh bằng DDIM η = 0 để kết quả tất định.

#### Chi phí tính toán

- Cài đo số tham số, MACs/FLOPs mỗi ảnh sinh và training FLOPs cho cả ba mô hình theo mục 2.5.
- Viết script hiệu chuẩn throughput để tính hệ số quy đổi `k_h` về T4.

#### Benchmark tốc độ

- Hoàn thiện `evaluation/speed.py` cho cả ba mô hình, chạy trên T4.
- Đo thời gian một ảnh, một batch, throughput và peak GPU memory.
- Warm-up trước khi đo; đồng bộ GPU; chạy nhiều lần và báo mean, standard deviation.
- Không tính thời gian load checkpoint vào sampling time, trừ khi báo cáo thành chỉ số riêng.
- Ghi rõ sampler và số sampling step của DDPM.

#### Tích hợp

- Tổng hợp chất lượng (mục 2.6) và chi phí (mục 2.5) vào bảng riêng cho CelebA, Fashion-MNIST và CIFAR-10.
- Dùng config tốt nhất đã chọn của mỗi mô hình trong bảng chính.
- Vẽ đường KID validation theo giờ GPU quy đổi và theo training FLOPs cho ba mô hình.
- Bổ sung bảng tuning tóm tắt các config đã thử.
- Trình bày ảnh mẫu, interpolation và nearest-neighbor cùng layout; phân tích ưu/nhược điểm.

#### Deliverables

- Source code, config, notebook/script train và sample DDPM.
- `models/interface.py`, `evaluation/speed.py`, `scripts/aggregate_results.py` (bảng, biểu đồ, ảnh ghép).
- `evaluation/flops.py` (tham số, MACs/FLOPs) và `evaluation/calibrate.py` (hiệu chuẩn phần cứng).
- `scripts/render_samples.py` (lưới ảnh mẫu không chọn lọc).
- `models/dummy.py`, `configs/eval/speed_example.yaml` và `scripts/smoke_test_benchmark.py` (smoke test bắt buộc, mục 4.4).
- Bảng config DDPM đã thử và config tốt nhất cho từng dataset.
- Ba bảng kết quả chính, biểu đồ, ảnh mẫu và báo cáo tổng hợp.

## 4. Chuẩn giao tiếp giữa các module

Mỗi mô hình cần cung cấp lệnh tương đương:

```bash
python scripts/train_<model>.py --config configs/<model>/<dataset>/<experiment>.yaml
python scripts/sample_<model>.py \
  --checkpoint <checkpoint-path> \
  --num-samples 50000 \
  --seed 42 \
  --output-dir outputs/samples/<dataset>/<model>/<run-name>
```

DDPM thêm `--sampler ddpm|ddim --steps <K>`.

### 4.1. Interface chung của mô hình

Code đánh giá (benchmark tốc độ, FID, interpolation, notebook) load mọi mô hình qua một interface duy nhất, định nghĩa trong `models/interface.py`.

Mỗi package `models/vae`, `models/gan`, `models/ddpm` phải cung cấp trong `__init__.py`:

```python
def load_model(checkpoint: str, device: str) -> GenerativeModel
```

Lớp trả về kế thừa `GenerativeModel` và cài ba thứ:

| Thành phần | Yêu cầu |
|---|---|
| `name`, `image_shape` | `"vae"`/`"dcgan"`/`"ddpm"` và `(C, H, W)` |
| `sample(n, seed, **kwargs)` | Trả về tensor float `(n, C, H, W)` trong [0, 1], trên device của mô hình. Cùng seed cho cùng ảnh. DDPM nhận thêm `sampler`, `steps` |
| `sampling_modules()` | Dict các module dùng khi sinh: decoder (VAE), generator (DCGAN), U-Net (DDPM). Dùng để đếm tham số và đo MACs |

DDPM override thêm `forward_passes_per_image(**kwargs)` để trả về số step (2 × số step nếu dùng CFG).

Để đo tham số và MACs (`evaluation/flops.py`), lớp mô hình cài thêm:

| Thành phần | Yêu cầu |
|---|---|
| `training_modules()` | Mọi module tham gia train, đặt tên theo vai trò: VAE `{"encoder", "decoder"}`, DCGAN `{"generator", "discriminator"}`, DDPM `{"unet"}` |
| `example_inputs()` | Input batch = 1 cho từng module trong `training_modules()`, ví dụ DCGAN `{"generator": (z,), "discriminator": (img,)}`, DDPM `{"unet": (x, t)}` |

Tên trong `sampling_modules()` cũng theo vai trò: `"decoder"` (VAE), `"generator"` (DCGAN), `"unet"` (DDPM).

Để hiệu chuẩn phần cứng (`evaluation/calibrate.py`), mỗi package cung cấp thêm:

```python
def build_trainer(config: str, device: str) -> Trainer
```

`Trainer` (trong `models/interface.py`) có `batch_size`, `image_shape`, `precision`, `num_gpus` và `train_step(images)` thực hiện đúng một iteration train như lúc train thật (DCGAN gồm cả bước D và bước G). Trọng số ngẫu nhiên là đủ.

Cách dùng từ notebook hoặc script:

```python
from models.interface import load_model, check_contract

model = load_model("ddpm", "path/to/ckpt.pt", device="cuda")
check_contract(model, sampler="ddim", steps=100)   # báo lỗi nếu sai interface
images = model.sample(64, seed=0, sampler="ddim", steps=100)
```

`scripts/sample_<model>.py` nên gọi chính `load_model` và `sample` này để ảnh dùng tính FID và ảnh dùng đo tốc độ đi qua cùng một đường code.

`models/dummy.py` là mô hình giả, vừa để chạy thử pipeline, vừa là bản cài mẫu đúng interface: package thật cài theo cùng cấu trúc (mục 4.4).

### 4.2. File kết quả

Mọi file kết quả là JSON có trường `kind` và các khóa ghép `dataset`, `model`, `run_name`, `label`. `label` phân biệt các biến thể sampler của cùng một checkpoint, ví dụ `ddim100`; VAE và DCGAN để trống. `scripts/aggregate_results.py` đọc các file này và sinh bảng.

| `kind` | Ai sinh | Nội dung |
|---|---|---|
| `run` | Script train của từng mô hình | Metadata bên dưới |
| `quality` | `evaluation/fid.py`, `evaluation/diversity.py` | Khối `eval` và khối `metrics` |
| `speed` | `evaluation/speed.py` | Thời gian, throughput, memory theo batch size |
| `compute` | `evaluation/flops.py` | Tham số khi sinh và khi train, MACs từng module, GMACs/GFLOPs mỗi ảnh, FLOPs train mỗi ảnh, công cụ và phiên bản |
| `calibration` | `evaluation/calibrate.py` | Throughput train của một config trên một phần cứng |
| `real_baseline` | `evaluation/fid.py` | Số liệu thật-với-thật của dataset (`fid_floor`, `lpips_real`) |

**Mọi file JSON kết quả chỉ được sinh ra khi chạy script, không viết tay và không sửa tay số liệu trong file.** Số sai thì sửa code hoặc config rồi chạy lại script. Script tổng hợp chỉ đọc đúng định dạng hiện tại: không nhận `kind` cũ (ví dụ `"reference"` thay cho `"real_baseline"`), và file `speed` thiếu trường `complete` bị coi là chưa đo xong.

File `quality`:

```json
{
  "kind": "quality",
  "model": "ddpm", "dataset": "cifar10", "run_name": "ddpm_base", "label": "ddim100",
  "sampler": "ddim", "sampling_steps": 100,
  "eval": {
    "eval_config": "configs/eval/fid.yaml", "eval_config_hash": "...",
    "library": "clean-fid==<version>", "commit": "...",
    "ref_train": "cifar10_train_32", "ref_test": "cifar10_test_32", "num_gen": 50000
  },
  "metrics": {
    "fid_train_50k": 0.0, "fid_test_10k_mean": 0.0, "fid_test_10k_std": 0.0,
    "kid_test_mean": 0.0, "kid_test_std": 0.0,
    "precision": 0.0, "recall": 0.0, "density": 0.0, "coverage": 0.0,
    "lpips_diversity": 0.0,
    "fid_train_10k": 0.0
  }
}
```

`fid_train_10k` chỉ bắt buộc với dòng DDPM chính (dùng cho bảng so sánh sampler). Dòng phụ so sánh sampler (mục 2.6) có dạng:

```json
{
  "kind": "quality",
  "model": "ddpm", "dataset": "cifar10", "run_name": "ddpm_base", "label": "ddpm500",
  "sampler": "ddpm", "sampling_steps": 500,
  "comparison": "sampler",
  "eval": { "...": "giống hệt dòng chính, trừ", "num_gen": 10000 },
  "metrics": {"fid_train_10k": 0.0, "kid_test_mean": 0.0, "kid_test_std": 0.0}
}
```

Run ở giai đoạn tuning chỉ cần `metrics.kid_val_mean`.

Ảnh dùng trong báo cáo lưu theo đường dẫn cố định để script tổng hợp ghép thành ảnh so sánh cùng layout:

```text
outputs/figures/<dataset>/<model>__<run_name>[__<label>]/
  samples.png            # scripts/render_samples.py: 64 ảnh, seed 42, không chọn lọc
  interpolation.png      # script interpolation của từng mô hình
  nearest_neighbor.png   # evaluation/fid.py hoặc diversity.py
```

Dùng hàm `save_grid` trong `scripts/render_samples.py` để mọi lưới ảnh cùng kiểu (nền trắng, phóng to nearest-neighbor).

Script tổng hợp từ chối ghi bảng nếu trong cùng một dataset:

- Khối `eval` của các dòng chính khác nhau ở bất kỳ trường nào.
- Dòng phụ so sánh sampler có `eval.num_gen` khác 10.000, có trường `eval` khác (ngoài `num_gen`) so với bảng chính, hoặc thiếu `fid_train_10k`/`kid_test_*`.
- `comparison` khác `"sampler"`, hoặc đặt cho VAE/DCGAN.
- Tốc độ được đo trên GPU, phiên bản PyTorch, precision hoặc batch size khác nhau (kể cả dòng phụ), hoặc không đo trên T4.
- MACs được đo bằng công cụ hoặc phiên bản khác nhau.
- Không quy đổi được giờ train về T4 vì thiếu file calibration, file calibration thiếu `dataset`, hoặc trùng file calibration/metadata run.
- Một dòng DDPM thiếu sampler và số step.
- Metadata run có trường do script tổng hợp tự tính (`params_m`, `gmacs_per_forward`, `train_flops`, `gpu_hours_t4`), dù giá trị đúng hay sai (mục 4.3).

Quên ghi `"comparison": "sampler"` ở dòng DDPM 500 thì dòng này rơi vào bảng chính và bị từ chối vì khác `num_gen`.

Ngoài bảng chất lượng và bảng chi phí (chỉ gồm dòng chính), script ghi bảng "Đánh đổi sampler của DDPM" nếu có dòng phụ, vẽ hai biểu đồ cho mỗi dataset (KID validation theo Thời gian - T4 và theo FLOPs train, từ trường `curve` của metadata) và ghép ảnh mẫu, interpolation, nearest-neighbor của các mô hình.

Bảng chính chỉ gồm các run có `stage = "full"`; run `overfit` và `tuning` vào bảng "Các config đã thử".

### 4.3. Metadata của run

Mỗi run phải xuất metadata tối thiểu:

```json
{
  "kind": "run",
  "model": "vae|dcgan|ddpm",
  "dataset": "celeba|fashion_mnist|cifar10",
  "run_name": "example_run",
  "stage": "overfit|tuning|full",
  "config": "path/to/config.yaml",
  "image_size": 64,
  "num_samples": 50000,
  "seed": 42,
  "sampler": "ddim",
  "sampling_steps": 100,
  "checkpoint": "checkpoint_name",
  "train_steps": 100000,
  "batch_size": 128,
  "precision": "fp32",
  "hardware": [
    {"device": "Tesla T4", "train_hours": 1.5},
    {"device": "Tesla P100-PCIE-16GB", "train_hours": 2.0}
  ],
  "curve": [
    {"step": 10000, "kid_val_mean": 0.031, "hardware_hours": {"Tesla T4": 0.35}},
    {"step": 20000, "kid_val_mean": 0.024, "hardware_hours": {"Tesla T4": 0.7}}
  ],
  "eval_config": "configs/eval/fid.yaml",
  "commit": "git_commit_hash"
}
```

`sampler` và `sampling_steps` chỉ áp dụng cho DDPM. Các giá trị số ở trên chỉ là ví dụ.

- `hardware[].device` ghi đúng "device key" mà `evaluation/calibrate.py` in ra (ví dụ `"Tesla T4"`, `"Tesla T4 x2"`). `train_hours` là giờ train thực tế trên phần cứng đó (mục 2.5).
- `curve` là KID validation tại các checkpoint của lần train đầy đủ (giai đoạn 2). `hardware_hours` là giờ train cộng dồn tới checkpoint đó, theo từng phần cứng.
- **Không được ghi** `params_m`, `gmacs_per_forward`, `train_flops`, `gpu_hours_t4` vào metadata: script tổng hợp lấy tham số và MACs từ file `compute`, tính FLOPs train = `train_steps` × `batch_size` × FLOPs train mỗi ảnh, và tính Thời gian - T4 từ `hardware` cùng các file `calibration`. Metadata có một trong các trường này thì script báo lỗi và không ghi bảng. Thiếu file `compute` thì các cột tham số, MACs/FLOPs và FLOPs train để trống (không lấy từ metadata).
- File `calibration` phải có `dataset`; script tổng hợp ghép calibration với run theo `(dataset, model, config)`. Mỗi `(model, run_name)` của một dataset chỉ có **một** file metadata; trùng thì script báo lỗi.

### 4.4. Smoke test bắt buộc trước khi chạy mô hình thật

**Quy tắc:** trước khi train đầy đủ, sinh bộ 50k ảnh, tính FID hay đo tốc độ trên T4, mỗi thành viên phải chạy smoke test với mô hình của mình và đạt `SMOKE TEST PASS`. Mỗi pull request thêm/sửa mô hình hoặc script đánh giá phải ghi kết quả smoke test.

Ba file dùng cho bước này:

| File | Vai trò |
|---|---|
| `models/dummy.py` | Mô hình giả và bản cài mẫu (`load_model`, `build_trainer`, `GenerativeModel`, `Trainer`); các biến thể lỗi có chủ đích (`DEFECTS`, `TRAINER_DEFECTS`) để chứng minh bộ kiểm tra bắt được lỗi; file `quality`/`real_baseline` giả đúng cấu trúc |
| `configs/eval/speed_example.yaml` | Spec mẫu: dataset, precision, batch size, danh sách target (checkpoint, `run_name`, `label`, `sample_kwargs`, `trainer_config`, `quality`, `run_metadata`) |
| `scripts/smoke_test_benchmark.py` | Chạy từng bước, dùng `assert` để kiểm tra cấu trúc output; sai ở đâu dừng ở đó và in lý do |

Cách chạy:

```bash
# 1. Kiểm tra pipeline bằng mô hình giả (phải PASS trên máy/Colab/Kaggle của mình trước)
python scripts/smoke_test_benchmark.py

# 2. Chép spec mẫu, sửa target của mô hình mình: bỏ loader_module, điền checkpoint
#    và trainer_config thật (checkpoint sau giai đoạn 0 là đủ), rồi chạy
python scripts/smoke_test_benchmark.py --spec configs/eval/speed_<tên>.yaml
```

Các bước smoke test kiểm tra:

1. **Spec**: khóa bắt buộc; DDPM có `sampler`, `steps` và `label`; `comparison: sampler` chỉ cho DDPM và phải có dòng DDPM chính cùng `run_name`.
2. **Interface**: `load_model` trả về `GenerativeModel`, tên module đúng vai trò, `image_shape` đúng dataset (mục 2.1), `sample()` trả về tensor float `(n, C, H, W)` trong [0, 1], cùng device (kể cả chỉ số GPU), cùng seed cho cùng ảnh, `forward_passes_per_image` và `example_inputs()` đúng.
3. **Mô hình sai**: chạy kiểm tra của bước 2 và kiểm tra `Trainer` trên từng biến thể lỗi trong `models/dummy.py` (sai thứ tự chiều, quên đổi [-1, 1] về [0, 1], uint8, NaN, bỏ qua seed, bỏ qua `n`, sai tên module, thiếu discriminator, sai `forward_passes_per_image`, `example_inputs()` sai batch, Trainer sai shape/precision/batch...). Mỗi lỗi phải bị bắt đúng chỗ; nếu không, bộ kiểm tra bị lỏng và lỗi tương tự của mô hình thật cũng sẽ lọt qua.
4. **Tốc độ**: `evaluation/speed.py` ghi file `speed` đúng khóa ghép và đủ trường; `warmup`/`repeats` riêng của từng target được áp dụng; target cùng checkpoint dùng chung mô hình; file được ghi sau mỗi vòng đo (`complete = false` cho tới khi đo xong) để phiên bị ngắt không mất số liệu.
5. **MACs/FLOPs**: file `compute` đúng công thức mục 2.5; fvcore và thop cho cùng MACs.
6. **Hiệu chuẩn**: `build_trainer` trả về `Trainer` đúng `batch_size`, `image_shape`, `precision`, `num_gpus` và `train_step` chạy được; file `calibration` đủ trường; đường dẫn config của Kaggle, Colab, Windows đều được đưa về cùng chuỗi `configs/...`.
7. **Ảnh mẫu**: `scripts/render_samples.py` ghi đúng đường dẫn mục 4.2.
8. **FID/diversity**: ảnh lưu PNG uint8 = `round(clamp(x) × 255)` đúng độ phân giải; `configs/eval/fid.yaml` đúng mục 2.6; file `quality` và `real_baseline` đủ khóa, đúng tên cache `<dataset>_<split>_<resolution>`, metric trong miền hợp lệ; dòng phụ so sánh sampler có `comparison`, `num_gen = 10000` và đủ `fid_train_10k`, `kid_test_*`.
9. **Metadata run**: file `run` đủ trường mục 4.3, run `full` có `curve` tăng dần.
10. **Tổng hợp** và 11. **Bắt lỗi** (chỉ với mô hình giả): `aggregate_results.py` ra bảng đúng; từ chối dữ liệu không công bằng và metadata run có số điền tay; cảnh báo file `speed` ghi dở hoặc thiếu `complete`, thiếu file `compute`; không đọc `kind` cũ; run ghi config đường dẫn tuyệt đối vẫn quy đổi được Thời gian - T4.

`evaluation/fid.py`, `evaluation/diversity.py` và `configs/eval/fid.yaml` chưa có thì bước 8 dùng metric giả trong `models/dummy.py` để thử cấu trúc và báo trong mục "Chưa kiểm tra được". Khi Quân hoàn thành hai file này, `fid.yaml` phải có các khóa trong `FID_CONFIG_EXPECTED` của smoke test (đổi tên khóa thì sửa dict này cùng lúc), và mỗi target thêm `quality: <file quality>` để smoke test kiểm tra file thật.

Không chạy smoke test bằng `python -O` (cờ này tắt `assert`).

## 5. Quản lý Git và artifact

### Không commit

- Dataset, file nén dữ liệu và cache (kể cả thư mục ảnh tham chiếu và FID statistics lớn).
- Checkpoint lớn (`*.pt`, `*.pth`, `*.ckpt`).
- Toàn bộ ảnh sinh trong các run thử nghiệm.
- Token, API key, `.env`, `kaggle.json` và đường dẫn chứa thông tin cá nhân.
- Log hoặc notebook output có dung lượng lớn.

### Có thể commit

- Source code, config và notebook đã dọn output.
- File index split (nhỏ).
- Một số ảnh mẫu đại diện có dung lượng hợp lý.
- Metric JSON/CSV, biểu đồ và bảng kết quả.
- README và hướng dẫn tái lập.

Branch đề xuất:

- `feature/gan-data-pipeline`.
- `feature/vae-evaluation`.
- `feature/ddpm-benchmark`.

Mỗi pull request phải ghi nội dung, cách chạy, dataset/config, kết quả kiểm tra, nơi lưu artifact bên ngoài và hạn chế còn lại.

## 6. Trình tự thực hiện

1. Lâm hoàn thành ba pipeline/config dữ liệu, cố định split và xuất ảnh tham chiếu. Song song, Quân hoàn thành pipeline KID/FID và cache statistics.
2. Cả nhóm kiểm tra cùng đọc được dữ liệu và chạy được pipeline đánh giá trên Colab/Kaggle; mỗi thành viên chạy `python scripts/smoke_test_benchmark.py` và đạt PASS (mục 4.4).
3. Trước khi chạy mô hình thật ở bất kỳ giai đoạn nào, chạy smoke test với spec chứa checkpoint của mô hình mình và đạt PASS (mục 4.4).
4. Giai đoạn 0: mỗi thành viên chạy kiểm tra overfit cho từng dataset.
5. Giai đoạn 1: tuning ngắn, chọn config bằng KID-5k trên validation.
6. Giai đoạn 2: train đầy đủ một lần với config đã chọn; tính KID validation tại các checkpoint.
7. Sinh bộ 50k ảnh chính thức bằng seed đã thống nhất.
8. Quân chạy FID/KID/diversity theo mục 2.6; Long đo FLOPs, benchmark tốc độ trên T4 và tổng hợp.
9. Báo cáo kết quả riêng cho từng dataset và kết luận chung.

## 7. Tiêu chí hoàn thành tối thiểu

- VAE, DCGAN và DDPM đều train và sinh được ảnh trên cả ba dataset.
- Mỗi cặp `mô hình × dataset` có config, log, checkpoint và hướng dẫn tái lập.
- Có bảng các config đã thử (giai đoạn 0 và 1) và lý do chọn config tốt nhất.
- Có FID_train-50k, FID_test-10k mean ± std, FID sàn, KID, Precision/Recall, LPIPS diversity và benchmark tốc độ cho cả ba mô hình trên từng dataset.
- Có số tham số, MACs/FLOPs mỗi ảnh, training FLOPs, giờ GPU (thực tế và quy đổi T4) và dung lượng checkpoint.
- Có latent interpolation và hình nearest-neighbor cho cả ba mô hình.
- Có bảng so sánh và ảnh mẫu riêng cho CelebA, Fashion-MNIST và CIFAR-10.
- Code chạy lại được trên Colab hoặc Kaggle.
- Dataset và checkpoint lớn không nằm trong Git.

## 8. Mở rộng

- Conditional DDPM:
  - Fashion-MNIST và CIFAR-10 dùng 10 lớp có sẵn.
  - CelebA chọn trước một số thuộc tính (ví dụ Male, Smiling, Young) và ghi vào config.
- Classifier-free guidance; ghi rõ guidance scale và tính chi phí sinh là 2 × K × F_UNet.
- Metric theo từng class/attribute cho Conditional DDPM.
- So sánh noise schedule.
- Train lại config tốt nhất với 2–3 seed train để đo biến thiên giữa các lần train.
- Fine-tune Stable Diffusion bằng LoRA theo phong cách riêng.
- Demo sinh ảnh.


## 9. Lưu ý  
- nếu có thắc mắc nào về TASK thì gửi thông tin về  cho Long 