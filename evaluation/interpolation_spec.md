# Đặc tả Quy chuẩn Nội suy Không gian ẩn (Latent Interpolation Specification)

Tài liệu đặc tả quy chuẩn chung cho cả 3 mô hình (**VAE**, **DCGAN**, **DDPM**) theo `TASK_ASSIGNMENT.md` (mục 2.6, 3.2, 4.2).

---

## 1. Mục tiêu
Nội suy trong không gian ẩn (latent space) nhằm kiểm tra tính liên tục, độ trơn tru và cấu trúc hình học của không gian biểu diễn:
- Không gian biểu diễn tốt phải có tính chuyển tiếp mượt mà, không xuất hiện ảnh rác, nhiễu đột ngột hoặc nhảy cóc giữa các đối tượng.
- Kiểm tra xem mô hình có học được phân phối liên tục hay chỉ ghi nhớ các điểm rời rạc (học thuộc).

---

## 2. Quy chuẩn thực hiện chung cho 3 mô hình

### 2.1. Số bước và Điểm đầu mút
* **Số bước nội suy**: Cố định đúng **10 bước** cho mỗi chuỗi chuyển tiếp:
  $$\alpha \in [0.0, 0.111, 0.222, 0.333, 0.444, 0.556, 0.667, 0.778, 0.889, 1.0]$$
* **Điểm đầu mút (Fixed Seeds)**:
  * Điểm bắt đầu ($z_1$): Lấy từ `seed = 42`.
  * Điểm kết thúc ($z_2$): Lấy từ `seed = 100`.
  * Mọi mô hình phải dùng cùng danh sách cặp seed này để kết quả so sánh được trực quan trên cùng layout.

### 2.2. Hai phương pháp nội suy
Cả 3 mô hình đều phải thực hiện cả hai phương pháp để so sánh:

1. **Nội suy tuyến tính (LERP - Linear Interpolation)**:
   $$z(\alpha) = (1 - \alpha) z_1 + \alpha z_2$$

2. **Nội suy cầu (SLERP - Spherical Linear Interpolation)**:
   Do không gian Gaussian nhiều chiều $\mathcal{N}(0, I)$ có mật độ xác suất tập trung chủ yếu trên mặt cầu bán kính $\sqrt{d}$ (hiện tượng Gaussian Annulus / Sphere), SLERP giúp bảo toàn độ dài vector ẩn $z$ và cho chất lượng ảnh trung gian mượt mà, tự nhiên hơn:
   $$z(\alpha) = \frac{\sin((1 - \alpha)\Omega)}{\sin\Omega} z_1 + \frac{\sin(\alpha\Omega)}{\sin\Omega} z_2$$
   với $\Omega = \arccos\left(\frac{z_1 \cdot z_2}{\|z_1\|_2 \|z_2\|_2}\right)$. Nếu $z_1$ và $z_2$ gần như trùng nhau ($\Omega \approx 0$), chuyển về LERP.

---

## 3. Quy chuẩn áp dụng cho từng mô hình

### 3.1. Đối với VAE (Quân)
VAE có cả Encoder $q_\phi(z|x)$ và Decoder $p_\theta(x|z)$, do đó cần thực hiện 2 loại nội suy:
1. **Nội suy Prior $z$ ngẫu nhiên**:
   * Lấy $z_1 \sim \mathcal{N}(0, I)$ (`seed=42`) và $z_2 \sim \mathcal{N}(0, I)$ (`seed=100`).
   * Nội suy $z(\alpha)$ bằng cả LERP và SLERP $\to$ đưa qua Decoder để tái tạo chuỗi ảnh.
2. **Nội suy Reconstruction giữa 2 ảnh test thật**:
   * Lấy 2 ảnh thật $x_1, x_2$ từ tập test split (theo index cố định).
   * Đưa qua Encoder để lấy vector trung bình $\mu_1 = \text{Encoder}(x_1)$ và $\mu_2 = \text{Encoder}(x_2)$.
   * Nội suy giữa $\mu_1$ và $\mu_2$ bằng SLERP / LERP $\to$ đưa qua Decoder.

### 3.2. Đối với DCGAN (Lâm)
* Lấy 2 vector nhiễu ngẫu nhiên $z_1, z_2 \sim \mathcal{N}(0, I)$ (hoặc $\mathcal{U}(-1, 1)$ tùy config Generator) theo seed 42 và 100.
* Nội suy $z(\alpha)$ bằng SLERP và LERP $\to$ sinh ảnh qua Generator: $G(z(\alpha))$.

### 3.3. Đối với DDPM (Long)
* Lấy 2 điểm nhiễu ban đầu $x_T^{(1)}, x_T^{(2)} \sim \mathcal{N}(0, I)$ theo seed 42 và 100.
* Nội suy trên không gian nhiễu ban đầu bằng SLERP: $x_T(\alpha) = \text{SLERP}(x_T^{(1)}, x_T^{(2)}, \alpha)$.
* Dùng **DDIM với $\eta = 0$** (100 bước) để khử nhiễu từ $x_T(\alpha) \to x_0(\alpha)$.
* *Lưu ý*: Bắt buộc dùng DDIM $\eta = 0$ (deterministic) để đường đi nội suy liên tục và không bị nhiễu loạn ngẫu nhiên ở mỗi bước khử nhiễu.

---

## 4. Định dạng và Đường dẫn xuất ảnh

Tất cả các mô hình phải xuất ảnh theo đúng quy ước của `scripts/aggregate_results.py`:
* **Đường dẫn**:
  `outputs/figures/<dataset>/<model>__<run_name>[__<label>]/interpolation.png`
* **Quy cách lưới**:
  * Dùng hàm `save_grid` trong `scripts/render_samples.py` với `nrow = 10`.
  * Mỗi hàng là 1 chuỗi chuyển tiếp 10 bước.
  * Ảnh nền trắng, phóng to nearest-neighbor để đồng bộ với `samples.png` và `nearest_neighbor.png`.
