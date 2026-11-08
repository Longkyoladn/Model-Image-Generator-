# Model-Image-Generator-
phân tích các mô hình sinh ảnh và so sánh tính hiệu quả của chúng. Đưa ra những cải tiến để tối ưu các mô hình 

## Project structure

```text
configs/                 # Cấu hình huấn luyện và đánh giá
datasets/preprocessing/  # Dữ liệu và pipeline tiền xử lý
models/
  gan/                   # Generative Adversarial Networks
  vae/                   # Variational Autoencoders
  ddpm/                  # Denoising Diffusion Probabilistic Models
evaluation/              # FID, diversity và benchmark tốc độ
outputs/
  samples/               # Ảnh sinh ra
  checkpoints/           # Trọng số mô hình
  metrics/               # Kết quả đánh giá
scripts/                 # Script huấn luyện, suy luận và tiện ích
```
