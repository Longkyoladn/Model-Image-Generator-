"""Tên mô hình, loại file, cột bảng và màu dùng chung trong package aggregate."""

from __future__ import annotations

MODELS = ["vae", "dcgan", "ddpm"]
MODEL_DISPLAY = {"vae": "VAE", "dcgan": "DCGAN", "ddpm": "DDPM"}
KINDS = ["run", "quality", "speed", "compute", "calibration", "real_baseline"]

# Các trường phải giống nhau giữa mọi mô hình trong cùng một dataset.
EVAL_KEYS = ["eval_config", "eval_config_hash", "library", "commit", "ref_train", "ref_test", "num_gen"]
SPEED_ENV_KEYS = ["gpu", "torch"]
RUN_REQUIRED = ["config", "seed", "commit", "train_steps", "batch_size", "hardware"]
# Script tổng hợp tự tính các trường này (mục 4.3); metadata run có chúng là số điền tay → lỗi.
SCRIPT_COMPUTED_RUN_FIELDS = ["params_m", "gmacs_per_forward", "train_flops", "gpu_hours_t4"]

# Dòng phụ "so sánh sampler" của DDPM (mục 2.6): file quality có "comparison": "sampler",
# sinh SAMPLER_COMPARE_NUM_GEN ảnh với cùng seed đầu của bộ chính. Dòng phụ không vào
# bảng chính mà vào bảng "Đánh đổi sampler của DDPM", so trên cùng số ảnh với
# metrics.fid_train_10k / kid_test của dòng DDPM chính (tính trên 10k ảnh đầu).
COMPARISON_SAMPLER = "sampler"
SAMPLER_COMPARE_NUM_GEN = 10000
SAMPLER_METRICS = ("fid_train_10k", "kid_test_mean", "kid_test_std")

FIGURE_NAMES = ("samples", "interpolation", "nearest_neighbor")
FIGURE_TITLES = {"samples": "Ảnh mẫu", "interpolation": "Latent interpolation", "nearest_neighbor": "Nearest-neighbor"}

# (khóa, tiêu đề, số chữ số thập phân, hướng tốt: "low" | "high" | None)
QUALITY_COLUMNS = [
    ("fid_train_50k", "FID_train-50k", 2, "low"),
    ("fid_test_10k", "FID_test-10k", 2, "low"),
    ("kid_test", "KID_test ×10³", 2, "low"),
    ("precision", "Precision", 3, "high"),
    ("recall", "Recall", 3, "high"),
    ("density", "Density", 3, "high"),
    ("coverage", "Coverage", 3, "high"),
    ("lpips_diversity", "LPIPS div.", 3, None),
]
# (khóa, tiêu đề, số chữ số thập phân hoặc "sci" | None)
COST_COLUMNS = [
    ("params_m", "Tham số sinh (M)", 2),
    ("params_train_m", "Tham số train (M)", 2),
    ("gmacs_per_image", "GMACs/ảnh", 3),
    ("gflops_per_image", "GFLOPs/ảnh", 3),
    ("train_flops", "Train FLOPs", "sci"),
    ("train_hours", "Giờ train thực tế", 2),
    ("gpu_hours_t4", "Thời gian - T4 (giờ)", 2),
    ("hardware", "Phần cứng train", None),
    ("train_steps", "Train step", 0),
    ("latency_ms", "Latency (ms, batch 1)", 3),
    ("throughput_img_s", "Throughput (ảnh/s)", 1),
    ("throughput_batch", "ở batch", 0),
    ("peak_memory_mb", "Peak mem (MB)", 0),
    ("checkpoint_mb", "Checkpoint (MB)", 1),
]

# Màu theo mô hình (cố định, không đổi theo thứ hạng): 3 slot đầu của bảng màu
# tham chiếu, đã kiểm tra phân biệt được với người mù màu khi cả 3 cùng xuất hiện.
MODEL_COLORS = {"vae": "#2a78d6", "dcgan": "#eb6834", "ddpm": "#1baf7a"}
MODEL_MARKERS = {"vae": "o", "dcgan": "s", "ddpm": "^"}
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
