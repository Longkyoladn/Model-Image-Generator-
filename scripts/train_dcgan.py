"""
Script huấn luyện mô hình DCGAN (Task 3.1 & 3.2 trong TASK_ASSIGNMENT.md).
- Hỗ trợ cả 3 bộ dữ liệu: Fashion-MNIST, CIFAR-10, CelebA.
- Quản lý cấu hình hoàn toàn bằng file YAML (configs/dcgan/<dataset>/<exp>.yaml).
- Training loop Minimax chuẩn (cập nhật D với label smoothing, cập nhật G).
- Giám sát trực quan bằng Fixed Noise (lưới 8x8 seed 42 theo mục 4.3 & 2.6).
- Hỗ trợ lưu Checkpoint định kỳ và Resume tiếp tục huấn luyện khi bị ngắt.
- Xuất file nhật ký loss.csv và metadata.json đúng chuẩn schema mục 4.3.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

# Đặt project root lên đầu sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
import yaml

from evaluation.common import git_commit, normalize_config, sync
from evaluation.calibrate import device_key
from datasets.preprocessing.builder import build_dataloader, load_yaml_config
from models.gan import DCGAN, weights_init

# Thử import save_grid từ scripts/render_samples.py để đồng nhất kiểu lưới ảnh của nhóm
try:
    from scripts.render_samples import save_grid
except ImportError:
    import torchvision.utils as vutils

    def save_grid(images: torch.Tensor, path: str | Path, nrow: int = 8, **kwargs) -> Path:
        images = images.detach().float().clamp(0, 1).cpu()
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        vutils.save_image(images, str(path), nrow=nrow, padding=2, normalize=False)
        return Path(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình DCGAN")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Đường dẫn file cấu hình YAML (ví dụ configs/dcgan/cifar10/base.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device huấn luyện (cuda hoặc cpu)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Đường dẫn file checkpoint .pt để tiếp tục huấn luyện",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Ghi đè số epochs cần train từ config",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Giới hạn số bước train tối đa mỗi epoch (tiện lợi cho dry-run / test)",
    )
    parser.add_argument(
        "--overfit_samples",
        type=int,
        default=None,
        help="Lấy cố định N ảnh đầu của tập train (64-512) cho bài kiểm tra Overfit (Mục 2.3 Giai đoạn 0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Ghi đè batch size từ config",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Ghi đè learning rate cho cả G và D",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=["overfit", "tuning", "full"],
        help="Giai đoạn thí nghiệm (overfit | tuning | full, theo mục 4.3)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Tên của run (mặc định lấy tên file config hoặc theo exp)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1. Nạp và xử lý file cấu hình
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {args.config}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Đọc các thông số
    ds_cfg = cfg.get("dataset", {})
    dataset_name = ds_cfg.get("name", "cifar10") if isinstance(ds_cfg, dict) else str(ds_cfg)
    ds_config_path = ds_cfg.get("dataset_config", f"configs/datasets_config/{dataset_name}.yaml") if isinstance(ds_cfg, dict) else f"configs/datasets_config/{dataset_name}.yaml"
    image_shape = tuple(cfg.get("image_shape", ds_cfg.get("image_shape", (3, 32, 32)) if isinstance(ds_cfg, dict) else (3, 32, 32)))

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    log_cfg = cfg.get("logging", {})

    latent_dim = model_cfg.get("latent_dim", 100)
    g_feat = model_cfg.get("generator_features", 64)
    d_feat = model_cfg.get("discriminator_features", 64)
    d_steps = model_cfg.get("d_steps", 1)

    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 40)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 64)
    lr_d = args.lr if args.lr is not None else train_cfg.get("lr_d", train_cfg.get("lr", 0.0002))
    lr_g = args.lr if args.lr is not None else train_cfg.get("lr_g", train_cfg.get("lr", 0.0002))
    beta1 = train_cfg.get("beta1", 0.5)
    beta2 = train_cfg.get("beta2", 0.999)
    label_smoothing = train_cfg.get("label_smoothing", 0.9)
    seed = train_cfg.get("seed", 42)
    num_workers = train_cfg.get("num_workers", 0)

    run_name = args.run_name or cfg_path.stem
    if run_name == "base":
        run_name = f"dcgan_{dataset_name}_base"

    # Thiết lập thư mục lưu trữ
    ckpt_dir = PROJECT_ROOT / log_cfg.get("checkpoint_dir", f"outputs/checkpoints/dcgan/{dataset_name}/{run_name}")
    sample_dir = PROJECT_ROOT / log_cfg.get("sample_dir", f"outputs/samples/dcgan/{dataset_name}/{run_name}")
    metric_dir = PROJECT_ROOT / log_cfg.get("metric_dir", f"outputs/metrics/dcgan/{dataset_name}/{run_name}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    save_every = log_cfg.get("save_every_epochs", 5)
    sample_every = log_cfg.get("sample_every_epochs", 1)
    num_fixed_samples = log_cfg.get("num_fixed_samples", 64)

    # Cố định Seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("\n" + "=" * 75)
    print(f"🚀 HUẤN LUYỆN DCGAN: {dataset_name.upper()} | RUN: {run_name}")
    print("=" * 75)
    print(f" - Config:           {normalize_config(str(cfg_path))}")
    print(f" - Image shape:      {image_shape}")
    print(f" - Latent Dim:       {latent_dim}")
    print(f" - Batch size:       {batch_size}")
    print(f" - Epochs:           {epochs}")
    print(f" - Device:           {device}")
    print(f" - Learning Rate:    D={lr_d}, G={lr_g} (Adam beta1={beta1})")
    print(f" - Label Smoothing:  {label_smoothing}")
    print(f" - Checkpoint dir:   {ckpt_dir}")

    # 2. Xây dựng DataLoader qua Pipeline dữ liệu Task 3.1
    print("\n📦 Nạp DataLoader...")
    train_loader, train_ds = build_dataloader(
        ds_config_path,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    if args.overfit_samples is not None and args.overfit_samples > 0:
        n_sub = min(args.overfit_samples, len(train_ds))
        train_ds = torch.utils.data.Subset(train_ds, list(range(n_sub)))
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=min(batch_size, n_sub),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f" - [Chế độ Overfit Mục 2.3] Cắt cố định {n_sub:,} ảnh đầu ({len(train_loader)} batches/epoch)")
    else:
        print(f" - Số lượng mẫu train: {len(train_ds):,} ảnh ({len(train_loader)} batches/epoch)")

    # 3. Khởi tạo mô hình DCGAN
    model = DCGAN(
        image_shape=image_shape,
        latent_dim=latent_dim,
        generator_features=g_feat,
        discriminator_features=d_feat,
        d_steps=d_steps,
    ).to(device)

    # Optimizers và Loss
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=lr_g, betas=(beta1, beta2))
    criterion = nn.BCEWithLogitsLoss()

    # Fixed Noise cố định để theo dõi sự tiến hóa qua từng epoch (seed 42)
    fixed_gen = torch.Generator(device=device).manual_seed(42)
    fixed_z = torch.randn(num_fixed_samples, latent_dim, generator=fixed_gen, device=device)

    start_epoch = 1
    total_steps = 0
    total_train_seconds = 0.0
    dev_str = device_key(device)
    checkpoint_history = []

    # 4. Hỗ trợ Resume nếu có checkpoint
    if args.resume:
        resume_p = Path(args.resume)
        if not resume_p.is_absolute():
            resume_p = PROJECT_ROOT / resume_p
        if resume_p.exists():
            print(f"🔄 Đang phục hồi từ checkpoint: {resume_p}...")
            ckpt = torch.load(resume_p, map_location=device, weights_only=False)
            model.generator.load_state_dict(ckpt["generator_state_dict"])
            model.discriminator.load_state_dict(ckpt["discriminator_state_dict"])
            if "optimizer_d_state_dict" in ckpt:
                opt_d.load_state_dict(ckpt["optimizer_d_state_dict"])
            if "optimizer_g_state_dict" in ckpt:
                opt_g.load_state_dict(ckpt["optimizer_g_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            total_steps = ckpt.get("total_steps", 0)
            total_train_seconds = ckpt.get("total_train_seconds", 0.0)
            print(f"   --> Phục hồi thành công! Tiếp tục từ Epoch {start_epoch}, Step {total_steps}")
        else:
            print(f"⚠️ Không tìm thấy file resume: {resume_p}, sẽ train mới từ đầu.")

    # File log Loss
    loss_log_path = metric_dir / "loss.csv"
    write_header = not loss_log_path.exists() or (start_epoch == 1)
    loss_file = open(loss_log_path, "a", newline="", encoding="utf-8")
    loss_writer = csv.writer(loss_file)
    if write_header:
        loss_writer.writerow(["epoch", "step", "loss_d", "loss_g", "loss_d_real", "loss_d_fake", "elapsed_sec"])
        loss_file.flush()

    # 5. Training Loop
    print("\n" + "=" * 75)
    print("🎯 BẮT ĐẦU TRAINING LOOP")
    print("=" * 75)

    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_start_time = time.time()
            model.generator.train()
            model.discriminator.train()

            running_loss_d = 0.0
            running_loss_g = 0.0
            running_loss_d_real = 0.0
            running_loss_d_fake = 0.0
            num_batches = len(train_loader)

            for batch_idx, (real_imgs, _) in enumerate(train_loader):
                if args.max_steps is not None and batch_idx >= args.max_steps:
                    break
                b = real_imgs.size(0)
                real_imgs = real_imgs.to(device)

                step_start_time = time.perf_counter()

                # ---------------------------------------------
                # 1. Cập nhật Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
                # ---------------------------------------------
                for _ in range(d_steps):
                    opt_d.zero_grad(set_to_none=True)

                    # Ảnh thật (Label smoothing)
                    output_real = model.discriminator(real_imgs).view(-1)
                    target_real = torch.full((b,), label_smoothing, device=device)
                    loss_d_real = criterion(output_real, target_real)

                    # Ảnh giả
                    z = torch.randn(b, latent_dim, device=device)
                    fake_imgs = model.generator(z)
                    output_fake = model.discriminator(fake_imgs.detach()).view(-1)
                    target_fake = torch.full((b,), 0.0, device=device)
                    loss_d_fake = criterion(output_fake, target_fake)

                    loss_d = loss_d_real + loss_d_fake
                    loss_d.backward()
                    opt_d.step()

                # ---------------------------------------------
                # 2. Cập nhật Generator: maximize log(D(G(z)))
                # ---------------------------------------------
                opt_g.zero_grad(set_to_none=True)
                output_fake_for_g = model.discriminator(fake_imgs).view(-1)
                target_g = torch.full((b,), 1.0, device=device)
                loss_g = criterion(output_fake_for_g, target_g)
                loss_g.backward()
                opt_g.step()

                sync(device)
                step_elapsed = time.perf_counter() - step_start_time
                total_train_seconds += step_elapsed
                total_steps += 1

                # Tích lũy loss
                running_loss_d += loss_d.item()
                running_loss_g += loss_g.item()
                running_loss_d_real += loss_d_real.item()
                running_loss_d_fake += loss_d_fake.item()

                # In tiến độ định kỳ
                if (batch_idx + 1) % max(1, num_batches // 5) == 0 or (batch_idx + 1) == num_batches:
                    cur_d = running_loss_d / (batch_idx + 1)
                    cur_g = running_loss_g / (batch_idx + 1)
                    speed = b / max(step_elapsed, 1e-4)
                    print(
                        f"Epoch [{epoch:03d}/{epochs:03d}] "
                        f"Step [{batch_idx + 1:04d}/{num_batches:04d}] "
                        f"| Loss D: {cur_d:.4f} (Real: {loss_d_real.item():.4f}, Fake: {loss_d_fake.item():.4f}) "
                        f"| Loss G: {cur_g:.4f} "
                        f"| Speed: {speed:.1f} img/s"
                    )

            # Trung bình loss của cả epoch
            avg_d = running_loss_d / num_batches
            avg_g = running_loss_g / num_batches
            avg_d_real = running_loss_d_real / num_batches
            avg_d_fake = running_loss_d_fake / num_batches
            epoch_duration = time.time() - epoch_start_time

            # Ghi loss ra CSV
            loss_writer.writerow([epoch, total_steps, avg_d, avg_g, avg_d_real, avg_d_fake, total_train_seconds])
            loss_file.flush()

            print(
                f"--> Hoàn thành Epoch {epoch:03d}/{epochs:03d} trong {epoch_duration:.2f}s "
                f"| Avg Loss D: {avg_d:.4f} | Avg Loss G: {avg_g:.4f}"
            )

            # 6. Sinh ảnh mẫu cố định (Fixed Noise Visual Monitoring)
            if (epoch % sample_every == 0) or (epoch == epochs):
                model.generator.eval()
                with torch.no_grad():
                    fixed_fakes = model.generator(fixed_z)
                    fixed_fakes_01 = torch.clamp((fixed_fakes + 1.0) / 2.0, 0.0, 1.0)
                    sample_path = sample_dir / f"epoch_{epoch:03d}.png"
                    save_grid(fixed_fakes_01, sample_path, nrow=8)
                    # Đồng thời cập nhật latest_sample.png
                    save_grid(fixed_fakes_01, sample_dir / "latest_sample.png", nrow=8)
                print(f"   🖼️ Đã lưu lưới ảnh mẫu: {sample_path}")

            # 7. Lưu Checkpoint định kỳ
            if (epoch % save_every == 0) or (epoch == epochs):
                ckpt_data = {
                    "config": {
                        "model": "dcgan",
                        "dataset": dataset_name,
                        "image_shape": list(image_shape),
                        "latent_dim": latent_dim,
                        "generator_features": g_feat,
                        "discriminator_features": d_feat,
                        "d_steps": d_steps,
                    },
                    "generator_state_dict": model.generator.state_dict(),
                    "discriminator_state_dict": model.discriminator.state_dict(),
                    "optimizer_d_state_dict": opt_d.state_dict(),
                    "optimizer_g_state_dict": opt_g.state_dict(),
                    "epoch": epoch,
                    "total_steps": total_steps,
                    "total_train_seconds": total_train_seconds,
                }
                ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt"
                torch.save(ckpt_data, ckpt_path)
                torch.save(ckpt_data, ckpt_dir / "checkpoint_latest.pt")
                checkpoint_history.append({
                    "epoch": epoch,
                    "step": total_steps,
                    "checkpoint": ckpt_path.name,
                    "hardware_hours": {dev_str: round(total_train_seconds / 3600.0, 6)},
                })
                print(f"   💾 Đã lưu checkpoint: {ckpt_path}")

    finally:
        loss_file.close()

    # 8. Lưu Checkpoint Final và Metadata (kind: "run" theo chuẩn Mục 4.3)
    final_ckpt_path = ckpt_dir / "checkpoint_final.pt"
    final_ckpt_data = {
        "config": {
            "model": "dcgan",
            "dataset": dataset_name,
            "image_shape": list(image_shape),
            "latent_dim": latent_dim,
            "generator_features": g_feat,
            "discriminator_features": d_feat,
            "d_steps": d_steps,
        },
        "generator_state_dict": model.generator.state_dict(),
        "discriminator_state_dict": model.discriminator.state_dict(),
        "optimizer_d_state_dict": opt_d.state_dict(),
        "optimizer_g_state_dict": opt_g.state_dict(),
        "epoch": epochs,
        "total_steps": total_steps,
        "total_train_seconds": total_train_seconds,
    }
    torch.save(final_ckpt_data, final_ckpt_path)
    print(f"\n🎉 Huấn luyện hoàn tất! Checkpoint cuối cùng: {final_ckpt_path}")

    # Metadata run (schema chuẩn mục 4.3 trong TASK_ASSIGNMENT.md)
    train_hours = total_train_seconds / 3600.0

    metadata = {
        "kind": "run",
        "model": "dcgan",
        "dataset": dataset_name,
        "run_name": run_name,
        "stage": args.stage,
        "config": normalize_config(str(cfg_path)),
        "image_size": image_shape[1],
        "num_samples": num_fixed_samples,
        "seed": seed,
        "checkpoint": final_ckpt_path.name,
        "train_steps": total_steps,
        "batch_size": batch_size,
        "precision": "fp32",
        "hardware": [
            {"device": dev_str, "train_hours": round(train_hours, 6)}
        ],
        "eval_config": "configs/eval/fid.yaml",
        "commit": git_commit() or "uncommitted",
    }

    meta_file = metric_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    ckpt_hist_file = metric_dir / "checkpoints.json"
    with open(ckpt_hist_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_history, f, indent=2, ensure_ascii=False)

    print(f"📄 Metadata run lưu tại: {meta_file}")
    print(f"📄 Lịch sử checkpoint (để điền curve KID) lưu tại: {ckpt_hist_file}")
    print("=" * 75)


if __name__ == "__main__":
    main()
