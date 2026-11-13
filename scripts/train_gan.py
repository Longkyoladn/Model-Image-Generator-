import os
import sys
import json
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

# Đặt project root lên đầu sys.path để ưu tiên package local
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.preprocessing.builder import build_dataloader, denormalize, load_yaml_config
from models.gan.dcgan import DCGANGenerator, DCGANDiscriminator
from models.gan.c_dcgan import ConditionalDCGANGenerator, ConditionalDCGANDiscriminator

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(config_path, resume_path=None):
    cfg = load_yaml_config(config_path)
    
    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    
    dataset_cfg_path = exp_cfg.get("dataset_config")
    run_name = exp_cfg.get("name", "dcgan_run")
    is_conditional = exp_cfg.get("conditional", False)
    seed = exp_cfg.get("random_seed", 42)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Khởi động huấn luyện DCGAN: {run_name}")
    print(f"Thiết bị: {device}")
    
    # 1. Nạp Dataset và DataLoader
    train_loader, train_ds = build_dataloader(
        dataset_cfg_path,
        split="train",
        batch_size=train_cfg.get("batch_size", 64),
        shuffle=True
    )
    
    ds_cfg = load_yaml_config(dataset_cfg_path)
    dataset_name = ds_cfg.get("dataset", {}).get("name", "unknown")
    image_size = model_cfg.get("image_size", 32)
    channels = model_cfg.get("channels", 1)
    latent_dim = model_cfg.get("latent_dim", 100)
    hidden_dim = model_cfg.get("hidden_dim", 64)
    num_classes = exp_cfg.get("num_classes", 10)
    
    # 2. Tạo thư mục output cho run này
    output_sample_dir = os.path.join("outputs", "samples", dataset_name, "dcgan", run_name)
    checkpoint_dir = os.path.join("outputs", "checkpoints", dataset_name, "dcgan", run_name)
    metric_dir = os.path.join("outputs", "metrics", dataset_name, "dcgan", run_name)
    os.makedirs(output_sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    
    # 3. Khởi tạo mô hình
    if is_conditional:
        embed_dim = model_cfg.get("embed_dim", 50)
        netG = ConditionalDCGANGenerator(
            num_classes=num_classes,
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            out_channels=channels,
            image_size=image_size,
            hidden_dim=hidden_dim
        ).to(device)
        
        netD = ConditionalDCGANDiscriminator(
            num_classes=num_classes,
            in_channels=channels,
            image_size=image_size,
            hidden_dim=hidden_dim
        ).to(device)
    else:
        netG = DCGANGenerator(
            latent_dim=latent_dim,
            out_channels=channels,
            image_size=image_size,
            hidden_dim=hidden_dim
        ).to(device)
        
        netD = DCGANDiscriminator(
            in_channels=channels,
            image_size=image_size,
            hidden_dim=hidden_dim
        ).to(device)
        
    params_g = count_parameters(netG)
    params_d = count_parameters(netD)
    print(f" - Tham số Generator:     {params_g:,}")
    print(f" - Tham số Discriminator: {params_d:,}")
    print(f" - Tổng tham số DCGAN:    {(params_g + params_d):,}")
    
    # 4. Optimizer và Criterion
    lr_g = train_cfg.get("lr_g", 0.0002)
    lr_d = train_cfg.get("lr_d", 0.0002)
    beta1 = train_cfg.get("beta1", 0.5)
    beta2 = train_cfg.get("beta2", 0.999)
    
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))
    criterion = nn.BCEWithLogitsLoss()
    
    # Cố định noise và label để theo dõi ảnh mẫu qua các epoch
    num_fixed = train_cfg.get("num_fixed_samples", 16)
    fixed_noise = torch.randn(num_fixed, latent_dim, 1, 1, device=device)
    if is_conditional:
        fixed_labels = torch.arange(num_classes, device=device).repeat(num_fixed // num_classes + 1)[:num_fixed]
    
    start_epoch = 1
    history = {"epoch": [], "loss_d": [], "loss_g": [], "d_x": [], "d_gz": [], "diversity": []}
    
    # 5. Khôi phục từ checkpoint nếu có
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        netG.load_state_dict(ckpt["netG_state_dict"])
        netD.load_state_dict(ckpt["netD_state_dict"])
        optimizerG.load_state_dict(ckpt["optimizerG_state_dict"])
        optimizerD.load_state_dict(ckpt["optimizerD_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", history)
        print(f" -> Đã khôi phục checkpoint từ epoch {ckpt['epoch']}")
        
    epochs = train_cfg.get("epochs", 25)
    start_time = time.time()
    
    # 6. Training Loop
    for epoch in range(start_epoch, epochs + 1):
        netG.train()
        netD.train()
        epoch_loss_d, epoch_loss_g = 0.0, 0.0
        epoch_dx, epoch_dgz = 0.0, 0.0
        
        for batch_idx, (real_imgs, labels) in enumerate(train_loader):
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            
            real_target = torch.ones(b_size, 1, device=device)
            fake_target = torch.zeros(b_size, 1, device=device)
            
            # --- (A) Huấn luyện Discriminator ---
            optimizerD.zero_grad()
            if is_conditional:
                out_real = netD(real_imgs, labels)
            else:
                out_real = netD(real_imgs)
            loss_d_real = criterion(out_real, real_target)
            
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            if is_conditional:
                fake_imgs = netG(noise, labels)
                out_fake = netD(fake_imgs.detach(), labels)
            else:
                fake_imgs = netG(noise)
                out_fake = netD(fake_imgs.detach())
            loss_d_fake = criterion(out_fake, fake_target)
            
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizerD.step()
            
            # --- (B) Huấn luyện Generator ---
            optimizerG.zero_grad()
            if is_conditional:
                out_fake_for_g = netD(fake_imgs, labels)
            else:
                out_fake_for_g = netD(fake_imgs)
            loss_g = criterion(out_fake_for_g, real_target)
            loss_g.backward()
            optimizerG.step()
            
            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            epoch_dx += torch.sigmoid(out_real).mean().item()
            epoch_dgz += torch.sigmoid(out_fake).mean().item()
            
        n_batches = len(train_loader)
        avg_d = epoch_loss_d / n_batches
        avg_g = epoch_loss_g / n_batches
        avg_dx = epoch_dx / n_batches
        avg_dgz = epoch_dgz / n_batches
        
        # Theo dõi độ đa dạng (Standard Deviation pixel-level) để phát hiện Mode Collapse
        with torch.no_grad():
            batch_std = fake_imgs.std().item()
            
        history["epoch"].append(epoch)
        history["loss_d"].append(avg_d)
        history["loss_g"].append(avg_g)
        history["d_x"].append(avg_dx)
        history["d_gz"].append(avg_dgz)
        history["diversity"].append(batch_std)
        
        mode_collapse_alert = ""
        if batch_std < 0.05:
            mode_collapse_alert = " ⚠️ CẢNH BÁO: Độ đa dạng thấp (Nguy cơ Mode Collapse)!"
            
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss D: {avg_d:.4f} | Loss G: {avg_g:.4f} | D(x): {avg_dx:.2f} | D(G(z)): {avg_dgz:.2f} | Std: {batch_std:.3f}{mode_collapse_alert}")
        
        # Sinh ảnh mẫu định kỳ
        if epoch % train_cfg.get("sample_interval_epochs", 1) == 0:
            netG.eval()
            with torch.no_grad():
                if is_conditional:
                    sample_imgs = netG(fixed_noise, fixed_labels).cpu()
                else:
                    sample_imgs = netG(fixed_noise).cpu()
                sample_display = denormalize(sample_imgs, norm_range="[-1, 1]")
                grid = vutils.make_grid(sample_display, nrow=4, padding=2)
                vutils.save_image(grid, os.path.join(output_sample_dir, f"epoch_{epoch:03d}.png"))
                
        # Lưu checkpoint
        if epoch % train_cfg.get("checkpoint_interval_epochs", 5) == 0 or epoch == epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pt")
            latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
            ckpt_data = {
                "epoch": epoch,
                "netG_state_dict": netG.state_dict(),
                "netD_state_dict": netD.state_dict(),
                "optimizerG_state_dict": optimizerG.state_dict(),
                "optimizerD_state_dict": optimizerD.state_dict(),
                "history": history,
                "config": cfg
            }
            torch.save(ckpt_data, ckpt_path)
            torch.save(ckpt_data, latest_path)
            
    total_time = time.time() - start_time
    print(f"\n✅ Hoàn thành huấn luyện trong: {total_time/60:.2f} phút.")
    
    # 7. Xuất metadata theo chuẩn Mục 4
    metadata_path = os.path.join(metric_dir, "metadata.json")
    meta_info = {
        "model": "dcgan",
        "dataset": dataset_name,
        "run_name": run_name,
        "config": config_path,
        "image_size": image_size,
        "epochs": epochs,
        "training_time_seconds": round(total_time, 2),
        "params_generator": params_g,
        "params_discriminator": params_d,
        "total_params": params_g + params_d,
        "seed": seed,
        "checkpoint": os.path.join(checkpoint_dir, "checkpoint_latest.pt"),
        "device": str(device)
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)
        
    with open(os.path.join(metric_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    print(f" -> Đã lưu Metadata & History tại: {metric_dir}")
    return meta_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện DCGAN / Conditional DCGAN")
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn checkpoint để tiếp tục huấn luyện")
    args = parser.parse_args()
    
    train(args.config, args.resume)
