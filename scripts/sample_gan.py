import os
import sys
import json
import argparse
import torch
import torchvision.utils as vutils
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.preprocessing.builder import denormalize
from models.gan.dcgan import DCGANGenerator
from models.gan.c_dcgan import ConditionalDCGANGenerator

def sample_images(checkpoint_path, num_samples=1000, seed=42, output_dir="outputs/samples", batch_size=64, class_id=None):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    print(f"\n🎨 Bắt đầu sinh {num_samples:,} ảnh từ checkpoint: {checkpoint_path}")
    print(f"Thiết bị: {device} | Seed: {seed}")
    
    # 1. Đọc Checkpoint & Cấu hình
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})
    
    is_conditional = exp_cfg.get("conditional", False)
    num_classes = exp_cfg.get("num_classes", 10)
    dataset_name = exp_cfg.get("name", "unknown").split("_")[1] if "_" in exp_cfg.get("name", "") else "unknown"
    
    image_size = model_cfg.get("image_size", 32)
    channels = model_cfg.get("channels", 1)
    latent_dim = model_cfg.get("latent_dim", 100)
    hidden_dim = model_cfg.get("hidden_dim", 64)
    
    # 2. Khởi tạo Generator
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
    else:
        netG = DCGANGenerator(
            latent_dim=latent_dim,
            out_channels=channels,
            image_size=image_size,
            hidden_dim=hidden_dim
        ).to(device)
        
    netG.load_state_dict(ckpt["netG_state_dict"])
    netG.eval()
    
    # 3. Quá trình sinh ảnh hàng loạt theo batch
    generated_count = 0
    first_batch_display = None
    
    with torch.no_grad():
        while generated_count < num_samples:
            current_b_size = min(batch_size, num_samples - generated_count)
            noise = torch.randn(current_b_size, latent_dim, 1, 1, device=device)
            
            if is_conditional:
                if class_id is not None:
                    labels = torch.full((current_b_size,), class_id, dtype=torch.long, device=device)
                else:
                    labels = torch.randint(0, num_classes, (current_b_size,), device=device)
                fake_imgs = netG(noise, labels)
            else:
                fake_imgs = netG(noise)
                
            fake_imgs_01 = denormalize(fake_imgs.cpu(), norm_range="[-1, 1]")
            
            if first_batch_display is None:
                first_batch_display = fake_imgs_01[:16]
                
            # Lưu từng ảnh riêng biệt cho Quân tính FID
            for i in range(current_b_size):
                img_tensor = fake_imgs_01[i]
                img_path = os.path.join(images_dir, f"sample_{generated_count + i + 1:05d}.png")
                vutils.save_image(img_tensor, img_path)
                
            generated_count += current_b_size
            if generated_count % (batch_size * 5) == 0 or generated_count == num_samples:
                print(f" -> Đã sinh {generated_count:,} / {num_samples:,} ảnh...")
                
    # 4. Lưu Grid 16 ảnh đại diện
    if first_batch_display is not None:
        grid = vutils.make_grid(first_batch_display, nrow=4, padding=2)
        vutils.save_image(grid, os.path.join(output_dir, "sample_grid.png"))
        
    # 5. Xuất metadata chuẩn Mục 4
    metadata = {
        "model": "dcgan",
        "dataset": dataset_name,
        "run_name": exp_cfg.get("name", "sample_run"),
        "config": exp_cfg.get("dataset_config", "unknown"),
        "image_size": image_size,
        "num_samples": num_samples,
        "seed": seed,
        "checkpoint": os.path.basename(checkpoint_path),
        "device": str(device)
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"✅ Hoàn tất sinh {num_samples:,} ảnh tại: {output_dir}")
    print(f" -> Metadata lưu tại: {os.path.join(output_dir, 'metadata.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh ảnh từ checkpoint DCGAN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn file checkpoint")
    parser.add_argument("--num-samples", type=int, default=1000, help="Số lượng ảnh cần sinh")
    parser.add_argument("--seed", type=int, default=42, help="Random seed cố định")
    parser.add_argument("--output-dir", type=str, default="outputs/samples/test_run", help="Thư mục xuất ảnh")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước batch khi sinh ảnh")
    parser.add_argument("--class-id", type=int, default=None, help="Lớp cần sinh (nếu là Conditional)")
    args = parser.parse_args()
    
    sample_images(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        class_id=args.class_id
    )
