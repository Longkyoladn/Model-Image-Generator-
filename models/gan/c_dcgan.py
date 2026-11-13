import torch
import torch.nn as nn
from .dcgan import weights_init, DCGANGenerator, DCGANDiscriminator

class ConditionalDCGANGenerator(nn.Module):
    """
    Conditional DCGAN Generator:
    Nhận vector z và nhãn lớp/thuộc tính y, sinh ảnh tương ứng với nhãn yêu cầu.
    """
    def __init__(self, num_classes=10, embed_dim=50, latent_dim=100, out_channels=3, image_size=64, hidden_dim=64):
        super(ConditionalDCGANGenerator, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        
        # Nhúng nhãn lớp y thành vector liên tục
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        
        # Generator backbone nhận đầu vào kết hợp [z, label_embed]
        total_latent = latent_dim + embed_dim
        self.generator = DCGANGenerator(
            latent_dim=total_latent,
            out_channels=out_channels,
            image_size=image_size,
            hidden_dim=hidden_dim
        )

    def forward(self, z, labels):
        # z: (B, latent_dim) hoặc (B, latent_dim, 1, 1)
        # labels: (B,)
        if len(z.shape) == 4:
            z = z.view(z.size(0), -1)
            
        c_emb = self.label_embedding(labels) # (B, embed_dim)
        input_combined = torch.cat([z, c_emb], dim=1) # (B, latent_dim + embed_dim)
        input_combined = input_combined.view(input_combined.size(0), -1, 1, 1)
        
        return self.generator(input_combined)

    def sample(self, num_samples=16, class_id=None, device="cpu"):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        if class_id is None:
            labels = torch.randint(0, self.num_classes, (num_samples,), device=device)
        else:
            labels = torch.full((num_samples,), class_id, dtype=torch.long, device=device)
        with torch.no_grad():
            return self.forward(z, labels), labels


class ConditionalDCGANDiscriminator(nn.Module):
    """
    Conditional DCGAN Discriminator:
    Nhận ảnh và nhãn lớp, đánh giá xem cặp (ảnh, nhãn) có phải là thật hay giả.
    """
    def __init__(self, num_classes=10, in_channels=3, image_size=64, hidden_dim=64):
        super(ConditionalDCGANDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Nhúng nhãn thành 1 kênh bản đồ không gian (1 x H x W)
        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)
        
        # Discriminator nhận (in_channels + 1) kênh
        self.discriminator = DCGANDiscriminator(
            in_channels=in_channels + 1,
            image_size=image_size,
            hidden_dim=hidden_dim
        )

    def forward(self, img, labels):
        # img: (B, C, H, W)
        # labels: (B,)
        b, _, h, w = img.shape
        c_map = self.label_embedding(labels).view(b, 1, h, w)
        img_cond = torch.cat([img, c_map], dim=1) # (B, C + 1, H, W)
        return self.discriminator(img_cond)

