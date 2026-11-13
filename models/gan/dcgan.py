import torch
import torch.nn as nn

def weights_init(m):
    """
    Khởi tạo trọng số DCGAN chuẩn theo paper Radford et al. (2016):
    - Conv, ConvTranspose: N(0.0, 0.02)
    - BatchNorm: N(1.0, 0.02), bias = 0
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class DCGANGenerator(nn.Module):
    """
    Generator chuẩn cho DCGAN, hỗ trợ linh hoạt cả kích thước 32x32 (Fashion-MNIST, CIFAR-10)
    và kích thước 64x64 (CelebA).
    """
    def __init__(self, latent_dim=100, out_channels=3, image_size=64, hidden_dim=64):
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        if image_size == 64:
            # Output 64x64: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
            self.model = nn.Sequential(
                # (B, latent_dim, 1, 1) -> (B, hidden_dim * 8, 4, 4)
                nn.ConvTranspose2d(latent_dim, hidden_dim * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim * 8),
                nn.ReLU(True),
                
                # (B, hidden_dim * 8, 4, 4) -> (B, hidden_dim * 4, 8, 8)
                nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_dim * 4),
                nn.ReLU(True),
                
                # (B, hidden_dim * 4, 8, 8) -> (B, hidden_dim * 2, 16, 16)
                nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_dim * 2),
                nn.ReLU(True),
                
                # (B, hidden_dim * 2, 16, 16) -> (B, hidden_dim, 32, 32)
                nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(True),
                
                # (B, hidden_dim, 32, 32) -> (B, out_channels, 64, 64)
                nn.ConvTranspose2d(hidden_dim, out_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        elif image_size == 32:
            # Output 32x32: 4x4 -> 8x8 -> 16x16 -> 32x32
            self.model = nn.Sequential(
                # (B, latent_dim, 1, 1) -> (B, hidden_dim * 4, 4, 4)
                nn.ConvTranspose2d(latent_dim, hidden_dim * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim * 4),
                nn.ReLU(True),
                
                # (B, hidden_dim * 4, 4, 4) -> (B, hidden_dim * 2, 8, 8)
                nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_dim * 2),
                nn.ReLU(True),
                
                # (B, hidden_dim * 2, 8, 8) -> (B, hidden_dim, 16, 16)
                nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(True),
                
                # (B, hidden_dim, 16, 16) -> (B, out_channels, 32, 32)
                nn.ConvTranspose2d(hidden_dim, out_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        else:
            raise ValueError(f"image_size {image_size} chưa được hỗ trợ. Chọn 32 hoặc 64.")
            
        self.apply(weights_init)

    def forward(self, z):
        if len(z.shape) == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.model(z)

    def sample(self, num_samples=16, device="cpu"):
        z = torch.randn(num_samples, self.latent_dim, 1, 1, device=device)
        with torch.no_grad():
            return self.forward(z)


class DCGANDiscriminator(nn.Module):
    """
    Discriminator chuẩn cho DCGAN, hỗ trợ ảnh 32x32 hoặc 64x64.
    Đầu ra là Logit thô (dùng với BCEWithLogitsLoss).
    """
    def __init__(self, in_channels=3, image_size=64, hidden_dim=64):
        super(DCGANDiscriminator, self).__init__()
        self.image_size = image_size
        
        if image_size == 64:
            self.model = nn.Sequential(
                # (B, in_channels, 64, 64) -> (B, hidden_dim, 32, 32) (không dùng BN tầng đầu)
                nn.Conv2d(in_channels, hidden_dim, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                
                # (B, hidden_dim, 32, 32) -> (B, hidden_dim * 2, 16, 16)
                nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_dim * 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                # (B, hidden_dim * 2, 16, 16) -> (B, hidden_dim * 4, 8, 8)
                nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_dim * 4),
                nn.LeakyReLU(0.2, inplace=True),
                
                # (B, hidden_dim * 4, 8, 8) -> (B, hidden_dim * 8, 4, 4)
                nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_dim * 8),
                nn.LeakyReLU(0.2, inplace=True),
                
                # (B, hidden_dim * 8, 4, 4) -> (B, 1, 1, 1)
                nn.Conv2d(hidden_dim * 8, 1, 4, 1, 0, bias=False)
            )
        elif image_size == 32:
            self.model = nn.Sequential(
                # (B, in_channels, 32, 32) -> (B, hidden_dim, 16, 16)
                nn.Conv2d(in_channels, hidden_dim, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                
                # (B, hidden_dim, 16, 16) -> (B, hidden_dim * 2, 8, 8)
                nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_dim * 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                # (B, hidden_dim * 2, 8, 8) -> (B, hidden_dim * 4, 4, 4)
                nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_dim * 4),
                nn.LeakyReLU(0.2, inplace=True),
                
                # (B, hidden_dim * 4, 4, 4) -> (B, 1, 1, 1)
                nn.Conv2d(hidden_dim * 4, 1, 4, 1, 0, bias=False)
            )
        else:
            raise ValueError(f"image_size {image_size} chưa được hỗ trợ. Chọn 32 hoặc 64.")
            
        self.apply(weights_init)

    def forward(self, img):
        out = self.model(img)
        return out.view(-1, 1)

