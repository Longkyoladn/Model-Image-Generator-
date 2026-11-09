"""Generated-image diversity evaluation utilities."""

import torch
import torch.nn.functional as F
import lpips # Cần pip install lpips

class DiversityCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        # Khởi tạo LPIPS (sử dụng AlexNet hoặc VGG làm backbone)
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_model.eval()

    def calculate_feature_diversity(self, features):
        """
        Tính độ phân tán (Variance) của features.
        Đầu vào: Tensor (N, D) - thường là feature từ InceptionV3
        """
        # Chuẩn hóa features về mean=0, var=1 để tránh ảnh hưởng scale
        features = F.normalize(features, p=2, dim=1)
        variance = torch.var(features, dim=0).mean().item()
        return variance

    def calculate_lpips_diversity(self, images, num_pairs=100):
        """
        Tính LPIPS giữa các cặp ảnh ngẫu nhiên trong tập sinh.
        LPIPS càng cao => càng đa dạng (nhưng cũng có thể là rác, cần đánh giá cùng FID).
        """
        n = images.shape[0]
        if n < 2:
            return 0.0
            
        # Lấy ngẫu nhiên các cặp
        indices = torch.randperm(n)
        img1 = images[indices[:num_pairs]]
        img2 = images[indices[num_pairs:num_pairs*2]]
        
        # Đảm bảo LPIPS nhận input [-1, 1]. Nếu data là [0, 1] thì cần đổi.
        # Giả định pipeline đã chuẩn hóa về [-1, 1] hoặc ta cần convert.
        if img1.min() >= 0:
            img1 = img1 * 2 - 1
            img2 = img2 * 2 - 1
            
        with torch.no_grad():
            dists = self.lpips_model(img1, img2)
            
        return dists.mean().item()

    def calculate_ms_ssim(self, images, num_pairs=100):
        """
        Tính MS-SSIM trung bình giữa các cặp ảnh ngẫu nhiên.
        MS-SSIM càng thấp => ảnh càng đa dạng (khác nhau nhiều).
        Lưu ý: Hàm ms_ssim của pytorch-msssim trả về độ tương đồng (1 là giống hệt).
        """
        n = images.shape[0]
        if n < 2:
            return 1.0  # Trả về max similarity nếu không đủ ảnh để so sánh
            
        # Lấy ngẫu nhiên các cặp
        indices = torch.randperm(n)
        img1 = images[indices[:num_pairs]]
        img2 = images[indices[num_pairs:num_pairs*2]]
        
        # Đảm bảo input nằm trong khoảng [0, 1]
        # Nếu pipeline trả về [-1, 1], cần convert lại
        if img1.min() < 0:
            img1 = (img1 + 1) / 2
            img2 = (img2 + 1) / 2
            
        # Tính MS-SSIM (chạy trên GPU nếu có)
        with torch.no_grad():
            # pytorch_msssim yêu cầu size > 160, nếu ảnh nhỏ (32x32, 28x28) cần resize tạm
            if img1.shape[-1] < 160:
                img1_resized = F.interpolate(img1, size=(160, 160), mode='bilinear', align_corners=False)
                img2_resized = F.interpolate(img2, size=(160, 160), mode='bilinear', align_corners=False)
                score = ms_ssim(img1_resized, img2_resized, data_range=1.0, size_average=True)
            else:
                score = ms_ssim(img1, img2, data_range=1.0, size_average=True)
                
        return score.item()