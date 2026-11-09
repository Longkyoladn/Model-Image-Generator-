"""Fréchet Inception Distance evaluation utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg
import numpy as np
import os
import hashlib

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load InceptionV3 pretrained
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        self.model.fc = nn.Identity() # Bỏ lớp classification cuối để lấy feature 2048 chiều
        self.model.eval()
        self.device = device
        self.model.to(device)

    def forward(self, x):
        # Yêu cầu của InceptionV3 là ảnh 299x299
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Xử lý kênh: Nếu là 1 kênh (Fashion-MNIST), nhân bản thành 3 kênh
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # Yêu cầu normalize của InceptionV3 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Giả định đầu vào x đang ở khoảng [0, 1] (sau khi qua dataloader đã normalize)
        # Nếu dataloader trả về [-1, 1], cần thêm bước (x + 1) / 2
        # TODO: Cần thống nhất về range của tensor đầu ra. Ở đây giả định [0, 1].
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        x = (x - mean) / std
        
        with torch.no_grad():
            features = self.model(x)
        return features

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Tính khoảng cách Frechet giữa 2 phân phối Gaussian"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

class FIDCalculator:
    def __init__(self, device='cuda', cache_dir='outputs/metrics/cache'):
        self.device = device
        self.extractor = InceptionFeatureExtractor(device)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, dataloader, dataset_name, split_name):
        # Tạo hash dựa trên tên dataset và split để cache
        key = f"{dataset_name}_{split_name}"
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"stats_{hash_key}.npz")

    def compute_statistics(self, dataloader, dataset_name, split_name, use_cache=True):
        """Tính mean và covariance của features"""
        cache_path = self._get_cache_path(dataloader, dataset_name, split_name)
        
        if use_cache and os.path.exists(cache_path):
            print(f"Loading cached stats from {cache_path}")
            data = np.load(cache_path)
            return data['mu'], data['sigma']

        print(f"Computing stats for {dataset_name} - {split_name}...")
        features = []
        for batch in dataloader:
            # Giả định batch là dict hoặc tuple (images, labels)
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(self.device)
            
            feat = self.extractor(imgs)
            features.append(feat.cpu().numpy())
            
        features = np.concatenate(features, axis=0)
        
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        if use_cache:
            np.savez(cache_path, mu=mu, sigma=sigma)
            
        return mu, sigma

    def calculate_fid(self, real_loader, fake_loader, dataset_name, use_cache=True):
        """Tính FID score tổng thể"""
        mu_real, sigma_real = self.compute_statistics(real_loader, dataset_name, "real", use_cache)
        mu_fake, sigma_fake = self.compute_statistics(fake_loader, dataset_name, "fake", use_cache)
        
        fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        return fid_score

    def calculate_fid_per_class(self, real_loader, fake_loader, class_indices, use_cache=True):
        """
        Tính FID cho từng class riêng biệt.
        Giả định dataloader trả về tuple (images, labels).
        """
        class_fid_scores = {}
        
        for cls_idx in class_indices:
            print(f"Calculating FID for class {cls_idx}...")
            
            # Lọc dataloader theo nhãn
            # Lưu ý: Cách này load lại toàn bộ dataset mỗi lần, sẽ chậm nếu dataset lớn.
            # Cách tối ưu hơn là tạo custom SubsetDataset trước khi vào vòng lặp, nhưng 
            # để đơn giản và dễ hiểu, ta lọc ngay trong batch.
            
            def filter_by_class(loader, target_class):
                filtered_batches = []
                for batch in loader:
                    imgs, labels = batch[0], batch[1]
                    mask = (labels == target_class)
                    if mask.any():
                        filtered_batches.append((imgs[mask], labels[mask]))
                
                if not filtered_batches:
                    raise ValueError(f"No samples found for class {target_class} in loader.")
                
                # Gộp tất cả ảnh cùng class thành 1 tensor lớn để tạo loader mới
                all_imgs = torch.cat([b[0] for b in filtered_batches], dim=0)
                all_labels = torch.cat([b[1] for b in filtered_batches], dim=0)
                dataset = torch.utils.data.TensorDataset(all_imgs, all_labels)
                return torch.utils.data.DataLoader(dataset, batch_size=real_loader.batch_size, shuffle=False)

            real_loader_cls = filter_by_class(real_loader, cls_idx)
            fake_loader_cls = filter_by_class(fake_loader, cls_idx)
            
            # Gọi lại hàm tính FID tổng thể cho class này
            # Sử dụng tên dataset kèm hậu tố class để cache riêng
            fid_score = self.calculate_fid(
                real_loader_cls, 
                fake_loader_cls, 
                dataset_name=f"cifar10_class_{cls_idx}", # Giả định tên dataset
                use_cache=use_cache
            )
            class_fid_scores[cls_idx] = fid_score
            
        mean_fid = sum(class_fid_scores.values()) / len(class_fid_scores)
        return class_fid_scores, mean_fid