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
        can_cache = use_cache and (split_name == "real")
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
        from torch.utils.data import Subset
        
        real_dataset = real_loader.dataset
        fake_dataset = fake_loader.dataset
        
        class_fid_scores = {}
        
        for cls_idx in class_indices:
            # Lấy danh sách index của class này
            # Xử lý an toàn: Kiểm tra xem dataset dùng thuộc tính 'targets' hay 'labels'
            try:
                real_targets = real_dataset.targets
                fake_targets = fake_dataset.targets
            except AttributeError:
                try:
                    real_targets = real_dataset.labels
                    fake_targets = fake_dataset.labels
                except AttributeError:
                    raise AttributeError("Dataset không có thuộc tính '.targets' hoặc '.labels'.")

            real_indices = [i for i, label in enumerate(real_targets) if label == cls_idx]
            fake_indices = [i for i, label in enumerate(fake_targets) if label == cls_idx]
            
            # Bỏ qua nếu không có đủ ảnh để tính FID (cần ít nhất 2 ảnh để tính covariance)
            if len(real_indices) < 2 or len(fake_indices) < 2:
                print(f"Warning: Bỏ qua class {cls_idx} vì không đủ số lượng ảnh (Real: {len(real_indices)}, Fake: {len(fake_indices)})")
                continue
            
            # Tạo Subset (chỉ lưu index, không copy ảnh vào RAM)
            real_subset = Subset(real_dataset, real_indices)
            fake_subset = Subset(fake_dataset, fake_indices)
            
            real_loader_cls = torch.utils.data.DataLoader(real_subset, batch_size=real_loader.batch_size, shuffle=False)
            fake_loader_cls = torch.utils.data.DataLoader(fake_subset, batch_size=fake_loader.batch_size, shuffle=False)
            
            ds_name = f"{real_dataset.__class__.__name__}_class_{cls_idx}"
            
            fid_score = self.calculate_fid(
                real_loader_cls, 
                fake_loader_cls, 
                dataset_name=ds_name, 
                use_cache=use_cache
            )
            class_fid_scores[cls_idx] = fid_score
            print(f" -> FID class {cls_idx}: {fid_score:.4f}")
            
        if not class_fid_scores:
            return {}, 0.0
            
        mean_fid = sum(class_fid_scores.values()) / len(class_fid_scores)
        return class_fid_scores, mean_fid