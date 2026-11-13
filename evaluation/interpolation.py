import torch
import torch.nn.functional as F
import imageio
import os

def linear_interpolate(z1, z2, steps=8):
    """Nội suy tuyến tính giữa 2 vector latent."""
    alphas = torch.linspace(0, 1, steps, device=z1.device).view(-1, 1)
    return z1.unsqueeze(0) * (1 - alphas) + z2.unsqueeze(0) * alphas

def slerp(z1, z2, steps=8, eps=1e-8):
    """Nội suy cầu (Spherical Linear Interpolation) — mượt hơn linear với VAE/GAN."""
    # Chuẩn hóa để tính góc
    z1_norm_val = z1.norm()
    z2_norm_val = z2.norm()
    z1_n = z1 / (z1_norm_val + eps)
    z2_n = z2 / (z2_norm_val + eps)
    
    dot = torch.clamp((z1_n * z2_n).sum(), -1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega) + eps

    alphas = torch.linspace(0, 1, steps, device=z1.device)
    zs = []
    for a in alphas:
        if omega.abs() < eps:
            z = z1 * (1 - a) + z2 * a
        else:
            # Nội suy hướng (slerp) và nội suy độ dài (lerp)
            dir_interp = (torch.sin((1 - a) * omega) / sin_omega) * z1_n + (torch.sin(a * omega) / sin_omega) * z2_n
            norm_interp = z1_norm_val * (1 - a) + z2_norm_val * a
            z = dir_interp * norm_interp
        zs.append(z)
    return torch.stack(zs)

def make_interpolation_grid(model, z1, z2, steps=8, method="slerp"):
    """Sinh ảnh nội suy và ghép thành grid ngang."""
    if method == "slerp":
        zs = slerp(z1, z2, steps)
    else:
        zs = linear_interpolate(z1, z2, steps)

    with torch.no_grad():
        imgs = model.decode(zs)  # giả định model có hàm decode(z)
    # imgs: (steps, C, H, W) trong [0, 1] hoặc [-1, 1]
    imgs = (imgs.clamp(-1, 1) + 1) / 2 if imgs.min() < 0 else imgs.clamp(0, 1)
    grid = torch.cat([imgs[i] for i in range(steps)], dim=-1)  # nối ngang
    return grid

def save_interpolation_gif(model, z1, z2, steps=16, method="slerp", path="outputs/samples/interp.gif"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if method == "slerp":
        zs = slerp(z1, z2, steps)
    else:
        zs = linear_interpolate(z1, z2, steps)

    with torch.no_grad():
        imgs = model.decode(zs)
    imgs = (imgs.clamp(-1, 1) + 1) / 2 if imgs.min() < 0 else imgs.clamp(0, 1)
    frames = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
    imageio.mimsave(path, frames, fps=4)

def make_conditional_interpolation_grid(model, z1, z2, label1, label2, steps=8, method="slerp", mode="same_class"):
    """
    Sinh ảnh nội suy có điều kiện (Conditional).
    mode="same_class": label cố định là label1 cho toàn bộ quá trình.
    mode="diff_class": label nội suy từ label1 sang label2 (dùng one-hot).
    """
    if method == "slerp":
        zs = slerp(z1, z2, steps)
    else:
        zs = linear_interpolate(z1, z2, steps)

    # Xử lý labels
    if mode == "same_class":
        # Tất cả ảnh dùng chung label1
        labels = torch.full((steps,), label1, dtype=torch.long, device=z1.device)
    elif mode == "diff_class":
        # Nội suy label (dùng one-hot để nội suy mượt)
        # Giả sử num_classes là 10
        num_classes = 10 
        one_hot1 = F.one_hot(torch.tensor([label1]), num_classes=num_classes).float().to(z1.device)
        one_hot2 = F.one_hot(torch.tensor([label2]), num_classes=num_classes).float().to(z1.device)
        
        alphas = torch.linspace(0, 1, steps, device=z1.device).view(-1, 1)
        mixed_one_hot = one_hot1 * (1 - alphas) + one_hot2 * alphas
        
        # Lấy argmax để quay về nhãn rời rạc (vì model thường nhận label rời rạc)
        # Hoặc nếu model hỗ trợ nhận one-hot trực tiếp thì truyền mixed_one_hot vào.
        labels = torch.argmax(mixed_one_hot, dim=1)
    else:
        raise ValueError("mode must be 'same_class' or 'diff_class'")

    with torch.no_grad():
        # Truyền thêm labels vào hàm decode
        if model_type == "ddpm":
            # zs lúc này đóng vai trò là nhiễu x_T
            try:
                imgs = model.sample_from_noise(zs, labels)
            except AttributeError:
                # Nếu model chưa có hàm sample_from_noise, gọi hàm helper
                imgs = ddpm_sampling_wrapper(model, zs, labels)
        else :
            try:
                imgs = model.decode(zs, labels)
            except TypeError:
                # Fallback nếu model không nhận label (như VAE thường)
                print("Model.decode does not accept labels. Falling back to unconditional decode.")
                imgs = model.decode(zs)
            
    imgs = (imgs.clamp(-1, 1) + 1) / 2 if imgs.min() < 0 else imgs.clamp(0, 1)
    grid = torch.cat([imgs[i] for i in range(steps)], dim=-1)
    return grid