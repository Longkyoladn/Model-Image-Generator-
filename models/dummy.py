"""Mô hình giả: bản cài mẫu đúng interface, dùng cho scripts/smoke_test_benchmark.py.

KHÔNG dùng cho kết quả báo cáo. File này có ba vai trò:

1. Chạy thử toàn bộ pipeline khi chưa có checkpoint thật (speed.py, flops.py,
   calibrate.py, render_samples.py, aggregate_results.py).
2. Làm **mẫu tham chiếu** cho package thật (`models/vae`, `models/gan`,
   `models/ddpm`): mỗi package phải cung cấp đúng các thành phần dưới đây,
   cùng tên, cùng kiểu trả về.

    Thành phần bắt buộc của package       | Trong file này
    --------------------------------------|---------------------------------
    load_model(checkpoint, device)        | load_model
    build_trainer(config, device)         | build_trainer
    lớp kế thừa GenerativeModel           | DummyModel
    lớp kế thừa Trainer                   | DummyTrainer

3. Làm **mô hình sai có chủ đích** (`DEFECTS`): mỗi biến thể cố tình mắc một
   lỗi hay gặp khi cài mô hình thật (sai shape, quên đổi [-1, 1] về [0, 1], bỏ
   qua seed, sai tên module...). Smoke test chạy bộ kiểm tra trên từng biến thể
   và bắt buộc phải bắt được lỗi; nếu không bắt được nghĩa là assert bị lỏng,
   khi đó mô hình thật mắc lỗi tương tự cũng sẽ lọt qua.

Phần cuối file (`dummy_quality_record`, `dummy_real_baseline`) sinh file
`kind = "quality"` / `"real_baseline"` đúng schema mục 4.2 bằng các metric
tính trên feature pixel đơn giản (không phải Inception). Chỉ để kiểm tra cấu
trúc file mà `evaluation/fid.py` và `evaluation/diversity.py` phải ghi ra.
"""

from __future__ import annotations

import torch
from torch import nn

from models.interface import GenerativeModel, Trainer

LATENT_DIM = 16

# Shape ảnh (C, H, W) theo dataset, đúng bảng ở mục 2.1 của TASK_ASSIGNMENT.md.
DATASET_SHAPES = {
    "cifar10": (3, 32, 32),
    "fashion_mnist": (1, 32, 32),
    "celeba": (3, 64, 64),
}

# Biến thể lỗi: tên -> (mô hình áp dụng, mô tả lỗi, đoạn chữ phải có trong thông báo lỗi).
# Đoạn chữ thứ ba bảo đảm lỗi bị bắt đúng chỗ, không phải bị bắt nhầm bởi assert khác.
DEFECTS = {
    "sai_thu_tu_chieu": ("vae", "sample() trả về (n, H, W, C) thay vì (n, C, H, W)", "shape"),
    "sai_mien_gia_tri": ("dcgan", "sample() trả về [-1, 1] (quên đổi output tanh về [0, 1])", "cần [0, 1]"),
    "sai_dtype": ("vae", "sample() trả về uint8 thay vì float", "dtype"),
    "co_nan": ("vae", "sample() có NaN (ví dụ chia cho 0 khi chuẩn hóa)", "NaN"),
    "bo_qua_seed": ("dcgan", "sample() không dùng make_generator(seed): cùng seed ra ảnh khác", "cùng seed"),
    "seed_co_dinh": ("vae", "sample() luôn dùng seed 0: khác seed ra ảnh giống nhau", "khác seed"),
    "sai_so_anh": ("vae", "sample(n) luôn sinh 2 ảnh, bỏ qua n", "sample(1)"),
    "sai_image_shape": ("vae", "image_shape khai báo sai so với dataset", "image_shape"),
    "sai_ten_module_sinh": ("dcgan", "sampling_modules() đặt tên 'net' thay vì 'generator'", "sampling_modules()"),
    "thieu_module_train": ("dcgan", "training_modules() thiếu discriminator", "training_modules()"),
    "sai_forward_passes": ("ddpm", "forward_passes_per_image() trả về 1 thay vì số step", "forward_passes_per_image"),
    "example_inputs_batch_2": ("vae", "example_inputs() dùng batch 2 thay vì 1", "batch = 1"),
    "example_inputs_thieu": ("vae", "example_inputs() thiếu input của encoder", "example_inputs()"),
    "sai_device": ("vae", "sample() trả ảnh về CPU khi mô hình nằm trên GPU (chỉ thử được khi có GPU)", "nằm trên"),
}
TRAINER_DEFECTS = {
    "trainer_sai_image_shape": ("Trainer.image_shape khác dataset", "image_shape"),
    "trainer_sai_precision": ("Trainer.precision = 'half' thay vì 'fp32'/'fp16'", "precision"),
    "trainer_sai_batch": ("Trainer.batch_size = 0", "batch_size"),
}


# --------------------------------------------------------------------------
# 1. Kiến trúc: tên module phải đúng vai trò trong TRAIN_ROLES
# --------------------------------------------------------------------------

class _UNet(nn.Module):
    """U-Net giả: nhận (x, t), trả về tensor cùng shape với x."""

    def __init__(self, c: int, width: int) -> None:
        super().__init__()
        self.inp = nn.Conv2d(c, width, 3, padding=1)
        self.time = nn.Linear(1, width)
        self.out = nn.Conv2d(width, c, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.inp(x) + self.time(t.float().view(-1, 1))[:, :, None, None]
        return self.out(torch.relu(h))


def _build_modules(name: str, image_shape: tuple[int, int, int], width: int = 32) -> dict[str, nn.Module]:
    """VAE: encoder + decoder; DCGAN: generator + discriminator; DDPM: unet."""
    c, h, w = image_shape
    pixels = c * h * w
    if name == "vae":
        return {
            # Encoder trả về [mu, logvar] nối lại: 2 × LATENT_DIM.
            "encoder": nn.Sequential(nn.Flatten(), nn.Linear(pixels, width), nn.ReLU(), nn.Linear(width, 2 * LATENT_DIM)),
            "decoder": nn.Sequential(nn.Linear(LATENT_DIM, width), nn.ReLU(), nn.Linear(width, pixels)),
        }
    if name == "dcgan":
        return {
            "generator": nn.Sequential(nn.Flatten(), nn.Linear(LATENT_DIM, width), nn.ReLU(), nn.Linear(width, pixels)),
            "discriminator": nn.Sequential(nn.Conv2d(c, width, 4, 2, 1), nn.LeakyReLU(0.2), nn.Flatten(),
                                           nn.Linear(width * (h // 2) * (w // 2), 1)),
        }
    if name == "ddpm":
        return {"unet": _UNet(c, width)}
    raise ValueError(f"Tên mô hình không hợp lệ: {name!r}")


# --------------------------------------------------------------------------
# 2. Lớp mô hình: kế thừa GenerativeModel
# --------------------------------------------------------------------------

class DummyModel(GenerativeModel):
    """`defect = None` là mô hình đúng; tên trong DEFECTS là mô hình sai có chủ đích."""

    _ROLE = {"vae": "decoder", "dcgan": "generator", "ddpm": "unet"}

    def __init__(self, name: str, image_shape: tuple[int, int, int], default_steps: int = 20,
                 defect: str | None = None) -> None:
        if defect is not None and defect not in DEFECTS:
            raise ValueError(f"defect không hợp lệ: {defect!r}")
        self.name = name                         # "vae" | "dcgan" | "ddpm"
        self.image_shape = tuple(image_shape)    # (C, H, W)
        self.default_steps = default_steps
        self.defect = defect
        self._true_shape = self.image_shape      # shape mà module được dựng
        self.modules = _build_modules(name, self.image_shape)
        if defect == "sai_image_shape":
            c, h, w = self.image_shape
            self.image_shape = (1 if c == 3 else 3, h, w)

    # Bắt buộc: module tham gia train, tên theo TRAIN_ROLES.
    def training_modules(self) -> dict[str, nn.Module]:
        if self.defect == "thieu_module_train":
            return self.sampling_modules()
        return dict(self.modules)

    # Bắt buộc: chỉ module dùng khi sinh, tên theo SAMPLING_ROLE.
    def sampling_modules(self) -> dict[str, nn.Module]:
        role = self._ROLE[self.name]
        return {"net" if self.defect == "sai_ten_module_sinh" else role: self.modules[role]}

    # Chuyển cả module train (encoder/discriminator) chứ không chỉ module sinh.
    def to(self, device):
        for m in self.modules.values():
            m.to(device)
        return self

    # DDPM: số forward mỗi ảnh = số sampling step.
    def forward_passes_per_image(self, **kwargs) -> int:
        if self.defect == "sai_forward_passes":
            return 1
        return int(kwargs.get("steps", self.default_steps)) if self.name == "ddpm" else 1

    # Input batch = 1 cho mỗi module trong training_modules(), nằm trên device.
    def example_inputs(self) -> dict[str, tuple]:
        d = self.device
        b = 2 if self.defect == "example_inputs_batch_2" else 1
        img = torch.zeros(b, *self._true_shape, device=d)
        z = torch.zeros(b, LATENT_DIM, device=d)
        if self.name == "vae":
            inputs = {"encoder": (img,), "decoder": (z,)}
        elif self.name == "dcgan":
            inputs = {"generator": (z,), "discriminator": (img,)}
        else:
            inputs = {"unet": (img, torch.zeros(b, dtype=torch.long, device=d))}
        if self.defect == "example_inputs_thieu":
            inputs.pop(next(iter(inputs)))
        return inputs

    # Bắt buộc: (n, C, H, W), float, trong [0, 1], trên device; cùng seed → cùng ảnh.
    @torch.no_grad()
    def sample(self, n: int, seed: int | None = None, **kwargs) -> torch.Tensor:
        d = self.defect
        if d == "bo_qua_seed":
            gen = None                      # Lỗi: dùng RNG toàn cục.
        elif d == "seed_co_dinh":
            gen = self.make_generator(0)    # Lỗi: bỏ qua seed truyền vào.
        else:
            gen = self.make_generator(seed)
        if d == "sai_so_anh":
            n = 2                           # Lỗi: bỏ qua n.
        shape = self._true_shape
        if self.name == "ddpm":
            sampler = kwargs.get("sampler", "ddpm")
            if sampler not in ("ddpm", "ddim"):
                raise ValueError(f"sampler phải là ddpm hoặc ddim, nhận {sampler!r}")
            steps = int(kwargs.get("steps", self.default_steps))
            x = torch.randn(n, *shape, generator=gen, device=self.device)
            t = torch.zeros(n, dtype=torch.long, device=self.device)
            for _ in range(steps):
                x = x - 0.1 * self.modules["unet"](x, t)
            x = torch.sigmoid(x)
        else:
            z = torch.randn(n, LATENT_DIM, generator=gen, device=self.device)
            net = self.modules["decoder" if self.name == "vae" else "generator"]
            x = torch.sigmoid(net(z)).view(n, *shape)
        return self._apply_output_defect(x)

    def _apply_output_defect(self, x: torch.Tensor) -> torch.Tensor:
        d = self.defect
        if d == "sai_thu_tu_chieu":
            return x.permute(0, 2, 3, 1)
        if d == "sai_mien_gia_tri":
            return x * 2 - 1
        if d == "sai_dtype":
            return (x * 255).round().to(torch.uint8)
        if d == "co_nan":
            x = x.clone()
            x.view(-1)[0] = float("nan")
            return x
        if d == "sai_device":
            return x.cpu()
        return x


# --------------------------------------------------------------------------
# 3. Checkpoint và load_model(checkpoint, device)
# --------------------------------------------------------------------------

def save_dummy_checkpoint(path: str, name: str, image_shape: tuple[int, int, int] = (3, 32, 32), seed: int = 0) -> None:
    """Checkpoint gồm `config` (đủ để dựng lại kiến trúc) và `state_dict` theo từng module."""
    torch.manual_seed(seed)
    model = DummyModel(name, image_shape)
    torch.save(
        {
            "config": {"name": name, "image_shape": list(image_shape)},
            "state_dict": {k: m.state_dict() for k, m in model.modules.items()},
        },
        path,
    )


def load_model(checkpoint: str, device: str) -> DummyModel:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = DummyModel(cfg["name"], tuple(cfg["image_shape"]))
    for k, m in model.modules.items():
        m.load_state_dict(ckpt["state_dict"][k])
    return model.to(device)


# --------------------------------------------------------------------------
# 4. Trainer và build_trainer(config, device) cho evaluation/calibrate.py
# --------------------------------------------------------------------------

class DummyTrainer(Trainer):
    """Một bước train giả: forward + backward + optimizer step qua mọi module."""

    def __init__(self, name: str, image_shape, batch_size: int, device: str, defect: str | None = None) -> None:
        if defect is not None and defect not in TRAINER_DEFECTS:
            raise ValueError(f"defect không hợp lệ: {defect!r}")
        self.model = DummyModel(name, image_shape).to(device)
        self.batch_size = batch_size
        self.image_shape = tuple(image_shape)
        self.precision = "fp32"
        self.num_gpus = 1
        if defect == "trainer_sai_image_shape":
            c, h, w = self.image_shape
            self.image_shape = (1 if c == 3 else 3, h, w)
        elif defect == "trainer_sai_precision":
            self.precision = "half"
        elif defect == "trainer_sai_batch":
            self.batch_size = 0
        params = [p for m in self.model.modules.values() for p in m.parameters()]
        self.opt = torch.optim.Adam(params, lr=1e-4)

    def train_step(self, images: torch.Tensor) -> None:
        m = self.model.modules
        n = images.shape[0]
        if self.model.name == "vae":
            mu = m["encoder"](images)[:, :LATENT_DIM]
            loss = (m["decoder"](mu) - images.flatten(1)).pow(2).mean()
        elif self.model.name == "dcgan":
            # Bước D và bước G gộp thành một loss cho gọn; mô hình thật phải tách hai bước.
            z = torch.randn(n, LATENT_DIM, device=images.device)
            fake = m["generator"](z).view_as(images)
            loss = m["discriminator"](images).mean() - m["discriminator"](fake).mean()
        else:
            t = torch.zeros(n, dtype=torch.long, device=images.device)
            loss = (m["unet"](images, t) - images).pow(2).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()


def build_trainer(config: str, device: str) -> DummyTrainer:
    """`config` của mô hình giả: "<tên>:<batch_size>[:<dataset>[:<defect>]]", ví dụ "ddpm:16:cifar10".

    Mô hình thật nhận đường dẫn file YAML trong configs/<model>/<dataset>/.
    """
    parts = config.split(":")
    name, batch = parts[0], int(parts[1])
    shape = DATASET_SHAPES[parts[2]] if len(parts) > 2 else (3, 32, 32)
    defect = parts[3] if len(parts) > 3 else None
    return DummyTrainer(name, shape, batch, device, defect)


# --------------------------------------------------------------------------
# 5. File quality / real_baseline giả, đúng schema của evaluation/fid.py
# --------------------------------------------------------------------------
# Metric tính trên feature pixel (avg-pool 4×4) thay cho InceptionV3, nên chỉ
# có giá trị kiểm tra cấu trúc. Tên khóa và miền giá trị giống hệt file thật.

def _features(images: torch.Tensor) -> torch.Tensor:
    x = images.float()
    if x.shape[1] == 1:  # Grayscale lặp thành 3 kênh như mục 2.6.
        x = x.repeat(1, 3, 1, 1)
    return nn.functional.adaptive_avg_pool2d(x, 4).flatten(1).double()


def _frechet(a: torch.Tensor, b: torch.Tensor) -> float:
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = torch.cov(a.T), torch.cov(b.T)
    # Tr(sqrt(cov_a · cov_b)) qua trị riêng (cov_a · cov_b có trị riêng thực, không âm).
    eig = torch.linalg.eigvals(cov_a @ cov_b).real.clamp(min=0)
    fid = (mu_a - mu_b).pow(2).sum() + cov_a.trace() + cov_b.trace() - 2 * eig.sqrt().sum()
    return max(float(fid), 0.0)


def _kid(a: torch.Tensor, b: torch.Tensor, subset: int, num_subsets: int, gen: torch.Generator) -> tuple[float, float]:
    d = a.shape[1]
    vals = []
    for _ in range(num_subsets):
        x = a[torch.randperm(len(a), generator=gen)[:subset]]
        y = b[torch.randperm(len(b), generator=gen)[:subset]]
        k = lambda u, v: (u @ v.T / d + 1) ** 3  # noqa: E731
        m = len(x)
        kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
        mmd = (kxx.sum() - kxx.trace()) / (m * (m - 1)) + (kyy.sum() - kyy.trace()) / (m * (m - 1)) - 2 * kxy.mean()
        vals.append(float(mmd))
    t = torch.tensor(vals)
    return float(t.mean()), float(t.std()) if len(vals) > 1 else 0.0


def _prdc(real: torch.Tensor, fake: torch.Tensor, k: int = 5) -> dict[str, float]:
    """Precision/Recall/Density/Coverage (Naeem et al. 2020), cùng định nghĩa thư viện prdc."""
    rr, ff, rf = torch.cdist(real, real), torch.cdist(fake, fake), torch.cdist(real, fake)
    r_rad = rr.kthvalue(k + 1, dim=1).values
    f_rad = ff.kthvalue(k + 1, dim=1).values
    inside_real = rf < r_rad[:, None]
    return {
        "precision": float(inside_real.any(0).float().mean()),
        "recall": float((rf < f_rad[None, :]).any(1).float().mean()),
        "density": float(inside_real.float().sum(0).mean() / k),
        "coverage": float((rf.min(1).values < r_rad).float().mean()),
    }


def _pair_distance(images: torch.Tensor, pairs: int, gen: torch.Generator) -> float:
    """Thay cho LPIPS: khoảng cách L1 trung bình mỗi pixel giữa các cặp ngẫu nhiên, trong [0, 1]."""
    i = torch.randint(len(images), (pairs,), generator=gen)
    j = torch.randint(len(images), (pairs,), generator=gen)
    keep = i != j
    return float((images[i[keep]] - images[j[keep]]).abs().mean())


def dummy_quality_record(model: str, dataset: str, run_name: str, label: str, sample_kwargs: dict,
                         fake: torch.Tensor, real_train: torch.Tensor, real_test: torch.Tensor,
                         eval_block: dict, splits: int = 5, seed: int = 0) -> dict:
    """File `kind = "quality"` đúng schema mục 4.2.

    `fake`, `real_*`: tensor (N, C, H, W) trong [0, 1]. Quy mô nhỏ hơn bộ 50k thật,
    nhưng chia phần giống mục 2.6 (FID_test chia `splits` phần, KID có std).
    """
    gen = torch.Generator().manual_seed(seed)
    ff, ftr, fte = _features(fake.cpu()), _features(real_train.cpu()), _features(real_test.cpu())
    chunks = ff.chunk(splits)
    fid_test = torch.tensor([_frechet(c, fte) for c in chunks])
    subset = max(2, min(len(ff), len(fte)) // 2)
    kid_mean, kid_std = _kid(ff, fte, subset, 10, gen)
    return {
        "kind": "quality",
        "model": model,
        "dataset": dataset,
        "run_name": run_name,
        "label": label,
        "sampler": sample_kwargs.get("sampler"),
        "sampling_steps": sample_kwargs.get("steps"),
        "eval": dict(eval_block),
        "num_real_train": len(real_train),
        "num_real_test": len(real_test),
        "metrics": {
            "fid_train_50k": _frechet(ff, ftr),
            "fid_test_10k_mean": float(fid_test.mean()),
            "fid_test_10k_std": float(fid_test.std()),
            "kid_test_mean": kid_mean,
            "kid_test_std": kid_std,
            **_prdc(ftr, ff),
            "lpips_diversity": _pair_distance(fake.cpu().float(), 200, gen),
        },
    }


def dummy_real_baseline(dataset: str, real_train: torch.Tensor, real_test: torch.Tensor, seed: int = 0) -> dict:
    """File `kind = "real_baseline"`: FID sàn (train với test) và diversity của ảnh thật."""
    gen = torch.Generator().manual_seed(seed)
    return {
        "kind": "real_baseline",
        "dataset": dataset,
        "metrics": {
            "fid_floor": _frechet(_features(real_train.cpu()), _features(real_test.cpu())),
            "lpips_real": _pair_distance(real_test.cpu().float(), 200, gen),
        },
    }
