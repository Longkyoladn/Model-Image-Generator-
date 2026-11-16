"""Interface chung cho VAE, DCGAN và DDPM.

Mỗi package mô hình (`models/vae`, `models/gan`, `models/ddpm`) phải cung cấp:

    def load_model(checkpoint: str, device: str) -> GenerativeModel

Code đánh giá (speed, FID, interpolation, notebook benchmark) chỉ gọi qua
`models.interface.load_model(...)` và các method của `GenerativeModel`, nên
không cần biết chi tiết bên trong từng mô hình.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod

import torch
from torch import nn

# Tên mô hình trong metadata -> module chứa hàm load_model.
MODEL_MODULES = {
    "vae": "models.vae",
    "dcgan": "models.gan",
    "ddpm": "models.ddpm",
}


class ContractError(RuntimeError):
    """Mô hình không tuân thủ interface chung."""


class GenerativeModel(ABC):
    """Lớp cơ sở mà mọi mô hình sinh ảnh phải kế thừa.

    Thuộc tính bắt buộc:
        name: "vae" | "dcgan" | "ddpm".
        image_shape: (C, H, W) của ảnh sinh ra.
    """

    name: str = ""
    image_shape: tuple[int, int, int]

    @abstractmethod
    def sample(self, n: int, seed: int | None = None, **kwargs) -> torch.Tensor:
        """Sinh `n` ảnh.

        Trả về tensor float, shape (n, C, H, W), giá trị trong [0, 1], nằm trên
        device của mô hình. Cùng `seed` và cùng `n` phải cho cùng kết quả.
        Dùng `self.make_generator(seed)` để tạo noise.

        `kwargs` là tham số riêng của sampler, ví dụ DDPM nhận
        `sampler="ddim"`, `steps=100`. VAE và DCGAN không cần kwargs.
        """

    @abstractmethod
    def sampling_modules(self) -> dict[str, nn.Module]:
        """Các module được dùng khi sinh ảnh.

        VAE trả về decoder, DCGAN trả về generator, DDPM trả về U-Net (bản EMA
        nếu sinh bằng EMA). Dùng để đếm tham số và đo MACs/FLOPs.
        """

    def forward_passes_per_image(self, **kwargs) -> int:
        """Số lần forward để sinh một ảnh. DDPM override: trả về số step."""
        return 1

    # --- Phần dưới đã cài sẵn, không cần override ---

    def _parameters(self) -> list[nn.Parameter]:
        seen: dict[int, nn.Parameter] = {}
        for module in self.sampling_modules().values():
            for p in module.parameters():
                seen[id(p)] = p
        return list(seen.values())

    @property
    def device(self) -> torch.device:
        return self._parameters()[0].device

    def to(self, device: str | torch.device) -> "GenerativeModel":
        for module in self.sampling_modules().values():
            module.to(device)
        return self

    def eval(self) -> "GenerativeModel":
        for module in self.sampling_modules().values():
            module.eval()
        return self

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self._parameters())

    def weight_bytes(self) -> int:
        """Bộ nhớ của tham số và buffer dùng khi sinh ảnh."""
        seen: dict[int, torch.Tensor] = {}
        for module in self.sampling_modules().values():
            for t in list(module.parameters()) + list(module.buffers()):
                seen[id(t)] = t
        return sum(t.numel() * t.element_size() for t in seen.values())

    def make_generator(self, seed: int | None) -> torch.Generator | None:
        if seed is None:
            return None
        return torch.Generator(device=self.device).manual_seed(int(seed))


def load_model(
    model: str,
    checkpoint: str,
    device: str | torch.device = "cpu",
    module: str | None = None,
) -> GenerativeModel:
    """Load một mô hình theo tên.

    `module` cho phép chỉ định module khác với mặc định (dùng cho test).
    """
    module_name = module or MODEL_MODULES.get(model)
    if module_name is None:
        raise ValueError(
            f"Mô hình không hợp lệ: {model!r}. Chọn một trong {sorted(MODEL_MODULES)}."
        )
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "load_model"):
        raise ContractError(
            f"{module_name} chưa có hàm load_model(checkpoint, device)."
        )
    obj = mod.load_model(str(checkpoint), str(device))
    if not isinstance(obj, GenerativeModel):
        raise ContractError(
            f"{module_name}.load_model phải trả về GenerativeModel, nhận được {type(obj).__name__}."
        )
    return obj.eval()


@torch.inference_mode()
def check_contract(model: GenerativeModel, n: int = 2, **kwargs) -> None:
    """Kiểm tra mô hình tuân thủ interface. Raise ContractError nếu sai."""
    problems: list[str] = []
    a = model.sample(n, seed=0, **kwargs)
    expected = (n, *model.image_shape)
    if tuple(a.shape) != expected:
        problems.append(f"shape {tuple(a.shape)}, cần {expected}")
    if not a.is_floating_point():
        problems.append(f"dtype {a.dtype}, cần kiểu float")
    elif not torch.isfinite(a).all():
        problems.append("có giá trị NaN/Inf")
    elif a.min() < 0 or a.max() > 1:
        problems.append(
            f"giá trị trong [{a.min().item():.3f}, {a.max().item():.3f}], cần [0, 1]"
        )
    if a.device.type != model.device.type:
        problems.append(f"ảnh nằm trên {a.device}, mô hình nằm trên {model.device}")
    if not problems:
        b = model.sample(n, seed=0, **kwargs)
        if not torch.allclose(a.float(), b.float(), atol=1e-4):
            problems.append("cùng seed nhưng ảnh khác nhau")
        c = model.sample(n, seed=1, **kwargs)
        if torch.allclose(a.float(), c.float(), atol=1e-6):
            problems.append("khác seed nhưng ảnh giống nhau")
    if problems:
        raise ContractError(
            f"Mô hình {model.name!r} vi phạm interface: " + "; ".join(problems)
        )
