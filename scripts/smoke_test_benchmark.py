"""Smoke test: kiểm tra cấu trúc output của mô hình và của mọi script đánh giá.

Mỗi bước dùng `assert` để kiểm tra đúng cấu trúc đã thống nhất trong
TASK_ASSIGNMENT.md (mục 2.6, 4.1, 4.2, 4.3). Sai ở bước nào thì dừng ở bước đó
và in lý do. Mọi file được ghi vào thư mục tạm, không ghi vào outputs/.

    # Mô hình giả (mặc định): chạy toàn bộ pipeline, gồm cả aggregate_results.py
    python scripts/smoke_test_benchmark.py

    # Mô hình thật: BẮT BUỘC PASS trước khi train/đánh giá thật (mục 5 của TASK_ASSIGNMENT.md)
    python scripts/smoke_test_benchmark.py --spec configs/eval/speed_<tên>.yaml

    # Chép bảng, biểu đồ, ảnh ghép ra một thư mục để xem
    python scripts/smoke_test_benchmark.py --keep smoke_out

Các bước:
     1. Spec            khóa bắt buộc của file YAML.
     2. Interface       load_model, tên module, sample() đúng shape/dtype/[0, 1]/device/seed.
     3. Mô hình sai     chạy kiểm tra bước 2 và Trainer trên các biến thể lỗi của
                        models/dummy.py (DEFECTS, TRAINER_DEFECTS); mỗi lỗi phải bị bắt.
     4. Tốc độ          evaluation/speed.py và cấu trúc file kind = "speed".
     5. MACs/FLOPs      evaluation/flops.py và cấu trúc file kind = "compute".
     6. Hiệu chuẩn      build_trainer, evaluation/calibrate.py, file kind = "calibration",
                        normalize_config đưa đường dẫn config của mọi máy về cùng một chuỗi.
     7. Ảnh mẫu         scripts/render_samples.py ghi đúng đường dẫn.
     8. FID/diversity   ảnh PNG uint8 theo mục 2.6, configs/eval/fid.yaml (nếu có),
                        file kind = "quality" và "real_baseline".
     9. Metadata run    file kind = "run" theo mục 4.3.
    10. Tổng hợp        aggregate_results.py ghi bảng đúng (chỉ với mô hình giả).
    11. Bắt lỗi         aggregate_results.py từ chối dữ liệu không công bằng và số điền tay
                        trong metadata run; cảnh báo file speed chưa đo xong; không đọc
                        kind cũ; ghép được config ghi đường dẫn tuyệt đối (chỉ với mô hình giả).

Với mô hình thật, bước 8 và 9 kiểm tra file trong `quality` / `run_metadata`
của target nếu có; chưa có thì báo BỎ QUA. Không chạy bằng `python -O`
(cờ này tắt assert).

Số liệu chất lượng của mô hình giả tính trên feature pixel, không phải
Inception: chỉ dùng để kiểm tra cấu trúc, không phải kết quả thật.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # gốc repo, xem configs/common.py
# Console Windows (cp1252) không in được tiếng Việt khi chạy qua pipe/CI.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from evaluation.calibrate import calibrate  # noqa: E402
from configs.common import ROOT, normalize_config  # noqa: E402
from evaluation.speed import run_benchmark  # noqa: E402
from models.dummy import (  # noqa: E402
    DATASET_SHAPES,
    DEFECTS,
    TRAINER_DEFECTS,
    DummyModel,
    build_trainer,
    dummy_quality_record,
    dummy_real_baseline,
    save_dummy_checkpoint,
)
from models.interface import (  # noqa: E402
    SAMPLING_ROLE,
    TRAIN_ROLES,
    GenerativeModel,
    Trainer,
    ContractError,
    check_contract,
    check_roles,
    load_model,
    load_trainer,
)
from scripts.render_samples import figure_dir, save_grid  # noqa: E402

DEFAULT_SPEC = ROOT / "configs" / "eval" / "speed_example.yaml"
FID_CONFIG = ROOT / "configs" / "eval" / "fid.yaml"
DUMMY = "models.dummy"
REF = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"

MODELS = tuple(TRAIN_ROLES)
DATASETS = tuple(DATASET_SHAPES)
STAGES = ("overfit", "tuning", "full")

# Giá trị mục 2.6 mà configs/eval/fid.yaml phải ghi. Tên khóa là đề xuất cho
# evaluation/fid.py; nếu đổi tên khóa trong fid.yaml thì sửa dict này cùng lúc.
FID_CONFIG_EXPECTED = {
    "library": "clean-fid",
    "mode": "clean",
    "feature_extractor": "inception_v3",
    "feature_dim": 2048,
    "feature_batch_size": 100,
    "num_gen_final": 50000,
    "num_gen_tuning": 5000,
    "fid_test_splits": 5,
    "fid_test_num_gen": 10000,
    "kid_subset_size": 1000,
    "kid_num_subsets": 100,
    "prdc_k": 5,
    "prdc_num": 10000,
    "lpips_net": "alex",
    "lpips_pairs": 2000,
    "sampler_compare_num_gen": 10000,
    "image_format": "png",
}
# Khối `eval` trong file quality (mục 4.2); phải giống nhau giữa các mô hình cùng dataset.
EVAL_KEYS = ("eval_config", "eval_config_hash", "library", "commit", "ref_train", "ref_test", "num_gen")
# Dòng phụ so sánh sampler của DDPM (mục 2.6): file quality có "comparison": "sampler".
COMPARISON_SAMPLER = "sampler"
SAMPLER_COMPARE_NUM_GEN = FID_CONFIG_EXPECTED["sampler_compare_num_gen"]
SAMPLER_METRICS = ("fid_train_10k", "kid_test_mean", "kid_test_std")
QUALITY_METRICS = (
    "fid_train_50k", "fid_test_10k_mean", "fid_test_10k_std", "kid_test_mean", "kid_test_std",
    "precision", "recall", "density", "coverage", "lpips_diversity",
)
SPEED_KEYS = (
    "kind", "model", "dataset", "run_name", "label", "checkpoint", "checkpoint_mb", "load_time_s",
    "model_shared", "sample_kwargs", "forward_passes_per_image", "image_shape", "params_m", "precision",
    "warmup", "repeats_requested", "complete", "seed", "env", "commit", "results",
)
SPEED_RESULT_KEYS = (
    "batch_size", "oom", "n_repeats", "time_per_batch_s_mean", "time_per_batch_s_std",
    "time_per_image_ms", "throughput_img_s", "peak_memory_mb",
)
COMPUTE_KEYS = (
    "kind", "model", "dataset", "run_name", "label", "checkpoint", "sample_kwargs", "tool", "tool_version",
    "params_m", "params_train_m", "gmacs_by_module", "gmacs_per_forward", "forward_passes_per_image",
    "gmacs_per_image", "gflops_per_image", "train_flops_per_image", "train_formula",
)
CALIBRATION_KEYS = (
    "kind", "model", "dataset", "config", "device", "num_gpus", "batch_size", "precision",
    "warmup_steps", "measured_steps", "elapsed_s", "step_time_ms", "throughput_img_s", "env",
)
ELEMENTWISE_OPS = {
    "aten::add", "aten::add_", "aten::sub", "aten::mul", "aten::mul_", "aten::div", "aten::neg", "aten::exp",
    "aten::sqrt", "aten::rsqrt", "aten::pow", "aten::relu", "aten::relu_", "aten::leaky_relu", "aten::silu",
    "aten::gelu", "aten::sigmoid", "aten::tanh", "aten::softmax", "aten::sin", "aten::cos", "aten::cat",
}
RUN_KEYS = (
    "kind", "model", "dataset", "run_name", "stage", "config", "seed", "commit",
    "train_steps", "batch_size", "precision", "hardware",
)
# Script tổng hợp tự tính các trường này; metadata run có chúng là số điền tay → lỗi (mục 4.3).
SCRIPT_COMPUTED_RUN_FIELDS = ("params_m", "gmacs_per_forward", "train_flops", "gpu_hours_t4")
# (đường dẫn config ghi trên một máy nào đó, chuỗi normalize_config phải trả về)
CONFIG_PATH_CASES = (
    ("configs/ddpm/cifar10/base.yaml", "configs/ddpm/cifar10/base.yaml"),
    ("./configs/ddpm/cifar10/base.yaml", "configs/ddpm/cifar10/base.yaml"),
    ("/kaggle/working/Model-Image-Generator-/configs/ddpm/cifar10/base.yaml", "configs/ddpm/cifar10/base.yaml"),
    ("/content/drive/MyDrive/repo/configs/vae/celeba/base.yaml", "configs/vae/celeba/base.yaml"),
    ("C:\\Users\\a\\repo\\configs\\dcgan\\fashion_mnist\\base.yaml", "configs/dcgan/fashion_mnist/base.yaml"),
    ("/a/configs/cu/configs/ddpm/cifar10/base.yaml", "configs/ddpm/cifar10/base.yaml"),
    ("/a/myconfigs/base.yaml", "/a/myconfigs/base.yaml"),
    ("vae:16:cifar10", "vae:16:cifar10"),
    (None, None),
)


# --------------------------------------------------------------------------
# Tiện ích
# --------------------------------------------------------------------------

def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env)


def must(cmd: list[str]) -> str:
    result = run(cmd)
    assert result.returncode == 0, f"Lệnh lỗi: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}"
    return result.stdout


def has_keys(rec: dict, keys, where: str) -> None:
    missing = [k for k in keys if k not in rec]
    assert not missing, f"{where}: thiếu khóa {missing}"


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def result_name(dataset: str, t: dict) -> str:
    """Tên file chung của speed/compute: <dataset>__<model>__<run_name>[__<label>].json"""
    return "__".join([dataset, t["model"], t["run_name"]] + ([t["label"]] if t["label"] else [])) + ".json"


def who(t: dict) -> str:
    return f"{t['model']}/{t['run_name']}" + (f"/{t['label']}" if t["label"] else "")


def importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# --------------------------------------------------------------------------
# Validator cho từng loại file (dùng được cho cả file của mô hình thật)
# --------------------------------------------------------------------------

def validate_spec(spec: dict) -> None:
    assert isinstance(spec, dict), "Spec phải là một mapping YAML."
    assert spec.get("dataset") in DATASETS, f"dataset phải là một trong {DATASETS}, nhận {spec.get('dataset')!r}"
    assert spec.get("precision", "fp32") in ("fp32", "fp16"), "precision phải là fp32 hoặc fp16"
    bs = spec.get("batch_sizes", [1, 64])
    assert bs and all(isinstance(b, int) and b > 0 for b in bs), f"batch_sizes phải là list số nguyên dương: {bs}"
    assert 1 in bs, "batch_sizes phải có 1 (dùng cho cột latency của bảng chi phí)"
    targets = spec.get("targets")
    assert isinstance(targets, list) and targets, "Spec phải có danh sách targets"
    seen = set()
    for i, t in enumerate(targets):
        where = f"targets[{i}]"
        has_keys(t, ("model", "checkpoint", "run_name"), where)
        assert t["model"] in MODELS, f"{where}: model phải là một trong {MODELS}"
        key = (t["model"], t["run_name"], t.get("label") or "")
        assert key not in seen, f"{where}: trùng (model, run_name, label) = {key}"
        seen.add(key)
        kw = t.get("sample_kwargs") or {}
        assert isinstance(kw, dict), f"{where}: sample_kwargs phải là mapping"
        if t["model"] == "ddpm":
            assert kw.get("sampler") in ("ddpm", "ddim"), f"{where}: DDPM phải ghi sample_kwargs.sampler (ddpm|ddim)"
            assert isinstance(kw.get("steps"), int) and kw["steps"] > 0, f"{where}: DDPM phải ghi sample_kwargs.steps"
            assert t.get("label"), f"{where}: DDPM phải có label phân biệt biến thể sampler, ví dụ ddim200"
        else:
            assert not t.get("label"), f"{where}: VAE/DCGAN để trống label"
        if t.get("comparison") is not None:
            assert t["comparison"] == COMPARISON_SAMPLER, f"{where}: comparison chỉ nhận {COMPARISON_SAMPLER!r}"
            assert t["model"] == "ddpm", f"{where}: comparison = {COMPARISON_SAMPLER!r} chỉ dùng cho DDPM"
        if t["checkpoint"] == "auto":
            assert t.get("loader_module") == DUMMY, f"{where}: checkpoint 'auto' chỉ dùng cho {DUMMY}"
        else:
            assert Path(t["checkpoint"]).exists(), f"{where}: không thấy checkpoint {t['checkpoint']}"
    main_ddpm = {t["run_name"] for t in targets if t["model"] == "ddpm" and t.get("comparison") is None}
    for t in targets:
        if t.get("comparison") == COMPARISON_SAMPLER:
            assert t["run_name"] in main_ddpm, (
                f"Dòng so sánh sampler {t['run_name']}/{t.get('label')} cần một dòng DDPM chính cùng run_name"
            )


def check_model(model: GenerativeModel, t: dict, dataset: str, device: torch.device) -> None:
    """Interface mục 4.1: kiểu, tên module, sample(), example_inputs()."""
    where = who(t)
    kw = t["sample_kwargs"]
    assert isinstance(model, GenerativeModel), f"{where}: load_model phải trả về GenerativeModel"
    assert model.name == t["model"], f"{where}: model.name = {model.name!r}, spec ghi {t['model']!r}"
    expected_shape = DATASET_SHAPES[dataset]
    assert tuple(model.image_shape) == expected_shape, (
        f"{where}: image_shape = {tuple(model.image_shape)}, dataset {dataset} cần {expected_shape} (mục 2.1)"
    )
    check_roles(model)
    assert model.device.type == device.type, f"{where}: mô hình nằm trên {model.device}, cần {device}"
    check_contract(model, **kw)

    # Kiểm tra thêm ngoài check_contract: batch 1, batch lẻ, dtype và device chính xác.
    with torch.inference_mode():
        for n in (1, 3):
            x = model.sample(n, seed=123, **kw)
            assert isinstance(x, torch.Tensor), f"{where}: sample() phải trả về torch.Tensor"
            assert tuple(x.shape) == (n, *expected_shape), f"{where}: sample({n}) có shape {tuple(x.shape)}"
            assert x.dtype in (torch.float32, torch.float16, torch.bfloat16), f"{where}: dtype {x.dtype}"
            assert x.device == model.device, f"{where}: ảnh trên {x.device}, mô hình trên {model.device}"
            assert float(x.min()) >= 0 and float(x.max()) <= 1, f"{where}: giá trị ngoài [0, 1]"

    passes = model.forward_passes_per_image(**kw)
    assert isinstance(passes, int) and passes >= 1, f"{where}: forward_passes_per_image phải là int ≥ 1"
    if t["model"] == "ddpm":
        assert passes in (kw["steps"], 2 * kw["steps"]), (
            f"{where}: forward_passes_per_image = {passes}, cần steps ({kw['steps']}) hoặc 2 × steps nếu CFG"
        )
    else:
        assert passes == 1, f"{where}: VAE/DCGAN chỉ có 1 forward mỗi ảnh"

    assert model.num_parameters() > 0, f"{where}: num_parameters() = 0"
    assert model.num_training_parameters() >= model.num_parameters(), f"{where}: tham số train < tham số sinh"
    if t["model"] != "ddpm":
        other = "encoder" if t["model"] == "vae" else "discriminator"
        assert model.num_training_parameters() > model.num_parameters(), f"{where}: tham số train phải gồm cả {other}"

    inputs = model.example_inputs()
    modules = model.training_modules()
    assert set(inputs) == set(modules), f"{where}: example_inputs() có {sorted(inputs)}, cần {sorted(modules)}"
    for name, args in inputs.items():
        assert isinstance(args, tuple), f"{where}: example_inputs()[{name!r}] phải là tuple"
        for a in args:
            if isinstance(a, torch.Tensor):
                assert a.shape[0] == 1, f"{where}: example_inputs()[{name!r}] phải có batch = 1"
                assert a.device == model.device, f"{where}: example_inputs()[{name!r}] nằm trên {a.device}"
        with torch.inference_mode():
            modules[name](*args)  # Phải forward được.


def check_trainer(trainer: Trainer, t: dict, dataset: str, device: torch.device) -> None:
    """Trainer mục 4.1: thuộc tính đúng kiểu và train_step chạy được trên batch đúng shape."""
    where = f"trainer {t['model']} ({t['trainer_config']})"
    assert isinstance(trainer, Trainer), f"{where}: build_trainer phải trả về Trainer"
    assert isinstance(trainer.batch_size, int) and trainer.batch_size > 0, f"{where}: batch_size = {trainer.batch_size!r}"
    expected = DATASET_SHAPES[dataset]
    assert tuple(trainer.image_shape) == expected, (
        f"{where}: image_shape = {tuple(trainer.image_shape)}, dataset {dataset} cần {expected}"
    )
    assert trainer.precision in ("fp32", "fp16"), f"{where}: precision = {trainer.precision!r}, cần fp32 hoặc fp16"
    assert isinstance(trainer.num_gpus, int) and trainer.num_gpus >= 1, f"{where}: num_gpus = {trainer.num_gpus!r}"
    trainer.train_step(torch.rand(trainer.batch_size, *expected, device=device) * 2 - 1)


def validate_speed(rec: dict, t: dict, dataset: str, spec: dict) -> None:
    where = f"speed {who(t)}"
    has_keys(rec, SPEED_KEYS, where)
    assert rec["kind"] == "speed", where
    assert (rec["model"], rec["dataset"], rec["run_name"], rec["label"]) == (t["model"], dataset, t["run_name"], t["label"]), (
        f"{where}: khóa ghép sai {(rec['model'], rec['dataset'], rec['run_name'], rec['label'])}"
    )
    assert rec["sample_kwargs"] == t["sample_kwargs"], f"{where}: sample_kwargs khác spec"
    assert rec["image_shape"] == list(DATASET_SHAPES[dataset]), f"{where}: image_shape {rec['image_shape']}"
    assert is_num(rec["checkpoint_mb"]) and rec["checkpoint_mb"] > 0, f"{where}: checkpoint_mb"
    assert is_num(rec["params_m"]) and rec["params_m"] > 0, f"{where}: params_m"
    assert rec["precision"] == spec.get("precision", "fp32"), f"{where}: precision"
    assert rec["warmup"] == t["warmup"], f"{where}: warmup {rec['warmup']}, target ghi {t['warmup']}"
    assert rec["repeats_requested"] == t["repeats"], f"{where}: repeats_requested {rec['repeats_requested']}"
    assert rec["complete"] is True, f"{where}: đo xong mà complete vẫn là {rec['complete']!r}"
    has_keys(rec["env"], ("device", "gpu", "torch"), f"{where}.env")
    sizes = [r["batch_size"] for r in rec["results"]]
    assert sizes == spec["batch_sizes"], f"{where}: batch_sizes {sizes}, cần {spec['batch_sizes']}"
    for r in rec["results"]:
        if r["oom"]:
            continue
        has_keys(r, SPEED_RESULT_KEYS, f"{where} batch {r['batch_size']}")
        assert r["n_repeats"] == t["repeats"], f"{where}: n_repeats {r['n_repeats']}, cần {t['repeats']}"
        assert r["time_per_batch_s_mean"] > 0 and r["time_per_batch_s_std"] >= 0, where
        assert abs(r["throughput_img_s"] - r["batch_size"] / r["time_per_batch_s_mean"]) < 1e-6 * r["throughput_img_s"], where
        assert abs(r["time_per_image_ms"] - r["time_per_batch_s_mean"] / r["batch_size"] * 1000) < 1e-9 + 1e-6 * r["time_per_image_ms"], where
        if rec["env"]["device"] == "cuda":
            assert is_num(r["peak_memory_mb"]) and r["peak_memory_mb"] > 0, f"{where}: thiếu peak_memory_mb trên GPU"


def validate_compute(rec: dict, t: dict, dataset: str) -> None:
    where = f"compute {who(t)}"
    has_keys(rec, COMPUTE_KEYS, where)
    assert rec["kind"] == "compute", where
    assert (rec["model"], rec["dataset"], rec["run_name"], rec["label"]) == (t["model"], dataset, t["run_name"], t["label"]), where
    roles = TRAIN_ROLES[t["model"]]
    assert set(rec["gmacs_by_module"]) == set(roles), f"{where}: gmacs_by_module có {sorted(rec['gmacs_by_module'])}"
    assert all(is_num(v) and v > 0 for v in rec["gmacs_by_module"].values()), f"{where}: MACs phải > 0"
    role = SAMPLING_ROLE[t["model"]]
    assert abs(rec["gmacs_per_forward"] - rec["gmacs_by_module"][role]) < 1e-12, f"{where}: gmacs_per_forward ≠ MACs của {role}"
    passes = rec["forward_passes_per_image"]
    if t["model"] == "ddpm":
        assert passes in (t["sample_kwargs"]["steps"], 2 * t["sample_kwargs"]["steps"]), f"{where}: forward_passes_per_image"
    assert abs(rec["gmacs_per_image"] - passes * rec["gmacs_per_forward"]) < 1e-9 * max(1, rec["gmacs_per_image"]), where
    assert abs(rec["gflops_per_image"] - 2 * rec["gmacs_per_image"]) < 1e-9 * max(1, rec["gflops_per_image"]), f"{where}: FLOPs ≠ 2 × MACs"
    m = {k: v * 1e9 for k, v in rec["gmacs_by_module"].items()}
    if t["model"] == "vae":
        expected = 2 * 3 * (m["encoder"] + m["decoder"])
    elif t["model"] == "dcgan":
        k = rec.get("d_steps") or 1
        expected = 2 * (k * (m["generator"] + 6 * m["discriminator"]) + 3 * m["generator"] + 3 * m["discriminator"])
    else:
        expected = 2 * 3 * m["unet"]
    assert abs(rec["train_flops_per_image"] - expected) <= 1e-6 * expected, f"{where}: train_flops_per_image sai công thức mục 2.5"
    assert rec["params_train_m"] >= rec["params_m"] > 0, f"{where}: params"
    # Op element-wise không có phép nhân-cộng nên fvcore bỏ qua là đúng; chỉ cảnh báo op còn lại.
    unsupported = {k: [op for op in v if op not in ELEMENTWISE_OPS] for k, v in (rec.get("unsupported_ops") or {}).items()}
    unsupported = {k: v for k, v in unsupported.items() if v}
    if unsupported:
        print(f"    Cảnh báo {where}: op không được đếm {unsupported} — MACs có thể thiếu.")


def validate_calibration(rec: dict, t: dict, dataset: str) -> None:
    where = f"calibration {who(t)}"
    has_keys(rec, CALIBRATION_KEYS, where)
    assert rec["kind"] == "calibration" and rec["model"] == t["model"] and rec["dataset"] == dataset, where
    assert isinstance(rec["device"], str) and rec["device"], f"{where}: device"
    assert rec["batch_size"] > 0 and rec["num_gpus"] >= 1 and rec["precision"] in ("fp32", "fp16"), where
    assert rec["elapsed_s"] > 0 and rec["throughput_img_s"] > 0, where
    expected = rec["measured_steps"] * rec["batch_size"] / rec["elapsed_s"]
    assert abs(rec["throughput_img_s"] - expected) < 1e-6 * expected, f"{where}: throughput ≠ steps × batch / elapsed"


def validate_png_folder(folder: Path, n: int, shape: tuple[int, int, int], originals: torch.Tensor) -> torch.Tensor:
    """Mục 2.6: PNG uint8 = round(clamp(x) × 255), đúng độ phân giải dataset. Trả về ảnh đọc lại trong [0, 1]."""
    files = sorted(folder.glob("*.png"))
    assert len(files) == n, f"{folder}: có {len(files)} file PNG, cần {n}"
    assert not list(folder.glob("*.jp*g")), f"{folder}: không được lưu JPEG (mục 2.6)"
    c, h, w = shape
    loaded = []
    for f in files:
        img = Image.open(f)
        assert img.format == "PNG", f"{f}: không phải PNG"
        assert img.size == (w, h), f"{f}: kích thước {img.size}, cần {(w, h)}"
        assert img.mode == ("L" if c == 1 else "RGB"), f"{f}: mode {img.mode}, cần {'L' if c == 1 else 'RGB'}"
        arr = np.asarray(img)
        assert arr.dtype == np.uint8, f"{f}: dtype {arr.dtype}, cần uint8"
        loaded.append(torch.from_numpy(arr.copy()).reshape(h, w, c).permute(2, 0, 1))
    stacked = torch.stack(loaded)
    expected = (originals.detach().float().clamp(0, 1).cpu() * 255).round().to(torch.uint8)
    assert torch.equal(stacked, expected), f"{folder}: PNG khác round(clamp(x) × 255)"
    return stacked.float() / 255


def save_png_folder(images: torch.Tensor, folder: Path) -> None:
    """Cách lưu ảnh sinh theo mục 2.6 (evaluation/fid.py nên dùng cùng cách)."""
    folder.mkdir(parents=True, exist_ok=True)
    arr = (images.detach().float().clamp(0, 1).cpu() * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    for i, a in enumerate(arr):
        Image.fromarray(a[:, :, 0] if a.shape[2] == 1 else a).save(folder / f"{i:06d}.png")


def validate_fid_config(cfg: dict) -> None:
    where = "configs/eval/fid.yaml"
    has_keys(cfg, FID_CONFIG_EXPECTED, where)
    for key, expected in FID_CONFIG_EXPECTED.items():
        assert cfg[key] == expected, f"{where}: {key} = {cfg[key]!r}, mục 2.6 cần {expected!r}"


def validate_quality(rec: dict, t: dict, dataset: str, num_gen: int | None = None) -> None:
    where = f"quality {who(t)}"
    has_keys(rec, ("kind", "model", "dataset", "run_name", "label", "eval", "metrics"), where)
    assert rec["kind"] == "quality", where
    assert (rec["model"], rec["dataset"], rec["run_name"], rec["label"] or "") == (t["model"], dataset, t["run_name"], t["label"]), (
        f"{where}: khóa ghép phải trùng với file speed/compute"
    )
    if t["model"] == "ddpm":
        assert rec.get("sampler") == t["sample_kwargs"]["sampler"], f"{where}: sampler khác spec"
        assert rec.get("sampling_steps") == t["sample_kwargs"]["steps"], f"{where}: sampling_steps khác spec"
    ev = rec["eval"]
    has_keys(ev, EVAL_KEYS, f"{where}.eval")
    assert str(ev["eval_config"]).replace("\\", "/").endswith("configs/eval/fid.yaml"), f"{where}: eval_config"
    assert str(ev["library"]).startswith("clean-fid"), f"{where}: library phải là clean-fid (mục 2.6)"
    shape = DATASET_SHAPES[dataset]
    for ref in ("ref_train", "ref_test"):
        expected = f"{dataset}_{ref[4:]}_{shape[1]}"
        assert ev[ref] == expected, f"{where}: eval.{ref} = {ev[ref]!r}, quy ước tên cache là {expected!r}"
    if num_gen is not None:
        assert ev["num_gen"] == num_gen, f"{where}: eval.num_gen = {ev['num_gen']}, cần {num_gen}"
    m = rec["metrics"]
    if rec.get("stage") == "tuning" or set(m) == {"kid_val_mean"}:
        assert is_num(m.get("kid_val_mean")), f"{where}: run tuning cần metrics.kid_val_mean"
        return
    comparison = t.get("comparison")
    assert rec.get("comparison") == comparison, (
        f"{where}: comparison = {rec.get('comparison')!r}, spec ghi {comparison!r}"
        " (dòng phụ so sánh sampler phải có comparison = 'sampler', dòng chính thì không)"
    )
    if comparison == COMPARISON_SAMPLER:
        assert ev["num_gen"] == SAMPLER_COMPARE_NUM_GEN, (
            f"{where}: dòng so sánh sampler cần eval.num_gen = {SAMPLER_COMPARE_NUM_GEN}, nhận {ev['num_gen']}"
        )
        has_keys(m, SAMPLER_METRICS, f"{where}.metrics")
        for k in SAMPLER_METRICS:
            assert is_num(m[k]), f"{where}: metrics.{k} = {m[k]!r} không phải số hữu hạn"
        assert m["fid_train_10k"] >= 0 and m["kid_test_std"] >= 0, f"{where}: metric âm"
        return
    if t["model"] == "ddpm":
        assert is_num(m.get("fid_train_10k")), (
            f"{where}: dòng DDPM chính cần metrics.fid_train_10k (FID trên 10k ảnh đầu) cho bảng so sánh sampler"
        )
    has_keys(m, QUALITY_METRICS, f"{where}.metrics")
    for k in QUALITY_METRICS:
        assert is_num(m[k]), f"{where}: metrics.{k} = {m[k]!r} không phải số hữu hạn"
    for k in ("fid_train_50k", "fid_test_10k_mean", "fid_test_10k_std", "kid_test_std", "density"):
        assert m[k] >= 0, f"{where}: metrics.{k} < 0"
    for k in ("precision", "recall", "coverage", "lpips_diversity"):
        assert 0 <= m[k] <= 1, f"{where}: metrics.{k} = {m[k]} ngoài [0, 1]"
    # KID không chệch có thể hơi âm khi hai phân phối rất gần nhau.
    assert m["kid_test_mean"] > -1e-2, f"{where}: kid_test_mean âm bất thường"


def validate_real_baseline(rec: dict, dataset: str) -> None:
    where = "real_baseline"
    has_keys(rec, ("kind", "dataset", "metrics"), where)
    assert rec["kind"] == "real_baseline" and rec["dataset"] == dataset, where
    has_keys(rec["metrics"], ("fid_floor", "lpips_real"), f"{where}.metrics")
    assert is_num(rec["metrics"]["fid_floor"]) and rec["metrics"]["fid_floor"] >= 0, f"{where}: fid_floor"
    assert 0 <= rec["metrics"]["lpips_real"] <= 1, f"{where}: lpips_real"


def validate_run(rec: dict, t: dict, dataset: str) -> None:
    where = f"run {t['model']}/{t['run_name']}"
    has_keys(rec, RUN_KEYS, where)
    assert rec["kind"] == "run" and rec["model"] == t["model"] and rec["dataset"] == dataset, where
    assert rec["run_name"] == t["run_name"], f"{where}: run_name khác spec"
    assert rec["stage"] in STAGES, f"{where}: stage phải là một trong {STAGES}"
    assert isinstance(rec["seed"], int), f"{where}: seed"
    assert isinstance(rec["train_steps"], int) and rec["train_steps"] > 0, f"{where}: train_steps"
    assert isinstance(rec["batch_size"], int) and rec["batch_size"] > 0, f"{where}: batch_size"
    assert rec["precision"] in ("fp32", "fp16"), f"{where}: precision"
    assert isinstance(rec["hardware"], list) and rec["hardware"], f"{where}: hardware phải là list không rỗng"
    devices = set()
    for h in rec["hardware"]:
        has_keys(h, ("device", "train_hours"), f"{where}.hardware[]")
        assert is_num(h["train_hours"]) and h["train_hours"] >= 0, f"{where}: train_hours"
        devices.add(h["device"])
    if rec["stage"] == "full":
        curve = rec.get("curve")
        assert isinstance(curve, list) and curve, f"{where}: run full phải có curve (KID validation tại các checkpoint)"
        steps = [p["step"] for p in curve]
        assert steps == sorted(steps) and len(set(steps)) == len(steps), f"{where}: curve phải tăng dần theo step"
        for p in curve:
            has_keys(p, ("step", "kid_val_mean", "hardware_hours"), f"{where}.curve[]")
            assert set(p["hardware_hours"]) <= devices, f"{where}: curve dùng device không có trong hardware"
    if t["model"] == "ddpm" and "sampler" in rec:
        assert rec["sampler"] in ("ddpm", "ddim"), f"{where}: sampler"


# --------------------------------------------------------------------------
# Các bước
# --------------------------------------------------------------------------

class Ctx:
    def __init__(self, spec: dict, tmp: Path, device: torch.device) -> None:
        self.spec = spec
        self.dataset = spec["dataset"]
        self.tmp = tmp
        self.device = device
        self.metrics = tmp / "metrics"
        self.figures = tmp / "figures"
        self.dummy = all(t.get("loader_module") == DUMMY for t in spec["targets"])
        smoke = spec.get("smoke") or {}
        self.batch_sizes = smoke.get("batch_sizes", [1, 4])
        self.warmup = smoke.get("warmup", 1)
        self.repeats = smoke.get("repeats", 2)
        self.num_quality = smoke.get("num_quality_images", 200)
        self.targets: list[dict] = []
        self.models: dict[tuple, GenerativeModel] = {}
        self.compute: dict[str, dict] = {}
        self.calibration: dict[str, dict] = {}
        self.eval_block: dict = {}
        self.skipped: list[str] = []


def step_spec(ctx: Ctx) -> None:
    validate_spec(ctx.spec)
    for raw in ctx.spec["targets"]:
        t = {
            "model": raw["model"],
            "run_name": raw["run_name"],
            "label": raw.get("label") or "",
            "loader_module": raw.get("loader_module"),
            "sample_kwargs": dict(raw.get("sample_kwargs") or {}),
            "repeats": min(int(raw.get("repeats", ctx.repeats)), ctx.repeats),
            "warmup": min(int(raw.get("warmup", ctx.warmup)), ctx.warmup),
            "trainer_config": raw.get("trainer_config"),
            "quality": raw.get("quality"),
            "comparison": raw.get("comparison"),
            "run_metadata": raw.get("run_metadata"),
            "checkpoint": raw["checkpoint"],
        }
        if t["checkpoint"] == "auto":
            path = ctx.tmp / "checkpoints" / f"{ctx.dataset}__{t['model']}__{t['run_name']}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                save_dummy_checkpoint(str(path), t["model"], DATASET_SHAPES[ctx.dataset])
            t["checkpoint"] = str(path)
        ctx.targets.append(t)
    print(f"    {len(ctx.targets)} target, dataset {ctx.dataset}, "
          f"{'mô hình giả' if ctx.dummy else 'mô hình thật'}, device {ctx.device}")


def step_interface(ctx: Ctx) -> None:
    for t in ctx.targets:
        key = (t["model"], t["checkpoint"])
        if key not in ctx.models:
            ctx.models[key] = load_model(t["model"], t["checkpoint"], ctx.device, module=t["loader_module"])
        check_model(ctx.models[key], t, ctx.dataset, ctx.device)
        print(f"    OK {who(t)}")


def step_defects(ctx: Ctx) -> None:
    """Bộ kiểm tra phải bắt được từng lỗi cài sẵn trong models/dummy.py.

    Bước này luôn chạy (kể cả với spec mô hình thật): nếu một assert bị viết lỏng,
    lỗi tương ứng của mô hình thật cũng sẽ lọt qua bước 2.
    """
    shape = DATASET_SHAPES[ctx.dataset]
    missed = []

    def expect_caught(name: str, needle: str, fn) -> None:
        try:
            fn()
        except (AssertionError, ContractError) as err:
            if needle not in str(err):
                missed.append(f"{name}: bị bắt nhưng sai chỗ, cần thông báo chứa {needle!r}, nhận: {err}")
            else:
                print(f"    OK {name}: {str(err).splitlines()[0][:110]}")
            return
        missed.append(f"{name}: KHÔNG bị bắt")

    for name, (model_name, desc, needle) in DEFECTS.items():
        if name == "sai_device" and ctx.device.type != "cuda":
            ctx.skipped.append(f"Biến thể lỗi {name}: cần GPU")
            continue
        kw = {"sampler": "ddim", "steps": 5} if model_name == "ddpm" else {}
        t = {"model": model_name, "run_name": f"loi_{name}", "label": "ddim5" if kw else "", "sample_kwargs": kw}
        model = DummyModel(model_name, shape, defect=name).to(ctx.device)
        expect_caught(name, needle, lambda: check_model(model, t, ctx.dataset, ctx.device))

    for name, (desc, needle) in TRAINER_DEFECTS.items():
        cfg = f"vae:4:{ctx.dataset}:{name}"
        t = {"model": "vae", "trainer_config": cfg}
        expect_caught(name, needle, lambda: check_trainer(build_trainer(cfg, str(ctx.device)), t, ctx.dataset, ctx.device))

    # Mô hình đúng phải qua được đúng các kiểm tra đó (không báo lỗi nhầm).
    for model_name in MODELS:
        kw = {"sampler": "ddim", "steps": 5} if model_name == "ddpm" else {}
        t = {"model": model_name, "run_name": "dung", "label": "ddim5" if kw else "", "sample_kwargs": kw,
             "trainer_config": f"{model_name}:4:{ctx.dataset}"}
        check_model(DummyModel(model_name, shape).to(ctx.device), t, ctx.dataset, ctx.device)
        check_trainer(build_trainer(t["trainer_config"], str(ctx.device)), t, ctx.dataset, ctx.device)

    assert not missed, "Bộ kiểm tra bị lỏng:\n  " + "\n  ".join(missed)


def step_speed(ctx: Ctx) -> None:
    spec = {
        "dataset": ctx.dataset,
        "precision": ctx.spec.get("precision", "fp32"),
        "batch_sizes": ctx.batch_sizes,
        "warmup": ctx.warmup,
        "repeats": ctx.repeats,
        "seed": ctx.spec.get("seed", 0),
        "targets": [
            {"model": t["model"], "loader_module": t["loader_module"], "checkpoint": t["checkpoint"],
             "run_name": t["run_name"], "label": t["label"], "sample_kwargs": t["sample_kwargs"],
             "repeats": t["repeats"], "warmup": t["warmup"]}
            for t in ctx.targets
        ],
    }
    out = ctx.metrics / "speed"
    partial_dir = ctx.tmp / "speed_partial"

    # Sau vòng đầu, file của mọi target phải đã có trên đĩa, ghi đúng tiến độ (mô phỏng phiên bị ngắt).
    def after_round(r: int, paths: list[Path]) -> None:
        if r != 0:
            return
        assert len(paths) == len(ctx.targets), "Vòng đầu phải ghi file cho mọi target"
        for t in ctx.targets:
            rec = read_json(out / result_name(ctx.dataset, t))
            assert rec["complete"] is (t["repeats"] == 1), f"{who(t)}: sau vòng 1, complete = {rec['complete']}"
            assert all(x["n_repeats"] == 1 for x in rec["results"] if not x["oom"]), f"{who(t)}: sau vòng 1 phải có 1 lần đo"
        shutil.copytree(out, partial_dir)

    written = run_benchmark(spec, out, str(ctx.device), on_round=after_round)
    assert partial_dir.exists(), "speed.py không gọi on_round sau vòng đầu"
    assert len(written) == len(ctx.targets), f"speed.py ghi {len(written)} file, cần {len(ctx.targets)}"
    assert not list(out.glob("*.tmp")), "speed.py để sót file tạm"
    names = {p.name for p in written}
    seen = set()
    for t in ctx.targets:
        name = result_name(ctx.dataset, t)
        assert name in names, f"speed.py không ghi {name}"
        rec = read_json(out / name)
        validate_speed(rec, t, ctx.dataset, spec)
        key = (t["model"], t["checkpoint"])
        assert rec["model_shared"] is (key in seen), (
            f"{who(t)}: model_shared = {rec['model_shared']}, target cùng checkpoint phải dùng chung mô hình"
        )
        seen.add(key)


def step_compute(ctx: Ctx) -> None:
    for t in ctx.targets:
        cmd = ["evaluation/flops.py", "--model", t["model"], "--checkpoint", t["checkpoint"], "--dataset", ctx.dataset,
               "--run-name", t["run_name"], "--output-dir", str(ctx.metrics / "compute")]
        if t["loader_module"]:
            cmd += ["--loader-module", t["loader_module"]]
        if t["label"]:
            cmd += ["--label", t["label"]]
        for k in ("sampler", "steps"):
            if k in t["sample_kwargs"]:
                cmd += [f"--{k}", str(t["sample_kwargs"][k])]
        must(cmd)
        rec = read_json(ctx.metrics / "compute" / result_name(ctx.dataset, t))
        validate_compute(rec, t, ctx.dataset)
        ctx.compute[who(t)] = rec

    # fvcore và thop phải cho cùng MACs (nếu cài cả hai).
    if not importable("thop"):
        ctx.skipped.append("So sánh fvcore/thop: chưa cài thop")
        return
    seen = set()
    for t in ctx.targets:
        if (t["model"], t["checkpoint"]) in seen:
            continue
        seen.add((t["model"], t["checkpoint"]))
        out = ctx.tmp / "thop"
        cmd = ["evaluation/flops.py", "--model", t["model"], "--checkpoint", t["checkpoint"], "--dataset", ctx.dataset,
               "--run-name", t["run_name"], "--tool", "thop", "--output-dir", str(out)]
        if t["loader_module"]:
            cmd += ["--loader-module", t["loader_module"]]
        must(cmd)
        th = read_json(out / f"{ctx.dataset}__{t['model']}__{t['run_name']}.json")
        fv = ctx.compute[who(t)]
        for name, g in fv["gmacs_by_module"].items():
            diff = abs(g - th["gmacs_by_module"][name]) / max(g, 1e-12)
            if ctx.dummy:
                assert diff < 1e-9, f"{who(t)}: fvcore {g} ≠ thop {th['gmacs_by_module'][name]} ({name})"
            elif diff > 0.05:
                print(f"    Cảnh báo {who(t)}/{name}: fvcore và thop lệch {diff:.1%} — kiểm tra op không được đếm.")


def step_calibration(ctx: Ctx) -> None:
    # Run train trên Kaggle/Colab/Windows và file calibration đo ở máy khác phải ra cùng chuỗi config,
    # kể cả khi máy chạy aggregate không có thư mục đó.
    for raw, expected in CONFIG_PATH_CASES:
        got = normalize_config(raw)
        assert got == expected, f"normalize_config({raw!r}) = {got!r}, cần {expected!r}"
    print(f"    OK normalize_config: {len(CONFIG_PATH_CASES)} đường dẫn")
    seen = set()
    for t in ctx.targets:
        cfg = t["trainer_config"]
        if not cfg:
            ctx.skipped.append(f"Hiệu chuẩn {who(t)}: target chưa có trainer_config")
            continue
        if (t["model"], cfg) in seen:
            continue
        seen.add((t["model"], cfg))
        rec = calibrate(t["model"], cfg, str(ctx.device), warmup=1, steps=3, module=t["loader_module"])
        rec["dataset"] = ctx.dataset
        check_trainer(load_trainer(t["model"], cfg, ctx.device, module=t["loader_module"]), t, ctx.dataset, ctx.device)
        validate_calibration(rec, t, ctx.dataset)
        path = write_json(ctx.metrics / "calibration" / f"{ctx.dataset}__{t['model']}__{rec['device']}.json", rec)
        ctx.calibration[t["model"]] = rec
        print(f"    OK {t['model']} ({cfg}): {rec['throughput_img_s']:.1f} ảnh/s trên {rec['device']} → {path.name}")


def step_samples(ctx: Ctx) -> None:
    for t in ctx.targets:
        cmd = ["scripts/render_samples.py", "--model", t["model"], "--checkpoint", t["checkpoint"],
               "--dataset", ctx.dataset, "--run-name", t["run_name"], "--figures-dir", str(ctx.figures),
               "--num", "16", "--nrow", "4"]
        if t["loader_module"]:
            cmd += ["--loader-module", t["loader_module"]]
        if t["label"]:
            cmd += ["--label", t["label"]]
        for k in ("sampler", "steps"):
            if k in t["sample_kwargs"]:
                cmd += [f"--{k}", str(t["sample_kwargs"][k])]
        must(cmd)
        path = figure_dir(ctx.figures, ctx.dataset, t["model"], t["run_name"], t["label"]) / "samples.png"
        assert path.exists(), f"render_samples.py không ghi {path}"
        img = Image.open(path)
        assert img.format == "PNG" and img.mode == "RGB", f"{path}: cần PNG RGB"


def step_quality(ctx: Ctx) -> None:
    # 7a. Code FID/diversity đã cài chưa.
    for mod in ("evaluation.fid", "evaluation.diversity"):
        funcs = [n for n, f in inspect.getmembers(__import__(mod, fromlist=["_"]), inspect.isfunction)
                 if not n.startswith("_")]
        if not funcs:
            ctx.skipped.append(f"{mod.replace('.', '/')}.py chưa cài — dùng metric giả trong models/dummy.py để thử schema")

    # 7b. configs/eval/fid.yaml đúng cấu hình mục 2.6.
    num_gen = None
    if FID_CONFIG.exists():
        cfg = yaml.safe_load(FID_CONFIG.read_text(encoding="utf-8")) or {}
        validate_fid_config(cfg)
        num_gen = cfg["num_gen_final"]
        print("    OK configs/eval/fid.yaml")
    else:
        ctx.skipped.append("configs/eval/fid.yaml chưa có — chưa kiểm tra được cấu hình mục 2.6")
        validate_fid_config(dict(FID_CONFIG_EXPECTED))  # Tự kiểm tra validator.

    # 7c. Ảnh sinh/ảnh thật lưu PNG uint8 đúng mục 2.6 và đọc lại không sai lệch.
    shape = DATASET_SHAPES[ctx.dataset]
    n = ctx.num_quality
    gen = torch.Generator().manual_seed(0)
    real_train = torch.rand(n, *shape, generator=gen)
    real_test = torch.rand(n // 2, *shape, generator=gen)
    save_png_folder(real_train, ctx.tmp / "ref" / "train")
    save_png_folder(real_test, ctx.tmp / "ref" / "test")
    real_train = validate_png_folder(ctx.tmp / "ref" / "train", n, shape, real_train)
    real_test = validate_png_folder(ctx.tmp / "ref" / "test", n // 2, shape, real_test)

    ctx.eval_block = {
        "eval_config": "configs/eval/fid.yaml",
        "eval_config_hash": "smoke",
        "library": "clean-fid (giả)",
        "commit": "smoke",
        "ref_train": f"{ctx.dataset}_train_{shape[1]}",
        "ref_test": f"{ctx.dataset}_test_{shape[1]}",
        "num_gen": num_gen or FID_CONFIG_EXPECTED["num_gen_final"],
    }
    for t in ctx.targets:
        model = ctx.models[(t["model"], t["checkpoint"])]
        # Dòng phụ so sánh sampler: 1/5 số ảnh (đóng vai 10k so với bộ 50k), cùng seed đầu.
        secondary = t["comparison"] == COMPARISON_SAMPLER
        count = n // 5 if secondary else n
        with torch.inference_mode():
            fake = model.sample(count, seed=42, **t["sample_kwargs"])
        folder = ctx.tmp / "samples" / ctx.dataset / t["model"] / (t["run_name"] + (f"__{t['label']}" if t["label"] else ""))
        save_png_folder(fake, folder)
        fake = validate_png_folder(folder, count, shape, fake)
        eval_block = dict(ctx.eval_block, num_gen=SAMPLER_COMPARE_NUM_GEN) if secondary else ctx.eval_block

        # 7d. File quality: file thật nếu target có, không thì sinh file giả đúng schema.
        if t["quality"]:
            rec = read_json(Path(t["quality"]))
            validate_quality(rec, t, ctx.dataset, SAMPLER_COMPARE_NUM_GEN if secondary else num_gen)
            print(f"    OK {t['quality']}")
        else:
            if not ctx.dummy:
                ctx.skipped.append(f"Quality {who(t)}: target chưa có file quality")
            rec = dummy_quality_record(t["model"], ctx.dataset, t["run_name"], t["label"], t["sample_kwargs"],
                                       fake, real_train, real_test, eval_block, comparison=t["comparison"])
            validate_quality(rec, t, ctx.dataset, eval_block["num_gen"])
        write_json(ctx.metrics / "quality" / result_name(ctx.dataset, t), rec)

    baseline = dummy_real_baseline(ctx.dataset, real_train, real_test)
    validate_real_baseline(baseline, ctx.dataset)
    write_json(ctx.metrics / "real_baseline.json", baseline)

    # 7e. Validator phải bắt được file sai.
    t0 = ctx.targets[0]
    good = read_json(ctx.metrics / "quality" / result_name(ctx.dataset, t0))
    bad_cases = {
        "thiếu metric": lambda r: r["metrics"].pop("recall"),
        "precision > 1": lambda r: r["metrics"].__setitem__("precision", 1.5),
        "thiếu eval.num_gen": lambda r: r["eval"].pop("num_gen"),
        "sai tên cache": lambda r: r["eval"].__setitem__("ref_test", "cifar10_test"),
        "sai run_name": lambda r: r.__setitem__("run_name", "khac"),
    }
    for name, mutate in bad_cases.items():
        bad = copy.deepcopy(good)
        mutate(bad)
        try:
            validate_quality(bad, t0, ctx.dataset)
        except AssertionError:
            continue
        raise AssertionError(f"validate_quality không bắt được lỗi: {name}")

    sec = next((t for t in ctx.targets if t["comparison"] == COMPARISON_SAMPLER), None)
    if sec is None:
        ctx.skipped.append("Dòng so sánh sampler: spec không có target comparison = sampler")
        return
    main_ddpm = next(t for t in ctx.targets if t["model"] == "ddpm" and t["comparison"] is None)
    sec_good = read_json(ctx.metrics / "quality" / result_name(ctx.dataset, sec))
    main_good = read_json(ctx.metrics / "quality" / result_name(ctx.dataset, main_ddpm))
    bad_cases = {
        "dòng phụ thiếu comparison": (sec, sec_good, lambda r: r.pop("comparison")),
        "dòng phụ num_gen 50000": (sec, sec_good, lambda r: r["eval"].__setitem__("num_gen", 50000)),
        "dòng phụ thiếu fid_train_10k": (sec, sec_good, lambda r: r["metrics"].pop("fid_train_10k")),
        "dòng DDPM chính thiếu fid_train_10k": (main_ddpm, main_good, lambda r: r["metrics"].pop("fid_train_10k")),
        "dòng chính bị đánh dấu comparison": (main_ddpm, main_good, lambda r: r.__setitem__("comparison", "sampler")),
    }
    for name, (t, good_rec, mutate) in bad_cases.items():
        bad = copy.deepcopy(good_rec)
        mutate(bad)
        try:
            validate_quality(bad, t, ctx.dataset)
        except AssertionError:
            continue
        raise AssertionError(f"validate_quality không bắt được lỗi: {name}")


def step_runs(ctx: Ctx) -> None:
    runs = {}
    for t in ctx.targets:
        runs.setdefault((t["model"], t["run_name"]), t)
    for (model, run_name), t in runs.items():
        if t["run_metadata"]:
            validate_run(read_json(Path(t["run_metadata"])), t, ctx.dataset)
            print(f"    OK {t['run_metadata']}")
            continue
        if not ctx.dummy:
            ctx.skipped.append(f"Metadata run {model}/{run_name}: target chưa có run_metadata")
            continue
        cal = ctx.calibration[model]
        ref = cal["device"]
        hardware = [{"device": ref, "train_hours": 1.0}]
        hw = lambda f: {ref: f}  # noqa: E731
        if model == "ddpm":
            hardware.append({"device": "Tesla P100-PCIE-16GB", "train_hours": 2.0})
            hw = lambda f: {ref: f, "Tesla P100-PCIE-16GB": 2 * f}  # noqa: E731
        base_kid = {"vae": 0.09, "dcgan": 0.04, "ddpm": 0.02}[model]
        rec = {
            "kind": "run", "model": model, "dataset": ctx.dataset, "run_name": run_name, "stage": "full",
            "config": cal["config"], "seed": 42, "commit": "smoke",
            "train_steps": 1000, "batch_size": cal["batch_size"], "precision": "fp32", "hardware": hardware,
            "curve": [
                {"step": s, "kid_val_mean": base_kid * (1 + 4 / (i + 1)), "hardware_hours": hw(s / 1000)}
                for i, s in enumerate((250, 500, 750, 1000))
            ],
            "eval_config": "configs/eval/fid.yaml",
        }
        validate_run(rec, t, ctx.dataset)
        write_json(ctx.metrics / "runs" / f"{run_name}.json", rec)

    if ctx.dummy:
        # Calibration giả cho P100 nhanh hơn máy tham chiếu 1,3 lần (DDPM train trên cả hai).
        ddpm = ctx.calibration["ddpm"]
        p100 = dict(ddpm, device="Tesla P100-PCIE-16GB", throughput_img_s=ddpm["throughput_img_s"] * 1.3)
        write_json(ctx.metrics / "calibration" / f"{ctx.dataset}__ddpm__p100.json", p100)
        # Interpolation / nearest-neighbor giả, đúng đường dẫn mục 4.2.
        c, h, w = DATASET_SHAPES[ctx.dataset]
        for model, run_name in runs:
            folder = figure_dir(ctx.figures, ctx.dataset, model, run_name)
            save_grid(torch.rand(10, c, h, w), folder / "interpolation.png", nrow=10)
            save_grid(torch.rand(16, c, h, w), folder / "nearest_neighbor.png", nrow=4)


def aggregate(ctx: Ctx, out: Path, reference: str = REF) -> subprocess.CompletedProcess:
    return run(["scripts/aggregate_results.py", "--inputs", str(ctx.metrics), "--figures-dir", str(ctx.figures),
                "--output-dir", str(out), "--reference-device", reference])


def step_aggregate(ctx: Ctx) -> None:
    out = ctx.tmp / "summary"
    result = aggregate(ctx, out)
    assert result.returncode == 0, f"aggregate_results.py lỗi:\n{result.stderr}"
    summary = (out / "summary.md").read_text(encoding="utf-8")
    main_targets = [t for t in ctx.targets if t["comparison"] is None]
    secondary = [t for t in ctx.targets if t["comparison"] == COMPARISON_SAMPLER]
    main_ddpm = sum(1 for t in main_targets if t["model"] == "ddpm")
    # Bảng chính (chất lượng + chi phí): mỗi dòng DDPM chính 2 lần. Bảng sampler: dòng chính tham chiếu + dòng phụ.
    ref_runs = {t["run_name"] for t in secondary}
    in_sampler_table = sum(1 for t in main_targets if t["model"] == "ddpm" and t["run_name"] in ref_runs) + len(secondary)
    assert summary.count("| DDPM (") == 2 * main_ddpm + in_sampler_table, "Số dòng DDPM trong các bảng sai"
    assert "Không phát hiện vấn đề" in summary, f"Bảng có cảnh báo:\n{result.stderr}"
    for t in ctx.targets:
        if t["model"] == "ddpm":
            text = f"{t['sample_kwargs']['sampler']}, {t['sample_kwargs']['steps']} step"
            assert text in summary, f"Bảng thiếu sampler/step của {who(t)}"

    with (out / f"{ctx.dataset}_quality.csv").open(encoding="utf-8") as f:
        quality = list(csv.DictReader(f))
    assert len(quality) == len(main_targets), "quality.csv chỉ gồm dòng chính, mỗi target một dòng"
    if secondary:
        assert "### Đánh đổi sampler của DDPM" in summary, "summary.md thiếu bảng so sánh sampler"
        with (out / f"{ctx.dataset}_sampler.csv").open(encoding="utf-8") as f:
            sampler_rows = list(csv.DictReader(f))
        assert len(sampler_rows) == in_sampler_table, "sampler.csv phải gồm dòng DDPM chính và các dòng phụ"
        assert all(r["fid_train_10k"] and r["latency_ms"] for r in sampler_rows), "sampler.csv thiếu FID-10k hoặc latency"
    with (out / f"{ctx.dataset}_cost.csv").open(encoding="utf-8") as f:
        cost = {row["run_name"]: row for row in csv.DictReader(f)}
    # DDPM: 1 giờ trên máy tham chiếu + 2 giờ P100 × 1,3 = 3,6 giờ quy đổi.
    assert abs(float(cost["ddpm_smoke"]["gpu_hours_t4"]) - 3.6) < 1e-6, cost["ddpm_smoke"]
    assert abs(float(cost["vae_smoke"]["gpu_hours_t4"]) - 1.0) < 1e-6, cost["vae_smoke"]
    dcgan = next(t for t in ctx.targets if t["model"] == "dcgan")
    expected_flops = 1000 * ctx.calibration["dcgan"]["batch_size"] * ctx.compute[who(dcgan)]["train_flops_per_image"]
    assert abs(float(cost["dcgan_smoke"]["train_flops"]) - expected_flops) < 1e-6 * expected_flops, cost["dcgan_smoke"]
    for name in ("kid_vs_gpu_hours", "kid_vs_train_flops", "samples", "interpolation", "nearest_neighbor"):
        assert (out / f"{ctx.dataset}_{name}.png").exists(), f"Thiếu {ctx.dataset}_{name}.png"
        assert f"({ctx.dataset}_{name}.png)" in summary, f"summary.md không nhúng {name}"
    if KEEP:
        shutil.copytree(out, KEEP, dirs_exist_ok=True)
        print(f"    Đã chép bảng, biểu đồ và ảnh vào {KEEP}")


def step_errors(ctx: Ctx) -> None:
    def expect_error(name: str, needle: str, reference: str = REF) -> None:
        out = ctx.tmp / f"bad_{name}"
        result = aggregate(ctx, out, reference)
        assert result.returncode == 1, f"[{name}] aggregate_results.py phải báo lỗi"
        assert needle in result.stderr, f"[{name}] cần thông báo chứa {needle!r}, nhận:\n{result.stderr}"
        assert not out.exists(), f"[{name}] không được ghi bảng khi có lỗi"
        print(f"    OK bắt lỗi: {name}")

    def expect_ok(name: str) -> tuple[Path, subprocess.CompletedProcess]:
        out = ctx.tmp / f"ok_{name}"
        result = aggregate(ctx, out)
        assert result.returncode == 0, f"[{name}] aggregate_results.py phải ghi được bảng:\n{result.stderr}"
        return out, result

    def expect_warning(name: str, needle: str) -> tuple[Path, subprocess.CompletedProcess]:
        out, result = expect_ok(name)
        assert needle in result.stderr, f"[{name}] cần cảnh báo chứa {needle!r}, nhận:\n{result.stderr}"
        print(f"    OK cảnh báo: {name}")
        return out, result

    cal_dir, runs_dir, q_dir = ctx.metrics / "calibration", ctx.metrics / "runs", ctx.metrics / "quality"
    p100_path = cal_dir / f"{ctx.dataset}__ddpm__p100.json"
    p100 = read_json(p100_path)

    if "T4" not in REF:
        expect_error("speed_khong_tren_T4", "phải đo trên T4", reference="T4")

    p100_path.unlink()
    expect_error("thieu_calibration", "không quy đổi được")

    write_json(p100_path, dict(p100, dataset="fashion_mnist"))
    expect_error("calibration_khac_dataset", "không quy đổi được")

    write_json(p100_path, {k: v for k, v in p100.items() if k != "dataset"})
    expect_error("calibration_thieu_dataset", "thiếu trường dataset")

    dup_cal = cal_dir / "dup_p100.json"
    write_json(p100_path, p100)
    write_json(dup_cal, p100)
    expect_error("trung_calibration", "Trùng file calibration")
    dup_cal.unlink()

    dup_run = runs_dir / "vae_smoke_copy.json"
    write_json(dup_run, read_json(runs_dir / "vae_smoke.json"))
    expect_error("trung_metadata_run", "Trùng metadata run")
    dup_run.unlink()

    vae = next(t for t in ctx.targets if t["model"] == "vae")
    bad_path = q_dir / result_name(ctx.dataset, vae)
    good = read_json(bad_path)
    write_json(bad_path, dict(good, eval={**good["eval"], "num_gen": 10000}))
    expect_error("khac_num_gen", "eval.num_gen")
    write_json(bad_path, good)

    sec = next((t for t in ctx.targets if t["comparison"] == COMPARISON_SAMPLER), None)
    if sec is not None:
        sec_path = q_dir / result_name(ctx.dataset, sec)
        sec_good = read_json(sec_path)
        # Bỏ dấu comparison: dòng 10k ảnh rơi vào bảng chính nên phải bị từ chối vì khác num_gen.
        write_json(sec_path, {k: v for k, v in sec_good.items() if k != "comparison"})
        expect_error("dong_phu_thieu_comparison", "eval.num_gen không giống nhau")
        write_json(sec_path, dict(sec_good, eval={**sec_good["eval"], "num_gen": 5000}))
        expect_error("dong_phu_sai_num_gen", "dòng so sánh sampler phải có eval.num_gen")
        write_json(sec_path, dict(sec_good, eval={**sec_good["eval"], "commit": "khac"}))
        expect_error("dong_phu_khac_eval", "khác bảng chính")
        write_json(sec_path, sec_good)
        write_json(bad_path, dict(good, comparison="sampler"))
        expect_error("comparison_tren_VAE", "chỉ dùng cho DDPM")
        write_json(bad_path, good)

    # Số điền tay trong metadata run bị từ chối, kể cả khi đúng bằng số script tính (VAE: 1 giờ T4).
    vae_run_path = runs_dir / "vae_smoke.json"
    vae_run = read_json(vae_run_path)
    manual_values = {"params_m": 1.0, "gmacs_per_forward": 0.1, "train_flops": 1e12, "gpu_hours_t4": 1.0}
    for key in SCRIPT_COMPUTED_RUN_FIELDS:
        write_json(vae_run_path, dict(vae_run, **{key: manual_values[key]}))
        expect_error(f"run_dien_tay ({key})", "không được điền tay")
    write_json(vae_run_path, vae_run)

    # File speed ghi dở (phiên bị ngắt sau vòng 1) hoặc thiếu `complete`: vẫn ghi bảng nhưng phải cảnh báo.
    vae_speed = ctx.metrics / "speed" / result_name(ctx.dataset, vae)
    full = read_json(vae_speed)
    write_json(vae_speed, read_json(ctx.tmp / "speed_partial" / vae_speed.name))
    expect_warning("speed_ghi_do", "chưa đo xong")
    write_json(vae_speed, {k: v for k, v in full.items() if k != "complete"})
    expect_warning("speed_thieu_complete", "thiếu trường complete")
    write_json(vae_speed, full)

    # Không đọc kind cũ: real_baseline ghi kind = "reference" thì bảng không có mức tham chiếu.
    baseline_path = ctx.metrics / "real_baseline.json"
    baseline = read_json(baseline_path)
    write_json(baseline_path, dict(baseline, kind="reference"))
    out, _ = expect_ok("kind_cu_reference")
    assert "Mức tham chiếu thật-với-thật" not in (out / "summary.md").read_text(encoding="utf-8"), (
        "[kind_cu_reference] file kind = 'reference' không được đọc như real_baseline"
    )
    print("    OK bỏ qua: kind_cu_reference")
    write_json(baseline_path, baseline)

    # Thiếu file compute: cảnh báo, các cột tham số và MACs/FLOPs để trống (không lấy từ metadata run).
    vae_compute = ctx.metrics / "compute" / result_name(ctx.dataset, vae)
    compute = read_json(vae_compute)
    vae_compute.unlink()
    out, _ = expect_warning("thieu_compute", "thiếu file compute")
    with (out / f"{ctx.dataset}_cost.csv").open(encoding="utf-8") as f:
        vae_cost = next(r for r in csv.DictReader(f) if r["run_name"] == vae["run_name"])
    empty = ("params_m", "params_train_m", "gmacs_per_image", "gflops_per_image", "train_flops")
    assert all(vae_cost[k] == "" for k in empty), f"[thieu_compute] các cột {empty} phải để trống: {vae_cost}"
    write_json(vae_compute, compute)

    # Run train trên Kaggle ghi config tuyệt đối, calibration P100 đo trên Windows, calibration máy
    # tham chiếu ghi đường dẫn tương đối: vẫn phải ghép được và ra đúng 3,6 giờ quy đổi của DDPM.
    ddpm_run_path = runs_dir / "ddpm_smoke.json"
    ddpm_run = read_json(ddpm_run_path)
    ddpm_ref_path = cal_dir / f"{ctx.dataset}__ddpm__{ctx.calibration['ddpm']['device']}.json"
    ddpm_ref = read_json(ddpm_ref_path)
    config = f"configs/ddpm/{ctx.dataset}/base.yaml"
    write_json(ddpm_run_path, dict(ddpm_run, config=f"/kaggle/working/Model-Image-Generator-/{config}"))
    write_json(ddpm_ref_path, dict(ddpm_ref, config=config))
    write_json(p100_path, dict(p100, config="C:\\Users\\a\\Model-Image-Generator-\\" + config.replace("/", "\\")))
    out, _ = expect_ok("config_duong_dan_tuyet_doi")
    with (out / f"{ctx.dataset}_cost.csv").open(encoding="utf-8") as f:
        ddpm_cost = next(r for r in csv.DictReader(f) if r["run_name"] == ddpm_run["run_name"])
    assert abs(float(ddpm_cost["gpu_hours_t4"]) - 3.6) < 1e-6, f"[config_duong_dan_tuyet_doi] {ddpm_cost}"
    print("    OK ghép config: config_duong_dan_tuyet_doi")
    write_json(ddpm_run_path, ddpm_run)
    write_json(ddpm_ref_path, ddpm_ref)
    write_json(p100_path, p100)

    # Sau khi hoàn nguyên, bảng phải ghi lại được.
    result = aggregate(ctx, ctx.tmp / "summary_again")
    assert result.returncode == 0, f"Hoàn nguyên xong vẫn lỗi:\n{result.stderr}"


STEPS = [
    ("Spec", step_spec, False),
    ("Interface mô hình", step_interface, False),
    ("Bộ kiểm tra bắt được mô hình sai", step_defects, False),
    ("Tốc độ sinh (speed.py)", step_speed, False),
    ("MACs/FLOPs (flops.py)", step_compute, False),
    ("Hiệu chuẩn (calibrate.py)", step_calibration, False),
    ("Ảnh mẫu (render_samples.py)", step_samples, False),
    ("FID/diversity", step_quality, False),
    ("Metadata run", step_runs, False),
    ("Tổng hợp (aggregate_results.py)", step_aggregate, True),
    ("Bắt lỗi công bằng", step_errors, True),
]


def main(spec_path: Path, device: str | None) -> int:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with tempfile.TemporaryDirectory() as tmp_name:
        ctx = Ctx(spec, Path(tmp_name), dev)
        for i, (title, fn, dummy_only) in enumerate(STEPS, 1):
            print(f"[{i}/{len(STEPS)}] {title}")
            if dummy_only and not ctx.dummy:
                ctx.skipped.append(f"Bước {i} ({title}): chỉ chạy với mô hình giả")
                print("    BỎ QUA")
                continue
            try:
                fn(ctx)
            except AssertionError as err:
                print(f"\nSMOKE TEST FAIL ở bước {i} ({title}):\n  {err}", file=sys.stderr)
                return 1
            except Exception:  # noqa: BLE001
                print(f"\nSMOKE TEST FAIL ở bước {i} ({title}) do exception:", file=sys.stderr)
                traceback.print_exc()
                return 1
            print("    PASS")
    if ctx.skipped:
        print("\nChưa kiểm tra được:")
        for s in ctx.skipped:
            print(f"  - {s}")
    print("\nSMOKE TEST PASS")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra cấu trúc output của mô hình và các script đánh giá.")
    parser.add_argument("--spec", default=str(DEFAULT_SPEC), help="File YAML theo mẫu configs/eval/speed_example.yaml.")
    parser.add_argument("--device", help="Mặc định: cuda nếu có, không thì cpu.")
    parser.add_argument("--keep", help="Thư mục để chép kết quả tổng hợp ra xem (chỉ với mô hình giả).")
    args = parser.parse_args()
    KEEP = args.keep
    sys.exit(main(Path(args.spec), args.device))
