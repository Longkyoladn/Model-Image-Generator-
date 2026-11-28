"""Script huấn luyện mô hình VAE theo quy trình 3 giai đoạn (TASK_ASSIGNMENT.md mục 2.3, 2.5, 3.2, 4.3).

Hỗ trợ:
- Giai đoạn 0 (`--stage overfit`): Học thuộc tập nhỏ (64-512 ảnh, mặc định 256), xuất lưới đối chiếu ảnh thật/tái tạo.
- Giai đoạn 1 (`--stage tuning`): Huấn luyện ngân sách ngắn, tự động tính KID-5k trên validation và ghi file quality tuning.
- Giai đoạn 2 (`--stage full`): Huấn luyện đầy đủ, tính KID validation định kỳ vào trường `curve`, chọn `best.pt` theo KID validation.
- Checkpoint & Resume (`--resume`): Khôi phục đầy đủ model, optimizer, GradScaler, bộ đếm step, lịch sử loss và giờ GPU cộng dồn.
- Ghi rõ riêng biệt `recon_loss`, `kl_loss`, `beta`, `loss_type` và xuất lưới ảnh reconstruction + random generation định kỳ.

Sử dụng:
    python scripts/train_vae.py --config configs/vae/cifar10/base.yaml --stage full --device cuda
"""

from __future__ import annotations

import argparse
from importlib.metadata import version as pkg_version
import json
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.common import git_commit, normalize_config, reference_split_dir, sync
from datasets.preprocessing.builder import build_dataloader
from evaluation.calibrate import device_key
from evaluation.fid import calculate_kid_tuning, load_config, write_json
from models.interface import load_trainer
from scripts.render_samples import figure_dir, save_grid


def _save_tensor_pngs(images: torch.Tensor, out_dir: Path, start_idx: int) -> None:
    """Lưu tensor (N, C, H, W) trong [0, 1] ra PNG uint8 chuẩn mục 2.6."""
    imgs_u8 = (images.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()
    for i in range(imgs_u8.shape[0]):
        img_t = imgs_u8[i]
        if img_t.shape[0] == 1:
            img = Image.fromarray(img_t.squeeze(0).numpy(), mode="L")
        else:
            img = Image.fromarray(img_t.permute(1, 2, 0).numpy(), mode="RGB")
        img.save(out_dir / f"{start_idx + i:06d}.png", format="PNG")


@torch.inference_mode()
def _save_visual_progress(
    model,
    fixed_real: torch.Tensor,
    recon_path: Path,
    samples_path: Path,
    seed: int = 42,
    num_samples: int = 64,
) -> None:
    """Lưu lưới đối chiếu ảnh thật/reconstruction và lưới ảnh sinh ngẫu nhiên từ prior."""
    model.eval()
    real_dev = fixed_real.to(model.device)
    mu, _ = model.encode(real_dev)
    recon = model.decode(mu)

    real_01 = ((real_dev + 1.0) / 2.0).clamp(0.0, 1.0)
    recon_01 = ((recon + 1.0) / 2.0).clamp(0.0, 1.0)
    comparison = torch.cat([real_01, recon_01], dim=0)
    save_grid(comparison, recon_path, nrow=real_01.shape[0])

    generated = model.sample(num_samples, seed=seed)
    save_grid(generated, samples_path, nrow=8)


def _evaluate_val_kid(
    model,
    val_dir: Path,
    eval_cfg_path: str,
    dataset: str,
    resolution: int,
    device: str,
    seed: int,
    cache_dir: str,
) -> tuple[float, float, dict, str]:
    """Sinh đủ số ảnh tuning theo configs/eval/fid.yaml và tính KID thật trên validation."""
    eval_cfg, eval_hash = load_config(eval_cfg_path)
    num_gen = int(eval_cfg.get("num_gen_tuning", 5000))
    batch_size = 128

    model.eval()
    with tempfile.TemporaryDirectory() as tmp:
        fake_dir = Path(tmp) / "fake_val"
        fake_dir.mkdir(parents=True, exist_ok=True)
        generated = 0
        with torch.inference_mode():
            while generated < num_gen:
                cur = min(batch_size, num_gen - generated)
                samples = model.sample(cur, seed=seed + generated)
                _save_tensor_pngs(samples, fake_dir, generated)
                generated += cur

        res = calculate_kid_tuning(
            fake_dir=fake_dir,
            val_dir=val_dir,
            cfg=eval_cfg,
            device=device,
            seed=seed,
            cache_dir=cache_dir,
            dataset=dataset,
            resolution=resolution,
            eval_config_hash=eval_hash,
        )
    return float(res["kid_val_mean"]), float(res["kid_val_std"]), eval_cfg, eval_hash


def _resolve_val_ref_dir(val_ref_arg: str | None, dataset: str, stage: str, no_val_kid: bool) -> Path | None:
    """Xác định thư mục validation tham chiếu (tự động dùng outputs/reference/<dataset>/val nếu có)."""
    if no_val_kid or stage == "overfit":
        return None
    if val_ref_arg is not None:
        p = Path(val_ref_arg)
        if not p.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục validation tham chiếu: {p}")
        return p
    default_dir = reference_split_dir(dataset, "val")
    if default_dir.is_dir() and any(default_dir.glob("*.png")):
        return default_dir
    return None


def _build_stage_dataloader(dataset: str, batch_size: int, stage: str, overfit_samples: int) -> DataLoader:
    """Tạo DataLoader cho stage tương ứng (overfit cắt tập nhỏ cố định 64-512 ảnh)."""
    ds_cfg_path = f"configs/datasets/{dataset}.yaml"
    full_loader, train_ds = build_dataloader(ds_cfg_path, split="train", batch_size=batch_size)
    if len(train_ds) == 0:
        raise RuntimeError(f"Tập dữ liệu '{dataset}' (split='train') rỗng.")

    if stage != "overfit":
        if len(full_loader) == 0:
            raise RuntimeError(f"DataLoader cho '{dataset}' (split='train') rỗng.")
        return full_loader

    n_sub = min(max(1, int(overfit_samples)), len(train_ds))
    subset = Subset(train_ds, list(range(n_sub)))
    eff_bs = min(batch_size, n_sub)
    return DataLoader(
        subset,
        batch_size=eff_bs,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=(n_sub >= eff_bs and n_sub % eff_bs == 0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình VAE (hỗ trợ overfit/tuning/full & resume).")
    parser.add_argument("--config", required=True, help="Đường dẫn file cấu hình YAML của VAE")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--stage", default="full", choices=["overfit", "tuning", "full"])
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=256,
        help="Số lượng ảnh dùng khi --stage overfit (mục 2.3 khuyến nghị 64-512, mặc định 256).",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None, help="Đường dẫn file checkpoint (.pt) để tiếp tục huấn luyện.")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument("--eval-config", default="configs/eval/fid.yaml")
    parser.add_argument("--cache-dir", default="outputs/cache/fid")
    parser.add_argument(
        "--val-ref-dir",
        default=None,
        help="Thư mục PNG validation thật để tính KID validation (mặc định tự tìm outputs/reference/<dataset>/val).",
    )
    parser.add_argument(
        "--no-val-kid",
        action="store_true",
        help="Tắt tính KID validation định kỳ tại các checkpoint.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Chế độ smoke với dữ liệu tổng hợp (không đọc dataset thật và KHÔNG ghi production artifact).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    dataset = cfg.get("dataset", "cifar10")
    run_name = args.run_name or f"vae_{dataset}_{args.stage}"
    train_cfg = cfg.get("training", {})
    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 50))
    save_interval = int(train_cfg.get("save_interval", 10))
    seed = int(train_cfg.get("seed", 42))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev = torch.device(args.device)
    trainer = load_trainer("vae", str(cfg_path), device=dev)
    hw_name = device_key(dev, trainer.num_gpus)

    print(
        f"--- Huấn luyện VAE [{dataset}] | Run: {run_name} | Stage: {args.stage} "
        f"| Device: {hw_name} ({trainer.precision}) ---"
    )
    print(
        f"    Kiến trúc: image_shape={trainer.image_shape}, latent_dim={trainer.model.latent_dim}, "
        f"base_channels={trainer.model.base_channels} | Loss: {trainer.loss_type} + {trainer.beta} * KL"
    )

    if args.smoke:
        print("[SMOKE MODE] Dùng dữ liệu tổng hợp (synthetic); không xuất checkpoint/metadata production.")
        dataloader = [
            (
                torch.rand(trainer.batch_size, *trainer.image_shape) * 2 - 1,
                torch.zeros(trainer.batch_size, dtype=torch.long),
            )
            for _ in range(4)
        ]
        val_ref_dir = None
    else:
        dataloader = _build_stage_dataloader(dataset, trainer.batch_size, args.stage, args.overfit_samples)
        val_ref_dir = _resolve_val_ref_dir(args.val_ref_dir, dataset, args.stage, args.no_val_kid)
        if args.stage == "full" and val_ref_dir is None:
            print(
                f"   [CẢNH BÁO] Chưa tìm thấy thư mục validation tham chiếu cho '{dataset}'. "
                "Metadata run 'full' sẽ thiếu trường 'curve' nếu không có KID validation!"
            )
        elif val_ref_dir is not None:
            print(f"   Validation KID tham chiếu: {val_ref_dir}")

    # Cố định 8 ảnh thật đầu tiên để theo dõi chất lượng reconstruction xuyên suốt quá trình train
    first_batch = next(iter(dataloader))
    first_imgs = first_batch[0] if isinstance(first_batch, (list, tuple)) else first_batch
    fixed_real = first_imgs[: min(8, first_imgs.shape[0])].detach().clone()

    ckpt_save_dir = Path(args.checkpoint_dir) / dataset / "vae" / run_name
    fig_save_dir = figure_dir(args.figures_dir, dataset, "vae", run_name)
    if not args.smoke:
        ckpt_save_dir.mkdir(parents=True, exist_ok=True)
        fig_save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    step_count = 0
    best_loss = float("inf")
    best_kid = float("inf")
    best_kid_std = 0.0
    hardware_seconds: dict[str, float] = {}
    curve: list[dict] = []
    loss_history: list[dict] = []
    last_eval_hash: str | None = None

    # Khôi phục trạng thái nếu truyền --resume
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy checkpoint để resume: {resume_path}")
        ckpt = torch.load(resume_path, map_location=dev, weights_only=False)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if trainer.scaler is not None and ckpt.get("scaler_state_dict") is not None:
            trainer.scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        step_count = int(ckpt.get("step", 0))
        raw_best_loss = ckpt.get("best_loss", ckpt.get("train_loss"))
        best_loss = float(raw_best_loss) if raw_best_loss is not None else float("inf")
        raw_best_kid = ckpt.get("best_kid")
        best_kid = float(raw_best_kid) if raw_best_kid is not None else float("inf")
        hardware_seconds = {str(k): float(v) for k, v in (ckpt.get("hardware_seconds") or {}).items()}
        curve = list(ckpt.get("curve") or [])
        loss_history = list(ckpt.get("loss_history") or [])
        print(f"   Đã khôi phục từ '{resume_path}' (tiếp tục từ epoch {start_epoch}/{epochs}, step {step_count}).")

    wall_start = time.perf_counter()

    for ep in range(start_epoch, epochs + 1):
        ep_wall_start = time.perf_counter()
        ep_step_seconds = 0.0
        ep_loss_sum = 0.0
        ep_recon_sum = 0.0
        ep_kl_sum = 0.0
        ep_batches = 0

        for batch in dataloader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            sync(dev)
            t0 = time.perf_counter()
            trainer.train_step(imgs)
            sync(dev)
            ep_step_seconds += time.perf_counter() - t0

            step_count += 1
            ep_loss_sum += trainer.last_metrics.get("loss", 0.0)
            ep_recon_sum += trainer.last_metrics.get("recon_loss", 0.0)
            ep_kl_sum += trainer.last_metrics.get("kl_loss", 0.0)
            ep_batches += 1

        hardware_seconds[hw_name] = hardware_seconds.get(hw_name, 0.0) + ep_step_seconds
        denom = max(ep_batches, 1)
        avg_loss = ep_loss_sum / denom
        avg_recon = ep_recon_sum / denom
        avg_kl = ep_kl_sum / denom
        ep_wall = time.perf_counter() - ep_wall_start

        epoch_log = {
            "epoch": ep,
            "step": step_count,
            "loss": round(avg_loss, 6),
            "recon_loss": round(avg_recon, 6),
            "kl_loss": round(avg_kl, 6),
            "beta": trainer.beta,
            "loss_type": trainer.loss_type,
            "step_time_s": round(ep_step_seconds, 4),
            "wall_time_s": round(ep_wall, 4),
        }
        loss_history.append(epoch_log)

        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(
                f"Epoch [{ep}/{epochs}] - Loss: {avg_loss:.4f} "
                f"(Recon[{trainer.loss_type}]: {avg_recon:.4f}, KL*{trainer.beta:g}: {avg_kl:.4f}) "
                f"- TrainStepTime: {ep_step_seconds:.2f}s (Wall: {ep_wall:.2f}s) - Steps: {step_count}"
            )

        is_save_point = (ep % save_interval == 0) or (ep == epochs)
        if not args.smoke and is_save_point:
            kid_val: float | None = None
            kid_std: float | None = None
            should_eval_kid = val_ref_dir is not None and (args.stage == "full" or ep == epochs)
            if should_eval_kid and val_ref_dir is not None:
                kid_val, kid_std, _, last_eval_hash = _evaluate_val_kid(
                    model=trainer.model,
                    val_dir=val_ref_dir,
                    eval_cfg_path=args.eval_config,
                    dataset=dataset,
                    resolution=trainer.image_shape[1],
                    device=args.device,
                    seed=seed,
                    cache_dir=args.cache_dir,
                )
                epoch_log["kid_val_mean"] = round(kid_val, 6)
                epoch_log["kid_val_std"] = round(kid_std, 6)
                hw_hours_snapshot = {k: round(v / 3600.0, 6) for k, v in hardware_seconds.items()}
                curve.append(
                    {
                        "step": step_count,
                        "kid_val_mean": kid_val,
                        "hardware_hours": hw_hours_snapshot,
                    }
                )
                print(f"   -> Validation KID tại step {step_count}: {kid_val:.6f} ± {kid_std:.6f}")

            # Xác định mô hình tốt nhất: ưu tiên KID validation nếu có, ngược lại dùng train_loss
            is_new_best = False
            if kid_val is not None:
                if kid_val <= best_kid:
                    best_kid = kid_val
                    best_kid_std = kid_std or 0.0
                    is_new_best = True
                best_loss = min(best_loss, avg_loss)
            elif avg_loss <= best_loss:
                best_loss = avg_loss
                is_new_best = True

            model_sd = trainer.model.state_dict()
            # Checkpoint phục vụ resume (gồm cả optimizer + scaler + lịch sử)
            full_ckpt_payload = {
                "epoch": ep,
                "step": step_count,
                "train_loss": float(avg_loss),
                "recon_loss": float(avg_recon),
                "kl_loss": float(avg_kl),
                "beta": trainer.beta,
                "loss_type": trainer.loss_type,
                "kid_val_mean": kid_val,
                "best_loss": float(best_loss),
                "best_kid": float(best_kid) if best_kid < float("inf") else None,
                "model_state_dict": model_sd,
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scaler_state_dict": trainer.scaler.state_dict() if trainer.scaler is not None else None,
                "includes_optimizer_state": True,
                "image_shape": trainer.image_shape,
                "latent_dim": trainer.model.latent_dim,
                "base_channels": trainer.model.base_channels,
                "config": normalize_config(str(cfg_path)),
                "hardware_seconds": dict(hardware_seconds),
                "curve": list(curve),
                "loss_history": list(loss_history),
            }
            ckpt_file = ckpt_save_dir / f"checkpoint_ep{ep}.pt"
            torch.save(full_ckpt_payload, ckpt_file)
            torch.save(full_ckpt_payload, ckpt_save_dir / "last.pt")

            # Lưu ảnh trực quan định kỳ (reconstruction + random generation)
            _save_visual_progress(
                model=trainer.model,
                fixed_real=fixed_real,
                recon_path=ckpt_save_dir / f"recon_ep{ep}.png",
                samples_path=ckpt_save_dir / f"samples_ep{ep}.png",
                seed=seed,
            )

            if is_new_best or not (ckpt_save_dir / "best.pt").exists():
                # best.pt dùng để sinh ảnh và đo dung lượng checkpoint (mục 2.5): KHÔNG kèm optimizer state
                inference_payload = {
                    "epoch": ep,
                    "step": step_count,
                    "train_loss": float(avg_loss),
                    "recon_loss": float(avg_recon),
                    "kl_loss": float(avg_kl),
                    "kid_val_mean": kid_val,
                    "model_state_dict": model_sd,
                    "includes_optimizer_state": False,
                    "image_shape": trainer.image_shape,
                    "latent_dim": trainer.model.latent_dim,
                    "base_channels": trainer.model.base_channels,
                    "config": normalize_config(str(cfg_path)),
                }
                torch.save(inference_payload, ckpt_save_dir / "best.pt")
                _save_visual_progress(
                    model=trainer.model,
                    fixed_real=fixed_real,
                    recon_path=fig_save_dir / "reconstruction.png",
                    samples_path=fig_save_dir / "samples.png",
                    seed=seed,
                )

            write_json({"run_name": run_name, "dataset": dataset, "epochs": loss_history}, ckpt_save_dir / "loss_history.json")

    if args.smoke:
        print(f"[SMOKE MODE] Hoàn tất {epochs} epoch ({step_count} steps). Không xuất checkpoint/metadata.")
        return

    total_wall_hours = (time.perf_counter() - wall_start) / 3600.0
    hardware_list = [
        {"device": dev_name, "train_hours": round(sec / 3600.0, 6)}
        for dev_name, sec in hardware_seconds.items()
    ]

    run_record: dict = {
        "kind": "run",
        "model": "vae",
        "dataset": dataset,
        "run_name": run_name,
        "stage": args.stage,
        "config": normalize_config(str(cfg_path)),
        "image_size": trainer.image_shape[1],
        "num_samples": 50000 if args.stage == "full" else (args.overfit_samples if args.stage == "overfit" else 5000),
        "seed": seed,
        "checkpoint": str(ckpt_save_dir / "best.pt"),
        "checkpoint_includes_optimizer": False,
        "train_steps": step_count,
        "batch_size": trainer.batch_size,
        "precision": trainer.precision,
        "loss_type": trainer.loss_type,
        "beta": trainer.beta,
        "final_loss": round(loss_history[-1]["loss"], 6) if loss_history else None,
        "final_recon_loss": round(loss_history[-1]["recon_loss"], 6) if loss_history else None,
        "final_kl_loss": round(loss_history[-1]["kl_loss"], 6) if loss_history else None,
        "wall_clock_hours": round(total_wall_hours, 6),
        "hardware": hardware_list,
        "eval_config": normalize_config(args.eval_config),
        "commit": git_commit(),
    }
    if curve:
        run_record["curve"] = curve

    runs_dir = Path(args.metrics_dir) / "runs"
    run_file = runs_dir / f"{dataset}__vae__{run_name}.json"
    write_json(run_record, run_file)

    # Nếu ở giai đoạn tuning và có KID validation, tự động ghi file quality tuning cho bảng "Các config đã thử"
    if args.stage == "tuning" and best_kid < float("inf"):
        _, eval_hash = load_config(args.eval_config)
        try:
            cleanfid_ver = pkg_version("clean-fid")
        except Exception:
            cleanfid_ver = "unknown"
        res_px = trainer.image_shape[1]
        tuning_quality = {
            "kind": "quality",
            "model": "vae",
            "dataset": dataset,
            "run_name": run_name,
            "label": "",
            "stage": "tuning",
            "eval": {
                "eval_config": normalize_config(args.eval_config),
                "eval_config_hash": last_eval_hash or eval_hash,
                "library": f"clean-fid=={cleanfid_ver}",
                "commit": git_commit() or "unknown",
                "ref_train": f"{dataset}_train_{res_px}",
                "ref_test": f"{dataset}_test_{res_px}",
                "ref_val": f"{dataset}_val_{res_px}",
                "num_gen": 5000,
            },
            "metrics": {
                "kid_val_mean": float(best_kid),
                "kid_val_std": float(best_kid_std),
            },
        }
        q_path = Path(args.metrics_dir) / "quality" / f"{dataset}__vae__{run_name}.json"
        write_json(tuning_quality, q_path)
        print(f"Tuning Quality JSON: {q_path}")

    print(f"Huấn luyện hoàn tất! Metadata run: {run_file}")


if __name__ == "__main__":
    main()
