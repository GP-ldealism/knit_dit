import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from knit.data import KnitDataset
from knit.diffusion_setup import (
    diffusion_loss,
    diffusion_settings,
    make_diffusion,
    sample_timesteps,
)
from knit.io import append_jsonl, load_json, save_json
from knit.model import build_dit, count_parameters, describe_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(
                    source[name].detach().to(value.device), alpha=1.0 - self.decay
                )
            else:
                value.copy_(source[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.module.state_dict()


def learning_rate_at(
    step: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
    schedule: str,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    if schedule == "constant":
        return base_lr
    if schedule == "cosine":
        progress_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / progress_steps)
        return min_lr + 0.5 * (base_lr - min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )
    raise ValueError(f"unknown lr schedule: {schedule}")


def make_dataset(config: dict[str, Any], split: str) -> KnitDataset:
    data_cfg = config["data"]
    labels = data_cfg[f"{split}_labels"]
    meta = data_cfg.get(f"{split}_meta")
    return KnitDataset(
        labels_path=labels,
        meta_path=meta,
        num_classes=int(config["in_channels"]),
    )


def make_loader(
    dataset: KnitDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def validation_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    diffusion: Any,
    device: torch.device,
    t_sample: str = "uniform",
    min_snr_gamma: float | None = None,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        x0 = batch["x0"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        t = sample_timesteps(
            x0.shape[0], diffusion.num_timesteps, t_sample, device=device
        )
        loss = diffusion_loss(diffusion, model, x0, t, y, min_snr_gamma=min_snr_gamma)
        total += float(loss.item()) * x0.shape[0]
        count += x0.shape[0]
    model.train()
    return total / count if count else float("nan")


def save_checkpoint(
    out: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
    ema: ModelEma | None,
    best_val_loss: float | None,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "config": config,
        "step": step,
        "best_val_loss": best_val_loss,
    }
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
        checkpoint["ema_decay"] = ema.decay
    torch.save(checkpoint, out / f"ckpt_{step:06d}.pt")
    torch.save(checkpoint, out / "model.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--out", default="runs/base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--overfit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--ckpt-every", type=int, default=None)
    parser.add_argument("--amp", choices=["none", "fp16"], default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--warmup-updates", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default=None)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--ema-start-step", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--stop-at-step", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument(
        "--noise-schedule",
        choices=["linear", "squaredcos_cap_v2"],
        default=None,
    )
    parser.add_argument("--predict-xstart", action="store_true", default=None)
    parser.add_argument("--predict-v", action="store_true", default=None)
    parser.add_argument("--t-sample", choices=["uniform", "logitnorm"], default=None)
    parser.add_argument("--min-snr-gamma", type=float, default=None)
    args = parser.parse_args()

    run_start = time.perf_counter()
    config = load_json(args.config)
    if args.updates is not None:
        config["updates"] = args.updates
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.log_every is not None:
        config["log_every"] = args.log_every
    if args.val_every is not None:
        config["val_every"] = args.val_every
    if args.ckpt_every is not None:
        config["ckpt_every"] = args.ckpt_every
    if args.amp is not None:
        config["amp"] = args.amp
    if args.grad_clip is not None:
        config["grad_clip"] = args.grad_clip
    if args.warmup_updates is not None:
        config["warmup_updates"] = args.warmup_updates
    if args.min_lr is not None:
        config["min_lr"] = args.min_lr
    if args.lr_schedule is not None:
        config["lr_schedule"] = args.lr_schedule
    if args.ema_decay is not None:
        config["ema_decay"] = args.ema_decay
    if args.ema_start_step is not None:
        config["ema_start_step"] = args.ema_start_step
    if args.max_seconds is not None:
        config["max_seconds"] = args.max_seconds
    if args.stop_at_step is not None:
        config["stop_at_step"] = args.stop_at_step
    if args.grad_accum_steps is not None:
        config["grad_accum_steps"] = args.grad_accum_steps
    if args.noise_schedule is not None:
        config["noise_schedule"] = args.noise_schedule
    if args.predict_xstart is not None:
        config["predict_xstart"] = bool(args.predict_xstart)
    if args.predict_v is not None:
        config["predict_v"] = bool(args.predict_v)
    if args.t_sample is not None:
        config["t_sample"] = args.t_sample
    if args.min_snr_gamma is not None:
        config["min_snr_gamma"] = args.min_snr_gamma

    set_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", config)

    train_dataset = make_dataset(config, "train")
    val_dataset = make_dataset(config, "val")
    if args.overfit is not None:
        n = min(int(args.overfit), len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(n)))
        val_dataset = Subset(val_dataset, list(range(min(n, len(val_dataset)))))
    if len(train_dataset) == 0:
        raise ValueError("empty training split")
    if len(val_dataset) == 0:
        val_dataset = train_dataset

    batch_size = int(config["batch_size"])
    num_workers = int(config.get("num_workers", 0))
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_dit(config).to(device)
    model_summary = describe_model(model)
    save_json(out / "model.json", model_summary)
    diffusion = make_diffusion(config)
    diff_settings = diffusion_settings(config)
    t_sample = str(config.get("t_sample", "uniform"))
    min_snr_gamma = config.get("min_snr_gamma")
    min_snr_gamma = None if min_snr_gamma is None else float(min_snr_gamma)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    amp_mode = str(config.get("amp", "none"))
    if amp_mode not in {"none", "fp16"}:
        raise ValueError(f"unknown amp mode: {amp_mode}")
    use_amp = amp_mode == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip = float(config.get("grad_clip", 0.0))
    warmup_updates = int(config.get("warmup_updates", 0))
    min_lr = float(config.get("min_lr", 0.0))
    lr_schedule = str(config.get("lr_schedule", "constant"))
    ema_decay = float(config.get("ema_decay", 0.0))
    ema_start_step = int(config.get("ema_start_step", 0))
    max_seconds = config.get("max_seconds")
    stop_at_step = config.get("stop_at_step")
    stop_at_step = None if stop_at_step is None else int(stop_at_step)
    grad_accum_steps = max(1, int(config.get("grad_accum_steps", 1)))

    start_step = 0
    best_val_loss: float | None = None
    resume_payload: dict[str, Any] = {}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            resume_payload = checkpoint
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "ema" in checkpoint:
            state = checkpoint["ema"]
        else:
            state = checkpoint
        model.load_state_dict(state)
        if isinstance(checkpoint, dict) and "opt" in checkpoint:
            optimizer.load_state_dict(checkpoint["opt"])
        if isinstance(checkpoint, dict):
            start_step = int(checkpoint.get("step", 0))
            best_val_loss = checkpoint.get("best_val_loss")
        print(f"resumed from {args.resume} at step {start_step}")

    ema: ModelEma | None = ModelEma(model, ema_decay) if ema_decay > 0 else None
    if ema is not None and "ema" in resume_payload:
        ema.module.load_state_dict(resume_payload["ema"])

    updates = int(config["updates"])
    effective_stop = updates if stop_at_step is None else min(updates, stop_at_step)
    log_every = int(config.get("log_every", 10))
    val_every = int(config.get("val_every", 500))
    ckpt_every = int(config.get("ckpt_every", 1000))
    log_path = out / "train.jsonl"
    data_iter = iter(train_loader)
    current_step = start_step
    model.train()

    print(
        f"device={device} params={count_parameters(model)} "
        f"train={len(train_dataset)} val={len(val_dataset)} updates={updates} "
        f"amp={amp_mode} ema={ema_decay} lr_schedule={lr_schedule} "
        f"grad_accum={grad_accum_steps} schedule={diff_settings['noise_schedule']} "
        f"predict_xstart={diff_settings['predict_xstart']} "
        f"predict_v={diff_settings['predict_v']} t_sample={t_sample} "
        f"min_snr_gamma={min_snr_gamma} stop_at_step={stop_at_step}"
    )

    for step in range(start_step + 1, effective_stop + 1):
        current_lr = learning_rate_at(
            step,
            base_lr=float(config["lr"]),
            min_lr=min_lr,
            warmup_steps=warmup_updates,
            total_steps=updates,
            schedule=lr_schedule,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        step_loss_sum = 0.0
        for _ in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            x0 = batch["x0"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            t = sample_timesteps(
                x0.shape[0], diffusion.num_timesteps, t_sample, device=device
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = diffusion_loss(
                    diffusion, model, x0, t, y, min_snr_gamma=min_snr_gamma
                )
            step_loss_sum += float(loss.detach().item())
            scaled_loss = loss / grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        grad_norm: float | None = None
        if scaler.is_enabled():
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip > 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                )
            optimizer.step()

        if ema is not None and step >= ema_start_step:
            ema.update(model)
        current_step = step

        if step % log_every == 0 or step == 1:
            record = {
                "step": step,
                "loss": step_loss_sum / grad_accum_steps,
                "lr": current_lr,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if grad_norm is not None:
                record["grad_norm"] = grad_norm
            if device.type == "cuda":
                record["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            append_jsonl(log_path, record)
            print(json.dumps(record, ensure_ascii=False))

        if step % val_every == 0:
            val_loss = validation_loss(
                model,
                val_loader,
                diffusion,
                device,
                t_sample=t_sample,
                min_snr_gamma=min_snr_gamma,
            )
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
            record = {"step": step, "val_loss": val_loss}
            append_jsonl(log_path, record)
            print(json.dumps(record, ensure_ascii=False))

        save_ema = ema if ema is not None and step >= ema_start_step else None
        if step % ckpt_every == 0:
            save_checkpoint(
                out, model, optimizer, config, step, save_ema, best_val_loss
            )

        if max_seconds is not None and time.perf_counter() - run_start >= float(
            max_seconds
        ):
            print(
                f"stopping at step {step}: max_seconds={float(max_seconds):.0f} reached",
                flush=True,
            )
            break

    if (
        stop_at_step is not None
        and current_step >= stop_at_step
        and current_step < updates
    ):
        print(
            f"stopped at stop_at_step={stop_at_step} (step {current_step})",
            flush=True,
        )

    save_ema = (
        ema if ema is not None and current_step >= ema_start_step else None
    )
    save_checkpoint(
        out, model, optimizer, config, current_step, save_ema, best_val_loss
    )
    summary = {
        "seed": int(args.seed),
        "device": str(device),
        "start_step": start_step,
        "last_step": current_step,
        "updates": updates,
        "stop_at_step": stop_at_step,
        "completed": current_step >= updates,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_val_loss": best_val_loss,
        "elapsed_sec": time.perf_counter() - run_start,
        "amp": amp_mode,
        "ema_decay": ema_decay if ema is not None else 0.0,
        "ema_start_step": ema_start_step,
        "lr_schedule": lr_schedule,
        "warmup_updates": warmup_updates,
        "grad_accum_steps": grad_accum_steps,
        "noise_schedule": diff_settings["noise_schedule"],
        "predict_xstart": diff_settings["predict_xstart"],
        "predict_v": diff_settings["predict_v"],
        "t_sample": t_sample,
        "min_snr_gamma": min_snr_gamma,
        "model": model_summary,
    }
    save_json(out / "train_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
