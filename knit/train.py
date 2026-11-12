import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from diffusion import create_diffusion
from knit.data import KnitDataset
from knit.io import append_jsonl, load_json, save_json
from knit.model import build_dit, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        x0 = batch["x0"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        t = torch.randint(
            0, diffusion.num_timesteps, (x0.shape[0],), device=device
        )
        loss = diffusion.training_losses(model, x0, t, {"y": y})["loss"].mean()
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
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "config": config,
        "step": step,
    }
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
    args = parser.parse_args()

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
    diffusion = create_diffusion(timestep_respacing="", learn_sigma=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "ema" in checkpoint:
            state = checkpoint["ema"]
        else:
            state = checkpoint
        model.load_state_dict(state)
        if isinstance(checkpoint, dict) and "opt" in checkpoint:
            optimizer.load_state_dict(checkpoint["opt"])
        start_step = int(checkpoint.get("step", 0)) if isinstance(checkpoint, dict) else 0
        print(f"resumed from {args.resume} at step {start_step}")

    updates = int(config["updates"])
    log_every = int(config.get("log_every", 10))
    val_every = int(config.get("val_every", 500))
    ckpt_every = int(config.get("ckpt_every", 1000))
    log_path = out / "train.jsonl"
    data_iter = iter(train_loader)
    model.train()

    print(
        f"device={device} params={count_parameters(model)} "
        f"train={len(train_dataset)} val={len(val_dataset)} updates={updates}"
    )

    for step in range(start_step + 1, updates + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        x0 = batch["x0"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        t = torch.randint(
            0, diffusion.num_timesteps, (x0.shape[0],), device=device
        )
        loss = diffusion.training_losses(model, x0, t, {"y": y})["loss"].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == 1:
            record = {
                "step": step,
                "loss": float(loss.item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if device.type == "cuda":
                record["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            append_jsonl(log_path, record)
            print(json.dumps(record, ensure_ascii=False))

        if step % val_every == 0:
            val_loss = validation_loss(model, val_loader, diffusion, device)
            record = {"step": step, "val_loss": val_loss}
            append_jsonl(log_path, record)
            print(json.dumps(record, ensure_ascii=False))

        if step % ckpt_every == 0:
            save_checkpoint(out, model, optimizer, config, step)

    save_checkpoint(out, model, optimizer, config, updates)
    print(f"finished at step {updates}; outputs in {out}")


if __name__ == "__main__":
    main()
