import argparse
import copy
import json
from pathlib import Path
from typing import Any

from knit.io import save_json


EXPECTED_DIFFUSION: dict[str, Any] = {
    "noise_schedule": "squaredcos_cap_v2",
    "predict_xstart": True,
    "predict_v": False,
    "learn_sigma": False,
}


def build_finetune_config(
    source: dict[str, Any],
    data_dir: str,
    base_data_dir: str,
    updates: int = 1000,
    lr: float = 1e-5,
    weight_decay: float = 0.0,
    warmup_updates: int = 0,
    min_lr: float = 0.0,
    lr_schedule: str = "constant",
    ema_decay: float = 0.99,
    ema_start_step: int = 0,
    val_every: int = 500,
    ckpt_every: int = 500,
    log_every: int = 10,
    max_seconds: float = 1800.0,
    grad_clip: float = 1.0,
    amp: str = "none",
    grad_accum_steps: int = 1,
    num_workers: int = 4,
) -> dict[str, Any]:
    mismatches = {
        key: {"source": source.get(key), "expected": expected}
        for key, expected in EXPECTED_DIFFUSION.items()
        if source.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"source config diffusion mismatch: {mismatches}")

    config = copy.deepcopy(source)
    config["data_dir"] = data_dir
    config["data"] = {
        "train_labels": f"{data_dir}/train_labels.npy",
        "train_meta": f"{data_dir}/train_meta.json",
        "val_labels": f"{data_dir}/val_labels.npy",
        "val_meta": f"{data_dir}/val_meta.json",
        "test_labels": f"{base_data_dir}/test_labels.npy",
        "test_meta": f"{base_data_dir}/test_meta.json",
        "palette": f"{base_data_dir}/palette.json",
    }
    config.update(
        {
            "batch_size": 32,
            "updates": int(updates),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "warmup_updates": int(warmup_updates),
            "min_lr": float(min_lr),
            "lr_schedule": lr_schedule,
            "ema_decay": float(ema_decay),
            "ema_start_step": int(ema_start_step),
            "val_every": int(val_every),
            "ckpt_every": int(ckpt_every),
            "log_every": int(log_every),
            "max_seconds": float(max_seconds),
            "grad_clip": float(grad_clip),
            "amp": amp,
            "grad_accum_steps": int(grad_accum_steps),
            "num_workers": int(num_workers),
        }
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--base-data-dir", required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--lr-schedule", default="constant")
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--ema-start-step", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", default="none")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.source, encoding="utf-8") as handle:
        source = json.load(handle)
    config = build_finetune_config(
        source,
        data_dir=args.data_dir,
        base_data_dir=args.base_data_dir,
        updates=args.updates,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_updates=args.warmup_updates,
        min_lr=args.min_lr,
        lr_schedule=args.lr_schedule,
        ema_decay=args.ema_decay,
        ema_start_step=args.ema_start_step,
        val_every=args.val_every,
        ckpt_every=args.ckpt_every,
        log_every=args.log_every,
        max_seconds=args.max_seconds,
        grad_clip=args.grad_clip,
        amp=args.amp,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, config)
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
