import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from knit import checker, feedback, repair
from knit.checkpoint import load_model
from knit.diffusion_setup import make_diffusion, merge_diffusion_config
from knit.encoding import CROSS_CLASSES, LabelSpec
from knit.io import (
    append_jsonl,
    load_json,
    save_json,
    save_labels_png,
    sha256_array,
)


def load_palette(path: str | None) -> list[int] | None:
    if not path:
        return None
    palette_path = Path(path)
    if not palette_path.is_file():
        return None
    payload = load_json(palette_path)
    palette = payload.get("palette") if isinstance(payload, dict) else payload
    return palette


def make_noise(
    num_samples: int,
    config: dict[str, Any],
    requests: str | None,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    shape = (
        num_samples,
        int(config["in_channels"]),
        int(config["image_size"]),
        int(config["image_size"]),
    )
    if requests:
        payload = np.load(requests, allow_pickle=False)
        noise = payload["noise"] if "noise" in payload else payload["xT"]
        noise = torch.from_numpy(noise).float()
        if noise.shape[0] < num_samples:
            raise ValueError(
                f"requests has {noise.shape[0]} samples, need {num_samples}"
            )
        noise = noise[:num_samples]
        y = payload["y"] if "y" in payload else None
        if y is not None:
            y = torch.from_numpy(y).long()[:num_samples]
        return noise.to(device), None if y is None else y.to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(shape, generator=generator).to(device)
    return noise, None


def repair_labels(
    labels: np.ndarray,
    budget: int,
    num_classes: int,
) -> np.ndarray:
    fixed: list[np.ndarray] = []
    for item in labels:
        report = checker.check(item, num_classes=num_classes)
        if report.ok:
            fixed.append(item)
            continue
        candidates = repair.propose(
            item, report, budget=budget, num_classes=num_classes
        )
        if not candidates:
            fixed.append(item)
            continue
        fixed.append(repair.rank(candidates, report))
    return np.stack(fixed)


def sample_a(
    args: argparse.Namespace,
    config: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    checkpoint_info: dict[str, Any],
) -> None:
    run_start = time.perf_counter()
    num_samples = int(args.num_samples)
    spec = LabelSpec(num_classes=int(config["in_channels"]))
    noise, y = make_noise(
        num_samples, config, args.requests, args.seed, device
    )
    if y is None:
        y = torch.zeros(num_samples, dtype=torch.long, device=device)
    diffusion = make_diffusion(config, f"ddim{args.ddim_steps}")
    steps = [int(item) for item in args.feedback_steps.split(",") if item]
    batch_size = int(args.batch_size)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preview_scale = int(args.preview_scale)
    preview_dir = out / "preview" if preview_scale > 1 else None
    palette = load_palette(args.palette)
    all_labels: list[np.ndarray] = []
    feedback_stats: dict[str, int] | None = (
        {
            "step": 0,
            "budget_left": 0,
            "checks": 0,
            "candidate_evaluations": 0,
            "corrections": 0,
            "corrected_samples": 0,
        }
        if args.method == "a2"
        else None
    )

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        chunk = end - start
        denoised_fn = None
        if args.method == "a2":
            denoised_fn = feedback.make_feedback_fn(
                steps=steps,
                budget=int(args.candidate_budget),
                num_classes=int(config["in_channels"]),
            )
        samples = diffusion.ddim_sample_loop(
            model,
            shape=(
                chunk,
                int(config["in_channels"]),
                int(config["image_size"]),
                int(config["image_size"]),
            ),
            noise=noise[start:end],
            clip_denoised=True,
            denoised_fn=denoised_fn,
            model_kwargs={"y": y[start:end]},
            device=device,
            progress=True,
        )
        if denoised_fn is not None and hasattr(denoised_fn, "stats"):
            chunk_stats = denoised_fn.stats
            if feedback_stats is not None:
                for key in (
                    "checks",
                    "candidate_evaluations",
                    "corrections",
                    "corrected_samples",
                ):
                    feedback_stats[key] += int(chunk_stats.get(key, 0))
                feedback_stats["step"] = int(chunk_stats.get("step", 0))
                feedback_stats["budget_left"] = int(
                    chunk_stats.get("budget_left", 0)
                )
        labels = samples.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
        if args.method == "a1":
            labels = repair_labels(
                labels,
                budget=int(args.candidate_budget),
                num_classes=int(config["in_channels"]),
            )
        all_labels.append(labels)
        for offset, item in enumerate(labels):
            sample_id = start + offset
            save_labels_png(item, out / f"{sample_id:06d}.png", palette=palette)
            if preview_dir is not None:
                save_labels_png(
                    item,
                    preview_dir / f"{sample_id:06d}.png",
                    palette=palette,
                    scale=preview_scale,
                )
            append_jsonl(
                out / "meta.jsonl",
                {
                    "id": sample_id,
                    "method": args.method,
                    "path": f"{sample_id:06d}.png",
                    "sha256": sha256_array(item),
                    "class_counts": spec.histogram(item).tolist(),
                    "cross_cells": int(np.isin(item, list(CROSS_CLASSES)).sum()),
                    "check": checker.check(
                        item, num_classes=int(config["in_channels"])
                    ).to_dict(),
                },
            )

    labels_array = np.concatenate(all_labels, axis=0)
    np.save(out / "labels.npy", labels_array)
    reports = checker.check_batch(
        labels_array, num_classes=int(config["in_channels"])
    )
    check_summary = checker.summarize(reports)
    save_json(out / "check_summary.json", check_summary)
    summary = {
        "method": args.method,
        "seed": int(args.seed),
        "num_samples": num_samples,
        "batch_size": batch_size,
        "preview_scale": preview_scale,
        "ddim_steps": int(args.ddim_steps),
        "checkpoint": checkpoint_info,
        "requests": args.requests,
        "candidate_budget": int(args.candidate_budget),
        "feedback_steps": steps,
        "elapsed_sec": time.perf_counter() - run_start,
        "config": config,
        "check": check_summary,
        "feedback_stats": feedback_stats,
    }
    save_json(out / "sample_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=["a0", "a1", "a2"], default="a0")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--requests", default=None)
    parser.add_argument("--candidate-budget", type=int, default=24)
    parser.add_argument("--feedback-steps", default="70,85,95")
    parser.add_argument("--palette", default="data/nik/processed/palette.json")
    parser.add_argument("--preview-scale", type=int, default=10)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--weights", choices=["auto", "ema", "model"], default="auto")
    args = parser.parse_args()
    if not 1 <= args.ddim_steps <= 1000:
        parser.error("--ddim-steps must be in 1..1000")

    config = load_json(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_info = load_model(
        config, args.ckpt, device, weights=args.weights
    )
    config = merge_diffusion_config(config, checkpoint_info.get("config"))
    sample_a(args, config, model, device, checkpoint_info)


if __name__ == "__main__":
    main()
