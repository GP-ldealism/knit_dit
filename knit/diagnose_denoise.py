import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from knit.checkpoint import load_model
from knit.diffusion_setup import (
    make_diffusion,
    merge_diffusion_config,
    model_output_to_eps,
    model_output_to_x0,
)
from knit.io import labels_to_onehot, load_json, save_json


def parse_timesteps(text: str) -> list[int]:
    values = sorted({int(item) for item in text.split(",") if item.strip()})
    if not values:
        raise ValueError("timesteps must not be empty")
    if any(value < 0 or value > 999 for value in values):
        raise ValueError("timesteps must be in 0..999")
    return values


def select_labels(
    labels: np.ndarray, num_samples: int, seed: int
) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError(f"expected [N,H,W] labels, got {labels.shape}")
    if labels.shape[0] == 0:
        raise ValueError("label file is empty")
    if num_samples >= labels.shape[0]:
        return labels
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(labels.shape[0], num_samples, replace=False))
    return labels[indices]


def diagnose(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    num_classes = int(config["in_channels"])
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    timesteps = parse_timesteps(args.timesteps)
    labels_raw = select_labels(
        np.load(args.labels, allow_pickle=False),
        int(args.num_samples),
        int(args.seed),
    )
    if labels_raw.max(initial=0) >= num_classes:
        raise ValueError(
            f"labels contain class >= {num_classes}; check --config in_channels"
        )
    x0_all = torch.from_numpy(
        labels_to_onehot(labels_raw, num_classes=num_classes).astype(np.float32)
    ).to(device)
    model, checkpoint_info = load_model(
        config, args.ckpt, device, weights=args.weights
    )
    config = merge_diffusion_config(config, checkpoint_info.get("config"))
    diffusion = make_diffusion(config)
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    noise_all = torch.randn(x0_all.shape, generator=generator).to(device)
    batch_size = int(args.batch_size)
    baseline = float((labels_raw == 0).mean())

    results: dict[str, Any] = {}
    with torch.no_grad():
        for timestep in timesteps:
            pixel_matches: list[np.ndarray] = []
            correct = {str(index): 0 for index in range(num_classes)}
            support = {str(index): 0 for index in range(num_classes)}
            label_counts = np.zeros(num_classes, dtype=np.int64)
            pred_counts = np.zeros(num_classes, dtype=np.int64)
            noise_mse_sum = 0.0
            x0_mse_sum = 0.0
            count = 0
            for start in range(0, x0_all.shape[0], batch_size):
                end = min(start + batch_size, x0_all.shape[0])
                x0 = x0_all[start:end]
                noise = noise_all[start:end]
                labels = labels_raw[start:end]
                t = torch.full(
                    (end - start,), timestep, dtype=torch.long, device=device
                )
                y = torch.zeros(end - start, dtype=torch.long, device=device)
                x_t = diffusion.q_sample(x0, t, noise=noise)
                model_output = model(x_t, t, y)
                eps = model_output_to_eps(diffusion, x_t, t, model_output)
                pred_x0 = model_output_to_x0(
                    diffusion, x_t, t, model_output
                ).clamp(-1, 1)
                predictions = pred_x0.argmax(dim=1).detach().cpu().numpy()
                chunk = end - start
                noise_mse_sum += float(((eps - noise) ** 2).mean().item()) * chunk
                x0_mse_sum += float(((pred_x0 - x0) ** 2).mean().item()) * chunk
                count += chunk
                matches = (predictions == labels).reshape(chunk, -1).mean(axis=1)
                pixel_matches.append(matches)
                for index in range(num_classes):
                    mask = labels == index
                    support[str(index)] += int(mask.sum())
                    correct[str(index)] += int((predictions[mask] == index).sum())
                label_counts += np.bincount(
                    labels.reshape(-1), minlength=num_classes
                )
                pred_counts += np.bincount(
                    predictions.reshape(-1), minlength=num_classes
                )
            matches = np.concatenate(pixel_matches)
            label_total = int(label_counts.sum())
            pred_total = int(pred_counts.sum())
            results[str(timestep)] = {
                "noise_mse": noise_mse_sum / count,
                "x0_mse": x0_mse_sum / count,
                "pixel_match": float(matches.mean()),
                "pixel_match_std": float(matches.std()),
                "pixel_match_min": float(matches.min()),
                "pixel_match_max": float(matches.max()),
                "class_accuracy": {
                    key: (correct[key] / support[key] if support[key] else None)
                    for key in correct
                },
                "class_support": support,
                "frequency": {
                    "label": {
                        str(index): float(label_counts[index] / label_total)
                        for index in range(num_classes)
                    },
                    "prediction": {
                        str(index): float(pred_counts[index] / pred_total)
                        for index in range(num_classes)
                    },
                },
            }
            print(
                json.dumps(
                    {
                        "timestep": timestep,
                        "noise_mse": results[str(timestep)]["noise_mse"],
                        "pixel_match": results[str(timestep)]["pixel_match"],
                        "baseline": baseline,
                    },
                    ensure_ascii=False,
                )
            )

    summary = {
        "checkpoint": checkpoint_info,
        "labels": str(args.labels),
        "num_samples": int(x0_all.shape[0]),
        "seed": int(args.seed),
        "timesteps": timesteps,
        "baseline_all_background_pixel_match": baseline,
        "results": results,
    }
    save_json(args.out, summary)
    print(f"saved {Path(args.out)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--timesteps", default="100,250,500,750,900")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--weights", choices=["auto", "ema", "model"], default="auto")
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    diagnose(args)


if __name__ == "__main__":
    main()
