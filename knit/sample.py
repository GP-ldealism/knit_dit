import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffusion import create_diffusion
from knit import checker, feedback, repair
from knit.io import append_jsonl, load_json, save_json, save_labels_png
from knit.model import build_dit


def load_model(config: dict[str, Any], ckpt: str, device: torch.device) -> torch.nn.Module:
    model = build_dit(config).to(device)
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "ema" in checkpoint:
        state = checkpoint["ema"]
    else:
        state = checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


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
) -> None:
    num_samples = int(args.num_samples)
    noise, y = make_noise(
        num_samples, config, args.requests, args.seed, device
    )
    if y is None:
        y = torch.zeros(num_samples, dtype=torch.long, device=device)
    diffusion = create_diffusion("ddim100", learn_sigma=False)
    steps = [int(item) for item in args.feedback_steps.split(",") if item]
    batch_size = int(args.batch_size)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    palette = load_palette(args.palette)
    all_labels: list[np.ndarray] = []

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
            clip_denoised=False,
            denoised_fn=denoised_fn,
            model_kwargs={"y": y[start:end]},
            device=device,
            progress=True,
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
            append_jsonl(
                out / "meta.jsonl",
                {
                    "id": sample_id,
                    "method": args.method,
                    "path": f"{sample_id:06d}.png",
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
    save_json(out / "check_summary.json", checker.summarize(reports))
    print(json.dumps(checker.summarize(reports), ensure_ascii=False, indent=2))


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
    args = parser.parse_args()

    config = load_json(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.ckpt, device)
    sample_a(args, config, model, device)


if __name__ == "__main__":
    main()
