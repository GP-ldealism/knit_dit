import argparse
from pathlib import Path
from typing import Any

import numpy as np

from knit.io import load_json, save_json


def make_requests(
    config: dict[str, Any],
    num_samples: int,
    seed: int,
    out: str | Path,
) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shape = (
        num_samples,
        int(config["in_channels"]),
        int(config["image_size"]),
        int(config["image_size"]),
    )
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape, dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int64)
    np.savez(out, noise=noise, y=y)
    save_json(
        out.with_suffix(".json"),
        {
            "path": str(out),
            "num_samples": num_samples,
            "seed": seed,
            "shape": list(shape),
            "config": config,
        },
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs/a/requests_s42.npz")
    args = parser.parse_args()

    config = load_json(args.config)
    path = make_requests(config, args.num_samples, args.seed, args.out)
    print(path)


if __name__ == "__main__":
    main()
