import argparse
import json
from pathlib import Path
from typing import Any

import torch

from knit.model import build_dit


def load_model(
    config: dict[str, Any],
    ckpt: str | Path,
    device: torch.device,
    weights: str = "auto",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    payload = checkpoint if isinstance(checkpoint, dict) else {}
    has_ema = "ema" in payload
    has_model = "model" in payload

    if weights == "auto":
        selected = "ema" if has_ema else "model" if has_model else "state_dict"
    elif weights == "ema":
        if not has_ema:
            raise ValueError(f"{ckpt}: requested EMA weights, but checkpoint has no 'ema'")
        selected = "ema"
    elif weights == "model":
        if not has_model:
            raise ValueError(f"{ckpt}: requested model weights, but checkpoint has no 'model'")
        selected = "model"
    else:
        raise ValueError(f"unknown weights selection: {weights}")

    if selected == "state_dict":
        state = checkpoint
    else:
        state = payload[selected]

    model = build_dit(config).to(device)
    model.load_state_dict(state)
    model.eval()
    info = {
        "path": str(ckpt),
        "weights": selected,
        "step": int(payload["step"]) if "step" in payload else None,
        "has_ema": has_ema,
        "has_optimizer": "opt" in payload,
        "config": payload.get("config"),
    }
    return model, info


def describe_checkpoint(
    ckpt: str | Path,
    expect_step: int | None = None,
    expect_ema: bool = False,
    expect_noise_schedule: str | None = None,
    expect_predict_xstart: bool | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    payload = checkpoint if isinstance(checkpoint, dict) else {}
    raw_config = payload.get("config")
    source_config = raw_config if isinstance(raw_config, dict) else {}

    info: dict[str, Any] = {
        "path": str(ckpt),
        "step": int(payload["step"]) if "step" in payload else None,
        "has_ema": "ema" in payload,
        "has_model": "model" in payload,
        "has_optimizer": "opt" in payload,
        "ema_key_count": len(payload["ema"]) if "ema" in payload else None,
        "model_key_count": len(payload["model"]) if "model" in payload else None,
        "config": {
            "noise_schedule": source_config.get("noise_schedule"),
            "predict_xstart": source_config.get("predict_xstart"),
            "predict_v": source_config.get("predict_v"),
            "learn_sigma": source_config.get("learn_sigma"),
            "in_channels": source_config.get("in_channels"),
            "hidden_size": source_config.get("hidden_size"),
        } if raw_config is not None else None,
    }

    errors: list[str] = []
    if expect_step is not None and info["step"] != expect_step:
        errors.append(f"expected step {expect_step}, got {info['step']}")
    if expect_ema and not info["has_ema"]:
        errors.append("missing 'ema' weights")
    if (
        expect_noise_schedule is not None
        and source_config.get("noise_schedule") != expect_noise_schedule
    ):
        errors.append(
            f"expected noise_schedule={expect_noise_schedule}, "
            f"got {source_config.get('noise_schedule')}"
        )
    if (
        expect_predict_xstart is not None
        and source_config.get("predict_xstart") is not expect_predict_xstart
    ):
        errors.append(
            f"expected predict_xstart={expect_predict_xstart}, "
            f"got {source_config.get('predict_xstart')}"
        )
    info["expect_ok"] = not errors
    info["expect_errors"] = errors
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--describe", required=True)
    parser.add_argument("--expect-step", type=int, default=None)
    parser.add_argument("--expect-ema", action="store_true")
    parser.add_argument("--expect-noise-schedule", default=None)
    parser.add_argument(
        "--expect-predict-xstart", choices=["true", "false"], default=None
    )
    args = parser.parse_args()

    predict_xstart = (
        args.expect_predict_xstart == "true"
        if args.expect_predict_xstart is not None
        else None
    )
    info = describe_checkpoint(
        args.describe,
        expect_step=args.expect_step,
        expect_ema=args.expect_ema,
        expect_noise_schedule=args.expect_noise_schedule,
        expect_predict_xstart=predict_xstart,
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if not info["expect_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
