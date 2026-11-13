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
