from typing import Any

from models import DiT


def build_dit(config: dict[str, Any]) -> DiT:
    image_size = int(config["image_size"])
    patch_size = int(config["patch_size"])
    if image_size % patch_size != 0:
        raise ValueError(
            f"image_size {image_size} must be divisible by patch_size {patch_size}"
        )
    hidden_size = int(config["hidden_size"])
    num_heads = int(config["num_heads"])
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        )
    return DiT(
        input_size=image_size,
        patch_size=patch_size,
        in_channels=int(config["in_channels"]),
        hidden_size=hidden_size,
        depth=int(config["depth"]),
        num_heads=num_heads,
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        class_dropout_prob=float(config.get("class_dropout_prob", 0.0)),
        num_classes=int(config.get("num_classes", 1)),
        learn_sigma=bool(config.get("learn_sigma", False)),
    )


def count_parameters(model: DiT) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
