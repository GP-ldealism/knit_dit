from dataclasses import asdict, dataclass
from typing import Any

import torch

from models import DiT


@dataclass(frozen=True)
class ModelSpec:
    image_size: int = 20
    patch_size: int = 2
    in_channels: int = 17
    hidden_size: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_ratio: float = 4.0
    num_classes: int = 1
    class_dropout_prob: float = 0.0
    learn_sigma: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ModelSpec":
        return cls(
            image_size=int(config["image_size"]),
            patch_size=int(config["patch_size"]),
            in_channels=int(config["in_channels"]),
            hidden_size=int(config["hidden_size"]),
            depth=int(config["depth"]),
            num_heads=int(config["num_heads"]),
            mlp_ratio=float(config.get("mlp_ratio", 4.0)),
            num_classes=int(config.get("num_classes", 1)),
            class_dropout_prob=float(config.get("class_dropout_prob", 0.0)),
            learn_sigma=bool(config.get("learn_sigma", False)),
        )

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.image_size <= 0:
            issues.append("image_size must be positive")
        if self.patch_size <= 0:
            issues.append("patch_size must be positive")
        if self.patch_size > 0 and self.image_size % self.patch_size != 0:
            issues.append(
                f"image_size {self.image_size} must be divisible by patch_size {self.patch_size}"
            )
        if self.in_channels <= 0:
            issues.append("in_channels must be positive")
        if self.hidden_size <= 0:
            issues.append("hidden_size must be positive")
        if self.depth <= 0:
            issues.append("depth must be positive")
        if self.num_heads <= 0:
            issues.append("num_heads must be positive")
        if self.mlp_ratio <= 0:
            issues.append("mlp_ratio must be positive")
        if self.num_heads > 0 and self.hidden_size % self.num_heads != 0:
            issues.append(
                f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
            )
        if self.num_classes <= 0:
            issues.append("num_classes must be positive")
        if not 0.0 <= self.class_dropout_prob <= 1.0:
            issues.append("class_dropout_prob must be in [0, 1]")
        return issues

    @property
    def tokens_per_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.tokens_per_side**2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dit(config: dict[str, Any]) -> DiT:
    spec = ModelSpec.from_config(config)
    issues = spec.validate()
    if issues:
        raise ValueError("; ".join(issues))
    return DiT(
        input_size=spec.image_size,
        patch_size=spec.patch_size,
        in_channels=spec.in_channels,
        hidden_size=spec.hidden_size,
        depth=spec.depth,
        num_heads=spec.num_heads,
        mlp_ratio=spec.mlp_ratio,
        class_dropout_prob=spec.class_dropout_prob,
        num_classes=spec.num_classes,
        learn_sigma=spec.learn_sigma,
    )


def count_parameters(model: DiT, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def describe_model(model: DiT) -> dict[str, Any]:
    patch_size = getattr(model, "patch_size", None)
    num_patches = int(getattr(model.x_embedder, "num_patches", 0))
    tokens_per_side = int(round(num_patches**0.5)) if num_patches else None
    image_size = (
        tokens_per_side * int(patch_size) if tokens_per_side and patch_size else None
    )
    return {
        "class": type(model).__name__,
        "parameters": count_parameters(model, trainable_only=False),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "image_size": image_size,
        "patch_size": int(patch_size) if patch_size is not None else None,
        "in_channels": int(getattr(model, "in_channels", 0)),
        "out_channels": int(getattr(model, "out_channels", 0)),
        "num_heads": int(getattr(model, "num_heads", 0)),
        "num_patches": num_patches,
    }


def check_forward(
    model: DiT,
    batch_size: int = 2,
    image_size: int = 20,
    in_channels: int = 17,
    timesteps: int = 1000,
    device: str | torch.device | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    target = torch.device(device or "cpu")
    model = model.to(target)
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(
        batch_size, in_channels, image_size, image_size, generator=generator
    ).to(target)
    t = torch.randint(0, timesteps, (batch_size,), device=target, dtype=torch.long)
    y = torch.zeros(batch_size, dtype=torch.long, device=target)
    with torch.no_grad():
        output = model(x, t, y)
    return {
        "input_shape": list(x.shape),
        "timestep_shape": list(t.shape),
        "label_shape": list(y.shape),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "finite": bool(torch.isfinite(output).all().item()),
        "min": float(output.min().item()),
        "max": float(output.max().item()),
    }
