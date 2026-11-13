"""Shared helpers for building the diffusion process consistently across
training, sampling and diagnosis.

The important invariant is that the noise schedule and the prediction target
(epsilon / x0 / v) must be identical at training and sampling time, otherwise
`pred_xstart` and the reverse trajectory silently disagree.
"""

from typing import Any

import torch

from diffusion import create_diffusion
from diffusion.gaussian_diffusion import ModelMeanType


DIFFUSION_CONFIG_KEYS = (
    "noise_schedule",
    "predict_xstart",
    "predict_v",
    "learn_sigma",
)

DEFAULT_NOISE_SCHEDULE = "linear"


def diffusion_settings(config: dict[str, Any]) -> dict[str, Any]:
    predict_xstart = bool(config.get("predict_xstart", False))
    predict_v = bool(config.get("predict_v", False))
    if predict_xstart and predict_v:
        raise ValueError("predict_xstart and predict_v are mutually exclusive")
    return {
        "noise_schedule": str(config.get("noise_schedule", DEFAULT_NOISE_SCHEDULE)),
        "predict_xstart": predict_xstart,
        "predict_v": predict_v,
        "learn_sigma": bool(config.get("learn_sigma", False)),
    }


def make_diffusion(config: dict[str, Any], timestep_respacing: str = "") -> Any:
    settings = diffusion_settings(config)
    return create_diffusion(
        timestep_respacing,
        noise_schedule=settings["noise_schedule"],
        predict_xstart=settings["predict_xstart"],
        predict_v=settings["predict_v"],
        learn_sigma=settings["learn_sigma"],
    )


def merge_diffusion_config(
    config: dict[str, Any], checkpoint_config: dict[str, Any] | None
) -> dict[str, Any]:
    """Overlay the diffusion settings stored in a checkpoint onto `config`.

    Architecture keys are left untouched; only the keys that decide the noise
    schedule and prediction target are copied so that sampling/diagnosis match
    how the checkpoint was trained.
    """
    if not isinstance(checkpoint_config, dict):
        return config
    merged = dict(config)
    for key in DIFFUSION_CONFIG_KEYS:
        if key in checkpoint_config:
            merged[key] = checkpoint_config[key]
    return merged


def sample_timesteps(
    batch_size: int,
    num_timesteps: int,
    mode: str = "uniform",
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample training timesteps.

    ``uniform`` is the classic uniform draw; ``logitnorm`` draws
    ``t = sigmoid(N(0, 1)) * T`` which puts more mass on the middle of the
    trajectory (SD3 / logit-normal schedule).
    """
    mode = (mode or "uniform").lower()
    if mode == "uniform":
        t = torch.randint(0, num_timesteps, (batch_size,), generator=generator)
    elif mode == "logitnorm":
        u = torch.randn(batch_size, generator=generator)
        t = (torch.sigmoid(u) * num_timesteps).long().clamp_(0, num_timesteps - 1)
    else:
        raise ValueError(f"unknown t_sample mode: {mode}")
    if device is not None:
        t = t.to(device)
    return t


def snr_loss_weight(diffusion: Any, t: torch.Tensor, gamma: float) -> torch.Tensor:
    """Min-SNR-gamma loss weight `min(SNR, gamma) / SNR` per sample.

    Only meaningful for epsilon prediction, where it down-weights the easy
    low-noise samples and gives the hard high-noise samples relatively more
    gradient.
    """
    abar = torch.as_tensor(
        diffusion.alphas_cumprod, dtype=torch.float32, device=t.device
    )[t]
    snr = abar / (1.0 - abar)
    return torch.clamp(snr, max=float(gamma)) / snr


def diffusion_loss(
    diffusion: Any,
    model: torch.nn.Module,
    x0: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    min_snr_gamma: float | None = None,
) -> torch.Tensor:
    terms = diffusion.training_losses(model, x0, t, {"y": y})
    losses = terms["loss"]
    if (
        min_snr_gamma is not None
        and min_snr_gamma > 0
        and diffusion.model_mean_type == ModelMeanType.EPSILON
    ):
        losses = losses * snr_loss_weight(diffusion, t, min_snr_gamma)
    return losses.mean()


def model_output_to_eps(
    diffusion: Any, x_t: torch.Tensor, t: torch.Tensor, model_output: torch.Tensor
) -> torch.Tensor:
    mean_type = diffusion.model_mean_type
    if mean_type == ModelMeanType.EPSILON:
        return model_output
    if mean_type == ModelMeanType.START_X:
        return diffusion._predict_eps_from_xstart(x_t, t, model_output)
    return diffusion._predict_eps_from_v(x_t, t, model_output)


def model_output_to_x0(
    diffusion: Any, x_t: torch.Tensor, t: torch.Tensor, model_output: torch.Tensor
) -> torch.Tensor:
    mean_type = diffusion.model_mean_type
    if mean_type == ModelMeanType.START_X:
        return model_output
    if mean_type == ModelMeanType.EPSILON:
        return diffusion._predict_xstart_from_eps(x_t, t, model_output)
    return diffusion._predict_xstart_from_v(x_t, t, model_output)
