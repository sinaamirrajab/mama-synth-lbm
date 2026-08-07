from __future__ import annotations

import copy
import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler

BridgeVariant = Literal["direct_peak"]
PredictionTarget = Literal["source_minus_target", "remaining_target", "clean_target"]


def load_flow_scheduler(pretrained_model_name_or_path: str, subfolder: str = "scheduler") -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)


def sample_bridge_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    *,
    timestep_sampling: str,
    n_samples: int,
    device: torch.device,
    deterministic_sigma_count: int = 501,
    deterministic_sigma_values: list[float] | None = None,
    deterministic_sigma_sampling: str = "uniform",
) -> torch.Tensor:
    if timestep_sampling == "uniform":
        indices = torch.randint(0, scheduler.config.num_train_timesteps, (n_samples,), device="cpu")
        return scheduler.timesteps[indices].to(device=device)

    if timestep_sampling != "deterministic_sigmas":
        raise ValueError(f"Unsupported timestep sampling mode: {timestep_sampling!r}.")

    if deterministic_sigma_values:
        requested = np.asarray(deterministic_sigma_values, dtype=np.float32)
    else:
        if deterministic_sigma_count < 2:
            raise ValueError("deterministic_sigma_count must be at least 2.")
        requested = np.linspace(1.0, 0.0, deterministic_sigma_count, dtype=np.float32)

    schedule_len = min(int(scheduler.sigmas.shape[0]), int(scheduler.timesteps.shape[0]))
    schedule_sigmas = scheduler.sigmas[:schedule_len].to(device=device, dtype=torch.float32)
    schedule_timesteps = scheduler.timesteps[:schedule_len].to(device=device)
    requested_sigmas = torch.as_tensor(requested, device=device, dtype=torch.float32)
    matched_indices = torch.argmin((schedule_sigmas[:, None] - requested_sigmas[None, :]).abs(), dim=0)
    matched_timesteps = schedule_timesteps[matched_indices]

    positive_timesteps = matched_timesteps[requested_sigmas > 1e-6]
    if positive_timesteps.numel() == 0:
        raise ValueError("deterministic sigma grid must contain at least one positive value.")
    if deterministic_sigma_sampling == "uniform":
        indices = torch.randint(0, positive_timesteps.shape[0], (n_samples,), device=device)
        return positive_timesteps[indices]
    if deterministic_sigma_sampling != "repeat":
        raise ValueError("deterministic_sigma_sampling must be 'uniform' or 'repeat'.")
    repeats = math.ceil(n_samples / positive_timesteps.shape[0])
    return positive_timesteps.repeat(repeats)[:n_samples]


def get_bridge_sigmas(
    scheduler: FlowMatchEulerDiscreteScheduler,
    timesteps: torch.Tensor,
    *,
    n_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    schedule_sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device=device)
    indices: list[int] = []
    for timestep in timesteps.to(device=device, dtype=schedule_timesteps.dtype).reshape(-1):
        matches = (schedule_timesteps == timestep).nonzero(as_tuple=False)
        if matches.numel():
            indices.append(int(matches[0].item()))
        else:
            indices.append(int(torch.argmin((schedule_timesteps - timestep).abs()).item()))
    sigmas = schedule_sigmas[indices].flatten()
    while sigmas.ndim < n_dim:
        sigmas = sigmas.unsqueeze(-1)
    return sigmas


def get_bridge_endpoints(
    source_latents: torch.Tensor,
    target_latents: torch.Tensor,
    bridge_variant: BridgeVariant,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bridge_variant != "direct_peak":
        raise ValueError("The minimal release trainer supports only bridge_variant='direct_peak'.")
    return source_latents, target_latents


def build_noisy_bridge_sample(
    *,
    source_latents: torch.Tensor,
    target_latents: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
    bridge_noise_sigma: float,
    bridge_variant: BridgeVariant,
) -> tuple[torch.Tensor, torch.Tensor]:
    bridge_start, bridge_target = get_bridge_endpoints(source_latents, target_latents, bridge_variant)
    sigmas = get_bridge_sigmas(
        scheduler,
        timesteps,
        n_dim=target_latents.ndim,
        dtype=target_latents.dtype,
        device=target_latents.device,
    )
    z_sigma = sigmas * bridge_start + (1.0 - sigmas) * bridge_target
    if bridge_noise_sigma > 0:
        z_sigma = z_sigma + bridge_noise_sigma * (sigmas * (1.0 - sigmas)).clamp_min(0.0).sqrt() * torch.randn_like(target_latents)
    return z_sigma, sigmas


def compute_bridge_prediction_target(
    *,
    prediction_target: PredictionTarget,
    noisy_latents: torch.Tensor,
    bridge_start_latents: torch.Tensor,
    bridge_target_latents: torch.Tensor,
) -> torch.Tensor:
    if prediction_target == "source_minus_target":
        return bridge_start_latents.float() - bridge_target_latents.float()
    if prediction_target == "remaining_target":
        return bridge_target_latents.float() - noisy_latents.float()
    if prediction_target == "clean_target":
        return bridge_target_latents.float()
    raise ValueError(f"Unknown prediction target: {prediction_target!r}.")


def compute_latent_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    latent_loss_type: str,
    sigmas: torch.Tensor | None = None,
    sigma_weighting: str = "none",
    sigma_min: float = 0.05,
) -> torch.Tensor:
    error = prediction.float() - target.float()
    if latent_loss_type == "mse":
        loss = error.square()
    elif latent_loss_type == "l1":
        loss = error.abs()
    elif latent_loss_type == "l1_mse":
        loss = 0.5 * (error.abs() + error.square())
    else:
        raise ValueError(f"Unknown latent loss type: {latent_loss_type!r}.")

    if sigma_weighting == "inverse_clamped":
        if sigmas is None:
            raise ValueError("sigma_weighting='inverse_clamped' requires sigmas.")
        loss = loss / sigmas.float().clamp_min(float(sigma_min))
    elif sigma_weighting != "none":
        raise ValueError(f"Unknown sigma weighting: {sigma_weighting!r}.")
    return loss.mean()


def compute_bridge_prediction_loss(
    *,
    model_pred: torch.Tensor,
    source_latents: torch.Tensor,
    target_latents: torch.Tensor,
    noisy_latents: torch.Tensor,
    sigmas: torch.Tensor,
    bridge_variant: BridgeVariant,
    prediction_target: PredictionTarget,
    latent_loss_type: str,
    latent_loss_sigma_weighting: str,
    latent_loss_sigma_min: float,
) -> torch.Tensor:
    bridge_start, bridge_target = get_bridge_endpoints(source_latents, target_latents, bridge_variant)
    target = compute_bridge_prediction_target(
        prediction_target=prediction_target,
        noisy_latents=noisy_latents,
        bridge_start_latents=bridge_start,
        bridge_target_latents=bridge_target,
    )
    return compute_latent_prediction_loss(
        model_pred,
        target,
        latent_loss_type=latent_loss_type,
        sigmas=sigmas,
        sigma_weighting=latent_loss_sigma_weighting,
        sigma_min=latent_loss_sigma_min,
    )


def predict_clean_latents(
    *,
    model_output: torch.Tensor,
    noisy_latents: torch.Tensor,
    prediction_target: PredictionTarget,
) -> torch.Tensor:
    if prediction_target == "remaining_target":
        return noisy_latents.float() + model_output.float()
    if prediction_target == "clean_target":
        return model_output.float()
    raise ValueError("source_minus_target does not provide a direct clean-target estimate without sigma.")


def build_inference_sigmas(num_inference_steps: int) -> np.ndarray:
    num_inference_steps = max(int(num_inference_steps), 1)
    return np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=np.float32)


@torch.no_grad()
def sample_lbm_latents(
    *,
    unet: torch.nn.Module,
    scheduler: FlowMatchEulerDiscreteScheduler,
    source_latents: torch.Tensor,
    conditioning_latents: torch.Tensor,
    num_inference_steps: int,
    bridge_noise_sigma: float,
    bridge_variant: BridgeVariant,
    prediction_target: PredictionTarget,
    generator: torch.Generator,
    use_autocast: bool,
) -> torch.Tensor:
    inference_scheduler = copy.deepcopy(scheduler)
    inference_scheduler.set_timesteps(sigmas=build_inference_sigmas(num_inference_steps), device=source_latents.device)
    sample, _ = get_bridge_endpoints(source_latents, source_latents, bridge_variant)
    last_clean_target = None

    for idx, timestep in enumerate(inference_scheduler.timesteps):
        timestep_batch = timestep.to(device=source_latents.device).repeat(sample.shape[0])
        denoiser_input = (
            inference_scheduler.scale_model_input(sample, timestep)
            if hasattr(inference_scheduler, "scale_model_input")
            else sample
        )
        model_input = torch.cat([denoiser_input, conditioning_latents.to(dtype=denoiser_input.dtype)], dim=1)
        with torch.autocast(source_latents.device.type, enabled=use_autocast):
            prediction = unet(model_input, timestep_batch, return_dict=False)[0]
        current_sigmas = get_bridge_sigmas(
            inference_scheduler,
            timestep_batch,
            n_dim=sample.ndim,
            dtype=sample.dtype,
            device=sample.device,
        )
        if prediction_target == "clean_target":
            last_clean_target = prediction.float()
            scheduler_prediction = ((sample.float() - prediction.float()) / current_sigmas.float().clamp_min(1e-4)).to(dtype=prediction.dtype)
        elif prediction_target == "remaining_target":
            last_clean_target = sample.float() + prediction.float()
            scheduler_prediction = (-prediction.float() / current_sigmas.float().clamp_min(1e-4)).to(dtype=prediction.dtype)
        else:
            scheduler_prediction = prediction

        sample = inference_scheduler.step(
            scheduler_prediction,
            timestep,
            sample,
            generator=generator,
            return_dict=False,
        )[0]
        if bridge_noise_sigma > 0 and idx < len(inference_scheduler.timesteps) - 1:
            next_timestep = inference_scheduler.timesteps[idx + 1].to(device=sample.device).repeat(sample.shape[0])
            next_sigmas = get_bridge_sigmas(
                inference_scheduler,
                next_timestep,
                n_dim=sample.ndim,
                dtype=sample.dtype,
                device=sample.device,
            )
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            sample = sample + bridge_noise_sigma * (next_sigmas * (1.0 - next_sigmas)).clamp_min(0.0).sqrt() * noise

    return last_clean_target if last_clean_target is not None else sample
