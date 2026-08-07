from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader

from .bridge import (
    build_noisy_bridge_sample,
    compute_bridge_prediction_loss,
    load_flow_scheduler,
    sample_bridge_timesteps,
    sample_lbm_latents,
)
from .data import (
    LatentManifestDataset,
    collate_latent_examples,
    conditioning_channel_count,
    filter_manifest_by_dataset_prefixes,
    normalize_dataset_prefixes,
    split_manifest,
)
from .modeling import build_pretrained_text_unet, build_random_unet, trainable_parameter_count

logger = logging.getLogger("mama_synth_lbm.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal MAMA-SYNTH latent bridge model.")
    parser.add_argument("--latent-manifest-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained-model-name-or-path", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--unet-init", choices=["pretrained_text", "random"], default="pretrained_text")
    parser.add_argument("--conditioning-mode", choices=["source", "source_only", "source_segmentation", "source_plus_segmentation"], default="source_segmentation")
    parser.add_argument("--include-datasets", nargs="*", default=None, help="Optional patient-id prefixes, e.g. DUKE ISPY2.")
    parser.add_argument("--target-kind", choices=["peak", "enhancement"], default="peak")
    parser.add_argument("--bridge-variant", choices=["direct_peak"], default="direct_peak")
    parser.add_argument("--prediction-target", choices=["source_minus_target", "remaining_target", "clean_target"], default="remaining_target")
    parser.add_argument("--timestep-sampling", choices=["uniform", "deterministic_sigmas"], default="deterministic_sigmas")
    parser.add_argument("--deterministic-sigma-count", type=int, default=501)
    parser.add_argument("--deterministic-sigma-values", type=float, nargs="*", default=None)
    parser.add_argument("--deterministic-sigma-sampling", choices=["uniform", "repeat"], default="uniform")
    parser.add_argument("--bridge-noise-sigma", type=float, default=0.005)
    parser.add_argument("--latent-loss-type", choices=["l1", "mse", "l1_mse"], default="l1_mse")
    parser.add_argument("--latent-loss-sigma-weighting", choices=["none", "inverse_clamped"], default="none")
    parser.add_argument("--latent-loss-sigma-min", type=float, default=0.05)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler", default="constant")
    parser.add_argument("--lr-warmup-steps", type=int, default=200)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=8000)
    parser.add_argument("--validation-steps", type=int, default=1000)
    parser.add_argument("--checkpointing-steps", type=int, default=1000)
    parser.add_argument("--validation-loss-batches", type=int, default=4)
    parser.add_argument("--validation-inference-steps", type=int, default=500)
    parser.add_argument("--rollout-validation-patients", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mixed_precision_dtype(device: torch.device, mode: str) -> torch.dtype | None:
    if device.type != "cuda" or mode == "no":
        return None
    return torch.float16 if mode == "fp16" else torch.bfloat16


def autocast_context(device: torch.device, mode: str):
    dtype = mixed_precision_dtype(device, mode)
    return torch.autocast(device.type, dtype=dtype, enabled=dtype is not None)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_checkpoint(args: argparse.Namespace, unet: torch.nn.Module, scheduler, step: int) -> None:
    checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step:07d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(checkpoint_dir / "unet")
    scheduler.save_pretrained(checkpoint_dir / "scheduler")
    save_json(
        checkpoint_dir / "training_state.json",
        {
            "step": step,
            "model_family": "lbm",
            "bridge_variant": args.bridge_variant,
            "prediction_target": args.prediction_target,
            "conditioning_mode": args.conditioning_mode,
            "unet_init": args.unet_init,
            "timestep_sampling": args.timestep_sampling,
            "deterministic_sigma_count": args.deterministic_sigma_count,
            "deterministic_sigma_values": args.deterministic_sigma_values,
            "deterministic_sigma_sampling": args.deterministic_sigma_sampling,
            "bridge_noise_sigma": args.bridge_noise_sigma,
        },
    )


def append_rollout_metrics(output_dir: Path, step: int, metrics: dict[str, float]) -> None:
    if not metrics:
        return
    path = output_dir / "checkpoint_rollout_metrics.csv"
    row = {"step": int(step), **metrics}
    fieldnames = ["step", *sorted(metrics)]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def validation_loss(
    args: argparse.Namespace,
    unet: torch.nn.Module,
    scheduler,
    loader: DataLoader,
    device: torch.device,
) -> float | None:
    if len(loader) == 0:
        return None
    unet.eval()
    losses: list[float] = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.validation_loss_batches:
            break
        source = batch["source_latents"].to(device=device)
        target = batch["target_latents"].to(device=device)
        cond = batch["conditioning_latents"].to(device=device)
        timesteps = sample_bridge_timesteps(
            scheduler,
            timestep_sampling=args.timestep_sampling,
            n_samples=target.shape[0],
            device=device,
            deterministic_sigma_count=args.deterministic_sigma_count,
            deterministic_sigma_values=args.deterministic_sigma_values,
            deterministic_sigma_sampling=args.deterministic_sigma_sampling,
        )
        noisy, sigmas = build_noisy_bridge_sample(
            source_latents=source,
            target_latents=target,
            timesteps=timesteps,
            scheduler=scheduler,
            bridge_noise_sigma=args.bridge_noise_sigma,
            bridge_variant=args.bridge_variant,
        )
        with autocast_context(device, args.mixed_precision):
            pred = unet(torch.cat([noisy, cond.to(dtype=noisy.dtype)], dim=1), timesteps, return_dict=False)[0]
            loss = compute_bridge_prediction_loss(
                model_pred=pred,
                source_latents=source,
                target_latents=target,
                noisy_latents=noisy,
                sigmas=sigmas,
                bridge_variant=args.bridge_variant,
                prediction_target=args.prediction_target,
                latent_loss_type=args.latent_loss_type,
                latent_loss_sigma_weighting=args.latent_loss_sigma_weighting,
                latent_loss_sigma_min=args.latent_loss_sigma_min,
            )
        losses.append(float(loss.detach().cpu()))
    unet.train()
    return float(np.mean(losses)) if losses else None


@torch.no_grad()
def rollout_metrics(
    args: argparse.Namespace,
    unet: torch.nn.Module,
    scheduler,
    dataset: LatentManifestDataset,
    device: torch.device,
) -> dict[str, float]:
    if len(dataset) == 0 or args.rollout_validation_patients <= 0:
        return {}
    unet.eval()
    indices = list(range(min(len(dataset), args.rollout_validation_patients)))
    rollout_steps = tuple(dict.fromkeys((1, 2, 4, max(int(args.validation_inference_steps), 1))))
    buckets: dict[str, list[float]] = {}
    for idx in indices:
        example = collate_latent_examples([dataset[idx]])
        source = example["source_latents"].to(device=device)
        target = example["target_latents"].to(device=device)
        cond = example["conditioning_latents"].to(device=device)
        for steps in rollout_steps:
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + idx)
            pred = sample_lbm_latents(
                unet=unet,
                scheduler=scheduler,
                source_latents=source,
                conditioning_latents=cond,
                num_inference_steps=steps,
                bridge_noise_sigma=args.bridge_noise_sigma,
                bridge_variant=args.bridge_variant,
                prediction_target=args.prediction_target,
                generator=generator,
                use_autocast=mixed_precision_dtype(device, args.mixed_precision) is not None,
            )
            error = pred.float() - target.float()
            buckets.setdefault(f"val_rollout_{steps}_l1", []).append(float(error.abs().mean().cpu()))
            buckets.setdefault(f"val_rollout_{steps}_rmse", []).append(float(torch.sqrt(error.square().mean()).cpu()))
    unet.train()
    return {key: float(np.mean(values)) for key, values in sorted(buckets.items())}


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "args.json", vars(args))

    manifest_path = Path(args.latent_manifest_path)
    manifest = pd.read_csv(manifest_path)
    manifest = filter_manifest_by_dataset_prefixes(manifest, normalize_dataset_prefixes(args.include_datasets))
    train_df, val_df = split_manifest(manifest)
    logger.info("Manifest rows: train=%s validation=%s", len(train_df), len(val_df))

    train_dataset = LatentManifestDataset(
        train_df,
        manifest_dir=manifest_path.parent,
        target_kind=args.target_kind,
        conditioning_mode=args.conditioning_mode,
    )
    val_dataset = LatentManifestDataset(
        val_df,
        manifest_dir=manifest_path.parent,
        target_kind=args.target_kind,
        conditioning_mode=args.conditioning_mode,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_latent_examples,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=max(0, min(args.num_workers, 2)),
        collate_fn=collate_latent_examples,
    )

    if len(train_loader) == 0:
        raise ValueError("Training loader is empty. Check the manifest and train batch size.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = load_flow_scheduler(args.pretrained_model_name_or_path)
    in_channels = 4 + conditioning_channel_count(args.conditioning_mode)
    if args.unet_init == "pretrained_text":
        unet = build_pretrained_text_unet(
            args.pretrained_model_name_or_path,
            in_channels=in_channels,
            revision=args.revision,
            variant=args.variant,
            prompt="",
        )
    else:
        unet = build_random_unet(in_channels=in_channels)
    unet.to(device)
    unet.train()
    logger.info("Trainable parameters: %.2fM", trainable_parameter_count(unet) / 1e6)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.mixed_precision == "fp16")

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    while global_step < args.max_train_steps:
        for batch in train_loader:
            source = batch["source_latents"].to(device=device)
            target = batch["target_latents"].to(device=device)
            cond = batch["conditioning_latents"].to(device=device)
            timesteps = sample_bridge_timesteps(
                scheduler,
                timestep_sampling=args.timestep_sampling,
                n_samples=target.shape[0],
                device=device,
                deterministic_sigma_count=args.deterministic_sigma_count,
                deterministic_sigma_values=args.deterministic_sigma_values,
                deterministic_sigma_sampling=args.deterministic_sigma_sampling,
            )
            noisy, sigmas = build_noisy_bridge_sample(
                source_latents=source,
                target_latents=target,
                timesteps=timesteps,
                scheduler=scheduler,
                bridge_noise_sigma=args.bridge_noise_sigma,
                bridge_variant=args.bridge_variant,
            )
            with autocast_context(device, args.mixed_precision):
                pred = unet(torch.cat([noisy, cond.to(dtype=noisy.dtype)], dim=1), timesteps, return_dict=False)[0]
                loss = compute_bridge_prediction_loss(
                    model_pred=pred,
                    source_latents=source,
                    target_latents=target,
                    noisy_latents=noisy,
                    sigmas=sigmas,
                    bridge_variant=args.bridge_variant,
                    prediction_target=args.prediction_target,
                    latent_loss_type=args.latent_loss_type,
                    latent_loss_sigma_weighting=args.latent_loss_sigma_weighting,
                    latent_loss_sigma_min=args.latent_loss_sigma_min,
                )
                loss = loss / max(int(args.gradient_accumulation_steps), 1)

            scaler.scale(loss).backward()
            if (global_step + 1) % max(int(args.gradient_accumulation_steps), 1) == 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step == 1 or global_step % 25 == 0:
                logger.info("step=%s loss=%.6f lr=%.3g", global_step, float(loss.detach().cpu()), lr_scheduler.get_last_lr()[0])

            if args.validation_steps > 0 and global_step % args.validation_steps == 0:
                val_loss = validation_loss(args, unet, scheduler, val_loader, device)
                metrics = rollout_metrics(args, unet, scheduler, val_dataset, device)
                if val_loss is not None:
                    metrics["val_loss"] = val_loss
                append_rollout_metrics(output_dir, global_step, metrics)
                logger.info("validation step=%s %s", global_step, " ".join(f"{k}={v:.6f}" for k, v in metrics.items()))

            if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                save_checkpoint(args, unet, scheduler, global_step)

            if global_step >= args.max_train_steps:
                break

    if args.checkpointing_steps <= 0 or global_step % args.checkpointing_steps != 0:
        save_checkpoint(args, unet, scheduler, global_step)
    logger.info("Done. Checkpoints written under %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
