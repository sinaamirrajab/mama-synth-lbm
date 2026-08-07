#!/usr/bin/env python3
"""MAMA-SYNTH direct-peak LBM submission entrypoint."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from runtime_lbm import LBMPredictor, resolve_lbm_inference_schedule, resolve_runtime_precision

# ---------------------------------------------------------------------------
# Path constants  (override via environment for local testing)
# ---------------------------------------------------------------------------
INPUT_PATH = Path(os.environ.get("MAMA_INPUT_DIR", "/input"))
OUTPUT_PATH = Path(os.environ.get("MAMA_OUTPUT_DIR", "/output"))

# Interface slugs — must match the challenge phase configuration on GC.
INPUT_SLUG = os.environ.get("MAMA_INPUT_SLUG", "pre-contrast-dce-mri-slice-breast")
OUTPUT_SLUG = os.environ.get(
    "MAMA_PREDICTION_SLUG", "synthetic-contrast-dce-mri-slice-breast"
)


def _find_input_image() -> Path:
    """Return the single input image file for this job."""
    search_dir = INPUT_PATH / "images" / INPUT_SLUG
    candidates: list[str] = []
    for ext in ("*.mha", "*.nii.gz", "*.nii"):
        candidates.extend(glob(str(search_dir / ext)))

    if not candidates:
        diag_lines: list[str] = [
            f"No image file found in: {search_dir}",
            f"INPUT_SLUG used: '{INPUT_SLUG}'",
            "",
            "Directory tree under /input/images (if it exists):",
        ]
        images_dir = INPUT_PATH / "images"
        if images_dir.exists():
            for entry in sorted(images_dir.rglob("*")):
                diag_lines.append(f"  {entry}")
        else:
            diag_lines.append(f"  {images_dir} does not exist!")
            diag_lines.append("")
            diag_lines.append("Contents of /input:")
            if INPUT_PATH.exists():
                for entry in sorted(INPUT_PATH.iterdir()):
                    diag_lines.append(f"  {entry}")
            else:
                diag_lines.append(f"  {INPUT_PATH} does not exist!")

        diag_lines += [
            "",
            "How to fix:",
            "  1. Check that the input interface slug in your GC algorithm",
            f"     settings exactly matches '{INPUT_SLUG}'.",
            "  2. Verify the phase's input interface type is 'Image'.",
            "  3. Override the slug at runtime: MAMA_INPUT_SLUG=<correct-slug>",
        ]
        raise FileNotFoundError("\n".join(diag_lines))

    if len(candidates) > 1:
        print(
            f"WARNING: {len(candidates)} files found in {search_dir}; using the first one.",
            file=sys.stderr,
        )
    return Path(candidates[0])


@lru_cache(maxsize=1)
def _get_predictor() -> LBMPredictor:
    return LBMPredictor(Path(__file__).resolve().parent)


def _log_predictor_config(predictor: LBMPredictor) -> None:
    config = predictor.config
    requested_steps = int(os.environ.get("MAMA_INFERENCE_STEPS", config.validation_inference_steps))
    print("Model:")
    if config.ensemble_models:
        print(
            f"  Ensemble        : {len(config.ensemble_models)} models / "
            f"{config.ensemble_strategy}"
        )
        for member_config in predictor.model_configs:
            member_steps = int(
                os.environ.get("MAMA_INFERENCE_STEPS", member_config.validation_inference_steps)
            )
            member_schedule = resolve_lbm_inference_schedule(
                member_config,
                num_inference_steps=member_steps,
            )
            sigmas = member_schedule.scheduler_sigmas
            if len(sigmas) <= 8:
                sigma_summary = str(list(sigmas))
            else:
                sigma_summary = f"{len(sigmas)} values from {sigmas[0]:.6g} to {sigmas[-1]:.6g}"
            print(
                "    - "
                f"{member_config.model_id or 'model'}: "
                f"step={member_config.checkpoint_dir.name}, "
                f"weight={member_config.weight:.6g}, "
                f"seed={member_config.seed}, "
                f"schedule={member_schedule.source}, sigmas={sigma_summary}"
            )
    else:
        schedule = resolve_lbm_inference_schedule(config, num_inference_steps=requested_steps)
    print(f"  Checkpoint      : {config.checkpoint_dir.name}")
    print(f"  UNet init       : {config.unet_init}")
    print(f"  Prediction      : {config.bridge_variant} / {config.prediction_target}")
    print(f"  Conditioning    : {config.conditioning_mode}")
    print(f"  Normalization   : {config.normalization_mode}")
    print(f"  VAE dir         : {config.vae_dir}")
    if config.vae_source_model_name_or_path is not None:
        print(f"  VAE source      : {config.vae_source_model_name_or_path}")
    print(f"  VAE variant     : {config.vae_variant}")
    if config.normalization_mode == "fixed_peak_zscore_window":
        print(
            "  Fixed z-window  : "
            f"[{config.fixed_peak_window_lower_zscore:.6f}, "
            f"{config.fixed_peak_window_upper_zscore:.6f}]"
        )
        if config.background_zscore is not None:
            print(f"  Background z    : {config.background_zscore:.6f}")
    if config.normalization_mode == "predicted_peak_p9995_logblend_a025_guard075_150":
        peak_cfg = config.predicted_peak_upper_config
        print("  Peak upper MLP  : " f"{peak_cfg.get('model_rel_path')}")
        print(
            "  Peak upper rule : "
            f"alpha={peak_cfg.get('logblend_alpha')}, "
            f"guard=[{peak_cfg.get('guard_min_delta_ratio')}, {peak_cfg.get('guard_max_delta_ratio')}]"
        )
        print(f"  Latent scale    : {config.latent_scale:.8f}")
        print(f"  Seg provider    : {config.segmentation_provider}")
    print(f"  Mixed precision : {config.mixed_precision}")
    print(f"  Decoded gray    : {config.decoded_gray_mode}")
    resolved_device = torch.device(
        os.environ.get("MAMA_DEVICE")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    resolved_precision = resolve_runtime_precision(resolved_device, config.mixed_precision)
    print(
        "  Runtime dtype   : "
        f"requested={resolved_precision.requested}, "
        f"resolved={resolved_precision.resolved}, "
        f"dtype={resolved_precision.dtype}, device={resolved_device}"
    )
    if config.output_clip_percentiles is not None:
        print(f"  Output clip     : percentiles={config.output_clip_percentiles}")
    if not config.ensemble_models:
        print(f"  Sigma source    : {schedule.source}")
        print(f"  Inference sigmas: {list(schedule.scheduler_sigmas)}")
    if config.unet_init == "pretrained_text":
        print(f"  Text encoder    : {config.text_encoder_dir}")
        print(f"  Tokenizer       : {config.tokenizer_dir}")


def _squeezed_image_array(image: sitk.Image) -> np.ndarray:
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    return array


def _print_array_stats(label: str, array: np.ndarray, *, background_value: float | None) -> None:
    values = array[np.isfinite(array)]
    if values.size == 0:
        print(f"  {label} stats: no finite values")
        return
    stats = [
        float(values.min()),
        float(np.percentile(values, 1.0)),
        float(np.percentile(values, 50.0)),
        float(np.percentile(values, 99.0)),
        float(values.max()),
    ]
    print(
        f"  {label} z stats: "
        f"min={stats[0]:.6f}, p1={stats[1]:.6f}, "
        f"p50={stats[2]:.6f}, p99={stats[3]:.6f}, max={stats[4]:.6f}"
    )
    if background_value is not None:
        background_fraction = float(np.isclose(array, float(background_value), atol=1e-6).mean())
        print(f"  {label} background fraction: {background_fraction:.6f}")


def run() -> int:
    """Main entry point."""
    print("=" * 50)
    print("MAMA-SYNTH Direct-Peak LBM Submission")
    print("=" * 50)

    predictor = _get_predictor()
    _log_predictor_config(predictor)

    input_file = _find_input_image()
    print(f"Input : {input_file}")

    image = sitk.ReadImage(str(input_file))
    print(
        f"  Size   : {image.GetSize()}\n"
        f"  Spacing: {image.GetSpacing()}\n"
        f"  Origin : {image.GetOrigin()}"
    )
    arr = _squeezed_image_array(image)
    print(f"  input z-score range: [{float(arr.min()):.3f}, {float(arr.max()):.3f}]")

    output_image = predictor.predict_sitk(image)
    config = predictor.config
    background_value = config.background_zscore
    _print_array_stats("Input", arr, background_value=background_value)
    _print_array_stats("Output", _squeezed_image_array(output_image), background_value=background_value)
    print(f"  output z-score range: [{float(_squeezed_image_array(output_image).min()):.3f}, {float(_squeezed_image_array(output_image).max()):.3f}]")
    output_dir = OUTPUT_PATH / "images" / OUTPUT_SLUG
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "output.mha"

    sitk.WriteImage(output_image, str(output_file), useCompression=True)
    print(f"Output: {output_file}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
