#!/usr/bin/env python3
"""Offline direct-peak LBM runtime for the MAMA-SYNTH submission container."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, UNet2DConditionModel, UNet2DModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

SOURCE_MINMAX_VAE_ENCODING_MODE = "source_minmax_pre_peak__positive_true_minmax_enhancement"
FIXED_PEAK_ZSCORE_WINDOW_VAE_ENCODING_MODE = (
    "fixed_peak_zscore_window_p999_pre_peak__positive_true_minmax_enhancement"
)
FIXED_SOURCE_TARGET_ZSCORE_WINDOW_VAE_ENCODING_MODE = "fixed_source_target_zscore_window_source_p9995_target_p9995"
EXPECTED_MANIFEST_VAE_ENCODING_MODE = SOURCE_MINMAX_VAE_ENCODING_MODE
EXPECTED_MANIFEST_VAE_ENCODING_MODES = {
    "source_minmax": SOURCE_MINMAX_VAE_ENCODING_MODE,
    "fixed_peak_zscore_window": FIXED_PEAK_ZSCORE_WINDOW_VAE_ENCODING_MODE,
    "fixed_source_target_zscore_window": FIXED_SOURCE_TARGET_ZSCORE_WINDOW_VAE_ENCODING_MODE,
}
PREDICTED_PEAK_UPPER_LOGBLEND_MODE = "predicted_peak_p9995_logblend_a025_guard075_150"
ADAPTIVE_SOURCE_TARGET_ZSCORE_WINDOW_PREFIX = "adaptive_source_target_zscore_window"
SUPPORTED_NORMALIZATION_MODES = frozenset(
    {*EXPECTED_MANIFEST_VAE_ENCODING_MODES, PREDICTED_PEAK_UPPER_LOGBLEND_MODE}
)
SUPPORTED_MIXED_PRECISION = frozenset({"auto", "no", "fp16", "bf16"})
SUPPORTED_DECODED_GRAY_MODES = frozenset({"channel_0", "luminance"})
SUPPORTED_CONDITIONING_MODES = frozenset({"source", "source_only", "source_segmentation", "source_plus_segmentation"})
SUPPORTED_SEGMENTATION_PROVIDERS = frozenset({"zero_mask", "nnunet_ensemble_best"})


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_optional_float_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null"}:
            return None
        if "," in text and not text.startswith("["):
            value = [item.strip() for item in text.split(",")]
        else:
            value = json.loads(text)
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return None
    return tuple(float(item) for item in value)


def _normalize_optional_percentile_pair(value: Any) -> tuple[float, float] | None:
    pair = _normalize_optional_float_tuple(value)
    if pair is None:
        return None
    if len(pair) != 2:
        raise ValueError(f"Expected two percentile values, got {pair!r}.")
    lower, upper = float(pair[0]), float(pair[1])
    if not (0.0 <= lower < upper <= 100.0):
        raise ValueError(f"Invalid percentile clip bounds: {pair!r}.")
    return lower, upper


def _normalize_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null", "nan"}:
            return None
        value = text
    return float(value)


def _normalize_normalization_mode(value: Any) -> str:
    return "source_minmax" if value is None else str(value).strip()


def _normalize_optional_string(value: Any) -> str | None:
    if value in {None, "", "none", "None", "null", "NaN", "nan"}:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text


def _normalize_mixed_precision(value: Any) -> str:
    text = "no" if value is None else str(value).strip().lower()
    aliases = {
        "": "no",
        "none": "no",
        "null": "no",
        "false": "no",
        "0": "no",
        "fp32": "no",
        "float32": "no",
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
    }
    text = aliases.get(text, text)
    if text not in SUPPORTED_MIXED_PRECISION:
        raise ValueError(
            "mixed_precision must be one of "
            f"{sorted(SUPPORTED_MIXED_PRECISION)}, got {value!r}."
        )
    return text


def _normalize_decoded_gray_mode(value: Any) -> str:
    text = "channel_0" if value is None else str(value).strip().lower()
    aliases = {"channel0": "channel_0", "first_channel": "channel_0", "rgb_channel_0": "channel_0"}
    text = aliases.get(text, text)
    if text not in SUPPORTED_DECODED_GRAY_MODES:
        raise ValueError(
            "decoded_gray_mode must be one of "
            f"{sorted(SUPPORTED_DECODED_GRAY_MODES)}, got {value!r}."
        )
    return text


@dataclass(frozen=True)
class SubmissionRuntimeConfig:
    app_root: Path
    model_id: str | None
    weight: float
    checkpoint_dir: Path
    unet_dir: Path
    scheduler_dir: Path
    vae_dir: Path
    text_encoder_dir: Path | None
    tokenizer_dir: Path | None
    model_family: str
    unet_init: str
    target_kind: str
    bridge_target_kind: str
    conditioning_mode: str
    bridge_variant: str
    prediction_target: str
    residual_base: str
    residual_reconstruction: str
    allow_noisy_clean_aux_losses: bool
    validation_inference_steps: int
    timestep_sampling: str | None
    deterministic_sigma_count: int | None
    deterministic_sigma_values: tuple[float, ...] | None
    bridge_noise_sigma: float
    mixed_precision: str
    decoded_gray_mode: str
    training_mixed_precision: str | None
    latent_cache_mixed_precision: str | None
    latent_cache_latent_kind: str | None
    latent_cache_vae_model_name_or_path: str | None
    latent_cache_vae_subfolder: str | None
    latent_cache_vae_variant: str | None
    variant: str | None
    vae_source_model_name_or_path: str | None
    vae_revision: str | None
    vae_variant: str | None
    image_size: int
    normalization_mode: str
    background_zscore: float | None
    fixed_peak_window_lower_zscore: float | None
    fixed_peak_window_upper_zscore: float | None
    fixed_source_window_lower_zscore: float | None
    fixed_source_window_upper_zscore: float | None
    fixed_target_window_lower_zscore: float | None
    fixed_target_window_upper_zscore: float | None
    latent_scale: float | None
    segmentation_provider: str
    predicted_peak_upper_config: dict[str, Any]
    validation_manifest_path: str | None
    manifest_vae_encoding_mode: str | None
    seed: int
    ensemble: bool
    ensemble_strategy: str
    ensemble_models: tuple[dict[str, Any], ...]
    output_clip_percentiles: tuple[float, float] | None

    @classmethod
    def from_file(cls, path: str | Path) -> "SubmissionRuntimeConfig":
        config_path = Path(path).resolve()
        payload = _load_json(config_path)
        app_root = config_path.parent
        return cls.from_payload(payload, app_root)

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        app_root: str | Path,
    ) -> "SubmissionRuntimeConfig":
        app_root = Path(app_root).resolve()
        checkpoint_dir = (app_root / payload["checkpoint_rel_dir"]).resolve()
        vae_dir = (app_root / payload["vae_dir"]).resolve()
        text_encoder_dir = (
            (app_root / payload["text_encoder_dir"]).resolve()
            if payload.get("text_encoder_dir") not in {None, "", "none", "null"}
            else None
        )
        tokenizer_dir = (
            (app_root / payload["tokenizer_dir"]).resolve()
            if payload.get("tokenizer_dir") not in {None, "", "none", "null"}
            else None
        )
        unet_dir = checkpoint_dir / "unet"
        scheduler_dir = checkpoint_dir / "scheduler"

        config = cls(
            app_root=app_root,
            model_id=_normalize_optional_string(payload.get("model_id")),
            weight=float(payload.get("weight", 1.0)),
            checkpoint_dir=checkpoint_dir,
            unet_dir=unet_dir,
            scheduler_dir=scheduler_dir,
            vae_dir=vae_dir,
            text_encoder_dir=text_encoder_dir,
            tokenizer_dir=tokenizer_dir,
            model_family=str(payload["model_family"]),
            unet_init=str(payload.get("unet_init", "random")),
            target_kind=str(payload["target_kind"]),
            bridge_target_kind=str(payload.get("bridge_target_kind", payload["target_kind"])),
            conditioning_mode=str(payload["conditioning_mode"]),
            bridge_variant=str(payload["bridge_variant"]),
            prediction_target=str(payload["prediction_target"]),
            residual_base=str(payload.get("residual_base", "auto")),
            residual_reconstruction=str(payload.get("residual_reconstruction", "signed")),
            allow_noisy_clean_aux_losses=bool(payload.get("allow_noisy_clean_aux_losses", False)),
            validation_inference_steps=int(payload["validation_inference_steps"]),
            timestep_sampling=(
                None
                if payload.get("timestep_sampling") in {None, "", "none", "null"}
                else str(payload["timestep_sampling"])
            ),
            deterministic_sigma_count=(
                int(payload["deterministic_sigma_count"])
                if payload.get("deterministic_sigma_count") is not None
                else None
            ),
            deterministic_sigma_values=_normalize_optional_float_tuple(payload.get("deterministic_sigma_values")),
            bridge_noise_sigma=float(payload.get("bridge_noise_sigma", 0.0)),
            mixed_precision=_normalize_mixed_precision(payload.get("mixed_precision", "fp16")),
            decoded_gray_mode=_normalize_decoded_gray_mode(payload.get("decoded_gray_mode", "channel_0")),
            training_mixed_precision=_normalize_optional_string(payload.get("training_mixed_precision")),
            latent_cache_mixed_precision=_normalize_optional_string(payload.get("latent_cache_mixed_precision")),
            latent_cache_latent_kind=_normalize_optional_string(payload.get("latent_cache_latent_kind")),
            latent_cache_vae_model_name_or_path=_normalize_optional_string(
                payload.get("latent_cache_vae_model_name_or_path")
            ),
            latent_cache_vae_subfolder=_normalize_optional_string(payload.get("latent_cache_vae_subfolder")),
            latent_cache_vae_variant=_normalize_optional_string(payload.get("latent_cache_vae_variant")),
            variant=(
                None
                if payload.get("variant") in {None, "", "none", "null"}
                else str(payload["variant"])
            ),
            vae_source_model_name_or_path=(
                None
                if payload.get("vae_source_model_name_or_path") in {None, "", "none", "null"}
                else str(payload["vae_source_model_name_or_path"])
            ),
            vae_revision=(
                None
                if payload.get("vae_revision") in {None, "", "none", "null"}
                else str(payload["vae_revision"])
            ),
            vae_variant=(
                None
                if payload.get("vae_variant", payload.get("variant")) in {None, "", "none", "null"}
                else str(payload.get("vae_variant", payload.get("variant")))
            ),
            image_size=int(payload.get("image_size", 512)),
            normalization_mode=_normalize_normalization_mode(
                payload.get("normalization_mode", "source_minmax")
            ),
            background_zscore=(
                None if payload.get("background_zscore") is None else float(payload["background_zscore"])
            ),
            fixed_peak_window_lower_zscore=_normalize_optional_float(
                payload.get("fixed_peak_window_lower_zscore")
            ),
            fixed_peak_window_upper_zscore=_normalize_optional_float(
                payload.get("fixed_peak_window_upper_zscore")
            ),
            fixed_source_window_lower_zscore=_normalize_optional_float(
                payload.get("fixed_source_window_lower_zscore", payload.get("fixed_peak_window_lower_zscore"))
            ),
            fixed_source_window_upper_zscore=_normalize_optional_float(
                payload.get("fixed_source_window_upper_zscore", payload.get("fixed_peak_window_upper_zscore"))
            ),
            fixed_target_window_lower_zscore=_normalize_optional_float(
                payload.get("fixed_target_window_lower_zscore", payload.get("fixed_peak_window_lower_zscore"))
            ),
            fixed_target_window_upper_zscore=_normalize_optional_float(
                payload.get("fixed_target_window_upper_zscore", payload.get("fixed_peak_window_upper_zscore"))
            ),
            latent_scale=_normalize_optional_float(payload.get("latent_scale")),
            segmentation_provider=str(payload.get("segmentation_provider", "zero_mask")),
            predicted_peak_upper_config=dict(payload.get("predicted_peak_upper_config") or {}),
            validation_manifest_path=(
                None
                if payload.get("validation_manifest_path") in {None, "", "none", "null"}
                else str(payload["validation_manifest_path"])
            ),
            manifest_vae_encoding_mode=(
                None
                if payload.get("manifest_vae_encoding_mode") in {None, "", "none", "null"}
                else str(payload["manifest_vae_encoding_mode"])
            ),
            seed=int(payload.get("seed", 42)),
            ensemble=bool(payload.get("ensemble", False)),
            ensemble_strategy=str(payload.get("ensemble_strategy", "single")),
            ensemble_models=tuple(payload.get("ensemble_models") or ()),
            output_clip_percentiles=_normalize_optional_percentile_pair(
                payload.get("output_clip_percentiles")
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.model_family != "lbm":
            raise ValueError(f"Only LBM checkpoints are supported, got {self.model_family!r}.")
        if self.bridge_variant != "direct_peak":
            raise ValueError(f"Only direct_peak checkpoints are supported, got {self.bridge_variant!r}.")
        if self.target_kind != "peak" or self.bridge_target_kind != "peak":
            raise ValueError(
                "The staged submission only supports peak-target direct-peak checkpoints."
            )
        if self.conditioning_mode not in SUPPORTED_CONDITIONING_MODES:
            raise ValueError(
                f"Only conditioning modes {sorted(SUPPORTED_CONDITIONING_MODES)} are supported, "
                f"got {self.conditioning_mode!r}."
            )
        if self.segmentation_provider not in SUPPORTED_SEGMENTATION_PROVIDERS:
            raise ValueError(
                f"Only segmentation providers {sorted(SUPPORTED_SEGMENTATION_PROVIDERS)} are supported, "
                f"got {self.segmentation_provider!r}."
            )
        if self.prediction_target not in {"source_minus_target", "remaining_target", "clean_target"}:
            raise ValueError(
                "The staged submission only supports prediction_target='source_minus_target', "
                "prediction_target='remaining_target', or prediction_target='clean_target'."
            )
        _normalize_mixed_precision(self.mixed_precision)
        _normalize_decoded_gray_mode(self.decoded_gray_mode)
        if self.latent_cache_latent_kind is not None and self.latent_cache_latent_kind != "mode":
            raise ValueError(
                "Runtime encodes VAE posterior.mode(), but the staged latent cache records "
                f"latent_cache_latent_kind={self.latent_cache_latent_kind!r}. Restage with a mode cache."
            )
        if self.normalization_mode not in SUPPORTED_NORMALIZATION_MODES:
            raise ValueError(
                "The staged submission only supports these normalization modes: "
                f"{sorted(SUPPORTED_NORMALIZATION_MODES)}. Got {self.normalization_mode!r}."
            )
        if self.normalization_mode == "fixed_peak_zscore_window":
            if self.fixed_peak_window_lower_zscore is None or self.fixed_peak_window_upper_zscore is None:
                raise ValueError(
                    "fixed_peak_zscore_window submissions must set fixed_peak_window_lower_zscore "
                    "and fixed_peak_window_upper_zscore in submission_config.json."
                )
            if self.fixed_peak_window_upper_zscore <= self.fixed_peak_window_lower_zscore:
                raise ValueError(
                    "fixed_peak_zscore_window upper bound must be greater than the lower bound: "
                    f"{self.fixed_peak_window_upper_zscore} <= {self.fixed_peak_window_lower_zscore}."
                )
        if self.normalization_mode == "fixed_source_target_zscore_window":
            required_values = {
                "fixed_source_window_lower_zscore": self.fixed_source_window_lower_zscore,
                "fixed_source_window_upper_zscore": self.fixed_source_window_upper_zscore,
                "fixed_target_window_lower_zscore": self.fixed_target_window_lower_zscore,
                "fixed_target_window_upper_zscore": self.fixed_target_window_upper_zscore,
            }
            missing = [name for name, value in required_values.items() if value is None]
            if missing:
                raise ValueError(
                    "fixed_source_target_zscore_window submissions must set " + ", ".join(missing)
                )
            if self.fixed_source_window_upper_zscore <= self.fixed_source_window_lower_zscore:
                raise ValueError("fixed source window upper bound must be greater than lower bound")
            if self.fixed_target_window_upper_zscore <= self.fixed_target_window_lower_zscore:
                raise ValueError("fixed target window upper bound must be greater than lower bound")
        if self.normalization_mode == PREDICTED_PEAK_UPPER_LOGBLEND_MODE:
            required_keys = {
                "model_rel_path",
                "lower_zscore",
                "adaptive_source_group_threshold",
                "adaptive_low_target_upper_zscore",
                "adaptive_high_target_upper_zscore",
                "logblend_alpha",
                "guard_min_delta_ratio",
                "guard_max_delta_ratio",
            }
            missing = sorted(required_keys - set(self.predicted_peak_upper_config))
            if missing:
                raise ValueError(
                    "predicted peak-upper normalization is missing config keys: "
                    + ", ".join(missing)
                )
            if self.latent_scale is None or self.latent_scale <= 0:
                raise ValueError("predicted peak-upper submissions must record a positive latent_scale.")
        if self.validation_manifest_path is None:
            raise ValueError(
                "submission_config.json must record validation_manifest_path for compatibility checks."
            )
        if "challenge_peak_slice_latents_full" in self.validation_manifest_path:
            raise ValueError(
                "The staged checkpoint still points to challenge_peak_slice_latents_full. "
                "Restage a submission-compatible checkpoint instead."
            )
        if self.normalization_mode == PREDICTED_PEAK_UPPER_LOGBLEND_MODE:
            if not str(self.manifest_vae_encoding_mode or "").startswith(ADAPTIVE_SOURCE_TARGET_ZSCORE_WINDOW_PREFIX):
                raise ValueError(
                    "The staged checkpoint is not submission-compatible: predicted peak-upper "
                    f"normalization expects a manifest_vae_encoding_mode starting with "
                    f"{ADAPTIVE_SOURCE_TARGET_ZSCORE_WINDOW_PREFIX!r}, got "
                    f"{self.manifest_vae_encoding_mode!r}."
                )
        else:
            expected_mode = EXPECTED_MANIFEST_VAE_ENCODING_MODES[self.normalization_mode]
            if self.manifest_vae_encoding_mode != expected_mode:
                raise ValueError(
                    "The staged checkpoint is not submission-compatible: expected "
                    f"manifest_vae_encoding_mode={expected_mode!r}, got "
                    f"{self.manifest_vae_encoding_mode!r}."
                )
        if not self.vae_dir.is_dir():
            raise FileNotFoundError(f"Offline VAE directory not found: {self.vae_dir}")
        vae_weight_candidates = [
            self.vae_dir / "diffusion_pytorch_model.safetensors",
            self.vae_dir / "diffusion_pytorch_model.fp16.safetensors",
            self.vae_dir / "diffusion_pytorch_model.bin",
        ]
        if not (self.vae_dir / "config.json").exists() or not any(path.exists() for path in vae_weight_candidates):
            raise FileNotFoundError(
                "Offline VAE assets must include config.json and a diffusion_pytorch_model weight file "
                f"under {self.vae_dir}."
            )
        if not self.unet_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint UNet directory not found: {self.unet_dir}")
        if not self.scheduler_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint scheduler directory not found: {self.scheduler_dir}")
        if self.unet_init == "pretrained_text":
            if self.text_encoder_dir is None or not self.text_encoder_dir.is_dir():
                raise FileNotFoundError(
                    f"Offline text encoder directory not found: {self.text_encoder_dir}"
                )
            if self.tokenizer_dir is None or not self.tokenizer_dir.is_dir():
                raise FileNotFoundError(
                    f"Offline tokenizer directory not found: {self.tokenizer_dir}"
                )


@dataclass(frozen=True)
class LoadedModelBundle:
    config: SubmissionRuntimeConfig
    peak_upper_normalizer: "PredictedPeakUpperNormalizer | None"
    vae: AutoencoderKL
    unet: nn.Module
    scheduler: FlowMatchEulerDiscreteScheduler
    device: torch.device
    requested_mixed_precision: str
    resolved_mixed_precision: str
    weight_dtype: torch.dtype
    use_autocast: bool


@dataclass(frozen=True)
class ResolvedRuntimePrecision:
    requested: str
    resolved: str
    dtype: torch.dtype


@dataclass(frozen=True)
class ResolvedLBMInferenceSchedule:
    source: str
    sigma_grid: tuple[float, ...] | None
    scheduler_sigmas: tuple[float, ...]


class PeakUpperMLP(nn.Module):
    """Small source-histogram regressor used by the predicted peak-upper normalizer."""

    def __init__(self, input_dim: int, hidden: tuple[int, int] = (128, 64), dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, int(hidden[0])),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden[0]), int(hidden[1])),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden[1]), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PredictedPeakUpperNormalizer:
    """Predicts a per-slice target upper z-score from source-only histogram features."""

    def __init__(self, config: SubmissionRuntimeConfig) -> None:
        payload = dict(config.predicted_peak_upper_config)
        model_path = (config.app_root / str(payload["model_rel_path"])).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Predicted peak-upper MLP not found: {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.lower_zscore = float(payload.get("lower_zscore", checkpoint.get("lower_zscore", -0.48832086263243285)))
        self.hist_edges = np.asarray(checkpoint["hist_edges"], dtype=np.float32)
        self.feature_columns = list(checkpoint["feature_columns"])
        self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        self.feature_std[self.feature_std < 1e-6] = 1.0
        hidden_raw = checkpoint.get("mlp_hidden", (128, 64))
        hidden = (int(hidden_raw[0]), int(hidden_raw[1]))
        dropout = float(checkpoint.get("mlp_dropout", 0.0))
        self.model = PeakUpperMLP(len(self.feature_columns), hidden=hidden, dropout=dropout)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.group_threshold = float(payload["adaptive_source_group_threshold"])
        self.low_target_upper = float(payload["adaptive_low_target_upper_zscore"])
        self.high_target_upper = float(payload["adaptive_high_target_upper_zscore"])
        self.logblend_alpha = float(payload["logblend_alpha"])
        self.guard_min_delta_ratio = float(payload["guard_min_delta_ratio"])
        self.guard_max_delta_ratio = float(payload["guard_max_delta_ratio"])

    def _foreground_from_source(self, source_zscore: np.ndarray) -> np.ndarray:
        foreground = source_zscore.astype(np.float32) > self.lower_zscore + 1e-4
        if int(foreground.sum()) < 16:
            foreground = np.ones_like(source_zscore, dtype=bool)
        return foreground

    def _feature_dict(self, source_zscore: np.ndarray) -> dict[str, float]:
        source_zscore = source_zscore.astype(np.float32)
        foreground = self._foreground_from_source(source_zscore)
        values = source_zscore[foreground].astype(np.float32)
        if values.size < 16:
            values = source_zscore.reshape(-1).astype(np.float32)
        hist_counts, _ = np.histogram(values, bins=self.hist_edges)
        hist = hist_counts.astype(np.float32) / max(float(hist_counts.sum()), 1.0)
        output = {f"hist_{idx:03d}": float(value) for idx, value in enumerate(hist.tolist())}
        output.update(
            {
                "source_mean": float(values.mean()),
                "source_std": float(values.std()),
                "source_median": float(np.median(values)),
                "source_p90": float(np.percentile(values, 90.0)),
                "source_p95": float(np.percentile(values, 95.0)),
                "source_p99": float(np.percentile(values, 99.0)),
                "source_p99p5": float(np.percentile(values, 99.5)),
                "source_p99p9": float(np.percentile(values, 99.9)),
                "source_min": float(values.min()),
                "source_max": float(values.max()),
                "source_fg_fraction": float(foreground.mean()),
                "source_below_hist_fraction": float((values < float(self.hist_edges[0])).mean()),
                "source_above_hist_fraction": float((values > float(self.hist_edges[-1])).mean()),
            }
        )
        return output

    def predict_upper(self, source_zscore_for_features: np.ndarray) -> float:
        features = self._feature_dict(source_zscore_for_features)
        vector = np.asarray([features[name] for name in self.feature_columns], dtype=np.float32)
        standardized = (vector - self.feature_mean) / self.feature_std
        with torch.inference_mode():
            pred_log_delta = float(self.model(torch.from_numpy(standardized[None, :])).item())
        predicted_upper = self.lower_zscore + float(np.exp(pred_log_delta))
        predicted_upper = max(predicted_upper, self.lower_zscore + 1e-3)
        source_p99p5 = float(features["source_p99p5"])
        adaptive_upper = self.high_target_upper if source_p99p5 >= self.group_threshold else self.low_target_upper
        adaptive_delta = max(adaptive_upper - self.lower_zscore, 1e-3)
        predicted_delta = max(predicted_upper - self.lower_zscore, 1e-3)
        blended_delta = float(
            np.exp(
                np.log(adaptive_delta)
                + self.logblend_alpha * (np.log(predicted_delta) - np.log(adaptive_delta))
            )
        )
        guarded_delta = float(
            np.clip(
                blended_delta,
                self.guard_min_delta_ratio * adaptive_delta,
                self.guard_max_delta_ratio * adaptive_delta,
            )
        )
        return float(self.lower_zscore + guarded_delta)


def cuda_bf16_supported(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    device_index = torch.cuda.current_device() if device.index is None else int(device.index)
    current_device = torch.cuda.current_device()
    try:
        if hasattr(torch.cuda, "is_bf16_supported"):
            if device_index != current_device:
                torch.cuda.set_device(device_index)
            return bool(torch.cuda.is_bf16_supported())
    except Exception:
        pass
    finally:
        if torch.cuda.current_device() != current_device:
            torch.cuda.set_device(current_device)
    try:
        major, _minor = torch.cuda.get_device_capability(device_index)
        return major >= 8
    except Exception:
        return False


def resolve_runtime_precision(device: torch.device, mixed_precision: str) -> ResolvedRuntimePrecision:
    override = os.environ.get("MAMA_MIXED_PRECISION")
    requested = _normalize_mixed_precision(override if override is not None else mixed_precision)

    if device.type == "cuda":
        if requested == "auto":
            if cuda_bf16_supported(device):
                return ResolvedRuntimePrecision(requested=requested, resolved="bf16", dtype=torch.bfloat16)
            return ResolvedRuntimePrecision(requested=requested, resolved="fp16", dtype=torch.float16)
        if requested == "bf16":
            if cuda_bf16_supported(device):
                return ResolvedRuntimePrecision(requested=requested, resolved="bf16", dtype=torch.bfloat16)
            return ResolvedRuntimePrecision(requested=requested, resolved="fp16", dtype=torch.float16)
        if requested == "fp16":
            return ResolvedRuntimePrecision(requested=requested, resolved="fp16", dtype=torch.float16)
    return ResolvedRuntimePrecision(requested=requested, resolved="no", dtype=torch.float32)


def resolve_weight_dtype(device: torch.device, mixed_precision: str) -> torch.dtype:
    return resolve_runtime_precision(device, mixed_precision).dtype


class ConstantTextConditionedUNet2DModel(nn.Module):
    """Expose an SD text-conditioned UNet with the no-text UNet call signature."""

    def __init__(self, unet: UNet2DConditionModel, encoder_hidden_states: torch.Tensor) -> None:
        super().__init__()
        self.unet = unet
        self.register_buffer("encoder_hidden_states", encoder_hidden_states.detach().float(), persistent=False)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, return_dict: bool = True, **kwargs):
        encoder_hidden_states = self.encoder_hidden_states.to(device=sample.device, dtype=sample.dtype)
        if encoder_hidden_states.shape[0] != sample.shape[0]:
            encoder_hidden_states = encoder_hidden_states.expand(sample.shape[0], -1, -1)
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


def build_constant_prompt_embeds(config: SubmissionRuntimeConfig) -> torch.Tensor:
    if config.tokenizer_dir is None or config.text_encoder_dir is None:
        raise ValueError("pretrained_text inference requires tokenizer_dir and text_encoder_dir.")
    tokenizer = CLIPTokenizer.from_pretrained(
        str(config.tokenizer_dir),
        local_files_only=True,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        str(config.text_encoder_dir),
        variant=config.variant,
        local_files_only=True,
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    inputs = tokenizer(
        [""],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        prompt_embeds = text_encoder(inputs.input_ids, return_dict=False)[0]
    return prompt_embeds


def load_unet_for_config(
    config: SubmissionRuntimeConfig,
    *,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> nn.Module:
    if config.unet_init == "pretrained_text":
        unet = UNet2DConditionModel.from_pretrained(
            str(config.unet_dir),
            local_files_only=True,
        )
        prompt_embeds = build_constant_prompt_embeds(config)
        wrapped_unet = ConstantTextConditionedUNet2DModel(
            unet=unet,
            encoder_hidden_states=prompt_embeds,
        )
        return wrapped_unet.to(device=device, dtype=weight_dtype)
    return UNet2DModel.from_pretrained(str(config.unet_dir), local_files_only=True).to(
        device=device,
        dtype=weight_dtype,
    )


def load_model_bundle(
    config: SubmissionRuntimeConfig,
    *,
    device_override: str | None = None,
) -> LoadedModelBundle:
    resolved_device = torch.device(
        device_override
        or os.environ.get("MAMA_DEVICE")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    precision = resolve_runtime_precision(resolved_device, config.mixed_precision)
    weight_dtype = precision.dtype
    use_autocast = resolved_device.type == "cuda" and weight_dtype in {torch.float16, torch.bfloat16}

    vae = AutoencoderKL.from_pretrained(
        str(config.vae_dir),
        variant=config.vae_variant,
        local_files_only=True,
    ).to(device=resolved_device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.eval()

    unet = load_unet_for_config(config, device=resolved_device, weight_dtype=weight_dtype)
    unet.eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(config.scheduler_dir),
        local_files_only=True,
    )

    peak_upper_normalizer = (
        PredictedPeakUpperNormalizer(config)
        if config.normalization_mode == PREDICTED_PEAK_UPPER_LOGBLEND_MODE
        else None
    )

    return LoadedModelBundle(
        config=config,
        peak_upper_normalizer=peak_upper_normalizer,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        device=resolved_device,
        requested_mixed_precision=precision.requested,
        resolved_mixed_precision=precision.resolved,
        weight_dtype=weight_dtype,
        use_autocast=use_autocast,
    )


def _squeeze_image_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got array with shape {array.shape}.")
    return array.astype(np.float32)


def _resample_sitk_image(
    image: sitk.Image,
    output_shape: tuple[int, int],
    *,
    is_mask: bool,
) -> sitk.Image:
    if image.GetDimension() != 2:
        raise ValueError(f"Expected a 2-D image, got dimension {image.GetDimension()}.")

    output_height, output_width = output_shape
    input_width, input_height = image.GetSize()
    input_spacing_x, input_spacing_y = image.GetSpacing()

    output_size = [int(output_width), int(output_height)]
    output_spacing = [
        float(input_spacing_x) * float(input_width) / float(output_width),
        float(input_spacing_y) * float(input_height) / float(output_height),
    ]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(output_size)
    resample_filter.SetOutputSpacing(output_spacing)
    resample_filter.SetOutputOrigin(image.GetOrigin())
    resample_filter.SetOutputDirection(image.GetDirection())
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetDefaultPixelValue(0)
    resample_filter.SetOutputPixelType(image.GetPixelID())
    resample_filter.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return resample_filter.Execute(image)


def _resize_float_array(
    array: np.ndarray,
    output_shape: tuple[int, int],
    *,
    resample: Image.Resampling,
) -> np.ndarray:
    image = Image.fromarray(array.astype(np.float32), mode="F")
    if image.size != (int(output_shape[1]), int(output_shape[0])):
        image = image.resize((int(output_shape[1]), int(output_shape[0])), resample)
    return np.asarray(image).astype(np.float32)


def _foreground_mask(array: np.ndarray, background_value: float, atol: float = 1e-6) -> np.ndarray:
    mask = ~np.isclose(array, background_value, atol=atol)
    if np.any(mask):
        return mask
    return np.ones_like(array, dtype=bool)


def _safe_minmax(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 1e-6
    lower = float(np.min(values.astype(np.float32)))
    upper = float(np.max(values.astype(np.float32)))
    if upper <= lower:
        upper = lower + 1e-6
    return lower, upper


def _normalize_minmax(
    array: np.ndarray,
    lower: float,
    upper: float,
    reference_mask: np.ndarray,
) -> np.ndarray:
    denominator = max(float(upper) - float(lower), 1e-6)
    output = np.zeros_like(array, dtype=np.float32)
    output[reference_mask] = (
        (array.astype(np.float32)[reference_mask] - float(lower)) / denominator
    )
    output[~reference_mask] = 0.0
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def _denormalize_minmax(
    array01: np.ndarray,
    lower: float,
    upper: float,
    reference_mask: np.ndarray,
    background_value: float,
) -> np.ndarray:
    denominator = max(float(upper) - float(lower), 1e-6)
    restored = np.clip(array01.astype(np.float32), 0.0, 1.0) * denominator + float(lower)
    restored = restored.astype(np.float32)
    restored[~reference_mask] = float(background_value)
    return restored


def estimate_background_value(
    source_zscore: np.ndarray,
    configured_background: float | None,
) -> float:
    if configured_background is not None:
        return float(configured_background)
    border = np.concatenate(
        [
            source_zscore[0, :].ravel(),
            source_zscore[-1, :].ravel(),
            source_zscore[:, 0].ravel(),
            source_zscore[:, -1].ravel(),
        ]
    )
    return float(np.median(border.astype(np.float32)))


def prepare_source_minmax_input(
    source_zscore: np.ndarray,
    *,
    background_value: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    source_mask = _foreground_mask(source_zscore, background_value)
    lower, upper = _safe_minmax(source_zscore[source_mask].astype(np.float32))
    source_model_01 = _normalize_minmax(source_zscore, lower, upper, source_mask)
    return source_model_01, source_mask, lower, upper


def prepare_fixed_peak_zscore_window_input(
    source_zscore: np.ndarray,
    *,
    background_value: float,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if upper <= lower:
        raise ValueError(f"Invalid fixed peak z-score window: lower={lower}, upper={upper}.")
    source_mask = _foreground_mask(source_zscore, background_value)
    denominator = max(float(upper) - float(lower), 1e-6)
    source_model_01 = np.clip(
        (source_zscore.astype(np.float32) - float(lower)) / denominator,
        0.0,
        1.0,
    ).astype(np.float32)
    return source_model_01, source_mask, float(lower), float(upper)


def prepare_predicted_peak_upper_input(
    source_zscore: np.ndarray,
    *,
    background_value: float,
    config: SubmissionRuntimeConfig,
    normalizer: PredictedPeakUpperNormalizer,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    source_mask = _foreground_mask(source_zscore, background_value)
    source_for_features = _resize_float_array(
        source_zscore.astype(np.float32),
        output_shape=(config.image_size, config.image_size),
        resample=Image.Resampling.BILINEAR,
    )
    lower = float(normalizer.lower_zscore)
    upper = float(normalizer.predict_upper(source_for_features))
    if upper <= lower:
        raise ValueError(f"Invalid predicted peak upper: lower={lower}, upper={upper}.")
    denominator = max(upper - lower, 1e-6)
    source_model_01 = np.clip((source_zscore.astype(np.float32) - lower) / denominator, 0.0, 1.0).astype(np.float32)
    return source_model_01, source_mask, lower, upper


def _nnunet_predict_command() -> list[str]:
    executable = shutil.which("nnUNetv2_predict")
    if executable:
        return [executable]
    # `python -m nnunetv2.inference.predict_from_raw_data` does not dispatch
    # the CLI entrypoint in some nnU-Net versions; it can fall through to a
    # hardcoded example path. Call the real entrypoint explicitly instead.
    return [
        sys.executable,
        "-c",
        "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()",
    ]


def run_nnunet_ensemble_best_segmentation(
    source_zscore: np.ndarray,
    config: SubmissionRuntimeConfig,
    *,
    output_shape: tuple[int, int],
    source_image: sitk.Image | None = None,
) -> np.ndarray:
    nnunet_root = config.app_root / "resources" / "nnunet"
    nnunet_results = nnunet_root / "nnUNet_results"
    nnunet_raw = nnunet_root / "nnUNet_raw"
    nnunet_preprocessed = nnunet_root / "nnUNet_preprocessed"
    model_root = nnunet_results / "Dataset001_BreastMRI" / "nnUNetTrainer__nnUNetPlans__2d"
    required_files = [model_root / "dataset.json", model_root / "plans.json"] + [
        model_root / f"fold_{fold}" / "checkpoint_best.pth" for fold in range(5)
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing staged nnU-Net resources: " + ", ".join(missing))

    source_zscore = _squeeze_image_array(source_zscore)
    case_id = "mama_case"
    with tempfile.TemporaryDirectory(prefix="mama_nnunet_in_", dir="/tmp") as tmp_in, tempfile.TemporaryDirectory(
        prefix="mama_nnunet_out_", dir="/tmp"
    ) as tmp_out:
        tmp_in_dir = Path(tmp_in)
        tmp_out_dir = Path(tmp_out)
        input_path = tmp_in_dir / f"{case_id}_0000.mha"
        if source_image is not None:
            # Preserve spacing/origin/direction for nnU-Net preprocessing. Recreating
            # the image from only the array silently changes spacing to 1.0.
            image_to_write = sitk.Image(source_image)
            if image_to_write.GetDimension() != 2:
                image_to_write = sitk.GetImageFromArray(source_zscore.astype(np.float32))
        else:
            image_to_write = sitk.GetImageFromArray(source_zscore.astype(np.float32))
        sitk.WriteImage(image_to_write, str(input_path))

        env = os.environ.copy()
        env["nnUNet_raw"] = str(nnunet_raw)
        env["nnUNet_preprocessed"] = str(nnunet_preprocessed)
        env["nnUNet_results"] = str(nnunet_results)
        cmd = _nnunet_predict_command() + [
            "-i",
            str(tmp_in_dir),
            "-o",
            str(tmp_out_dir),
            "-d",
            "001",
            "-c",
            "2d",
            "-tr",
            "nnUNetTrainer",
            "-f",
            "0",
            "1",
            "2",
            "3",
            "4",
            "-chk",
            "checkpoint_best.pth",
            "-npp",
            "1",
            "-nps",
            "1",
        ]
        if torch.cuda.is_available():
            cmd += ["-device", "cuda"]
        else:
            cmd += ["-device", "cpu"]
        print("Running nnU-Net segmentation:", " ".join(cmd))
        result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"nnUNetv2_predict failed with exit code {result.returncode}")

        predicted_path = tmp_out_dir / f"{case_id}.mha"
        if not predicted_path.exists():
            raise FileNotFoundError(f"nnU-Net output mask not found: {predicted_path}")
        mask = _squeeze_image_array(sitk.GetArrayFromImage(sitk.ReadImage(str(predicted_path))))

    mask01 = (mask > 0.5).astype(np.float32)
    return _resize_float_array(mask01, output_shape=output_shape, resample=Image.Resampling.NEAREST).astype(np.float32)


def build_segmentation_model01(
    config: SubmissionRuntimeConfig,
    *,
    source_zscore: np.ndarray,
    output_shape: tuple[int, int],
    source_image: sitk.Image | None = None,
) -> np.ndarray | None:
    if config.conditioning_mode not in {"source_segmentation", "source_plus_segmentation"}:
        return None
    if config.segmentation_provider == "zero_mask":
        return np.zeros(output_shape, dtype=np.float32)
    if config.segmentation_provider == "nnunet_ensemble_best":
        return run_nnunet_ensemble_best_segmentation(
            source_zscore,
            config,
            output_shape=output_shape,
            source_image=source_image,
        )
    raise ValueError(f"Unsupported segmentation_provider: {config.segmentation_provider!r}")


def restore_peak_zscore_from_source_minmax(
    generated_peak_01: np.ndarray,
    *,
    lower: float,
    upper: float,
    shared_mask: np.ndarray,
    background_value: float,
) -> np.ndarray:
    return _denormalize_minmax(
        generated_peak_01,
        lower,
        upper,
        shared_mask,
        background_value,
    )


def restore_peak_zscore_from_fixed_peak_zscore_window(
    generated_peak_01: np.ndarray,
    *,
    lower: float,
    upper: float,
    shared_mask: np.ndarray,
    background_value: float,
) -> np.ndarray:
    return _denormalize_minmax(
        generated_peak_01,
        lower,
        upper,
        shared_mask,
        background_value,
    )


def resize_model_space_array(
    array01: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    return _resize_float_array(
        np.clip(array01.astype(np.float32), 0.0, 1.0),
        output_shape=output_shape,
        resample=Image.Resampling.BILINEAR,
    )


def resolve_output_clip_percentiles(config: SubmissionRuntimeConfig) -> tuple[float, float] | None:
    override = os.environ.get("MAMA_OUTPUT_CLIP_PERCENTILES")
    if override is not None:
        return _normalize_optional_percentile_pair(override)
    return config.output_clip_percentiles


def apply_final_output_percentile_clip(
    prediction_zscore: np.ndarray,
    source_zscore: np.ndarray,
    *,
    config: SubmissionRuntimeConfig,
) -> np.ndarray:
    percentiles = resolve_output_clip_percentiles(config)
    if percentiles is None:
        return prediction_zscore.astype(np.float32)

    background_value = estimate_background_value(source_zscore, config.background_zscore)
    foreground_mask = _foreground_mask(source_zscore, background_value)
    finite_mask = foreground_mask & np.isfinite(prediction_zscore)
    values = prediction_zscore[finite_mask].astype(np.float32)
    if values.size == 0:
        return prediction_zscore.astype(np.float32)

    lower_value, upper_value = np.percentile(values, percentiles).astype(np.float32)
    if not np.isfinite(lower_value) or not np.isfinite(upper_value) or upper_value <= lower_value:
        return prediction_zscore.astype(np.float32)

    clipped = prediction_zscore.astype(np.float32, copy=True)
    clipped[foreground_mask] = np.clip(clipped[foreground_mask], lower_value, upper_value)
    clipped[~foreground_mask] = float(background_value)
    return clipped.astype(np.float32)


def gray01_to_rgb_tensor(array01: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(array01.astype(np.float32))[None, None, ...]
    return tensor.repeat(1, 3, 1, 1)


def encode_gray01_to_latent(
    vae: AutoencoderKL,
    array01: np.ndarray,
    *,
    device: torch.device,
    weight_dtype: torch.dtype,
    latent_scale: float | None = None,
) -> torch.Tensor:
    tensor = gray01_to_rgb_tensor(array01).to(device=device, dtype=weight_dtype)
    scale = float(vae.config.scaling_factor if latent_scale is None else latent_scale)
    with torch.inference_mode():
        posterior = vae.encode(tensor * 2.0 - 1.0, return_dict=False)[0]
        latents = posterior.mode() * scale
    return latents


def decode_latents_to_gray01(
    vae: AutoencoderKL,
    latents: torch.Tensor,
    *,
    decoded_gray_mode: str = "channel_0",
    latent_scale: float | None = None,
) -> np.ndarray:
    mode = _normalize_decoded_gray_mode(decoded_gray_mode)
    vae_param = next(vae.parameters())
    latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
    scale = float(vae.config.scaling_factor if latent_scale is None else latent_scale)
    with torch.inference_mode():
        decoded = vae.decode(latents / scale, return_dict=False)[0]
    decoded = decoded.float() / 2.0 + 0.5
    if mode == "channel_0":
        gray = decoded[:, 0:1]
    elif mode == "luminance":
        gray = 0.2989 * decoded[:, 0:1] + 0.5870 * decoded[:, 1:2] + 0.1140 * decoded[:, 2:3]
    else:
        raise AssertionError(f"Unsupported decoded gray mode after validation: {mode!r}")
    return gray.clamp(0.0, 1.0)[0, 0].cpu().numpy().astype(np.float32)


def build_inference_sigmas(num_inference_steps: int) -> np.ndarray:
    num_inference_steps = max(int(num_inference_steps), 1)
    return np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=np.float32)


def build_deterministic_sigma_grid(
    *,
    sigma_count: int | None = None,
    sigma_values: list[float] | tuple[float, ...] | None = None,
) -> np.ndarray:
    if sigma_values is not None:
        sigma_grid = np.asarray(sigma_values, dtype=np.float32).reshape(-1)
    else:
        resolved_count = 9 if sigma_count is None else int(sigma_count)
        if resolved_count < 2:
            raise ValueError("deterministic sigma count must be at least 2.")
        sigma_grid = np.linspace(1.0, 0.0, resolved_count, dtype=np.float32)

    if sigma_grid.size == 0:
        raise ValueError("deterministic sigma grid must contain at least one value.")
    if not np.all(np.isfinite(sigma_grid)):
        raise ValueError("deterministic sigma grid must contain only finite values.")
    if np.any((sigma_grid < 0.0) | (sigma_grid > 1.0)):
        raise ValueError("deterministic sigma values must lie in [0, 1].")

    sigma_grid = np.sort(sigma_grid.astype(np.float32))[::-1]
    deduplicated: list[float] = []
    for value in sigma_grid.tolist():
        if not deduplicated or not np.isclose(value, deduplicated[-1], atol=1e-6):
            deduplicated.append(float(value))
    sigma_grid = np.asarray(deduplicated, dtype=np.float32)

    if sigma_grid[0] < 1.0 - 1e-6:
        sigma_grid = np.concatenate([np.asarray([1.0], dtype=np.float32), sigma_grid])

    return sigma_grid.astype(np.float32)


def resolve_lbm_inference_schedule(
    config: SubmissionRuntimeConfig,
    *,
    num_inference_steps: int,
) -> ResolvedLBMInferenceSchedule:
    if config.timestep_sampling == "deterministic_sigmas":
        sigma_grid = build_deterministic_sigma_grid(
            sigma_count=config.deterministic_sigma_count,
            sigma_values=config.deterministic_sigma_values,
        )
        scheduler_sigmas = sigma_grid[sigma_grid > 1e-6]
        if scheduler_sigmas.size == 0:
            raise ValueError("LBM inference requires at least one positive sigma value.")
        return ResolvedLBMInferenceSchedule(
            source="checkpoint_deterministic",
            sigma_grid=tuple(float(value) for value in sigma_grid.tolist()),
            scheduler_sigmas=tuple(float(value) for value in scheduler_sigmas.tolist()),
        )

    linear_sigmas = build_inference_sigmas(num_inference_steps)
    return ResolvedLBMInferenceSchedule(
        source="linear_steps",
        sigma_grid=None,
        scheduler_sigmas=tuple(float(value) for value in linear_sigmas.tolist()),
    )


def get_bridge_sigmas(
    scheduler: FlowMatchEulerDiscreteScheduler,
    timesteps: torch.Tensor,
    *,
    n_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    scheduler_timesteps = scheduler.timesteps.to(device=device)
    scheduler_sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    indices = []
    for timestep in timesteps:
        matching = torch.nonzero(torch.isclose(scheduler_timesteps, timestep), as_tuple=False)
        if matching.numel() == 0:
            raise ValueError(f"Timestep {float(timestep)} not found in scheduler timesteps.")
        indices.append(int(matching[0].item()))
    sigma = scheduler_sigmas[indices]
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def get_bridge_endpoints(
    source_latents: torch.Tensor,
    target_latents: torch.Tensor,
    bridge_variant: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bridge_variant == "direct_peak":
        return source_latents, target_latents
    raise ValueError(f"Unsupported bridge_variant: {bridge_variant!r}")


def predict_clean_latents_from_bridge_with_config(
    model_output: torch.Tensor,
    noisy_latents: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    prediction_target: str,
    bridge_noise_sigma: float,
    bridge_start_latents: torch.Tensor | None,
    allow_noisy_clean_aux_losses: bool,
    sigma_eps: float = 1e-4,
) -> torch.Tensor | None:
    if prediction_target == "source_minus_target":
        return noisy_latents.float() - model_output.float() * sigmas.float()
    if prediction_target == "remaining_target":
        return noisy_latents.float() + model_output.float()
    if prediction_target == "vector_field":
        return None
    if prediction_target in {"clean_target", "enhancement"}:
        return model_output.float()
    if prediction_target == "noise":
        if bridge_start_latents is None:
            raise ValueError("bridge_start_latents must be provided when prediction_target='noise'.")
        if bridge_noise_sigma <= 0:
            raise ValueError("prediction_target='noise' requires bridge_noise_sigma > 0.")
        sigma = sigmas.float().clamp(sigma_eps, 1.0 - sigma_eps)
        scale = bridge_noise_sigma * (sigma * (1.0 - sigma)).clamp_min(0.0).sqrt()
        numerator = noisy_latents.float() - sigma * bridge_start_latents.float() - scale * model_output.float()
        return numerator / (1.0 - sigma).clamp_min(sigma_eps)
    raise ValueError(f"Unknown prediction_target: {prediction_target}")


def build_conditioning_latents(
    config: SubmissionRuntimeConfig,
    *,
    source_latents: torch.Tensor,
    segmentation_latents: torch.Tensor | None = None,
) -> torch.Tensor:
    if config.conditioning_mode == "source":
        return source_latents.new_empty((source_latents.shape[0], 0, *source_latents.shape[-2:]))
    if config.conditioning_mode == "source_only":
        return source_latents
    if config.conditioning_mode == "source_segmentation":
        if segmentation_latents is None:
            raise ValueError("source_segmentation conditioning requires segmentation_latents.")
        return segmentation_latents
    if config.conditioning_mode == "source_plus_segmentation":
        if segmentation_latents is None:
            raise ValueError("source_plus_segmentation conditioning requires segmentation_latents.")
        return torch.cat([source_latents, segmentation_latents], dim=1)
    raise ValueError(f"Unsupported conditioning_mode: {config.conditioning_mode!r}")


def sample_lbm_prediction(
    bundle: LoadedModelBundle,
    *,
    source_latents: torch.Tensor,
    conditioning_latents: torch.Tensor,
    num_inference_steps: int,
    generator: torch.Generator,
    inference_sigmas: np.ndarray | None = None,
) -> np.ndarray:
    scheduler = copy.deepcopy(bundle.scheduler)
    prediction_target = bundle.config.prediction_target
    sigmas = build_inference_sigmas(num_inference_steps) if inference_sigmas is None else np.asarray(
        inference_sigmas,
        dtype=np.float32,
    )
    scheduler.set_timesteps(sigmas=sigmas, device=bundle.device)

    source_latents = source_latents.to(device=bundle.device, dtype=bundle.weight_dtype)
    conditioning_latents = conditioning_latents.to(device=bundle.device, dtype=bundle.weight_dtype)
    bridge_start_latents, _ = get_bridge_endpoints(
        source_latents=source_latents,
        target_latents=source_latents,
        bridge_variant=bundle.config.bridge_variant,
    )
    sample = bridge_start_latents.to(dtype=source_latents.dtype)
    last_model_output = None
    last_clean_target_latents = None
    last_bridge_state = None
    last_sigmas = None

    with torch.inference_mode():
        for idx, timestep in enumerate(scheduler.timesteps):
            timestep_batch = timestep.to(device=bundle.device).repeat(sample.shape[0])
            denoiser_input = (
                scheduler.scale_model_input(sample, timestep)
                if hasattr(scheduler, "scale_model_input")
                else sample
            )
            model_input = torch.cat([denoiser_input, conditioning_latents.to(dtype=denoiser_input.dtype)], dim=1)
            autocast_context = (
                torch.autocast(bundle.device.type, dtype=bundle.weight_dtype)
                if bundle.use_autocast
                else nullcontext()
            )
            with autocast_context:
                prediction = bundle.unet(model_input, timestep_batch, return_dict=False)[0]
            current_sigmas = get_bridge_sigmas(
                scheduler=scheduler,
                timesteps=timestep_batch,
                n_dim=sample.ndim,
                dtype=sample.dtype,
                device=bundle.device,
            )
            if prediction_target == "clean_target":
                last_clean_target_latents = prediction
                scheduler_prediction = (
                    (sample.float() - prediction.float()) / current_sigmas.float().clamp_min(1e-4)
                ).to(dtype=prediction.dtype)
            elif prediction_target == "remaining_target":
                last_clean_target_latents = sample.float() + prediction.float()
                scheduler_prediction = (
                    -prediction.float() / current_sigmas.float().clamp_min(1e-4)
                ).to(dtype=prediction.dtype)
            else:
                scheduler_prediction = prediction
            last_model_output = prediction
            last_bridge_state = sample
            last_sigmas = current_sigmas
            sample = scheduler.step(scheduler_prediction, timestep, sample, generator=generator, return_dict=False)[0]

            if bundle.config.bridge_noise_sigma > 0 and idx < len(scheduler.timesteps) - 1:
                next_timestep = scheduler.timesteps[idx + 1].to(device=bundle.device).repeat(sample.shape[0])
                next_sigmas = get_bridge_sigmas(
                    scheduler=scheduler,
                    timesteps=next_timestep,
                    n_dim=sample.ndim,
                    dtype=sample.dtype,
                    device=bundle.device,
                )
                sample = sample + (
                    bundle.config.bridge_noise_sigma
                    * (next_sigmas * (1.0 - next_sigmas)).clamp_min(0.0).sqrt()
                    * torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
                )
                sample = sample.to(dtype=source_latents.dtype)

    if prediction_target in {"clean_target", "remaining_target"} and last_clean_target_latents is not None:
        return decode_latents_to_gray01(
            bundle.vae,
            last_clean_target_latents,
            decoded_gray_mode=bundle.config.decoded_gray_mode,
            latent_scale=bundle.config.latent_scale,
        )

    if (
        prediction_target == "source_minus_target"
        and bundle.config.bridge_noise_sigma == 0
        and last_model_output is not None
        and last_bridge_state is not None
        and last_sigmas is not None
    ):
        pred_clean_latents = predict_clean_latents_from_bridge_with_config(
            model_output=last_model_output,
            noisy_latents=last_bridge_state,
            sigmas=last_sigmas,
            prediction_target=prediction_target,
            bridge_noise_sigma=bundle.config.bridge_noise_sigma,
            bridge_start_latents=bridge_start_latents,
            allow_noisy_clean_aux_losses=bundle.config.allow_noisy_clean_aux_losses,
        )
        if pred_clean_latents is not None:
            return decode_latents_to_gray01(
                bundle.vae,
                pred_clean_latents,
                decoded_gray_mode=bundle.config.decoded_gray_mode,
                latent_scale=bundle.config.latent_scale,
            )

    return decode_latents_to_gray01(
        bundle.vae,
        sample,
        decoded_gray_mode=bundle.config.decoded_gray_mode,
        latent_scale=bundle.config.latent_scale,
    )


class LBMPredictor:
    """Thin offline predictor used by the Grand Challenge entrypoint."""

    def __init__(self, app_root: str | Path | None = None) -> None:
        resolved_root = Path(app_root).resolve() if app_root is not None else Path(__file__).resolve().parent
        self.config = SubmissionRuntimeConfig.from_file(resolved_root / "submission_config.json")
        self.model_configs = (
            tuple(
                SubmissionRuntimeConfig.from_payload(model_payload, resolved_root)
                for model_payload in self.config.ensemble_models
            )
            if self.config.ensemble_models
            else (self.config,)
        )
        if os.environ.get("MAMA_SEED") is not None and len({config.seed for config in self.model_configs}) > 1:
            raise ValueError(
                "MAMA_SEED overrides every ensemble member seed. Unset MAMA_SEED to use the "
                "distinct per-model seeds recorded in submission_config.json."
            )
        self._bundle: LoadedModelBundle | None = None

    @property
    def bundle(self) -> LoadedModelBundle:
        if self._bundle is None:
            self._bundle = load_model_bundle(self.model_configs[0])
        return self._bundle

    def _predict_array_with_bundle(
        self,
        source_zscore: np.ndarray,
        *,
        config: SubmissionRuntimeConfig,
        bundle: LoadedModelBundle,
        seed: int | None = None,
        source_image: sitk.Image | None = None,
    ) -> np.ndarray:
        source_zscore = _squeeze_image_array(source_zscore)
        background_value = estimate_background_value(source_zscore, config.background_zscore)
        if config.normalization_mode == "source_minmax":
            source_model_01, shared_mask, lower, upper = prepare_source_minmax_input(
                source_zscore,
                background_value=background_value,
            )
        elif config.normalization_mode == "fixed_peak_zscore_window":
            if (
                config.fixed_peak_window_lower_zscore is None
                or config.fixed_peak_window_upper_zscore is None
            ):
                raise ValueError("fixed_peak_zscore_window configuration is missing fixed bounds.")
            source_model_01, shared_mask, lower, upper = prepare_fixed_peak_zscore_window_input(
                source_zscore,
                background_value=background_value,
                lower=config.fixed_peak_window_lower_zscore,
                upper=config.fixed_peak_window_upper_zscore,
            )
        elif config.normalization_mode == "fixed_source_target_zscore_window":
            if (
                config.fixed_source_window_lower_zscore is None
                or config.fixed_source_window_upper_zscore is None
                or config.fixed_target_window_lower_zscore is None
                or config.fixed_target_window_upper_zscore is None
            ):
                raise ValueError("fixed_source_target_zscore_window configuration is missing source/target bounds.")
            source_model_01, shared_mask, _source_lower, _source_upper = prepare_fixed_peak_zscore_window_input(
                source_zscore,
                background_value=background_value,
                lower=config.fixed_source_window_lower_zscore,
                upper=config.fixed_source_window_upper_zscore,
            )
            lower = config.fixed_target_window_lower_zscore
            upper = config.fixed_target_window_upper_zscore
        elif config.normalization_mode == PREDICTED_PEAK_UPPER_LOGBLEND_MODE:
            if bundle.peak_upper_normalizer is None:
                raise ValueError("Predicted peak-upper normalizer was not loaded for this model.")
            source_model_01, shared_mask, lower, upper = prepare_predicted_peak_upper_input(
                source_zscore,
                background_value=background_value,
                config=config,
                normalizer=bundle.peak_upper_normalizer,
            )
        else:
            raise ValueError(f"Unsupported normalization_mode: {config.normalization_mode!r}")
        source_model_01_resized = resize_model_space_array(
            source_model_01,
            output_shape=(config.image_size, config.image_size),
        )

        source_latents = encode_gray01_to_latent(
            bundle.vae,
            source_model_01_resized,
            device=bundle.device,
            weight_dtype=bundle.weight_dtype,
            latent_scale=config.latent_scale,
        )
        segmentation_model_01 = build_segmentation_model01(
            config,
            source_zscore=source_zscore,
            output_shape=(config.image_size, config.image_size),
            source_image=source_image,
        )
        segmentation_latents = None
        if segmentation_model_01 is not None:
            segmentation_latents = encode_gray01_to_latent(
                bundle.vae,
                segmentation_model_01,
                device=bundle.device,
                weight_dtype=bundle.weight_dtype,
                latent_scale=config.latent_scale,
            )
        conditioning_latents = build_conditioning_latents(
            config,
            source_latents=source_latents,
            segmentation_latents=segmentation_latents,
        )

        resolved_schedule = resolve_lbm_inference_schedule(
            config,
            num_inference_steps=int(os.environ.get("MAMA_INFERENCE_STEPS", config.validation_inference_steps)),
        )
        generator = torch.Generator(device=bundle.device)
        generator.manual_seed(int(os.environ.get("MAMA_SEED", seed if seed is not None else config.seed)))

        generated_model_01 = sample_lbm_prediction(
            bundle,
            source_latents=source_latents,
            conditioning_latents=conditioning_latents,
            num_inference_steps=len(resolved_schedule.scheduler_sigmas),
            generator=generator,
            inference_sigmas=np.asarray(resolved_schedule.scheduler_sigmas, dtype=np.float32),
        )
        generated_model_01_original = resize_model_space_array(
            generated_model_01,
            output_shape=source_zscore.shape,
        )
        if config.normalization_mode in {"fixed_peak_zscore_window", "fixed_source_target_zscore_window", PREDICTED_PEAK_UPPER_LOGBLEND_MODE}:
            generated_peak_z = restore_peak_zscore_from_fixed_peak_zscore_window(
                generated_model_01_original,
                lower=lower,
                upper=upper,
                shared_mask=shared_mask,
                background_value=background_value,
            )
        else:
            generated_peak_z = restore_peak_zscore_from_source_minmax(
                generated_model_01_original,
                lower=lower,
                upper=upper,
                shared_mask=shared_mask,
                background_value=background_value,
            )
        generated_peak_z = generated_peak_z.astype(np.float32)
        generated_peak_z[~shared_mask] = background_value
        return generated_peak_z

    def predict_array(
        self,
        source_zscore: np.ndarray,
        *,
        seed: int | None = None,
        source_image: sitk.Image | None = None,
    ) -> np.ndarray:
        source_zscore = _squeeze_image_array(source_zscore)
        if len(self.model_configs) == 1:
            prediction = self._predict_array_with_bundle(
                source_zscore,
                config=self.model_configs[0],
                bundle=self.bundle,
                seed=seed,
                source_image=source_image,
            )
            return apply_final_output_percentile_clip(prediction, source_zscore, config=self.config)

        predictions: list[np.ndarray] = []
        weights: list[float] = []
        for member_index, config in enumerate(self.model_configs, start=1):
            resolved_schedule = resolve_lbm_inference_schedule(
                config,
                num_inference_steps=int(
                    os.environ.get("MAMA_INFERENCE_STEPS", config.validation_inference_steps)
                ),
            )
            effective_seed = int(
                os.environ.get("MAMA_SEED", seed if seed is not None else config.seed)
            )
            print(
                "Running ensemble member "
                f"{member_index}/{len(self.model_configs)}: "
                f"id={config.model_id or 'model'}, "
                f"weight={config.weight:.6g}, "
                f"seed={effective_seed}, "
                f"schedule={resolved_schedule.source}/"
                f"{len(resolved_schedule.scheduler_sigmas)} steps"
            )
            bundle = load_model_bundle(config)
            try:
                predictions.append(
                    self._predict_array_with_bundle(
                        source_zscore,
                        config=config,
                        bundle=bundle,
                        seed=seed,
                        source_image=source_image,
                    )
                )
                weights.append(float(config.weight))
            finally:
                del bundle
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        stacked = np.stack(predictions, axis=0).astype(np.float32)
        if self.config.ensemble_strategy == "weighted_mean_zscore":
            weight_array = np.asarray(weights, dtype=np.float32)
            if float(np.sum(weight_array)) <= 0:
                weight_array = np.ones_like(weight_array, dtype=np.float32)
            prediction = np.average(stacked, axis=0, weights=weight_array).astype(np.float32)
            print(
                "Ensemble aggregation: weighted_mean_zscore "
                f"weights={weight_array.tolist()} sum={float(np.sum(weight_array)):.6g}"
            )
        else:
            prediction = np.mean(stacked, axis=0).astype(np.float32)
            print(
                f"Ensemble aggregation: mean across {len(self.model_configs)} models"
            )
        return apply_final_output_percentile_clip(prediction, source_zscore, config=self.config)

    def predict_sitk(self, image: sitk.Image, *, seed: int | None = None) -> sitk.Image:
        if image.GetDimension() != 2:
            raise ValueError(f"Expected a 2-D image, got dimension {image.GetDimension()}.")

        original_array = _squeeze_image_array(sitk.GetArrayFromImage(image))
        prediction_original = self.predict_array(original_array, seed=seed, source_image=image)

        output_image = sitk.GetImageFromArray(prediction_original.astype(np.float32))
        output_image.CopyInformation(image)
        return output_image
