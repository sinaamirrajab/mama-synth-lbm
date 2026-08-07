from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from torch.utils.data import Dataset

PeakTargetKind = Literal["peak", "enhancement"]
LBMConditioningMode = Literal[
    "source",
    "source_only",
    "source_segmentation",
    "source_plus_segmentation",
]

TARGET_LATENT_KEY = {
    "peak": "peak_latent",
    "enhancement": "enhancement_latent",
}

EXTRA_CONDITIONING_KEYS = {
    "source": [],
    "source_only": ["image_0_latent"],
    "source_segmentation": ["segmentation_latent"],
    "source_plus_segmentation": ["image_0_latent", "segmentation_latent"],
}


def conditioning_channel_count(conditioning_mode: LBMConditioningMode) -> int:
    return 4 * len(EXTRA_CONDITIONING_KEYS[conditioning_mode])


def normalize_dataset_prefixes(values: list[str] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    prefixes: list[str] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            prefix = part.strip().upper()
            if prefix:
                prefixes.append(prefix)
    prefixes = list(dict.fromkeys(prefixes))
    return tuple(prefixes) if prefixes else None


def infer_dataset_id(patient_id: object) -> str:
    return str(patient_id).strip().upper().split("_", 1)[0]


def filter_manifest_by_dataset_prefixes(
    manifest: pd.DataFrame,
    prefixes: tuple[str, ...] | None,
) -> pd.DataFrame:
    if prefixes is None:
        return manifest
    if "patient_id" not in manifest.columns:
        raise KeyError("--include-datasets requires a patient_id column.")
    dataset_ids = manifest["patient_id"].map(infer_dataset_id)
    return manifest.loc[dataset_ids.isin(prefixes)].copy().reset_index(drop=True)


def split_manifest(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "split" not in manifest.columns:
        return manifest.reset_index(drop=True), manifest.iloc[0:0].copy()
    split = manifest["split"].astype(str).str.lower().str.strip()
    train = manifest.loc[split.eq("train")].copy().reset_index(drop=True)
    val = manifest.loc[split.isin({"val", "valid", "validation"})].copy().reset_index(drop=True)
    if train.empty:
        raise ValueError("No training rows found in latent manifest.")
    return train, val


def _load_torch_payload(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@lru_cache(maxsize=2048)
def load_latent_payload(path: str) -> dict:
    return _load_torch_payload(Path(path))


class LatentManifestDataset(Dataset):
    """Dataset over precomputed MAMA-SYNTH latent slices."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        manifest_dir: Path,
        target_kind: PeakTargetKind = "peak",
        conditioning_mode: LBMConditioningMode = "source_segmentation",
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.manifest_dir = Path(manifest_dir)
        self.target_kind = target_kind
        self.conditioning_mode = conditioning_mode

    def __len__(self) -> int:
        return len(self.manifest)

    def _resolve_latent_path(self, value: object) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = self.manifest_dir / path
        return path

    def __getitem__(self, index: int) -> dict:
        row = self.manifest.iloc[index]
        payload = load_latent_payload(str(self._resolve_latent_path(row["latent_path"])))

        source_latents = payload["image_0_latent"].float()
        target_latents = payload[TARGET_LATENT_KEY[self.target_kind]].float()
        conditioning_keys = EXTRA_CONDITIONING_KEYS[self.conditioning_mode]
        if conditioning_keys:
            conditioning_latents = torch.cat([payload[key].float() for key in conditioning_keys], dim=0)
        else:
            conditioning_latents = source_latents.new_empty((0, *source_latents.shape[-2:]))

        return {
            "source_latents": source_latents,
            "target_latents": target_latents,
            "conditioning_latents": conditioning_latents,
            "segmentation_latent": payload.get("segmentation_latent", torch.zeros_like(source_latents)).float(),
            "patient_id": str(row.get("patient_id", index)),
            "latent_path": str(self._resolve_latent_path(row["latent_path"])),
        }


def collate_latent_examples(examples: list[dict]) -> dict:
    return {
        "source_latents": torch.stack([example["source_latents"] for example in examples]),
        "target_latents": torch.stack([example["target_latents"] for example in examples]),
        "conditioning_latents": torch.stack([example["conditioning_latents"] for example in examples]),
        "segmentation_latent": torch.stack([example["segmentation_latent"] for example in examples]),
        "patient_id": [example["patient_id"] for example in examples],
        "latent_path": [example["latent_path"] for example in examples],
    }
