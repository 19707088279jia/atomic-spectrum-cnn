"""1D preprocessing and masked multilabel data utilities for NASA Z-903 spectra."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

TARGETS = ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb")
WAVELENGTHS = np.linspace(180.0, 960.0, 23401, dtype=np.float32)


def read_z903_spectrum(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read and validate one exported Z-903 wavelength/intensity vector."""
    frame = pd.read_csv(path, usecols=["wavelength", "intensity"])
    wavelengths = pd.to_numeric(frame["wavelength"], errors="coerce").to_numpy(dtype=np.float32)
    intensity = pd.to_numeric(frame["intensity"], errors="coerce").to_numpy(dtype=np.float32)
    if wavelengths.shape != (23401,) or intensity.shape != (23401,):
        raise ValueError(f"Expected 23401 channels in {path}, got {wavelengths.shape} and {intensity.shape}")
    if not np.isfinite(wavelengths).all():
        raise ValueError(f"Non-finite wavelength calibration in {path}")
    intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.allclose(wavelengths, WAVELENGTHS, rtol=0.0, atol=1e-4):
        raise ValueError(f"Unexpected wavelength calibration in {path}")
    return wavelengths, intensity


def total_area_normalize(intensity: np.ndarray) -> np.ndarray:
    """Normalize by non-negative total area, preserving wavelength positions."""
    values = np.nan_to_num(np.asarray(intensity, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    area = float(np.trapezoid(values))
    return values / area if area > 0.0 else np.zeros_like(values)


def robust_intensity_scale(intensity: np.ndarray) -> np.ndarray:
    """Center by median and scale by IQR without baseline correction."""
    values = np.nan_to_num(np.asarray(intensity, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25.0, 75.0])
    scale = float(q75 - q25)
    if scale <= 0.0:
        scale = float(np.std(values)) or 1.0
    return (values - median) / scale


def preprocess_spectrum(intensity: np.ndarray, option: Literal["area", "robust"] = "area") -> np.ndarray:
    if option == "area":
        return total_area_normalize(intensity)
    if option == "robust":
        return robust_intensity_scale(intensity)
    raise ValueError(f"Unknown preprocessing option: {option}")


def masked_binary_cross_entropy(logits: Tensor, labels: Tensor, label_mask: Tensor) -> Tensor:
    """Compute BCE only where the corresponding composition label is known."""
    if logits.shape != labels.shape or labels.shape != label_mask.shape:
        raise ValueError(f"logits, labels, and label_mask must have equal shapes, got {logits.shape}, {labels.shape}, {label_mask.shape}")
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    mask = label_mask.to(dtype=loss.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denominator


class Z903Dataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Load one averaged spectrum and six partially observed labels per row."""

    def __init__(self, manifest_csv: str | Path, preprocess: Literal["area", "robust"] = "area") -> None:
        self.manifest = pd.read_csv(manifest_csv)
        self.preprocess = preprocess
        self.label_columns = [f"{target}_raw" if target != "Mg" else "MgO_raw" for target in TARGETS]
        self.mask_columns = [f"{target}_mask" for target in TARGETS]
        required = {"spectrum_path", *self.label_columns, *self.mask_columns}
        missing = required.difference(self.manifest.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        row = self.manifest.iloc[index]
        _, intensity = read_z903_spectrum(row["spectrum_path"])
        processed = preprocess_spectrum(intensity, self.preprocess)
        labels = pd.to_numeric(row[self.label_columns], errors="coerce").to_numpy(dtype=np.float32)
        masks = pd.to_numeric(row[self.mask_columns], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        labels = np.nan_to_num(labels, nan=0.0)
        return torch.from_numpy(processed[None, :]), torch.from_numpy(labels), torch.from_numpy(masks)


def build_z903_dataloader(manifest_csv: str | Path, batch_size: int = 16, preprocess: Literal["area", "robust"] = "area", shuffle: bool = False, num_workers: int = 0) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    return DataLoader(Z903Dataset(manifest_csv, preprocess=preprocess), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


__all__ = ["TARGETS", "WAVELENGTHS", "Z903Dataset", "build_z903_dataloader", "masked_binary_cross_entropy", "preprocess_spectrum", "read_z903_spectrum", "robust_intensity_scale", "total_area_normalize"]