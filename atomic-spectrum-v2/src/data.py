"""Numerical CSV loading and built-in demo metadata."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from .labels import METADATA_COLUMNS, TARGETS, composition_label

EXPECTED_POINTS = 23401
WAVELENGTH_RANGE = (180.0, 960.0)
ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "data" / "demo"
METADATA_PATH = ROOT / "data" / "metadata" / "libs_metadata.xlsx"
SPLIT_DIR = ROOT / "data" / "splits"


def load_spectrum(source: str | Path | BinaryIO) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate a CSV with wavelength and intensity columns."""
    frame = pd.read_csv(source)
    required = {"wavelength", "intensity"}
    if not required.issubset(frame.columns):
        raise ValueError("CSV must contain exactly usable 'wavelength' and 'intensity' columns")
    wavelengths = pd.to_numeric(frame["wavelength"], errors="coerce").to_numpy(dtype=np.float32)
    intensity = pd.to_numeric(frame["intensity"], errors="coerce").to_numpy(dtype=np.float32)
    if wavelengths.shape != (EXPECTED_POINTS,) or intensity.shape != (EXPECTED_POINTS,):
        raise ValueError(f"Expected {EXPECTED_POINTS} wavelength/intensity rows")
    if not np.isfinite(wavelengths).all() or not np.isfinite(intensity).all():
        raise ValueError("CSV contains non-finite wavelength or intensity values")
    expected = np.linspace(*WAVELENGTH_RANGE, EXPECTED_POINTS, dtype=np.float32)
    if not np.allclose(wavelengths, expected, rtol=0.0, atol=1e-4):
        raise ValueError("CSV wavelength calibration must span 180-960 nm at the NASA Z-903 spacing")
    return wavelengths, intensity


def load_demo(name: str) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Load a built-in demo spectrum and its composition metadata."""
    metadata = pd.read_csv(DEMO_DIR / "demo_ground_truth.csv").set_index("sample_name")
    if name not in metadata.index:
        raise ValueError(f"Unknown demo sample: {name}")
    row = metadata.loc[name]
    path = DEMO_DIR / str(row["filename"])
    return (*load_spectrum(path), row)


def demo_labels(row: pd.Series) -> dict[str, str]:
    """Return composition-derived labels without treating missing values as negative."""
    labels = {}
    for element in TARGETS:
        value = pd.to_numeric(row[element], errors="coerce")
        labels[element] = composition_label(element, float(value) if pd.notna(value) else None)
    return labels


def load_metadata() -> pd.DataFrame:
    """Load canonical NASA composition metadata keyed by normalized target ID."""
    frame = pd.read_excel(METADATA_PATH, sheet_name=0)
    frame["target_id"] = frame["PELLET NAME"].astype(str).str.strip().str.lower()
    return frame.set_index("target_id")


def load_regression_split(split: str) -> pd.DataFrame:
    """Load one existing target-group split and join canonical metadata."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown regression split: {split}")
    frame = pd.read_csv(SPLIT_DIR / f"z903_{split}.csv")
    metadata = load_metadata()
    values = metadata.reindex(frame["target_id"].str.lower())
    for element, column in METADATA_COLUMNS.items():
        frame[element] = pd.to_numeric(values[column].to_numpy(), errors="coerce")
    return frame


__all__ = ["EXPECTED_POINTS", "WAVELENGTH_RANGE", "DEMO_DIR", "METADATA_PATH", "SPLIT_DIR", "load_spectrum", "load_demo", "demo_labels", "load_metadata", "load_regression_split"]
