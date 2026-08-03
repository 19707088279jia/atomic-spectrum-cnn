"""Import NIST LIBS CSV files and render spectrum images."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALLOWED_CLASSES = ("Fe", "Cu", "Na", "Ca", "Mg")
SPLIT_BY_SOURCE = {"001": "train", "002": "validation", "003": "test"}
DEFAULT_WAVELENGTH_RANGE = (200.0, 600.0)
DEFAULT_GRID_POINTS = 1200


def _normalize_header(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def _apply_width_variation(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a small width change without changing the overall shape semantics."""
    if values.size < 3:
        return values

    width_scale = rng.uniform(0.9, 1.15)
    radius = max(1, int(round(2 * width_scale)))
    kernel = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(kernel**2) / (2.0 * (max(0.6, width_scale) ** 2)))
    kernel = kernel / kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def read_nist_spectrum(
    path: str | Path,
    element: str,
    grid_points: int = DEFAULT_GRID_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a NIST CSV spectrum and return sorted, interpolated data."""
    path = Path(path)
    if element not in ALLOWED_CLASSES:
        raise ValueError(f"Unsupported element: {element}")

    rows = _read_csv_rows(path)
    if len(rows) < 2:
        raise ValueError(f"CSV contains no data rows: {path}")

    header = [cell.strip() for cell in rows[0]]
    normalized_header = [_normalize_header(cell) for cell in header]

    wavelength_index = None
    intensity_index = None
    for index, name in enumerate(normalized_header):
        if name == "wavelength (nm)":
            wavelength_index = index
        if name == "sum":
            intensity_index = index

    if wavelength_index is None or intensity_index is None:
        if len(normalized_header) >= 2 and normalized_header[0] == "s" and normalized_header[1] == "sum":
            wavelength_index = 0
            intensity_index = 1
        else:
            raise ValueError(f"Could not detect wavelength/intensity columns in {path}")

    wavelengths: list[float] = []
    intensities: list[float] = []

    for row in rows[1:]:
        if not row:
            continue
        if len(row) <= max(wavelength_index, intensity_index):
            continue
        wavelength = _parse_float(row[wavelength_index])
        intensity = _parse_float(row[intensity_index])
        if wavelength is None or intensity is None:
            continue
        if not math.isfinite(wavelength) or not math.isfinite(intensity):
            continue
        if wavelength < DEFAULT_WAVELENGTH_RANGE[0] or wavelength > DEFAULT_WAVELENGTH_RANGE[1]:
            continue
        wavelengths.append(float(wavelength))
        intensities.append(float(intensity))

    if len(wavelengths) < 2:
        raise ValueError(f"Not enough valid rows in {path}")

    order = np.argsort(wavelengths)
    wavelengths_array = np.asarray(wavelengths, dtype=float)[order]
    intensities_array = np.asarray(intensities, dtype=float)[order]

    unique_mask = np.ones_like(wavelengths_array, dtype=bool)
    if wavelengths_array.size > 1:
        unique_mask[1:] = np.diff(wavelengths_array) != 0.0
    wavelengths_array = wavelengths_array[unique_mask]
    intensities_array = intensities_array[unique_mask]

    if wavelengths_array.size < 2:
        raise ValueError(f"Not enough unique wavelengths in {path}")

    grid = np.linspace(DEFAULT_WAVELENGTH_RANGE[0], DEFAULT_WAVELENGTH_RANGE[1], grid_points)
    interpolated = np.interp(grid, wavelengths_array, intensities_array)
    if np.any(np.isnan(interpolated)):
        raise ValueError(f"Interpolation produced invalid values for {path}")

    min_value = float(interpolated.min())
    max_value = float(interpolated.max())
    if max_value > min_value:
        interpolated = (interpolated - min_value) / (max_value - min_value)
    else:
        interpolated = np.zeros_like(interpolated)

    return grid, interpolated


def assign_split(source_name: str) -> str:
    """Assign a source CSV to a split based on its numeric suffix."""
    match = re.search(r"(?:_|\s)(\d{3})\.csv$", source_name)
    if not match:
        raise ValueError(f"Cannot determine split from source name: {source_name}")
    suffix = match.group(1)
    split = SPLIT_BY_SOURCE.get(suffix)
    if split is None:
        raise ValueError(f"Unsupported source index suffix: {suffix}")
    return split


def render_spectrum_image(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    output_path: str | Path,
    dpi: int = 120,
    rng: np.random.Generator | None = None,
) -> None:
    """Render a spectrum image without labels, legends, or axis decorations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if rng is None:
        rng = np.random.default_rng(0)

    width = 8.0
    height = 3.2
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    ax.plot(wavelengths, intensities, color="black", linewidth=1.1, solid_capstyle="round")
    ax.set_xlim(DEFAULT_WAVELENGTH_RANGE[0], DEFAULT_WAVELENGTH_RANGE[1])
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    if output_path.exists() and output_path.stat().st_size <= 0:
        raise AssertionError("Rendered image file is empty")


def _stable_seed(element: str, split: str, source_file: str, index: int, base_seed: int) -> int:
    payload = f"{element}:{split}:{source_file}:{index}:{base_seed}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def generate_manifest(
    manifest_path: str | Path,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Write a manifest CSV and return the rows."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "element",
                "source_file",
                "output_file",
                "split",
                "source_index",
                "Te_eV",
                "Ne_cm3",
                "resolution",
                "augmentation_seed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def import_nist_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    copies_per_source: int = 20,
    seed: int = 0,
    classes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Import NIST CSV spectra into split-based image datasets."""
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    classes = list(classes or ALLOWED_CLASSES)

    for element in classes:
        if element not in ALLOWED_CLASSES:
            raise ValueError(f"Unsupported element: {element}")

    manifest_rows: list[dict[str, Any]] = []
    source_files = []
    for element in classes:
        element_dir = input_root / element
        if not element_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {element_dir}")
        source_files.extend(sorted(element_dir.glob("*.csv")))

    if not source_files:
        raise ValueError(f"No input CSV files found under {input_root}")

    if copies_per_source <= 0:
        raise ValueError("copies_per_source must be positive")

    for source_path in source_files:
        element = source_path.parent.name
        split = assign_split(source_path.name)
        stem = source_path.stem
        match = re.search(r"(\d{3})$", stem)
        if not match:
            raise ValueError(f"Cannot determine source index from {source_path}")
        source_index = int(match.group(1))
        te_eV = 0.7 if source_index == 1 else 1.0 if source_index == 2 else 1.3
        wavelengths, intensities = read_nist_spectrum(source_path, element=element)

        for copy_index in range(copies_per_source):
            rng = np.random.default_rng(
                _stable_seed(element, split, source_path.name, copy_index, seed)
            )

            augmented = intensities.copy()
            augmented = augmented + rng.uniform(-0.01, 0.01, size=augmented.shape)
            augmented = augmented + rng.uniform(-0.02, 0.02)
            augmented = augmented + rng.uniform(-0.02, 0.02) * (
                (wavelengths - wavelengths.mean()) / (wavelengths.max() - wavelengths.min() + 1e-8)
            )
            augmented = augmented * rng.uniform(0.9, 1.1)
            if augmented.size >= 3:
                shifted_wavelengths = wavelengths + rng.uniform(-0.4, 0.4)
                shifted_wavelengths = np.clip(shifted_wavelengths, 200.0, 600.0)
                shifted_profile = np.interp(shifted_wavelengths, wavelengths, augmented)
                augmented = shifted_profile
            augmented = _apply_width_variation(augmented, rng)
            augmented = np.clip(augmented, 0.0, 1.0)

            output_path = output_root / split / element / f"{source_path.stem}_{copy_index:04d}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            render_spectrum_image(wavelengths, augmented, output_path, dpi=120, rng=rng)
            manifest_rows.append(
                {
                    "element": element,
                    "source_file": str(source_path.relative_to(input_root)).replace("\\", "/"),
                    "output_file": str(output_path.relative_to(output_root)).replace("\\", "/"),
                    "split": split,
                    "source_index": source_index,
                    "Te_eV": te_eV,
                    "Ne_cm3": 1e17,
                    "resolution": 500,
                    "augmentation_seed": int(rng.bit_generator.state["state"].get("state")) if False else _stable_seed(element, split, source_path.name, copy_index, seed),
                }
            )

    manifest_path = output_root / "manifest.csv"
    generate_manifest(manifest_path, manifest_rows)
    return manifest_rows
