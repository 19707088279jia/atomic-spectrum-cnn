"""Synthetic atomic-spectrum image generation for software testing only."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Approximate, deliberately simplified peak positions for software demonstration.
# These are not a substitute for a validated spectroscopy database.
DEFAULT_PEAKS: dict[str, tuple[float, ...]] = {
    "Fe": (248.3, 302.1, 358.1, 372.0, 404.6, 438.4, 527.0),
    "Cu": (324.8, 327.4, 510.6, 515.3, 521.8, 578.2),
    "Na": (330.2, 568.3, 568.8, 589.0, 589.6, 615.4),
    "Ca": (393.4, 396.8, 422.7, 443.5, 616.2, 643.9),
    "Mg": (279.6, 280.3, 285.2, 383.8, 517.3, 518.4),
}


def _stable_seed(element: str, split: str, index: int, base_seed: int) -> int:
    payload = f"{element}:{split}:{index}:{base_seed}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def make_spectrum(
    element: str,
    wavelength_min: float,
    wavelength_max: float,
    points: int,
    rng: np.random.Generator,
    peaks: dict[str, tuple[float, ...]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simplified synthetic spectrum with randomized peak heights and noise."""
    peak_table = peaks or DEFAULT_PEAKS
    if element not in peak_table:
        raise KeyError(f"No synthetic peak definition for element: {element}")

    x = np.linspace(wavelength_min, wavelength_max, points)
    baseline_slope = rng.uniform(-0.02, 0.02)
    y = 0.04 + baseline_slope * ((x - x.mean()) / (wavelength_max - wavelength_min))
    y = y + rng.normal(0.0, rng.uniform(0.003, 0.012), size=points)

    wavelength_shift = rng.normal(0.0, 0.25)
    for peak in peak_table[element]:
        if not wavelength_min <= peak <= wavelength_max:
            continue
        height = rng.uniform(0.35, 1.0)
        width = rng.uniform(0.35, 1.3)
        center = peak + wavelength_shift + rng.normal(0.0, 0.06)
        y += height * np.exp(-0.5 * ((x - center) / width) ** 2)

    # Add weak nuisance peaks so the classifier cannot rely on a single perfect line.
    for _ in range(rng.integers(1, 4)):
        center = rng.uniform(wavelength_min, wavelength_max)
        height = rng.uniform(0.02, 0.10)
        width = rng.uniform(0.4, 1.8)
        y += height * np.exp(-0.5 * ((x - center) / width) ** 2)

    y = np.clip(y, 0.0, None)
    max_value = float(y.max())
    if max_value > 0:
        y = y / max_value
    return x, y


def save_spectrum_image(
    x: np.ndarray,
    y: np.ndarray,
    output_path: str | Path,
    dpi: int,
    rng: np.random.Generator,
) -> None:
    """Save a spectrum image without title, legend, text labels, or class leakage."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width = rng.uniform(5.6, 6.4)
    height = rng.uniform(3.0, 3.6)
    line_width = rng.uniform(0.8, 1.5)
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.plot(x, y, linewidth=line_width)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def generate_dataset(
    root: str | Path,
    classes: Iterable[str],
    samples_per_split: dict[str, int],
    wavelength_min: float,
    wavelength_max: float,
    points: int,
    dpi: int,
    seed: int,
) -> list[Path]:
    """Generate train/validation/test folders and return created paths."""
    root_path = Path(root)
    created: list[Path] = []
    for split, count in samples_per_split.items():
        if count <= 0:
            raise ValueError(f"Sample count for {split} must be positive")
        for element in classes:
            if element not in DEFAULT_PEAKS:
                raise KeyError(
                    f"Synthetic generator supports {sorted(DEFAULT_PEAKS)}, got {element}"
                )
            for index in range(count):
                rng = np.random.default_rng(_stable_seed(element, split, index, seed))
                x, y = make_spectrum(
                    element,
                    wavelength_min,
                    wavelength_max,
                    points,
                    rng,
                )
                path = root_path / split / element / f"{element}_{index:04d}.png"
                save_spectrum_image(x, y, path, dpi, rng)
                created.append(path)
    return created
