#!/usr/bin/env python
"""Render NASA Z-903 numerical spectra as neutral scientific PNG plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atomic_spectrum_ai.z903 import read_z903_spectrum  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "demo_images"
FIGURE_SIZE = (10.0, 4.0)
DPI = 150
WAVELENGTH_LIMITS = (180.0, 960.0)


def plot_z903_spectrum(input_path: str | Path, output_path: str | Path) -> Path:
    """Plot one validated Z-903 CSV without changing its numerical values."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    wavelengths, intensity = read_z903_spectrum(input_file)

    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI, facecolor="white")
    axis.set_facecolor("white")
    axis.plot(wavelengths, intensity, color="black", linewidth=0.8)
    axis.set_xlim(*WAVELENGTH_LIMITS)
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Intensity")
    figure.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=DPI, facecolor="white")
    plt.close(figure)
    return output_file


def expand_inputs(inputs: list[str]) -> list[Path]:
    """Expand CSV files and directories for convenient batch conversion."""
    paths: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.csv")))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Input spectrum not found: {path}")
    if not paths:
        raise ValueError("No CSV spectra found in the supplied inputs")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more Z-903 CSV files, or directories containing CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"PNG output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = expand_inputs(args.inputs)
    for input_path in input_paths:
        output_path = args.output_dir / f"{input_path.stem}.png"
        plot_z903_spectrum(input_path, output_path)
        print(output_path)


if __name__ == "__main__":
    main()