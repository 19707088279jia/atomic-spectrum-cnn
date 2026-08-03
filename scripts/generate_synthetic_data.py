#!/usr/bin/env python
"""Generate synthetic spectrum images for pipeline verification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atomic_spectrum_ai.config import load_config
from atomic_spectrum_ai.synthetic import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quickstart.yaml")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="Override every split count for a very small test dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    synthetic = config.get("synthetic")
    if not synthetic:
        raise ValueError("The selected config has no synthetic section")

    split_counts = dict(synthetic["samples_per_class"])
    if args.samples_per_class is not None:
        if args.samples_per_class <= 0:
            raise ValueError("--samples-per-class must be positive")
        split_counts = {split: args.samples_per_class for split in split_counts}

    created = generate_dataset(
        root=config["data"]["root"],
        classes=config["data"]["classes"],
        samples_per_split=split_counts,
        wavelength_min=float(synthetic["wavelength_min"]),
        wavelength_max=float(synthetic["wavelength_max"]),
        points=int(synthetic["points"]),
        dpi=int(synthetic["image_dpi"]),
        seed=int(config["project"]["seed"]),
    )
    print(f"Created {len(created)} synthetic images under {config['data']['root']}")
    print("Synthetic data verifies software only; it is not scientific validation.")


if __name__ == "__main__":
    main()
