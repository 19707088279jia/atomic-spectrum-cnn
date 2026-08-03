#!/usr/bin/env python
"""Import NIST LIBS CSV spectra into image datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atomic_spectrum_ai.config import load_config
from atomic_spectrum_ai.nist_import import import_nist_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/nist_raw")
    parser.add_argument("--output", default="data/nist_images")
    parser.add_argument("--copies-per-source", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="configs/nist_data.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    nist_config = config.get("nist", {})
    input_dir = args.input if args.input != "data/nist_raw" else str(nist_config.get("input_dir", args.input))
    output_dir = args.output if args.output != "data/nist_images" else str(nist_config.get("output_dir", args.output))
    copies_per_source = args.copies_per_source if args.copies_per_source != 20 else int(nist_config.get("copies_per_source", args.copies_per_source))
    default_seed = int(config.get("project", {}).get("seed", args.seed))
    seed = args.seed if args.seed != 0 else int(nist_config.get("seed", default_seed))

    manifest_rows = import_nist_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        copies_per_source=copies_per_source,
        seed=seed,
        classes=["Fe", "Cu", "Na", "Ca", "Mg"],
    )
    print(f"Created {len(manifest_rows)} NIST image variants")
    print(f"Manifest written to {Path(args.output) / 'manifest.csv'}")


if __name__ == "__main__":
    main()
