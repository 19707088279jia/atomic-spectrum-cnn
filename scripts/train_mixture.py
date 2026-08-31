#!/usr/bin/env python
"""Train a small mixture model for smoke testing."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atomic_spectrum_ai.mixture_model import build_mixture_model  # noqa: E402
from atomic_spectrum_ai.mixture_training import save_checkpoint, train_simple  # noqa: E402
from scripts.generate_mixture_data import generate_samples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/mixture_smoke")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path("data/mixture")
    # generate small dataset
    manifest = generate_samples(str(data_root), n_samples=args.n, seed=0, split="train")
    model = build_mixture_model(num_classes=6, pretrained=False)
    metrics = train_simple(model, str(manifest), root_dir=str(data_root), epochs=args.epochs, batch_size=8)
    chk = out / "mixture_smoke.pt"
    save_checkpoint(model, str(chk), class_names=["Zn", "Mn", "Cd", "Mg", "Cu", "Pb"])
    print("Saved checkpoint:", chk)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
