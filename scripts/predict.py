#!/usr/bin/env python
"""Predict an element from one spectrum image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atomic_spectrum_ai.inference import load_model_checkpoint, predict_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Accepted for CLI compatibility; checkpoint stores config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model, checkpoint, device = load_model_checkpoint(args.checkpoint, args.device)
    with Image.open(image_path) as image:
        results = predict_image(image, model, checkpoint, device, args.top_k)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
