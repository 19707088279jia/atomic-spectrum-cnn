#!/usr/bin/env python
"""Train a configured CNN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atomic_spectrum_ai.config import load_config
from atomic_spectrum_ai.training import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quickstart.yaml")
    args = parser.parse_args()
    metrics = train_from_config(load_config(args.config))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
