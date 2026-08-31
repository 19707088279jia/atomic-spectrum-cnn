#!/usr/bin/env python
"""Train the final corrected no-consistency hybrid on train/validation only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from train_z903_hybrid import run_variant, seed

from atomic_spectrum_ai.hybrid_windows import HybridWindowDataset

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb")

def main() -> None:
    seed(20260816)
    train = HybridWindowDataset(ROOT / "data/processed/z903_train.csv", ROOT / "data/processed/z903_target_windows.csv")
    val = HybridWindowDataset(ROOT / "data/processed/z903_val.csv", ROOT / "data/processed/z903_target_windows.csv")
    labels = np.asarray(train.labels)
    masks = np.asarray(train.masks)
    weights = torch.tensor([float(((labels[:, i] <= .5) & (masks[:, i] > 0)).sum() / max(1, ((labels[:, i] > .5) & (masks[:, i] > 0)).sum())) for i in range(6)], dtype=torch.float32)
    args = type("Args", (), {"max_epochs": 20, "patience": 5})()
    result = run_variant("z903_hybrid_final_corrected_regions", 0.0, train, val, weights, args)
    output = ROOT / "outputs/z903"
    (output / "final_hybrid_summary.json").write_text(json.dumps({k: v for k, v in result.items() if k not in ("model", "rows")}, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("model", "rows")}, indent=2))

if __name__ == "__main__":
    main()
