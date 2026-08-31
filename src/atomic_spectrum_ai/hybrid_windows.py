"""Cached full-spectrum and target-specific-window dataset for hybrid training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from atomic_spectrum_ai.z903 import TARGETS, WAVELENGTHS, preprocess_spectrum, read_z903_spectrum


class HybridWindowDataset(Dataset):
    def __init__(self, manifest: str | Path, windows: str | Path) -> None:
        self.frame = pd.read_csv(manifest)
        selected = pd.read_csv(windows)
        self.indices = {}
        for target in TARGETS:
            chosen = selected[(selected.target == target) & selected.selected]
            parts = [
                np.flatnonzero(
                    (WAVELENGTHS >= row.window_start_nm) & (WAVELENGTHS <= row.window_end_nm)
                )
                for row in chosen.itertuples()
            ]
            self.indices[target] = np.concatenate(parts) if parts else np.array([], dtype=int)
        self.full, self.branches, self.labels, self.masks = [], [], [], []
        thresholds = (100, 500, 0.05, 5, 25, 10)
        for _, row in self.frame.iterrows():
            _, intensity = read_z903_spectrum(row.spectrum_path)
            values = preprocess_spectrum(intensity, "robust").astype(np.float32)
            self.full.append(values[None, :])
            self.branches.append([values[self.indices[target]][None, :] for target in TARGETS])
            self.labels.append(
                np.asarray(
                    [
                        float(row[f"{target}_raw" if target != "Mg" else "MgO_raw"]) > threshold
                        for target, threshold in zip(TARGETS, thresholds, strict=True)
                    ],
                    dtype=np.float32,
                )
            )
            self.masks.append(
                np.asarray([row[f"{target}_mask"] for target in TARGETS], dtype=np.float32)
            )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.full[index]),
            [torch.from_numpy(value) for value in self.branches[index]],
            torch.from_numpy(self.labels[index]),
            torch.from_numpy(self.masks[index]),
        )
