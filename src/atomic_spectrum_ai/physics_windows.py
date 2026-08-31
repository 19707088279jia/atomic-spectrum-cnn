"""Dataset for target-specific robust-scaled Z-903 wavelength windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from atomic_spectrum_ai.z903 import TARGETS, WAVELENGTHS, preprocess_spectrum, read_z903_spectrum


class PhysicsWindowDataset(Dataset):
    def __init__(self, manifest: str | Path, windows: str | Path) -> None:
        self.frame = pd.read_csv(manifest)
        window_frame = pd.read_csv(windows)
        self.window_indices = {}
        for target in TARGETS:
            selected = window_frame[(window_frame["target"] == target) & (window_frame["selected"])]
            indices = [np.flatnonzero((WAVELENGTHS >= row.window_start_nm) & (WAVELENGTHS <= row.window_end_nm)) for row in selected.itertuples()]
            self.window_indices[target] = np.concatenate(indices) if indices else np.asarray([], dtype=int)
        self.branch_values = []
        self.labels = []
        self.masks = []
        for _, row in self.frame.iterrows():
            _, intensity = read_z903_spectrum(row["spectrum_path"])
            values = preprocess_spectrum(intensity, "robust")
            self.branch_values.append([values[self.window_indices[target]].astype(np.float32) for target in TARGETS])
            self.labels.append(np.asarray([float(row[f"{target}_raw" if target != "Mg" else "MgO_raw"]) > threshold for target, threshold in zip(TARGETS, (100, 500, 0.05, 5, 25, 10), strict=True)], dtype=np.float32))
            self.masks.append(np.asarray([row[f"{target}_mask"] for target in TARGETS], dtype=np.float32))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        branches = [torch.from_numpy(values[None, :]) for values in self.branch_values[index]]
        labels = self.labels[index]
        masks = self.masks[index]
        return branches, torch.from_numpy(labels), torch.from_numpy(masks)