"""Inference for the frozen numerical NASA Z-903 1D-CNN."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .models.z903_cnn import Z903CNN
from .z903 import TARGETS, preprocess_spectrum, read_z903_spectrum

Z903_CNN_CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / "z903_cnn_robust_best.pt"
Z903_THRESHOLDS = {
    "Zn": 0.7050,
    "Mn": 0.2350,
    "Cd": 0.5000,
    "Mg": 0.3550,
    "Cu": 0.5050,
    "Pb": 0.4050,
}


def load_z903_cnn(
    checkpoint_path: str | Path = Z903_CNN_CHECKPOINT,
    device: str | torch.device = "cpu",
) -> tuple[Z903CNN, dict[str, Any], torch.device]:
    """Load the frozen robust-preprocessed numerical Z-903 checkpoint."""
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"NASA Z-903 CNN checkpoint not found: {checkpoint_file}")
    resolved_device = torch.device(device)
    checkpoint = torch.load(checkpoint_file, map_location=resolved_device, weights_only=False)
    if tuple(checkpoint.get("targets", ())) != TARGETS:
        raise ValueError(f"Unexpected NASA Z-903 checkpoint targets: {checkpoint.get('targets')}")
    if checkpoint.get("preprocess") != "robust":
        raise ValueError("NASA Z-903 checkpoint is not the robust-preprocessed model")
    model = Z903CNN(output_size=len(TARGETS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, checkpoint, resolved_device


def predict_z903_csv(
    csv_path: str | Path,
    model: Z903CNN,
    device: torch.device,
) -> dict[str, dict[str, float | bool]]:
    """Run robust numerical preprocessing and 1D-CNN inference on one CSV."""
    _, intensity = read_z903_spectrum(csv_path)
    processed = preprocess_spectrum(intensity, option="robust")
    tensor = torch.from_numpy(np.asarray(processed, dtype=np.float32)).reshape(1, 1, 23401).to(device)
    if tuple(tensor.shape) != (1, 1, 23401):
        raise ValueError(f"Expected NASA Z-903 tensor [1, 1, 23401], got {tuple(tensor.shape)}")
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor))[0].cpu().numpy()
    return {
        element: {
            "probability": float(probabilities[index]),
            "decision": bool(probabilities[index] >= Z903_THRESHOLDS[element]),
        }
        for index, element in enumerate(TARGETS)
    }


__all__ = ["Z903_CNN_CHECKPOINT", "Z903_THRESHOLDS", "load_z903_cnn", "predict_z903_csv"]