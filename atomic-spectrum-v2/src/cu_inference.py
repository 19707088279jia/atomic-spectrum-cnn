"""Inference for the frozen Cu 1-500 ppm supported-range concentration model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import EXPECTED_POINTS
from .model import Z903CNN
from .preprocessing import robust_scale

REPO_ROOT = Path(__file__).resolve().parents[2]
CONCENTRATION_CHECKPOINT = REPO_ROOT / "models" / "cu_supported_range_final.pt"
SUPPORTED_RANGE_PPM = (1.0, 500.0)
EXISTING_DETECTION_THRESHOLD_PPM = 25.0
HIGH_CONCENTRATION_NOTE = (
    "High-Cu samples above the supported range may be underestimated because "
    "high-concentration training data are limited."
)


def load_cu_concentration(path: str | Path = CONCENTRATION_CHECKPOINT) -> tuple[Z903CNN, dict[str, object]]:
    """Load the frozen Cu supported-range concentration CNN."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Cu concentration checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("model_name") != "Z903CNN"
        or checkpoint.get("output_size") != 1
        or checkpoint.get("preprocess") != "robust"
    ):
        raise ValueError("Cu concentration checkpoint does not match the frozen inference contract")
    model = Z903CNN(output_size=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_cu_concentration(intensity: np.ndarray, model: Z903CNN, checkpoint: dict[str, object]) -> float:
    """Apply the frozen Cu CNN and invert its target transform without clipping to the supported range."""
    tensor = torch.from_numpy(robust_scale(intensity)).reshape(1, 1, EXPECTED_POINTS)
    with torch.inference_mode():
        scaled = float(model(tensor).numpy().ravel()[0])
    mean, scale = float(checkpoint["target_mean"]), float(checkpoint["target_scale"])
    prediction = scaled * scale + mean
    if checkpoint.get("target_transform") == "log1p":
        prediction = float(np.expm1(prediction))
    else:
        prediction = float(prediction)
    if not np.isfinite(prediction):
        raise ValueError("Cu concentration prediction is not finite")
    return max(0.0, prediction)


__all__ = [
    "CONCENTRATION_CHECKPOINT",
    "SUPPORTED_RANGE_PPM",
    "EXISTING_DETECTION_THRESHOLD_PPM",
    "HIGH_CONCENTRATION_NOTE",
    "load_cu_concentration",
    "predict_cu_concentration",
]
