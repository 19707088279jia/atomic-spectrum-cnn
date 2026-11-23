"""Inference for the dedicated Fe2O3-equivalent concentration model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import EXPECTED_POINTS
from .model import ConcentrationCNN
from .preprocessing import robust_scale

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FE_CHECKPOINT = REPOSITORY_ROOT / "reports" / "fe_analysis" / "fe_cnn_best.pt"
FE_VALIDATION_RANGE_WT_PERCENT = (0.12, 17.4)


def load_fe_checkpoint(
    path: str | Path = FE_CHECKPOINT,
    device: str | torch.device = "cpu",
) -> tuple[ConcentrationCNN, dict[str, Any], torch.device]:
    """Load and validate the Fe concentration checkpoint."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Fe concentration checkpoint not found: {checkpoint_path}")
    resolved_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    if checkpoint.get("target") != "Fe2O3*":
        raise ValueError(f"Unexpected Fe concentration target: {checkpoint.get('target')}")
    if checkpoint.get("preprocess") != "robust":
        raise ValueError("Fe concentration checkpoint must use robust preprocessing")
    if int(checkpoint.get("input_shape", [0, 0, 0])[-1]) != EXPECTED_POINTS:
        raise ValueError("Fe concentration checkpoint does not match the Z-903 spectrum length")
    model = ConcentrationCNN(output_size=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, checkpoint, resolved_device


def predict_fe_concentration(
    intensity: np.ndarray,
    model: ConcentrationCNN,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> float:
    """Predict Fe2O3-equivalent concentration in wt% from a full spectrum."""
    values = np.asarray(intensity, dtype=np.float32)
    if values.shape != (EXPECTED_POINTS,):
        raise ValueError(f"Expected {EXPECTED_POINTS} intensity values, got {values.shape}")
    processed = robust_scale(values)
    tensor = torch.from_numpy(processed).reshape(1, 1, EXPECTED_POINTS).to(device)
    with torch.inference_mode():
        scaled = float(model(tensor).reshape(-1)[0].cpu().item())
    mean = float(checkpoint["transform_mean"])
    scale = float(checkpoint["transform_scale"])
    return float(max(np.expm1(scaled * scale + mean), 0.0))


__all__ = [
    "FE_CHECKPOINT",
    "FE_VALIDATION_RANGE_WT_PERCENT",
    "load_fe_checkpoint",
    "predict_fe_concentration",
]
