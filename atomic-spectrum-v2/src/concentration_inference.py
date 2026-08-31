"""Inference for the separate concentration-regression checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .labels import TARGETS
from .model import ConcentrationCNN
from .regression import TargetTransform

CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "z903_concentration_best.pt"


def load_concentration_checkpoint(path: str | Path = CHECKPOINT) -> tuple[ConcentrationCNN, TargetTransform, dict[str, object]]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if tuple(checkpoint.get("targets", ())) != TARGETS:
        raise ValueError("Concentration checkpoint targets do not match V2 targets")
    model = ConcentrationCNN(output_size=len(TARGETS))
    model.load_state_dict(checkpoint["model_state_dict"])
    transform = TargetTransform(np.asarray(checkpoint["transform_means"], dtype=np.float32), np.asarray(checkpoint["transform_scales"], dtype=np.float32))
    model.eval()
    return model, transform, checkpoint


def predict_concentrations(intensity: np.ndarray, model: ConcentrationCNN, transform: TargetTransform) -> dict[str, float]:
    from .preprocessing import robust_scale

    tensor = torch.from_numpy(robust_scale(intensity)).reshape(1, 1, 23401)
    with torch.inference_mode():
        scaled = model(tensor).numpy()
    values = transform.inverse(scaled)[0]
    return {element: float(values[index]) for index, element in enumerate(TARGETS)}


__all__ = ["CHECKPOINT", "load_concentration_checkpoint", "predict_concentrations"]
