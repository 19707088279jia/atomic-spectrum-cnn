"""Frozen NASA Z-903 numerical CNN inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .labels import CNN_THRESHOLDS, TARGETS
from .model import Z903CNN
from .preprocessing import robust_scale

CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "z903_cnn_robust_best.pt"


def load_checkpoint(path: str | Path = CHECKPOINT) -> tuple[Z903CNN, dict[str, object]]:
    """Load the copied validated checkpoint on CPU."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("preprocess") != "robust" or tuple(checkpoint.get("targets", ())) != TARGETS:
        raise ValueError("Checkpoint metadata does not match the validated six-target robust model")
    model = Z903CNN(output_size=len(TARGETS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict(intensity: np.ndarray, model: Z903CNN) -> dict[str, dict[str, float | str]]:
    """Run robust preprocessing and return six independent sigmoid outputs."""
    values = robust_scale(intensity)
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).reshape(1, 1, 23401)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor))[0].numpy()
    return {
        element: {
            "probability": float(probabilities[index]),
            "threshold": CNN_THRESHOLDS[element],
            "result": "DETECTED" if probabilities[index] >= CNN_THRESHOLDS[element] else "NOT DETECTED",
        }
        for index, element in enumerate(TARGETS)
    }


__all__ = ["CHECKPOINT", "load_checkpoint", "predict"]
