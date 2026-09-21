"""Inference for the frozen Ag two-stage classification and concentration models."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from .data import EXPECTED_POINTS
from .mgo_ridge_inference import BIN_SIZE, bin_spectrum
from .model import Z903CNN
from .preprocessing import robust_scale

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_CHECKPOINT = REPO_ROOT / "models" / "ag_classifier_final.joblib"
CONCENTRATION_CHECKPOINT = REPO_ROOT / "models" / "ag_concentration_final.pt"
LOW_CONCENTRATION_NOTE = (
    "Ag quantitative estimation is experimental and is better supported in the "
    "low-concentration region."
)
ZERO_CLASS_NOTE = (
    "Ag quantitative estimate not reported because the sample was classified as "
    "Ag zero-class."
)


def load_ag_classifier(path: str | Path = CLASSIFIER_CHECKPOINT) -> dict[str, object]:
    """Load the frozen Ag stage-1 metadata-zero/metadata-positive classifier."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Ag classifier checkpoint not found: {checkpoint_path}")
    artifact = joblib.load(checkpoint_path)
    required = {"model", "scaler", "bin_size", "preprocess", "threshold"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"Ag classifier artifact is missing fields: {sorted(missing)}")
    if artifact["bin_size"] != BIN_SIZE or artifact["preprocess"] != "robust":
        raise ValueError("Ag classifier artifact does not match the frozen inference contract")
    return artifact


def classify_ag(intensity: np.ndarray, artifact: dict[str, object]) -> dict[str, object]:
    """Return the Ag stage-1 classification: metadata-zero or metadata-positive."""
    features = bin_spectrum(robust_scale(intensity)).reshape(1, -1)
    scaled_features = artifact["scaler"].transform(features)
    probability = float(artifact["model"].predict_proba(scaled_features)[0, 1])
    threshold = float(artifact["threshold"])
    return {"probability": probability, "threshold": threshold, "positive": probability >= threshold}


def load_ag_concentration(path: str | Path = CONCENTRATION_CHECKPOINT) -> tuple[Z903CNN, dict[str, object]]:
    """Load the frozen Ag stage-2 concentration CNN."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Ag concentration checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("model_name") != "Z903CNN"
        or checkpoint.get("output_size") != 1
        or checkpoint.get("preprocess") != "robust"
        or checkpoint.get("target_transform") != "log1p(Ag_ppm)"
    ):
        raise ValueError("Ag concentration checkpoint does not match the frozen inference contract")
    model = Z903CNN(output_size=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_ag_concentration(intensity: np.ndarray, model: Z903CNN, checkpoint: dict[str, object]) -> float:
    """Apply the frozen Ag CNN and invert the log1p(Ag_ppm) target transform."""
    tensor = torch.from_numpy(robust_scale(intensity)).reshape(1, 1, EXPECTED_POINTS)
    with torch.inference_mode():
        scaled = float(model(tensor).numpy().ravel()[0])
    mean, scale = float(checkpoint["target_mean"]), float(checkpoint["target_scale"])
    prediction = float(np.expm1(scaled * scale + mean))
    if not np.isfinite(prediction):
        raise ValueError("Ag concentration prediction is not finite")
    return max(0.0, prediction)


__all__ = [
    "CLASSIFIER_CHECKPOINT",
    "CONCENTRATION_CHECKPOINT",
    "LOW_CONCENTRATION_NOTE",
    "ZERO_CLASS_NOTE",
    "load_ag_classifier",
    "classify_ag",
    "load_ag_concentration",
    "predict_ag_concentration",
]
