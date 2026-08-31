"""Inference for the frozen final MgO Ridge pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .data import EXPECTED_POINTS, WAVELENGTH_RANGE
from .preprocessing import robust_scale

CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "mgo_ridge_final.joblib"
BIN_SIZE = 25
FEATURE_COUNT = 936
TARGET_UNIT = "wt%"
TARGET_LABEL = "MgO"


def bin_spectrum(values: np.ndarray, bin_size: int = BIN_SIZE) -> np.ndarray:
    """Average-pool robust-scaled Z-903 values into the frozen feature count."""
    if len(values) != EXPECTED_POINTS:
        raise ValueError(f"Expected {EXPECTED_POINTS} spectral values, got {len(values)}")
    usable = (len(values) // bin_size) * bin_size
    features = values[:usable].reshape(-1, bin_size).mean(axis=1)
    if len(features) != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} frozen Ridge features, got {len(features)}")
    return features


def load_pipeline(path: str | Path = CHECKPOINT) -> dict[str, object]:
    """Load and validate the frozen MgO Ridge artifact without fitting anything."""
    pipeline = joblib.load(path)
    required = {"model", "scaler", "alpha", "bin_size", "n_features", "expected_points", "wavelength_range_nm", "target"}
    missing = required.difference(pipeline)
    if missing:
        raise ValueError(f"MgO Ridge artifact is missing fields: {sorted(missing)}")
    if pipeline["alpha"] != 100.0 or pipeline["bin_size"] != BIN_SIZE or pipeline["n_features"] != FEATURE_COUNT or pipeline["expected_points"] != EXPECTED_POINTS or tuple(pipeline["wavelength_range_nm"]) != WAVELENGTH_RANGE or pipeline["target"] != "MgO wt%":
        raise ValueError("MgO Ridge artifact does not match the finalized frozen inference contract")
    return pipeline


def predict_mgo(intensity: np.ndarray, pipeline: dict[str, object]) -> float:
    """Apply frozen preprocessing and return one MgO concentration in wt%."""
    features = bin_spectrum(robust_scale(intensity)).reshape(1, -1)
    scaled_features = pipeline["scaler"].transform(features)
    prediction = float(pipeline["model"].predict(scaled_features)[0])
    if not np.isfinite(prediction):
        raise ValueError("MgO Ridge prediction is not finite")
    return max(0.0, prediction)


def build_display(predicted: float, ground_truth: float | None = None) -> dict[str, object]:
    """Build MgO display data, adding truth fields only for built-in demos."""
    display = {
        "Predicted MgO": predicted,
        "Unit": TARGET_UNIT,
        "Model": "Ridge Regression",
        "Status": "Experimental quantitative estimate",
    }
    if ground_truth is not None:
        display["Ground Truth MgO"] = ground_truth
        display["Absolute Error"] = abs(predicted - ground_truth)
    return display


__all__ = ["CHECKPOINT", "BIN_SIZE", "FEATURE_COUNT", "TARGET_UNIT", "TARGET_LABEL", "bin_spectrum", "load_pipeline", "predict_mgo", "build_display"]
