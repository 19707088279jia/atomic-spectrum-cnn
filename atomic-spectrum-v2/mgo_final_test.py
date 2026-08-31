"""One-time final untouched-test evaluation of the frozen MgO Ridge model.

The model-selection process is finished. This script fits the single frozen
configuration selected in mgo_iteration2.py (Ridge, alpha = 100, 936-bin
robust-scaled spectral features, training-only standardization) and evaluates
it exactly once on the existing untouched test split. No hyperparameter is
retuned here, no preprocessing is changed, and no test information is used to
fit the model or its preprocessing statistics. This script does not modify
the detection model or z903_concentration_best.pt.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mgo_iteration2 import BIN_SIZE, bin_spectrum, compute_metrics, quantile_bands
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.concentration_inference import load_concentration_checkpoint, predict_concentrations
from src.data import EXPECTED_POINTS, WAVELENGTH_RANGE, load_regression_split, load_spectrum
from src.preprocessing import robust_scale

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "concentration" / "mgo_final_test"
MODEL_PATH = ROOT / "models" / "mgo_ridge_final.joblib"
FROZEN_ALPHA = 100.0

# Declared for the fixed comparison; independently recomputed below for verification.
EXISTING_CNN_TEST_REFERENCE = {"known": 386, "mae": 2.8023, "rmse": 5.4403, "r2": 0.4933}

INTERPRETATION_BANDS = (
    (0.80, float("inf"), "Strong experimental quantitative performance"),
    (0.70, 0.80, "Good experimental quantitative performance"),
    (0.50, 0.70, "Moderate quantitative performance; further external validation needed"),
    (0.0, 0.50, "Weak quantitative performance"),
    (float("-inf"), 0.0, "No useful quantitative generalization"),
)


def interpret_r2(r2: float) -> str:
    for low, high, label in INTERPRETATION_BANDS:
        if low < r2 <= high or (low == float("-inf") and r2 <= high):
            return label
    return "No useful quantitative generalization"


def load_features(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (binned features, raw intensity, MgO truth, target_id) for known rows."""
    frame = load_regression_split(split)
    truth = pd.to_numeric(frame["Mg"], errors="coerce").to_numpy(dtype=np.float64)
    known = np.isfinite(truth)
    binned, raw, target_ids = [], [], []
    for path, target_id in zip(frame.loc[known, "spectrum_path"], frame.loc[known, "target_id"], strict=True):
        _, intensity = load_spectrum(path)
        raw.append(intensity)
        binned.append(bin_spectrum(robust_scale(intensity)))
        target_ids.append(str(target_id))
    return np.asarray(binned, dtype=np.float64), np.asarray(raw, dtype=np.float32), truth[known], target_ids


def banded_metrics_with_bias(y: np.ndarray, p: np.ndarray) -> dict[str, dict[str, float | int]]:
    return {name: compute_metrics(y[mask], p[mask]) for name, mask in quantile_bands(y).items() if mask.any()}


def scatter_with_annotation(y: np.ndarray, p: np.ndarray, mae: float, r2: float) -> None:
    limit = float(max(y.max(), p.max())) * 1.05
    figure, axis = plt.subplots(figsize=(5.5, 5.5), dpi=150)
    axis.scatter(y, p, s=16, alpha=0.6, color="#176b87")
    axis.plot([0, limit], [0, limit], color="black", linewidth=1, label="y = x")
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("Ground truth MgO (wt%)")
    axis.set_ylabel("Predicted MgO (wt%)")
    axis.set_title("Final untouched test: MgO Ridge (alpha=100)")
    axis.text(0.03, 0.95, f"R2 = {r2:.4f}\nMAE = {mae:.4f} wt%", transform=axis.transAxes, va="top", ha="left", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})
    axis.legend(loc="lower right")
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "test_scatter.png")
    plt.close(figure)


def residual_plot(y: np.ndarray, p: np.ndarray) -> None:
    residuals = p - y
    figure, axis = plt.subplots(figsize=(5.5, 4.5), dpi=150)
    axis.scatter(y, residuals, s=16, alpha=0.6, color="#c1440e")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Ground truth MgO (wt%)")
    axis.set_ylabel("Residual, predicted - ground truth (wt%)")
    axis.set_title("Final untouched test: MgO Ridge residuals")
    figure.tight_layout()
    figure.savefig(OUTPUT / "test_residuals.png")
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    x_train, _raw_train, y_train, _train_ids = load_features("train")
    x_test, raw_test, y_test, test_ids = load_features("test")

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = Ridge(alpha=FROZEN_ALPHA).fit(x_train_scaled, y_train)
    test_pred = model.predict(x_test_scaled)

    test_metrics = compute_metrics(y_test, test_pred)
    test_bands = banded_metrics_with_bias(y_test, test_pred)

    median_value = float(np.median(y_train))
    median_pred = np.full_like(y_test, median_value)
    median_metrics = compute_metrics(y_test, median_pred)

    cnn_model, transform, checkpoint = load_concentration_checkpoint()
    cnn_pred = np.asarray([predict_concentrations(intensity, cnn_model, transform)["Mg"] for intensity in raw_test])
    cnn_metrics = compute_metrics(y_test, cnn_pred)
    cnn_bands = banded_metrics_with_bias(y_test, cnn_pred)

    mae_below_reference = bool(test_metrics["mae"] < EXISTING_CNN_TEST_REFERENCE["mae"])
    top10_bias_reduced = bool(abs(test_bands["top_10"]["bias"]) < abs(cnn_bands["top_10"]["bias"]))
    top5_bias_reduced = bool(abs(test_bands["top_5"]["bias"]) < abs(cnn_bands["top_5"]["bias"]))
    interpretation = interpret_r2(test_metrics["r2"])

    scatter_with_annotation(y_test, test_pred, test_metrics["mae"], test_metrics["r2"])
    residual_plot(y_test, test_pred)

    pd.DataFrame({"target_id": test_ids, "ground_truth_wt_pct": y_test, "prediction_wt_pct": test_pred, "absolute_error_wt_pct": np.abs(test_pred - y_test), "split": "test"}).to_csv(OUTPUT / "test_predictions.csv", index=False)

    results = {
        "frozen_config": {"model": "Ridge", "alpha": FROZEN_ALPHA, "bin_size": BIN_SIZE, "n_features": int(x_train.shape[1]), "expected_points": EXPECTED_POINTS, "wavelength_range_nm": list(WAVELENGTH_RANGE)},
        "n_train": int(len(y_train)),
        "test": {"metrics": test_metrics, "bands": test_bands},
        "median_baseline_test": median_metrics,
        "existing_cnn_test_declared_reference": EXISTING_CNN_TEST_REFERENCE,
        "existing_cnn_test_recomputed": {"metrics": cnn_metrics, "bands": cnn_bands},
        "interpretation": interpretation,
        "ridge_mae_below_cnn_reference": mae_below_reference,
        "top10_bias_reduced_vs_cnn": top10_bias_reduced,
        "top5_bias_reduced_vs_cnn": top5_bias_reduced,
        "model_path": str(MODEL_PATH),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "alpha": FROZEN_ALPHA,
            "bin_size": BIN_SIZE,
            "n_features": int(x_train.shape[1]),
            "expected_points": EXPECTED_POINTS,
            "wavelength_range_nm": WAVELENGTH_RANGE,
            "preprocessing": "robust_scale then average-pool bin_size channels per bin",
            "target": "MgO wt%",
        },
        MODEL_PATH,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
