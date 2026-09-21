#!/usr/bin/env python
"""Develop a dedicated Cu supported-range (1-500 ppm) regression model.

Phase 2 of the Cu quantitative LIBS project. Uses only the frozen TRAIN and
VALIDATION splits; TEST is never opened by this script. Does not modify the
six-element detection CNN, the existing 25 ppm Cu detection threshold, MgO
models, Ag models, or Streamlit. Missing Cu labels are preserved as missing
(never filled with zero).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_z903_training_manifest import find_column, load_metadata, normalize  # noqa: E402, I001

from atomic_spectrum_ai.models.z903_cnn import Z903CNN  # noqa: E402
from atomic_spectrum_ai.z903 import WAVELENGTHS, preprocess_spectrum, read_z903_spectrum  # noqa: E402

OUTPUT = ROOT / "reports" / "concentration" / "cu_models"
MODELS = ROOT / "models"
BIN_SIZE = 25
SEED = 20260914
LOW_PPM = 1.0
HIGH_PPM = 500.0
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
PLS_COMPONENTS_GRID = (4, 8, 12, 16, 20)
CNN_EPOCHS = 25
CNN_PATIENCE = 5

# Existing exploratory Cu detection threshold. Reported only; never modified here.
EXISTING_CU_DETECTION_THRESHOLD_PPM = 25.0

# Fixed reporting ranges. First range in each tuple is inclusive on both ends.
PART_A_RANGES: tuple[tuple[str, float, float], ...] = (
    ("1-10 ppm", 1.0, 10.0),
    (">10-25 ppm", 10.0, 25.0),
    (">25-50 ppm", 25.0, 50.0),
    (">50-100 ppm", 50.0, 100.0),
    (">100-250 ppm", 100.0, 250.0),
    (">250-500 ppm", 250.0, 500.0),
)
PART_F_RANGES: tuple[tuple[str, float, float], ...] = (
    ("1-25 ppm", 1.0, 25.0),
    (">25-100 ppm", 25.0, 100.0),
    (">100-250 ppm", 100.0, 250.0),
    (">250-500 ppm", 250.0, 500.0),
)
MIN_SUBGROUP_SUPPORT = 5

# Strong Cu I resonance lines; used only as a validation-only sanity reference.
CU_NIST_LINES_NM: tuple[float, ...] = (324.754, 327.396)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="data/metadata/libs_metadata.xlsx")
    parser.add_argument("--splits-dir", default="data/processed")
    parser.add_argument("--output-dir", default="reports/concentration/cu_models")
    parser.add_argument("--epochs", type=int, default=CNN_EPOCHS)
    return parser.parse_args()


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def average_pool(values: np.ndarray) -> np.ndarray:
    usable = (values.shape[-1] // BIN_SIZE) * BIN_SIZE
    return values[..., :usable].reshape(*values.shape[:-1], -1, BIN_SIZE).mean(axis=-1)


def load_train_validation_cu(splits_dir: Path, metadata_path: Path) -> tuple[pd.DataFrame, str]:
    """Load only frozen TRAIN and VALIDATION rows with Cu labels. TEST is never opened."""
    metadata, _, _ = load_metadata(metadata_path)
    cu_column = find_column(list(metadata.columns), "Cu")
    metadata_by_key = metadata.set_index("_sample_key")
    frames = []
    for split in ("train", "val"):
        path = splits_dir / f"z903_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Frozen {split} split not found: {path}")
        frame = pd.read_csv(path)
        required = {"target_id", "group_id", "spectrum_path"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Frozen {split} split is missing columns: {sorted(missing)}")
        if frame["target_id"].duplicated().any() or frame["group_id"].duplicated().any():
            raise ValueError(f"Frozen {split} split has duplicate target IDs or groups")
        frame = frame[["target_id", "group_id", "spectrum_path"]].copy()
        frame["split"] = split
        frame["_sample_key"] = frame["target_id"].map(normalize)
        if not frame["_sample_key"].isin(metadata_by_key.index).all():
            raise ValueError(f"Frozen {split} split contains unmatched metadata target IDs")
        # Missing Cu labels are preserved as NaN, never filled with zero.
        frame["cu_ppm"] = pd.to_numeric(frame["_sample_key"].map(metadata_by_key[cu_column]), errors="coerce")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["target_id"].duplicated().any() or combined["group_id"].duplicated().any():
        raise ValueError("Target/sample leakage detected between frozen train and validation splits")
    return combined, cu_column


def filter_supported_range(frame: pd.DataFrame, low: float = LOW_PPM, high: float = HIGH_PPM) -> pd.DataFrame:
    """Rows with a known Cu label within [low, high] ppm, inclusive. Missing labels are excluded, never zeroed."""
    known = frame[frame["cu_ppm"].notna()]
    return known[(known["cu_ppm"] >= low) & (known["cu_ppm"] <= high)].copy()


def numeric_stats_ppm(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0, "minimum": None, "median": None, "maximum": None, "percentile_25": None, "percentile_75": None, "percentile_90": None, "percentile_95": None}
    return {
        "count": int(numeric.size),
        "minimum": float(numeric.min()),
        "median": float(numeric.median()),
        "maximum": float(numeric.max()),
        "percentile_25": float(numeric.quantile(0.25)),
        "percentile_75": float(numeric.quantile(0.75)),
        "percentile_90": float(numeric.quantile(0.90)),
        "percentile_95": float(numeric.quantile(0.95)),
    }


def range_counts(values: pd.Series, ranges: tuple[tuple[str, float, float], ...]) -> dict[str, int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    counts: dict[str, int] = {}
    for index, (label, low, high) in enumerate(ranges):
        mask = (numeric >= low) & (numeric <= high) if index == 0 else (numeric > low) & (numeric <= high)
        counts[label] = int(mask.sum())
    return counts


def load_spectra(frame: pd.DataFrame) -> np.ndarray:
    spectra = []
    for path in frame["spectrum_path"]:
        _, intensity = read_z903_spectrum(path)
        spectra.append(preprocess_spectrum(intensity, "robust"))
    return np.asarray(spectra, dtype=np.float32)


def regression_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    errors = predicted - y_true
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    has_variation = y_true.size > 1 and np.unique(y_true).size > 1 and np.unique(predicted).size > 1
    return {
        "n": int(y_true.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median_absolute_error": float(np.median(np.abs(errors))),
        "r2": float(1 - np.sum(errors**2) / denominator) if denominator else float("nan"),
        "pearson": float(pearsonr(y_true, predicted).statistic) if has_variation else float("nan"),
        "spearman": float(spearmanr(y_true, predicted).statistic) if has_variation else float("nan"),
        "bias": float(np.mean(errors)),
    }


def banded_metrics(y_true: np.ndarray, predicted: np.ndarray, ranges: tuple[tuple[str, float, float], ...]) -> dict[str, dict[str, float | int | str]]:
    result: dict[str, dict[str, float | int | str]] = {}
    for index, (label, low, high) in enumerate(ranges):
        mask = (y_true >= low) & (y_true <= high) if index == 0 else (y_true > low) & (y_true <= high)
        if mask.sum() < MIN_SUBGROUP_SUPPORT:
            result[label] = {"n": int(mask.sum()), "status": "Insufficient validation support"}
        else:
            result[label] = regression_metrics(y_true[mask], predicted[mask])
    return result


def apply_target_transform(values: np.ndarray, mode: Literal["log1p", "raw"]) -> np.ndarray:
    return np.log1p(values) if mode == "log1p" else values.astype(np.float64)


def invert_target_transform(values: np.ndarray, mode: Literal["log1p", "raw"]) -> np.ndarray:
    result = np.expm1(values) if mode == "log1p" else values
    return np.clip(result, 0.0, None)


def fit_ridge_candidates(
    pooled_train: np.ndarray, y_train: np.ndarray, pooled_val: np.ndarray, y_val: np.ndarray
) -> tuple[Ridge, StandardScaler, str, float, dict[str, Any]]:
    scaler = StandardScaler().fit(pooled_train)
    scaled_train, scaled_val = scaler.transform(pooled_train), scaler.transform(pooled_val)
    grid: list[dict[str, Any]] = []
    best: tuple[Ridge, str, float, dict[str, Any]] | None = None
    for mode in ("log1p", "raw"):
        target_train = apply_target_transform(y_train, mode)
        for alpha in RIDGE_ALPHAS:
            model = Ridge(alpha=alpha, solver="lsqr").fit(scaled_train, target_train)
            prediction = invert_target_transform(model.predict(scaled_val), mode)
            metrics = regression_metrics(y_val, prediction)
            entry = {"target_mode": mode, "alpha": alpha, "metrics": metrics}
            grid.append(entry)
            if best is None or metrics["mae"] < best[3]["metrics"]["mae"]:
                best = (model, mode, alpha, entry)
    assert best is not None
    model, mode, alpha, entry = best
    return model, scaler, mode, alpha, {"grid": grid, "selected": entry}


def fit_pls_candidates(
    pooled_train: np.ndarray, y_train: np.ndarray, pooled_val: np.ndarray, y_val: np.ndarray
) -> tuple[PLSRegression, StandardScaler, str, int, dict[str, Any]]:
    scaler = StandardScaler().fit(pooled_train)
    scaled_train, scaled_val = scaler.transform(pooled_train), scaler.transform(pooled_val)
    components_candidates = [c for c in PLS_COMPONENTS_GRID if c < len(y_train)] or [max(2, min(2, len(y_train) - 1))]
    grid: list[dict[str, Any]] = []
    best: tuple[PLSRegression, str, int, dict[str, Any]] | None = None
    for mode in ("log1p", "raw"):
        target_train = apply_target_transform(y_train, mode)
        for components in components_candidates:
            model = PLSRegression(n_components=components).fit(scaled_train, target_train)
            prediction = invert_target_transform(model.predict(scaled_val).ravel(), mode)
            metrics = regression_metrics(y_val, prediction)
            entry = {"target_mode": mode, "components": components, "metrics": metrics}
            grid.append(entry)
            if best is None or metrics["mae"] < best[3]["metrics"]["mae"]:
                best = (model, mode, components, entry)
    assert best is not None
    model, mode, components, entry = best
    return model, scaler, mode, components, {"grid": grid, "selected": entry}


def train_one_cnn(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
    mode: Literal["log1p", "raw"], epochs: int,
) -> tuple[Z903CNN, np.ndarray, int, float, float]:
    model = Z903CNN(output_size=1)
    target_train = apply_target_transform(y_train, mode)
    mean, scale = float(np.mean(target_train)), float(np.std(target_train)) or 1.0
    normalized = ((target_train - mean) / scale).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train[:, None, :]), torch.from_numpy(normalized[:, None])), batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()
    best_state, best_mae, best_epoch, patience_counter = None, np.inf, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            scaled = model(torch.from_numpy(x_val[:, None, :])).numpy().ravel()
        prediction = invert_target_transform(scaled * scale + mean, mode)
        mae = float(np.mean(np.abs(prediction - y_val)))
        if mae < best_mae:
            best_mae, best_epoch, patience_counter = mae, epoch, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= CNN_PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, best_state, best_epoch, mean, scale  # type: ignore[return-value]


def predict_cnn(model: Z903CNN, spectra: np.ndarray, mode: Literal["log1p", "raw"], mean: float, scale: float) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        scaled = model(torch.from_numpy(spectra[:, None, :])).numpy().ravel()
    return invert_target_transform(scaled * scale + mean, mode)


def fit_cnn_candidates(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, epochs: int
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for mode in ("log1p", "raw"):
        model, _, best_epoch, mean, scale = train_one_cnn(x_train, y_train, x_val, y_val, mode, epochs)
        prediction = predict_cnn(model, x_val, mode, mean, scale)
        metrics = regression_metrics(y_val, prediction)
        results[mode] = {"model": model, "best_epoch": best_epoch, "mean": mean, "scale": scale, "metrics": metrics, "prediction": prediction}
    winner_mode = min(results, key=lambda mode: results[mode]["metrics"]["mae"])
    return {"winner_mode": winner_mode, "candidates": results}


def select_final_model(candidate_metrics: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """Select ONE final model from validation metrics only. Disqualifies severe rank/collapse failures first."""
    eligible = {
        name: metrics for name, metrics in candidate_metrics.items()
        if not (np.isnan(metrics.get("pearson", float("nan"))) and metrics["n"] > 1) and metrics.get("pearson", 0.0) > 0
    }
    pool = eligible if eligible else candidate_metrics
    disqualifier_note = "" if eligible else " (no candidate had a positive Pearson correlation; selecting by lowest MAE only)"
    winner = min(pool, key=lambda name: pool[name]["mae"])
    reason = (
        f"Selected '{winner}' by lowest validation MAE ({pool[winner]['mae']:.4f} ppm) among candidates with a "
        f"positive validation Pearson correlation{disqualifier_note}. R2={pool[winner]['r2']:.4f}, "
        f"Pearson={pool[winner]['pearson']:.4f}, Spearman={pool[winner]['spearman']:.4f}."
    )
    return winner, reason


def out_of_range_counts(frame: pd.DataFrame) -> dict[str, int]:
    known = frame[frame["cu_ppm"].notna()]
    return {
        "train_over_500": int(((known["split"] == "train") & (known["cu_ppm"] > HIGH_PPM)).sum()),
        "val_over_500": int(((known["split"] == "val") & (known["cu_ppm"] > HIGH_PPM)).sum()),
    }


def warning_classifier_feasible(train_over: int, val_over: int, min_train: int = 10, min_val: int = 5) -> bool:
    return train_over >= min_train and val_over >= min_val


def occlusion_check(predict_fn: Callable[[np.ndarray], np.ndarray], spectra: np.ndarray, output: Path, model_name: str) -> str:
    baseline = predict_fn(spectra)
    lines = []
    for wavelength in CU_NIST_LINES_NM:
        masked = spectra.copy()
        window = np.abs(WAVELENGTHS - wavelength) <= 0.5
        masked[:, window] = 0.0
        change = predict_fn(masked) - baseline
        lines.append((wavelength, float(np.mean(change)), float(np.mean(np.abs(change)))))
    body = [
        "# Cu Supported-Range Model Physics Sanity Check", "",
        f"A validation-only wavelength-occlusion diagnostic was applied to the selected '{model_name}' Cu model. "
        "It is not a physical validation or proof of line identification. NIST wavelengths were not forced "
        "into the model in any way.", "",
        "| Occluded region (Cu I reference) | Mean prediction change (ppm) | Mean absolute prediction change (ppm) |",
        "| --- | ---: | ---: |",
    ]
    body.extend(f"| {wavelength:.3f} nm +/- 0.5 nm | {mean_change:.4f} | {absolute_change:.4f} |" for wavelength, mean_change, absolute_change in lines)
    body.extend([
        "",
        "Occluding these reference regions changing the prediction is consistent with, but does not prove, "
        "the model using physically relevant Cu emission regions. Matrix correlations and blended lines "
        "remain possible confounds that this check cannot rule out.",
    ])
    path = output / "physics_sanity_check.md"
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    summary = ", ".join(f"{wavelength:.2f} nm: mean change {mean_change:+.4f} ppm" for wavelength, mean_change, _ in lines)
    return f"Validation-only CNN-style wavelength occlusion at Cu I reference lines ({summary}); see physics_sanity_check.md."


def save_plots(y_true: np.ndarray, predicted: np.ndarray, output: Path) -> None:
    limit = max(float(y_true.max()), float(predicted.max())) * 1.05

    figure, axis = plt.subplots(figsize=(4.5, 4.5), dpi=150)
    axis.scatter(y_true, predicted, color="#176b87", s=28)
    axis.plot([0, limit], [0, limit], color="black", linewidth=1)
    axis.set(xlabel="True Cu (ppm)", ylabel="Predicted Cu (ppm)")
    figure.tight_layout()
    figure.savefig(output / "supported_range_scatter.png")
    plt.close(figure)

    log_true, log_pred = np.log1p(y_true), np.log1p(predicted)
    log_limit = max(float(log_true.max()), float(log_pred.max())) * 1.05
    figure, axis = plt.subplots(figsize=(4.5, 4.5), dpi=150)
    axis.scatter(log_true, log_pred, color="#4c78a8", s=28)
    axis.plot([0, log_limit], [0, log_limit], color="black", linewidth=1)
    axis.set(xlabel="log1p(True Cu ppm)", ylabel="log1p(Predicted Cu ppm)")
    figure.tight_layout()
    figure.savefig(output / "supported_range_log_scatter.png")
    plt.close(figure)

    residual = predicted - y_true
    figure, axis = plt.subplots(figsize=(5, 4), dpi=150)
    axis.scatter(y_true, residual, color="#e15759", s=28)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="True Cu (ppm)", ylabel="Prediction residual (ppm)", xlim=(0, limit))
    figure.tight_layout()
    figure.savefig(output / "supported_range_residuals.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 4), dpi=150)
    axis.scatter(y_true, np.abs(residual), color="#f28e2b", s=28)
    axis.set(xlabel="True Cu (ppm)", ylabel="Absolute error (ppm)", xlim=(0, limit))
    figure.tight_layout()
    figure.savefig(output / "error_vs_concentration.png")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    frame, cu_column = load_train_validation_cu(ROOT / args.splits_dir, ROOT / args.metadata)
    train_supported = filter_supported_range(frame[frame["split"] == "train"])
    val_supported = filter_supported_range(frame[frame["split"] == "val"])

    # ---- Part A ----
    part_a = {
        "train": {**numeric_stats_ppm(train_supported["cu_ppm"]), "range_counts": range_counts(train_supported["cu_ppm"], PART_A_RANGES)},
        "validation": {**numeric_stats_ppm(val_supported["cu_ppm"]), "range_counts": range_counts(val_supported["cu_ppm"], PART_A_RANGES)},
    }

    x_train, x_val = load_spectra(train_supported), load_spectra(val_supported)
    y_train, y_val = train_supported["cu_ppm"].to_numpy(dtype=np.float64), val_supported["cu_ppm"].to_numpy(dtype=np.float64)
    pooled_train, pooled_val = average_pool(x_train), average_pool(x_val)

    # ---- Part B: median baseline ----
    median_prediction = np.full(y_val.shape, np.median(y_train))
    baseline_metrics = regression_metrics(y_val, median_prediction)

    # ---- Part C: Ridge ----
    ridge_model, ridge_scaler, ridge_mode, ridge_alpha, ridge_details = fit_ridge_candidates(pooled_train, y_train, pooled_val, y_val)
    ridge_prediction = invert_target_transform(ridge_model.predict(ridge_scaler.transform(pooled_val)), ridge_mode)
    ridge_metrics = regression_metrics(y_val, ridge_prediction)

    # ---- Part D: PLSR ----
    pls_model, pls_scaler, pls_mode, pls_components, pls_details = fit_pls_candidates(pooled_train, y_train, pooled_val, y_val)
    pls_prediction = invert_target_transform(pls_model.predict(pls_scaler.transform(pooled_val)).ravel(), pls_mode)
    pls_metrics = regression_metrics(y_val, pls_prediction)

    # ---- Part E: 1D-CNN ----
    cnn_result = fit_cnn_candidates(x_train, y_train, x_val, y_val, args.epochs)
    cnn_mode = cnn_result["winner_mode"]
    cnn_candidate = cnn_result["candidates"][cnn_mode]
    cnn_model, cnn_mean, cnn_scale = cnn_candidate["model"], cnn_candidate["mean"], cnn_candidate["scale"]
    cnn_prediction = cnn_candidate["prediction"]
    cnn_metrics = cnn_candidate["metrics"]

    # ---- Part F: comparison ----
    predictions_by_model = {
        "median_baseline": median_prediction,
        "ridge": ridge_prediction,
        "pls": pls_prediction,
        "cnn": cnn_prediction,
    }
    metrics_by_model = {
        "median_baseline": baseline_metrics,
        "ridge": ridge_metrics,
        "pls": pls_metrics,
        "cnn": cnn_metrics,
    }
    banded_by_model = {name: banded_metrics(y_val, prediction, PART_F_RANGES) for name, prediction in predictions_by_model.items()}

    # ---- Part G: selection ----
    winner_name, selection_reason = select_final_model(metrics_by_model)
    winner_prediction = predictions_by_model[winner_name]
    winner_metrics = metrics_by_model[winner_name]

    contract: dict[str, Any] = {
        "element": "Cu",
        "unit": "ppm",
        "supported_range_ppm": [LOW_PPM, HIGH_PPM],
        "input_length": 23401,
        "wavelength_range_nm": [180.0, 960.0],
        "existing_cu_detection_threshold_ppm": EXISTING_CU_DETECTION_THRESHOLD_PPM,
        "threshold_modified": False,
        "winner_model": winner_name,
        "preprocessing": "robust median/IQR scaling per spectrum" + ("" if winner_name == "cnn" or winner_name == "median_baseline" else f"; average-pooled bin_size={BIN_SIZE}; StandardScaler fit on train only"),
    }
    if winner_name == "ridge":
        contract["target_transformation"] = f"{ridge_mode}(Cu_ppm), train-only scaling; alpha={ridge_alpha}"
    elif winner_name == "pls":
        contract["target_transformation"] = f"{pls_mode}(Cu_ppm), train-only scaling; components={pls_components}"
    elif winner_name == "cnn":
        contract["target_transformation"] = f"{cnn_mode}(Cu_ppm), train-only standardization"
    else:
        contract["target_transformation"] = "train-only median of Cu_ppm within supported range"
    contract["validation_metrics"] = winner_metrics
    contract["known_limitations"] = (
        "Restricted to the 1-500 ppm supported range determined from train/validation label density; "
        "1-500 ppm is a data-support boundary, not a physical detection limit. Out-of-range behavior above "
        "500 ppm is not calibrated; see out_of_range_analysis.md."
    )
    contract["selection_reason"] = selection_reason
    contract["test_set_status"] = "NOT USED FOR MODEL SELECTION OR PERFORMANCE EVALUATION"

    if winner_name == "ridge":
        joblib.dump({"model": ridge_model, "scaler": ridge_scaler, "bin_size": BIN_SIZE, "preprocess": "robust", "target_transform": ridge_mode, "alpha": ridge_alpha}, MODELS / "cu_supported_range_final.joblib")
    elif winner_name == "pls":
        joblib.dump({"model": pls_model, "scaler": pls_scaler, "bin_size": BIN_SIZE, "preprocess": "robust", "target_transform": pls_mode, "components": pls_components}, MODELS / "cu_supported_range_final.joblib")
    elif winner_name == "cnn":
        torch.save({"model_state_dict": cnn_model.state_dict(), "model_name": "Z903CNN", "output_size": 1, "preprocess": "robust", "target_transform": cnn_mode, "target_mean": cnn_mean, "target_scale": cnn_scale, "best_epoch": cnn_candidate["best_epoch"]}, MODELS / "cu_supported_range_final.pt")
    else:
        joblib.dump({"model_name": "median_baseline", "prediction_ppm": float(np.median(y_train))}, MODELS / "cu_supported_range_final.joblib")

    # ---- Part H: out-of-range analysis ----
    over_counts = out_of_range_counts(frame)
    feasible = warning_classifier_feasible(over_counts["train_over_500"], over_counts["val_over_500"])
    warning_result: dict[str, Any] = {"train_over_500": over_counts["train_over_500"], "val_over_500": over_counts["val_over_500"], "feasible": feasible}
    if feasible:
        known = frame[frame["cu_ppm"].notna() & (frame["cu_ppm"] >= LOW_PPM)]
        train_known, val_known = known[known["split"] == "train"], known[known["split"] == "val"]
        x_all_train, x_all_val = load_spectra(train_known), load_spectra(val_known)
        pooled_all_train, pooled_all_val = average_pool(x_all_train), average_pool(x_all_val)
        classifier_scaler = StandardScaler().fit(pooled_all_train)
        y_class_train = (train_known["cu_ppm"].to_numpy() > HIGH_PPM).astype(int)
        y_class_val = (val_known["cu_ppm"].to_numpy() > HIGH_PPM).astype(int)
        classifier = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED).fit(classifier_scaler.transform(pooled_all_train), y_class_train)
        probability = classifier.predict_proba(classifier_scaler.transform(pooled_all_val))[:, 1]
        predicted = (probability >= 0.5).astype(int)
        matrix = confusion_matrix(y_class_val, predicted, labels=[0, 1]).tolist()
        warning_result["balanced_accuracy"] = float(balanced_accuracy_score(y_class_val, predicted))
        warning_result["confusion_matrix"] = matrix
        warning_result["conclusion"] = "A logistic-regression out-of-range warning classifier was fitted and evaluated on validation."
        joblib.dump({"model": classifier, "scaler": classifier_scaler, "bin_size": BIN_SIZE, "preprocess": "robust", "positive_class": "Cu > 500 ppm"}, MODELS / "cu_out_of_range_warning.joblib")
    else:
        warning_result["conclusion"] = "Insufficient data to reliably identify out-of-range high-Cu samples."

    out_of_range_lines = [
        "# Cu Out-of-Range Analysis", "",
        f"- TRAIN known Cu labels > {HIGH_PPM:g} ppm: {over_counts['train_over_500']}",
        f"- VALIDATION known Cu labels > {HIGH_PPM:g} ppm: {over_counts['val_over_500']}",
        f"- Warning classifier feasibility threshold: at least 10 TRAIN and 5 VALIDATION known labels above {HIGH_PPM:g} ppm.",
        f"- Warning model feasible: {feasible}", "",
        f"**Conclusion:** {warning_result['conclusion']}", "",
    ]
    if feasible:
        out_of_range_lines.extend([
            f"- Logistic-regression balanced accuracy (validation): {warning_result['balanced_accuracy']:.4f}",
            f"- Confusion matrix [[TN, FP], [FN, TP]]: {warning_result['confusion_matrix']}",
        ])
    out_of_range_lines.extend([
        "",
        "Predictions from the supported-range regression model must not be trusted for spectra flagged as "
        "out-of-range (or, absent a feasible classifier, for any spectrum suspected to exceed 500 ppm); a "
        "true high-Cu sample could otherwise be silently regressed into the 1-500 ppm supported range.",
        "", "TEST was not used anywhere in this analysis.",
    ])
    (output / "out_of_range_analysis.md").write_text("\n".join(out_of_range_lines) + "\n", encoding="utf-8")

    # ---- Part I: physics sanity check ----
    if winner_name == "ridge":
        predict_fn: Callable[[np.ndarray], np.ndarray] = lambda spectra: invert_target_transform(ridge_model.predict(ridge_scaler.transform(average_pool(spectra))), ridge_mode)  # noqa: E731
    elif winner_name == "pls":
        predict_fn = lambda spectra: invert_target_transform(pls_model.predict(pls_scaler.transform(average_pool(spectra))).ravel(), pls_mode)  # noqa: E731
    elif winner_name == "cnn":
        predict_fn = lambda spectra: predict_cnn(cnn_model, spectra, cnn_mode, cnn_mean, cnn_scale)  # noqa: E731
    else:
        constant = float(np.median(y_train))
        predict_fn = lambda spectra: np.full(spectra.shape[0], constant)  # noqa: E731
    physics_summary = occlusion_check(predict_fn, x_val, output, winner_name)

    # ---- Part J: outputs ----
    save_plots(y_val, winner_prediction, output)
    pd.DataFrame({
        "target_id": val_supported["target_id"].to_numpy(),
        "true_cu_ppm": y_val,
        "median_baseline_ppm": median_prediction,
        "ridge_ppm": ridge_prediction,
        "pls_ppm": pls_prediction,
        "cnn_ppm": cnn_prediction,
        "winner_model": winner_name,
        "winner_prediction_ppm": winner_prediction,
    }).to_csv(output / "supported_range_validation_predictions.csv", index=False)

    range_performance_rows = []
    for model_name, bands in banded_by_model.items():
        for band_label, band_metrics in bands.items():
            row = {"model": model_name, "range": band_label, **band_metrics}
            range_performance_rows.append(row)
    pd.DataFrame(range_performance_rows).to_csv(output / "range_performance.csv", index=False)

    comparison = {
        "development_scope": "Frozen train and validation only; z903_test.csv was never opened by this script.",
        "cu_metadata_column": cu_column,
        "supported_range_ppm": [LOW_PPM, HIGH_PPM],
        "part_a_support": part_a,
        "models": {
            "median_baseline": {"metrics": baseline_metrics, "bands": banded_by_model["median_baseline"]},
            "ridge": {"target_mode": ridge_mode, "alpha": ridge_alpha, "metrics": ridge_metrics, "bands": banded_by_model["ridge"], "grid": ridge_details["grid"]},
            "pls": {"target_mode": pls_mode, "components": pls_components, "metrics": pls_metrics, "bands": banded_by_model["pls"], "grid": pls_details["grid"]},
            "cnn": {"target_mode": cnn_mode, "best_epoch": cnn_candidate["best_epoch"], "metrics": cnn_metrics, "bands": banded_by_model["cnn"], "other_target_mode_metrics": cnn_result["candidates"][[m for m in ("log1p", "raw") if m != cnn_mode][0]]["metrics"]},
        },
        "winner": winner_name,
        "selection_reason": selection_reason,
        "out_of_range_analysis": warning_result,
        "physics_sanity_check": physics_summary,
        "test_set_status": "NOT USED FOR MODEL SELECTION OR PERFORMANCE EVALUATION",
    }
    (output / "model_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (output / "model_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Cu Supported-Range (1-500 ppm) Model Comparison", "",
        "Train/validation only. TEST was never opened by this script. See `model_comparison.json` for full grids and per-model band metrics.",
        "", "## Winner", "",
        f"- Model: {winner_name}",
        f"- Reason: {selection_reason}", "",
        "## Summary", "",
        "| Model | MAE | RMSE | Median AE | R2 | Pearson | Spearman | Bias |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in metrics_by_model.items():
        md_lines.append(f"| {name} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | {metrics['median_absolute_error']:.4f} | {metrics['r2']:.4f} | {metrics['pearson']:.4f} | {metrics['spearman']:.4f} | {metrics['bias']:.4f} |")
    md_lines.extend(["", "See `range_performance.csv` for per-concentration-band metrics for every model."])
    (output / "model_comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("CU SUPPORTED-RANGE MODEL DEVELOPMENT")
    print("TRAIN/VALIDATION ONLY")
    print()
    print("SUPPORTED RANGE:")
    print(f"{LOW_PPM:g}-{HIGH_PPM:g} ppm")
    print()
    print("TRAIN SUPPORT")
    print(f"n: {part_a['train']['count']}")
    print(f"Median: {part_a['train']['median']}")
    print(f"P95: {part_a['train']['percentile_95']}")
    print()
    print("VALIDATION SUPPORT")
    print(f"n: {part_a['validation']['count']}")
    print(f"Median: {part_a['validation']['median']}")
    print(f"P95: {part_a['validation']['percentile_95']}")
    print()
    print("MODEL COMPARISON")
    print()
    print("Median baseline")
    print(f"MAE: {baseline_metrics['mae']}")
    print(f"RMSE: {baseline_metrics['rmse']}")
    print(f"R2: {baseline_metrics['r2']}")
    print()
    print("Ridge")
    print(f"MAE: {ridge_metrics['mae']}")
    print(f"RMSE: {ridge_metrics['rmse']}")
    print(f"R2: {ridge_metrics['r2']}")
    print(f"Pearson: {ridge_metrics['pearson']}")
    print(f"Spearman: {ridge_metrics['spearman']}")
    print()
    print("PLSR")
    print(f"MAE: {pls_metrics['mae']}")
    print(f"RMSE: {pls_metrics['rmse']}")
    print(f"R2: {pls_metrics['r2']}")
    print(f"Pearson: {pls_metrics['pearson']}")
    print(f"Spearman: {pls_metrics['spearman']}")
    print()
    print("CNN")
    print(f"MAE: {cnn_metrics['mae']}")
    print(f"RMSE: {cnn_metrics['rmse']}")
    print(f"R2: {cnn_metrics['r2']}")
    print(f"Pearson: {cnn_metrics['pearson']}")
    print(f"Spearman: {cnn_metrics['spearman']}")
    print()
    print("WINNER")
    print(f"Model: {winner_name}")
    print(f"MAE: {winner_metrics['mae']}")
    print(f"RMSE: {winner_metrics['rmse']}")
    print(f"Median AE: {winner_metrics['median_absolute_error']}")
    print(f"R2: {winner_metrics['r2']}")
    print(f"Pearson: {winner_metrics['pearson']}")
    print(f"Spearman: {winner_metrics['spearman']}")
    print(f"Bias: {winner_metrics['bias']}")
    print()
    print("OUT-OF-RANGE ANALYSIS")
    print(f"Train >500 ppm: {over_counts['train_over_500']}")
    print(f"Validation >500 ppm: {over_counts['val_over_500']}")
    print(f"Warning model feasible: {feasible}")
    print(f"Result: {warning_result['conclusion']}")
    print()
    print(f"PHYSICS SANITY CHECK:\n{physics_summary}")
    print()
    print("TEST STATUS:")
    print("NOT USED FOR MODEL SELECTION OR PERFORMANCE EVALUATION")
    print()
    print("Files created/modified:")
    for name in (
        "model_comparison.json", "model_comparison.md", "supported_range_validation_predictions.csv",
        "supported_range_scatter.png", "supported_range_residuals.png", "supported_range_log_scatter.png",
        "error_vs_concentration.png", "range_performance.csv", "out_of_range_analysis.md",
        "physics_sanity_check.md", "model_contract.json",
    ):
        print(output / name)
    print(MODELS / "cu_supported_range_final.*")
    if feasible:
        print(MODELS / "cu_out_of_range_warning.joblib")


if __name__ == "__main__":
    main()
