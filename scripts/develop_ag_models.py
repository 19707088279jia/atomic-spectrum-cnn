#!/usr/bin/env python
"""Develop Ag reference-metadata models using frozen train/validation rows only."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_z903_training_manifest import find_column, load_metadata  # noqa: E402, I001

from atomic_spectrum_ai.models.z903_cnn import Z903CNN  # noqa: E402
from atomic_spectrum_ai.z903 import WAVELENGTHS, preprocess_spectrum, read_z903_spectrum  # noqa: E402


OUTPUT = ROOT / "reports" / "concentration" / "ag_models"
MODELS = ROOT / "models"
BIN_SIZE = 25
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
PLS_COMPONENTS = (2, 4, 8, 12, 16, 24, 32, 48)
SEED = 20260831


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="data/metadata/libs_metadata.xlsx")
    parser.add_argument("--splits-dir", default="data/processed")
    parser.add_argument("--output-dir", default="reports/concentration/ag_models")
    parser.add_argument("--epochs", type=int, default=30)
    return parser.parse_args()


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def average_pool(values: np.ndarray) -> np.ndarray:
    usable = (values.shape[-1] // BIN_SIZE) * BIN_SIZE
    return values[..., :usable].reshape(*values.shape[:-1], -1, BIN_SIZE).mean(axis=-1)


def load_train_validation(splits_dir: Path, metadata_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    metadata, _, _ = load_metadata(metadata_path)
    ag_column = find_column(list(metadata.columns), "Ag")
    train_path = splits_dir / "z903_train.csv"
    val_path = splits_dir / "z903_val.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Both frozen z903_train.csv and z903_val.csv are required")
    # Deliberately construct only train/validation inputs: the frozen test split is never read.
    frame = load_frozen_splits_train_validation(train_path, val_path, metadata, ag_column)
    return frame[frame["split"] == "train"].copy(), frame[frame["split"] == "val"].copy(), ag_column


def load_frozen_splits_train_validation(
    train_path: Path, val_path: Path, metadata: pd.DataFrame, ag_column: str
) -> pd.DataFrame:
    metadata_by_key = metadata.set_index("_sample_key")
    frames = []
    for split, path in (("train", train_path), ("val", val_path)):
        frame = pd.read_csv(path)
        required = {"target_id", "group_id", "spectrum_path"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Frozen {split} split is missing columns: {sorted(missing)}")
        if frame["target_id"].duplicated().any() or frame["group_id"].duplicated().any():
            raise ValueError(f"Frozen {split} split has duplicate target IDs or groups")
        frame = frame[["target_id", "group_id", "spectrum_path"]].copy()
        frame["split"] = split
        frame["_sample_key"] = frame["target_id"].map(
            lambda value: "".join(char.lower() for char in str(value) if char.isalnum())
        )
        if not frame["_sample_key"].isin(metadata_by_key.index).all():
            raise ValueError(f"Frozen {split} split contains unmatched metadata target IDs")
        frame["ag_ppm"] = pd.to_numeric(
            frame["_sample_key"].map(metadata_by_key[ag_column]), errors="coerce"
        )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["target_id"].duplicated().any() or combined["group_id"].duplicated().any():
        raise ValueError("Target/sample leakage detected between frozen train and validation splits")
    return combined


def load_spectra(frame: pd.DataFrame) -> np.ndarray:
    spectra = []
    for path in frame["spectrum_path"]:
        _, intensity = read_z903_spectrum(path)
        spectra.append(preprocess_spectrum(intensity, "robust"))
    return np.asarray(spectra, dtype=np.float32)


def classifier_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else float("nan")
    return {
        "n": int(y_true.size),
        "threshold": float(threshold),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, predicted, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
    }


def validation_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    return float(max(candidates, key=lambda value: (balanced_accuracy_score(y_true, probabilities >= value), -value)))


def regression_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    errors = predicted - y_true
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": int(y_true.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median_absolute_error": float(np.median(np.abs(errors))),
        "r2": float(1 - np.sum(errors**2) / denominator) if denominator else float("nan"),
        "pearson": float(pearsonr(y_true, predicted).statistic)
        if y_true.size > 1 and np.unique(y_true).size > 1 and np.unique(predicted).size > 1
        else float("nan"),
        "spearman": float(spearmanr(y_true, predicted).statistic)
        if y_true.size > 1 and np.unique(y_true).size > 1 and np.unique(predicted).size > 1
        else float("nan"),
    }


def support_counts(frame: pd.DataFrame) -> dict[str, int]:
    known = frame["ag_ppm"].dropna()
    return {
        "known": int(known.size), "zero": int((known == 0).sum()), "positive": int((known > 0).sum()),
        "positive_gt_1_ppm": int((known > 1).sum()), "positive_gt_10_ppm": int((known > 10).sum()),
    }


def make_cnn() -> Z903CNN:
    return Z903CNN(output_size=1)


def train_cnn_classifier(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, epochs: int) -> tuple[Z903CNN, np.ndarray, int]:
    model = make_cnn()
    positive_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train[:, None, :]), torch.from_numpy(y_train[:, None].astype(np.float32))), batch_size=16, shuffle=True)
    best_state, best_score, best_epoch = None, -np.inf, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(torch.from_numpy(x_val[:, None, :]))).numpy().ravel()
        threshold = validation_threshold(y_val, probabilities)
        score = balanced_accuracy_score(y_val, probabilities >= threshold)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(torch.from_numpy(x_val[:, None, :]))).numpy().ravel()
    return model, probabilities, best_epoch


def train_cnn_regressor(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, epochs: int) -> tuple[Z903CNN, np.ndarray, int, float, float]:
    model = make_cnn()
    mean, scale = float(np.mean(np.log1p(y_train))), float(np.std(np.log1p(y_train))) or 1.0
    target = ((np.log1p(y_train) - mean) / scale).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train[:, None, :]), torch.from_numpy(target[:, None])), batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()
    best_state, best_mae, best_epoch = None, np.inf, 0
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
        prediction = np.expm1(scaled * scale + mean).clip(0.0)
        score = np.mean(np.abs(prediction - y_val))
        if score < best_mae:
            best_mae, best_epoch = score, epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        scaled = model(torch.from_numpy(x_val[:, None, :])).numpy().ravel()
    return model, np.expm1(scaled * scale + mean).clip(0.0), best_epoch, mean, scale


def predict_cnn(model: Z903CNN, spectra: np.ndarray, regression_mean: float | None = None, regression_scale: float | None = None) -> np.ndarray:
    with torch.inference_mode():
        output = model(torch.from_numpy(spectra[:, None, :])).numpy().ravel()
    if regression_mean is None:
        return torch.sigmoid(torch.from_numpy(output)).numpy()
    return np.expm1(output * regression_scale + regression_mean).clip(0.0)


def banded_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, dict[str, float | int | str]]:
    bands = {"0-0.1 ppm": y_true <= 0.1, ">0.1-0.5 ppm": (y_true > 0.1) & (y_true <= 0.5), ">0.5-1 ppm": (y_true > 0.5) & (y_true <= 1), ">1 ppm": y_true > 1}
    return {name: regression_metrics(y_true[mask], predicted[mask]) if mask.sum() >= 5 else {"n": int(mask.sum()), "status": "Insufficient validation support"} for name, mask in bands.items()}


def save_stage1_plot(metrics: dict[str, float | int], output: Path) -> None:
    matrix = np.array([[metrics["true_negative"], metrics["false_positive"]], [metrics["false_negative"], metrics["true_positive"]]])
    figure, axis = plt.subplots(figsize=(4, 3.5), dpi=150)
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks([0, 1], ["Predicted zero", "Predicted positive"])
    axis.set_yticks([0, 1], ["True zero", "True positive"])
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output / "stage1_confusion_matrix.png")
    plt.close(figure)


def save_stage2_plots(y_true: np.ndarray, predicted: np.ndarray, output: Path) -> None:
    limit = max(float(y_true.max()), float(predicted.max())) * 1.05
    plots = (("stage2_scatter.png", y_true, predicted, "True Ag (ppm)", "Predicted Ag (ppm)"), ("stage2_log_scatter.png", np.log1p(y_true), np.log1p(predicted), "log1p(True Ag ppm)", "log1p(Predicted Ag ppm)"))
    for name, horizontal, vertical, xlabel, ylabel in plots:
        figure, axis = plt.subplots(figsize=(4.5, 4.5), dpi=150)
        axis.scatter(horizontal, vertical, color="#176b87", s=28)
        plot_limit = max(float(horizontal.max()), float(vertical.max())) * 1.05
        axis.plot([0, plot_limit], [0, plot_limit], color="black", linewidth=1)
        axis.set(xlabel=xlabel, ylabel=ylabel)
        figure.tight_layout()
        figure.savefig(output / name)
        plt.close(figure)
    residual = predicted - y_true
    for name, vertical, ylabel in (("stage2_residuals.png", residual, "Prediction residual (ppm)"), ("stage2_error_vs_concentration.png", np.abs(residual), "Absolute error (ppm)")):
        figure, axis = plt.subplots(figsize=(5, 4), dpi=150)
        axis.scatter(y_true, vertical, color="#e15759", s=28)
        axis.axhline(0, color="black", linewidth=1)
        axis.set(xlabel="True Ag (ppm)", ylabel=ylabel, xlim=(0, limit))
        figure.tight_layout()
        figure.savefig(output / name)
        plt.close(figure)


def occlusion_check(model: Z903CNN, spectra: np.ndarray, output: Path) -> str:
    baseline = predict_cnn(model, spectra)
    lines = []
    for wavelength in (328.07, 338.29):
        masked = spectra.copy()
        window = np.abs(WAVELENGTHS - wavelength) <= 0.5
        masked[:, window] = 0.0
        change = predict_cnn(model, masked) - baseline
        lines.append((wavelength, float(np.mean(change)), float(np.mean(np.abs(change)))))
    body = ["# Ag CNN Physics Sanity Check", "", "A validation-only wavelength-occlusion diagnostic was applied to the trained Ag CNN candidate. It is not a physical validation or evidence of line identification.", "", "| Occluded region | Mean probability change | Mean absolute probability change |", "| --- | ---: | ---: |"]
    body.extend(f"| {wavelength:.2f} nm +/- 0.5 nm | {mean_change:.6f} | {absolute_change:.6f} |" for wavelength, mean_change, absolute_change in lines)
    body.extend(["", "The candidate was not forced to use these regions. Any predictive performance remains separate from demonstrating Ag-specific spectral causality; matrix correlations and blended lines remain possible shortcuts."])
    path = output / "physics_sanity_check.md"
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return "Validation-only CNN wavelength occlusion at 328.07 and 338.29 nm; see physics_sanity_check.md."


def main() -> None:
    args = parse_args()
    set_seed()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    train, validation, ag_column = load_train_validation(ROOT / args.splits_dir, ROOT / args.metadata)
    train_known, val_known = train[train["ag_ppm"].notna()].copy(), validation[validation["ag_ppm"].notna()].copy()
    x_train, x_val = load_spectra(train_known), load_spectra(val_known)
    y_train_class, y_val_class = (train_known["ag_ppm"].to_numpy() > 0).astype(int), (val_known["ag_ppm"].to_numpy() > 0).astype(int)
    pooled_train, pooled_val = average_pool(x_train), average_pool(x_val)
    scaler = StandardScaler().fit(pooled_train)
    logistic = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED).fit(scaler.transform(pooled_train), y_train_class)
    logistic_prob = logistic.predict_proba(scaler.transform(pooled_val))[:, 1]
    logistic_threshold = validation_threshold(y_val_class, logistic_prob)
    cnn_class, cnn_prob, cnn_class_epoch = train_cnn_classifier(x_train, y_train_class, x_val, y_val_class, args.epochs)
    cnn_threshold = validation_threshold(y_val_class, cnn_prob)
    majority_probability = np.full(y_val_class.shape, y_train_class.mean())
    stage1 = {
        "majority_baseline": classifier_metrics(y_val_class, majority_probability, 1.0),
        "logistic_regression": classifier_metrics(y_val_class, logistic_prob, logistic_threshold),
        "cnn": {**classifier_metrics(y_val_class, cnn_prob, cnn_threshold), "best_epoch": cnn_class_epoch},
    }
    stage1_winner = max(("majority_baseline", "logistic_regression", "cnn"), key=lambda name: stage1[name]["balanced_accuracy"])
    if stage1_winner == "logistic_regression":
        classifier_probability, classifier_threshold = logistic_prob, logistic_threshold
        joblib.dump({"model": logistic, "scaler": scaler, "bin_size": BIN_SIZE, "preprocess": "robust", "target": "Ag > 0 ppm reference metadata", "threshold": classifier_threshold}, MODELS / "ag_classifier_final.joblib")
    elif stage1_winner == "cnn":
        classifier_probability, classifier_threshold = cnn_prob, cnn_threshold
        torch.save({"model_state_dict": cnn_class.state_dict(), "model_name": "Z903CNN", "output_size": 1, "preprocess": "robust", "threshold": classifier_threshold, "best_epoch": cnn_class_epoch}, MODELS / "ag_classifier_final.pt")
    else:
        classifier_probability, classifier_threshold = majority_probability, 1.0
        joblib.dump({"model_name": "majority_baseline", "positive_probability": float(y_train_class.mean()), "threshold": 1.0}, MODELS / "ag_classifier_final.joblib")
    winning_stage1_metrics = stage1[stage1_winner]
    save_stage1_plot(winning_stage1_metrics, output)

    train_positive, val_positive = train_known[train_known["ag_ppm"] > 0].copy(), val_known[val_known["ag_ppm"] > 0].copy()
    x_train_positive = x_train[train_known["ag_ppm"].to_numpy() > 0]
    x_val_positive = x_val[val_known["ag_ppm"].to_numpy() > 0]
    y_train_positive, y_val_positive = train_positive["ag_ppm"].to_numpy(), val_positive["ag_ppm"].to_numpy()
    positive_scaler = StandardScaler().fit(average_pool(x_train_positive))
    classical_train, classical_val = positive_scaler.transform(average_pool(x_train_positive)), positive_scaler.transform(average_pool(x_val_positive))
    median_prediction = np.full(y_val_positive.shape, np.median(y_train_positive))
    ridge_grid = []
    for alpha in RIDGE_ALPHAS:
        prediction = np.expm1(Ridge(alpha=alpha, solver="lsqr").fit(classical_train, np.log1p(y_train_positive)).predict(classical_val)).clip(0.0)
        ridge_grid.append({"alpha": alpha, "metrics": regression_metrics(y_val_positive, prediction)})
    ridge_best = min(ridge_grid, key=lambda row: row["metrics"]["mae"])
    ridge = Ridge(alpha=ridge_best["alpha"], solver="lsqr").fit(classical_train, np.log1p(y_train_positive))
    ridge_prediction = np.expm1(ridge.predict(classical_val)).clip(0.0)
    pls_grid = []
    for components in (component for component in PLS_COMPONENTS if component < len(y_train_positive)):
        prediction = np.expm1(PLSRegression(n_components=components).fit(classical_train, np.log1p(y_train_positive)).predict(classical_val).ravel()).clip(0.0)
        pls_grid.append({"components": components, "metrics": regression_metrics(y_val_positive, prediction)})
    pls_best = min(pls_grid, key=lambda row: row["metrics"]["mae"])
    pls = PLSRegression(n_components=pls_best["components"]).fit(classical_train, np.log1p(y_train_positive))
    pls_prediction = np.expm1(pls.predict(classical_val).ravel()).clip(0.0)
    cnn_regression, cnn_regression_prediction, cnn_regression_epoch, target_mean, target_scale = train_cnn_regressor(x_train_positive, y_train_positive, x_val_positive, y_val_positive, args.epochs)
    stage2 = {"positive_median_baseline": regression_metrics(y_val_positive, median_prediction), "ridge_regression": {**regression_metrics(y_val_positive, ridge_prediction), "selected_alpha": ridge_best["alpha"], "grid": ridge_grid}, "pls_regression": {**regression_metrics(y_val_positive, pls_prediction), "selected_components": pls_best["components"], "grid": pls_grid}, "cnn_regression": {**regression_metrics(y_val_positive, cnn_regression_prediction), "best_epoch": cnn_regression_epoch}}
    stage2_winner = min(stage2, key=lambda name: stage2[name]["mae"])
    stage2_predictions = {"positive_median_baseline": median_prediction, "ridge_regression": ridge_prediction, "pls_regression": pls_prediction, "cnn_regression": cnn_regression_prediction}
    final_regression_prediction = stage2_predictions[stage2_winner]
    if stage2_winner == "ridge_regression":
        joblib.dump({"model": ridge, "scaler": positive_scaler, "bin_size": BIN_SIZE, "preprocess": "robust", "target_transform": "log1p(Ag_ppm)", "selected_alpha": ridge_best["alpha"]}, MODELS / "ag_concentration_final.joblib")
    elif stage2_winner == "pls_regression":
        joblib.dump({"model": pls, "scaler": positive_scaler, "bin_size": BIN_SIZE, "preprocess": "robust", "target_transform": "log1p(Ag_ppm)", "selected_components": pls_best["components"]}, MODELS / "ag_concentration_final.joblib")
    elif stage2_winner == "cnn_regression":
        torch.save({"model_state_dict": cnn_regression.state_dict(), "model_name": "Z903CNN", "output_size": 1, "preprocess": "robust", "target_transform": "log1p(Ag_ppm)", "target_mean": target_mean, "target_scale": target_scale, "best_epoch": cnn_regression_epoch}, MODELS / "ag_concentration_final.pt")
    else:
        joblib.dump({"model_name": "positive_median_baseline", "prediction_ppm": float(np.median(y_train_positive))}, MODELS / "ag_concentration_final.joblib")
    stage2[stage2_winner]["bands"] = banded_metrics(y_val_positive, final_regression_prediction)
    save_stage2_plots(y_val_positive, final_regression_prediction, output)
    pd.DataFrame({"target_id": val_known["target_id"], "true_ag_ppm": val_known["ag_ppm"], "probability_positive": classifier_probability, "predicted_positive": classifier_probability >= classifier_threshold}).to_csv(output / "stage1_validation_predictions.csv", index=False)
    pd.DataFrame({"target_id": val_positive["target_id"], "true_ag_ppm": y_val_positive, "predicted_ag_ppm": final_regression_prediction}).to_csv(output / "stage2_validation_predictions.csv", index=False)
    if stage2_winner == "ridge_regression":
        stage2_all = np.expm1(ridge.predict(positive_scaler.transform(average_pool(x_val)))).clip(0.0)
    elif stage2_winner == "pls_regression":
        stage2_all = np.expm1(pls.predict(positive_scaler.transform(average_pool(x_val))).ravel()).clip(0.0)
    elif stage2_winner == "cnn_regression":
        stage2_all = predict_cnn(cnn_regression, x_val, target_mean, target_scale)
    else:
        stage2_all = np.full(y_val_class.shape, np.median(y_train_positive))
    pipeline_prediction = np.where(classifier_probability >= classifier_threshold, stage2_all, 0.0)
    pipeline_metrics = regression_metrics(val_known["ag_ppm"].to_numpy(), pipeline_prediction)
    pipeline_positive_metrics = regression_metrics(y_val_positive, pipeline_prediction[y_val_class == 1])
    pd.DataFrame({"target_id": val_known["target_id"], "true_ag_ppm": val_known["ag_ppm"], "stage1_probability_positive": classifier_probability, "stage1_predicted_positive": classifier_probability >= classifier_threshold, "stage2_predicted_if_positive_ppm": stage2_all, "final_predicted_ag_ppm": pipeline_prediction}).to_csv(output / "pipeline_validation_predictions.csv", index=False)
    supported = {name: {"train": int(((y_train_positive <= 0.1) if name == "0-0.1 ppm" else (y_train_positive > 0.1) & (y_train_positive <= 0.5) if name == ">0.1-0.5 ppm" else (y_train_positive > 0.5) & (y_train_positive <= 1) if name == ">0.5-1 ppm" else y_train_positive > 1).sum()), "validation": int(((y_val_positive <= 0.1) if name == "0-0.1 ppm" else (y_val_positive > 0.1) & (y_val_positive <= 0.5) if name == ">0.1-0.5 ppm" else (y_val_positive > 0.5) & (y_val_positive <= 1) if name == ">0.5-1 ppm" else y_val_positive > 1).sum())} for name in ("0-0.1 ppm", ">0.1-0.5 ppm", ">0.5-1 ppm", ">1 ppm")}
    meaningful = [name for name, counts in supported.items() if counts["train"] >= 5 and counts["validation"] >= 5]
    physics = occlusion_check(cnn_class, x_val, output)
    comparison: dict[str, Any] = {"development_scope": "Frozen train and validation only; z903_test.csv was not read.", "metadata_column": ag_column, "units": "ppm", "support": {"train": support_counts(train), "validation": support_counts(validation), "positive_range_counts": supported}, "stage1": {"winner": stage1_winner, "models": stage1}, "stage2": {"winner": stage2_winner, "models": stage2}, "pipeline_validation": {"all_known": pipeline_metrics, "true_positive_only": pipeline_positive_metrics}, "physics_sanity_check": physics}
    contract = {"units": "ppm", "expected_input_length": 23401, "wavelength_range_nm": [180.0, 960.0], "preprocessing": "robust median/IQR scaling per spectrum", "stage1_target_definition": "0 when Ag reference metadata is exactly 0 ppm; 1 when Ag reference metadata is > 0 ppm", "stage1_threshold": classifier_threshold, "stage1_winner": stage1_winner, "stage2_target_transformation": "log1p(Ag_ppm), train-positive samples only; train-only standardization for CNN", "stage2_winner": stage2_winner, "supported_concentration_ranges": meaningful, "high_concentration_warning": "Outside well-supported training range", "validation_metrics": {"stage1": winning_stage1_metrics, "stage2": stage2[stage2_winner], "pipeline": pipeline_metrics}, "test_set_status": "NOT USED FOR MODEL SELECTION OR PERFORMANCE EVALUATION"}
    (output / "model_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (output / "model_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    (output / "model_comparison.md").write_text("# Ag Model Development Comparison\n\nTrain/validation only. See `model_comparison.json` for all candidates, grids, metrics, and subgroup support.\n\n## Winners\n\n- Stage 1: " + stage1_winner + "\n- Stage 2: " + stage2_winner + "\n- Supported quantitative ranges (at least 5 train and 5 validation positives): " + (", ".join(meaningful) if meaningful else "none") + "\n", encoding="utf-8")
    print("AG MODEL DEVELOPMENT — TRAIN/VALIDATION ONLY")
    for label, values in (("TRAIN SUPPORT", support_counts(train)), ("VALIDATION SUPPORT", support_counts(validation))):
        print(f"\n{label}")
        print(f"Known: {values['known']}")
        print(f"Zero: {values['zero']}")
        print(f"Positive: {values['positive']}")
        print(f"Positive >1 ppm: {values['positive_gt_1_ppm']}")
        print(f"Positive >10 ppm: {values['positive_gt_10_ppm']}")
    print(f"\nSTAGE 1 WINNER\nModel: {stage1_winner}\nThreshold: {classifier_threshold}\nBalanced accuracy: {winning_stage1_metrics['balanced_accuracy']}\nSensitivity: {winning_stage1_metrics['recall_sensitivity']}\nSpecificity: {winning_stage1_metrics['specificity']}\nF1: {winning_stage1_metrics['f1']}\nROC-AUC: {winning_stage1_metrics['roc_auc']}\nPR-AUC: {winning_stage1_metrics['pr_auc']}")
    winner2 = stage2[stage2_winner]
    print(f"\nSTAGE 2 WINNER\nModel: {stage2_winner}\nMAE: {winner2['mae']}\nRMSE: {winner2['rmse']}\nMedian AE: {winner2['median_absolute_error']}\nR²: {winner2['r2']}\nPearson: {winner2['pearson']}\nSpearman: {winner2['spearman']}")
    print(f"\nFULL TWO-STAGE VALIDATION\nMAE: {pipeline_metrics['mae']}\nRMSE: {pipeline_metrics['rmse']}\nMedian AE: {pipeline_metrics['median_absolute_error']}\nR²: {pipeline_metrics['r2']}\n\nSUPPORTED QUANTITATIVE RANGE: {', '.join(meaningful) if meaningful else 'None with the stated support rule'}\n\nPHYSICS SANITY CHECK: {physics}\n\nTEST SET STATUS:\nNOT USED FOR MODEL SELECTION OR PERFORMANCE EVALUATION\n\nFiles created/modified:\n{output}\n{MODELS / 'ag_classifier_final.*'}\n{MODELS / 'ag_concentration_final.*'}")


if __name__ == "__main__":
    main()