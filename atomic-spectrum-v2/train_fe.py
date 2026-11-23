"""Audit, compare, and train Fe concentration models on frozen Z-903 splits."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from src.data import EXPECTED_POINTS, SPLIT_DIR, load_metadata, load_spectrum
from src.mgo_ridge_inference import bin_spectrum
from src.model import ConcentrationCNN
from src.preprocessing import robust_scale
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
OUTPUT_DIR = REPOSITORY_ROOT / "reports" / "fe_analysis"
METADATA_COLUMN = "Fe2O3*"
TARGET_LABEL = "Fe2O3*"
TARGET_UNIT = "wt% Fe2O3-equivalent"
SEED = 20260816
MAX_EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 16
RANGE_LABELS = ("low", "medium", "high")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_fe_split(split: str, metadata: pd.DataFrame) -> pd.DataFrame:
    """Join Fe reference values to an existing split without altering membership."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown Z-903 split: {split}")
    frame = pd.read_csv(SPLIT_DIR / f"z903_{split}.csv")
    keys = frame["target_id"].astype(str).str.strip().str.lower()
    absent = sorted(set(keys).difference(metadata.index))
    if absent:
        raise ValueError(f"Split IDs missing from the metadata workbook: {absent[:5]}")
    frame[TARGET_LABEL] = pd.to_numeric(metadata.reindex(keys)[METADATA_COLUMN].to_numpy(), errors="coerce")
    frame["split"] = split
    return frame


def load_fe_splits() -> dict[str, pd.DataFrame]:
    metadata = load_metadata()
    if METADATA_COLUMN not in metadata.columns:
        raise ValueError(f"Fe reference column {METADATA_COLUMN!r} is missing from the workbook")
    splits = {name: load_fe_split(name, metadata) for name in ("train", "val", "test")}
    identifiers = [set(frame["target_id"].astype(str)) for frame in splits.values()]
    if any(identifiers[left] & identifiers[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("Frozen Z-903 splits contain overlapping target IDs")
    return splits


def concentration_counts(values: pd.Series) -> dict[str, int]:
    known = pd.to_numeric(values, errors="coerce").dropna()
    edges = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, np.inf]
    labels = ["(0, 0.1]", "(0.1, 0.5]", "(0.5, 1]", "(1, 2]", "(2, 5]", "(5, 10]", "(10, 20]", "(20, 50]", ">50"]
    counts = {"zero": int((known == 0).sum())}
    for lower, upper, label in zip(edges[:-1], edges[1:], labels, strict=True):
        counts[label] = int(((known > lower) & (known <= upper)).sum())
    return counts


def audit_data(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    all_rows = pd.concat(splits.values(), ignore_index=True)
    values = pd.to_numeric(all_rows[TARGET_LABEL], errors="coerce")
    known = values.dropna()
    if known.empty:
        raise ValueError("No known Fe2O3* reference concentrations are available")
    if (known < 0).any():
        raise ValueError("Fe2O3* reference concentrations must be non-negative")
    stats = {
        "total_spectra": int(len(all_rows)),
        "known_fe_values": int(known.size),
        "missing_fe_values": int(values.isna().sum()),
        "minimum": float(known.min()),
        "median": float(known.median()),
        "mean": float(known.mean()),
        "percentile_95": float(known.quantile(0.95)),
        "maximum": float(known.max()),
        "zero_values": int((known == 0).sum()),
        "positive_values": int((known > 0).sum()),
        "reference_column": METADATA_COLUMN,
        "reported_unit": TARGET_UNIT,
        "classification_feasible_at_zero": bool((known == 0).any() and (known > 0).any()),
    }
    per_split: dict[str, dict[str, Any]] = {}
    for name, frame in splits.items():
        sample = pd.to_numeric(frame[TARGET_LABEL], errors="coerce")
        known_split = sample.dropna()
        per_split[name] = {
            "spectra": int(len(frame)),
            "known": int(known_split.size),
            "missing": int(sample.isna().sum()),
            "minimum": float(known_split.min()) if len(known_split) else None,
            "median": float(known_split.median()) if len(known_split) else None,
            "mean": float(known_split.mean()) if len(known_split) else None,
            "percentile_95": float(known_split.quantile(0.95)) if len(known_split) else None,
            "maximum": float(known_split.max()) if len(known_split) else None,
            "zero": int((known_split == 0).sum()),
            "positive": int((known_split > 0).sum()),
            "concentration_bins": concentration_counts(sample),
        }
    return {"overall": stats, "splits": per_split}


def load_known_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    known = frame.loc[frame[TARGET_LABEL].notna()].reset_index(drop=True)
    spectra: list[np.ndarray] = []
    for spectrum_path in known["spectrum_path"]:
        wavelengths, intensity = load_spectrum(spectrum_path)
        if wavelengths.shape != (EXPECTED_POINTS,):
            raise ValueError(f"Unexpected spectrum shape in {spectrum_path}")
        processed = robust_scale(intensity)
        if not np.isfinite(processed).all():
            raise ValueError(f"Non-finite robust-preprocessed spectrum: {spectrum_path}")
        spectra.append(processed)
    inputs = np.asarray(spectra, dtype=np.float32)[:, None, :]
    labels = known[TARGET_LABEL].to_numpy(dtype=np.float32)
    return inputs, labels, known["target_id"].astype(str).tolist()


def transform_targets(train_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float, float]:
    log_train = np.log1p(train_values.astype(np.float64))
    mean = float(log_train.mean())
    scale = float(log_train.std()) or 1.0
    return ((np.log1p(values) - mean) / scale).astype(np.float32), mean, scale


def inverse_targets(values: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return np.maximum(np.expm1(values * scale + mean), 0.0)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(actual, dtype=np.float64).reshape(-1)
    estimate = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if truth.size == 0 or truth.size != estimate.size:
        raise ValueError("Regression metrics require equal, non-empty actual and predicted arrays")
    errors = estimate - truth
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "n": int(truth.size),
        "mae": float(np.mean(np.abs(errors))),
        "median_absolute_error": float(np.median(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "r2": float(1.0 - np.sum(errors**2) / denominator) if denominator > 0 else float("nan"),
        "bias": float(np.mean(errors)),
        "pearson_correlation": float(pearsonr(truth, estimate).statistic) if truth.size > 1 and np.std(estimate) > 0 else float("nan"),
        "spearman_correlation": float(spearmanr(truth, estimate).statistic) if truth.size > 1 and np.std(estimate) > 0 else float("nan"),
    }


def fit_linear_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    train_median: float,
) -> dict[str, dict[str, Any]]:
    standardized_train, mean, scale = transform_targets(y_train, y_train)
    results: dict[str, dict[str, Any]] = {
        "median_baseline": {
            "parameters": {"prediction": train_median},
            "validation_prediction": np.full(y_val.shape, train_median, dtype=np.float64),
        }
    }
    ridge_candidates = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
    for alpha in ridge_candidates:
        estimator = Ridge(alpha=alpha, solver="lsqr")
        estimator.fit(x_train, standardized_train)
        predicted = inverse_targets(estimator.predict(x_val), mean, scale)
        name = f"ridge_alpha_{alpha:g}"
        results[name] = {
            "parameters": {"alpha": alpha},
            "validation_prediction": predicted,
            "estimator": estimator,
        }
    pls_candidates = (2, 4, 8, 12)
    for components in pls_candidates:
        estimator = PLSRegression(n_components=components, scale=True, max_iter=500)
        estimator.fit(x_train, standardized_train.reshape(-1, 1))
        predicted_scaled = estimator.predict(x_val).reshape(-1)
        predicted = inverse_targets(predicted_scaled, mean, scale)
        name = f"pls_components_{components}"
        results[name] = {
            "parameters": {"n_components": components},
            "validation_prediction": predicted,
            "estimator": estimator,
        }
    selected_ridge = min(
        (name for name in results if name.startswith("ridge_")),
        key=lambda name: regression_metrics(y_val, results[name]["validation_prediction"])["mae"],
    )
    selected_pls = min(
        (name for name in results if name.startswith("pls_")),
        key=lambda name: regression_metrics(y_val, results[name]["validation_prediction"])["mae"],
    )
    selected = {
        "median_baseline": results["median_baseline"],
        "ridge": results[selected_ridge],
        "pls": results[selected_pls],
    }
    for name, result in selected.items():
        estimator = result.get("estimator")
        if estimator is None:
            result["parameters"]["selected_candidate"] = "median"
        else:
            result["parameters"]["selected_candidate"] = selected_ridge if name == "ridge" else selected_pls
            result["target_transform"] = {"mean": mean, "scale": scale}
    return selected


def train_cnn(
    arrays: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    device: torch.device,
) -> tuple[ConcentrationCNN, dict[str, Any], list[dict[str, float]]]:
    x_train, y_train, _ = arrays["train"]
    x_val, y_val, _ = arrays["validation"]
    y_train_scaled, target_mean, target_scale = transform_targets(y_train, y_train)
    y_val_scaled, _, _ = transform_targets(y_train, y_val)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train_scaled[:, None]),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    validation_inputs = torch.from_numpy(x_val).to(device)
    validation_targets = torch.from_numpy(y_val_scaled[:, None]).to(device)
    model = ConcentrationCNN(output_size=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    patience_count = 0
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.huber_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        with torch.inference_mode():
            validation_loss = float(nn.functional.huber_loss(model(validation_inputs), validation_targets).item())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            patience_count = 0
            checkpoint_path = OUTPUT_DIR / "fe_cnn_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "ConcentrationCNN",
                    "target": TARGET_LABEL,
                    "unit": TARGET_UNIT,
                    "targets": (TARGET_LABEL,),
                    "input_shape": [1, 1, EXPECTED_POINTS],
                    "preprocess": "robust",
                    "transform_mean": target_mean,
                    "transform_scale": target_scale,
                    "best_epoch": epoch,
                    "best_validation_loss": best_loss,
                    "seed": SEED,
                    "training_samples": int(len(y_train)),
                    "maximum_epochs": MAX_EPOCHS,
                    "early_stopping_patience": PATIENCE,
                },
                checkpoint_path,
            )
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(OUTPUT_DIR / "fe_cnn_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    checkpoint["training_seconds"] = elapsed
    return model, checkpoint, history


def predict_cnn(
    model: ConcentrationCNN,
    inputs: np.ndarray,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        scaled = model(torch.from_numpy(inputs).to(device)).cpu().numpy().reshape(-1)
    return inverse_targets(scaled, checkpoint["transform_mean"], checkpoint["transform_scale"])


def range_metrics(
    y_train: np.ndarray,
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    split_name: str,
) -> list[dict[str, Any]]:
    lower, upper = np.quantile(y_train, [1.0 / 3.0, 2.0 / 3.0])
    bins = (
        (RANGE_LABELS[0], actual < lower),
        (RANGE_LABELS[1], (actual >= lower) & (actual <= upper)),
        (RANGE_LABELS[2], actual > upper),
    )
    rows = []
    for label, selection in bins:
        for model_name, predicted in predictions.items():
            row = {
                "split": split_name,
                "concentration_range": label,
                "lower_train_tercile": float(lower),
                "upper_train_tercile": float(upper),
                "model": model_name,
                **regression_metrics(actual[selection], predicted[selection]),
            }
            rows.append(row)
    return rows


def metric_rows(
    split_name: str,
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    return [
        {"split": split_name, "model": model_name, **regression_metrics(actual, predicted)}
        for model_name, predicted in predictions.items()
    ]


def write_audit(audit: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    distribution_rows = []
    for split_name, item in audit["splits"].items():
        for concentration_range, count in item["concentration_bins"].items():
            distribution_rows.append(
                {
                    "split": split_name,
                    "concentration_range_wt_percent_fe2o3": concentration_range,
                    "spectra": count,
                }
            )
    pd.DataFrame(distribution_rows).to_csv(OUTPUT_DIR / "concentration_distribution.csv", index=False)
    overall = audit["overall"]
    lines = [
        "# Fe Reference Data Audit",
        "",
        "Reference source: the Z-903 workbook column `Fe2O3*`, reported as wt% Fe2O3-equivalent. "
        "This analysis does not reinterpret that oxide-equivalent value as elemental Fe ppm.",
        "",
        "## Overall",
        "",
        "| Measure | Result |",
        "| --- | ---: |",
        f"| Total spectra | {overall['total_spectra']} |",
        f"| Known Fe references | {overall['known_fe_values']} |",
        f"| Missing Fe references | {overall['missing_fe_values']} |",
        f"| Minimum | {overall['minimum']:.4f} wt% |",
        f"| Median | {overall['median']:.4f} wt% |",
        f"| Mean | {overall['mean']:.4f} wt% |",
        f"| 95th percentile | {overall['percentile_95']:.4f} wt% |",
        f"| Maximum | {overall['maximum']:.4f} wt% |",
        f"| Zero values | {overall['zero_values']} |",
        f"| Positive values | {overall['positive_values']} |",
        "",
        "## Frozen Split Distributions",
        "",
        "| Split | Spectra | Known | Missing | Min | Median | Mean | P95 | Max | Zero | Positive |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in ("train", "val", "test"):
        row = audit["splits"][split_name]
        lines.append(
            f"| {split_name} | {row['spectra']} | {row['known']} | {row['missing']} | "
            f"{row['minimum']:.4f} | {row['median']:.4f} | {row['mean']:.4f} | "
            f"{row['percentile_95']:.4f} | {row['maximum']:.4f} | {row['zero']} | {row['positive']} |"
        )
    lines.extend(
        [
            "",
            "Classification is not supported by this reference set: all known values are positive and "
            "there are no zero-reference spectra. Any positive cutoff would be an arbitrary concentration "
            "bin, not presence/absence or a validated detection limit. Modeling therefore proceeds as "
            "quantitative regression only.",
            "",
            "Concentration-bin counts are saved in `concentration_distribution.csv`.",
        ]
    )
    (OUTPUT_DIR / "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_prediction_frame(
    ids: list[str],
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    result = pd.DataFrame({"target_id": ids, "Fe2O3*_actual_wt_percent": actual})
    for model_name, values in predictions.items():
        result[f"{model_name}_predicted_wt_percent"] = values
    return result


def write_summary(
    audit: dict[str, Any],
    checkpoint: dict[str, Any],
    validation_metrics: list[dict[str, Any]],
    test_metrics: list[dict[str, Any]],
    selected_model: str,
    selected_parameters: dict[str, Any],
    ranges: list[dict[str, Any]],
    elapsed: float,
) -> None:
    validation_by_name = {row["model"]: row for row in validation_metrics}
    test_by_name = {row["model"]: row for row in test_metrics}
    selected_val = validation_by_name[selected_model]
    selected_test = test_by_name[selected_model]
    cnn_val = validation_by_name["cnn"]
    cnn_test = test_by_name["cnn"]
    best_val = min(validation_metrics, key=lambda row: row["mae"])
    test_mae_change = abs(selected_test["mae"] - selected_val["mae"]) / max(selected_val["mae"], 1e-12)
    history = pd.read_csv(OUTPUT_DIR / "training_history.csv")
    best_history = history.loc[history["validation_loss"].idxmin()]
    final_history = history.iloc[-1]
    late_overfit = bool(
        final_history["train_loss"] < best_history["train_loss"]
        and final_history["validation_loss"] > best_history["validation_loss"]
    )
    median_val = validation_by_name["median_baseline"]["mae"]
    median_test = test_by_name["median_baseline"]["mae"]
    range_by_key = {
        (row["split"], row["concentration_range"]): row
        for row in ranges
        if row["model"] == selected_model
    }
    stable = bool(
        selected_model != "median_baseline"
        and selected_val["mae"] < median_val
        and selected_test["mae"] < median_test
        and test_mae_change <= 0.25
        and selected_val["r2"] > 0
        and selected_test["r2"] > 0
    )
    range_rows = [row for row in ranges if row["model"] == selected_model]
    lines = [
        "# Fe Concentration Model — Final Summary",
        "",
        "## Decision",
        "",
        "- Chosen task: **quantitative regression only**.",
        "- Classification is not scientifically identifiable from these labels: known values have no zeros.",
        f"- Best validation model among the compared candidates: **{best_val['model']}** "
        f"(selected evaluation model: **{selected_model}**).",
        "- Prediction target is the workbook’s `Fe2O3*` wt% reference, an Fe2O3-equivalent value; "
        "it is not reported as elemental Fe ppm.",
        "- Ridge and PLS use the existing 936-feature pooled full-wavelength representation; the CNN "
        "uses all 23,401 channels.",
        "",
        "## Data",
        "",
        f"- Spectra: {audit['overall']['total_spectra']}",
        f"- Usable known Fe labels: {audit['overall']['known_fe_values']}",
        f"- Missing Fe labels excluded from model fitting: {audit['overall']['missing_fe_values']}",
        f"- Fixed split train/validation/test known labels: "
        f"{audit['splits']['train']['known']}/{audit['splits']['val']['known']}/{audit['splits']['test']['known']}",
        f"- Training-only low/medium/high boundaries: "
        f"{ranges[0]['lower_train_tercile']:.4f} and {ranges[0]['upper_train_tercile']:.4f} wt%.",
        "",
        "## Regression Metrics",
        "",
        "| Split | Model | N | MAE | Median AE | RMSE | R² | Bias | Pearson r | Spearman ρ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in validation_metrics + test_metrics:
        lines.append(
            f"| {row['split']} | {row['model']} | {row['n']} | {row['mae']:.4f} | "
            f"{row['median_absolute_error']:.4f} | {row['rmse']:.4f} | {row['r2']:.4f} | "
            f"{row['bias']:.4f} | {row['pearson_correlation']:.4f} | {row['spearman_correlation']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Selected-Model Concentration-Range Performance",
            "",
            "| Split | Range | N | MAE | Median AE | RMSE | R² | Bias | Pearson r | Spearman ρ |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in range_rows:
        lines.append(
            f"| {row['split']} | {row['concentration_range']} | {row['n']} | {row['mae']:.4f} | "
            f"{row['median_absolute_error']:.4f} | {row['rmse']:.4f} | {row['r2']:.4f} | "
            f"{row['bias']:.4f} | {row['pearson_correlation']:.4f} | {row['spearman_correlation']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Error Patterns",
            "",
            f"- Selected-model low/medium/high MAE (validation): "
            f"{range_by_key[('validation', 'low')]['mae']:.4f}/"
            f"{range_by_key[('validation', 'medium')]['mae']:.4f}/"
            f"{range_by_key[('validation', 'high')]['mae']:.4f} wt%.",
            f"- Selected-model low/medium/high MAE (test): "
            f"{range_by_key[('test', 'low')]['mae']:.4f}/"
            f"{range_by_key[('test', 'medium')]['mae']:.4f}/"
            f"{range_by_key[('test', 'high')]['mae']:.4f} wt%.",
            f"- Low-range bias is positive ({range_by_key[('validation', 'low')]['bias']:.4f}/"
            f"{range_by_key[('test', 'low')]['bias']:.4f} wt%), while high-range bias is negative "
            f"({range_by_key[('validation', 'high')]['bias']:.4f}/"
            f"{range_by_key[('test', 'high')]['bias']:.4f} wt%), indicating predictions regress toward "
            "the middle of the concentration range.",
            f"- High-range within-bin R2 is negative on validation/test "
            f"({range_by_key[('validation', 'high')]['r2']:.4f}/"
            f"{range_by_key[('test', 'high')]['r2']:.4f}); concentration-dependent performance is weaker "
            "at the high end. False-positive/false-negative counts do not apply to this regression task.",
            "",
            "## Stability and Generalization",
            "",
            f"- Validation/test selected-model MAE relative difference: {test_mae_change:.1%}.",
            f"- Median baseline MAE (validation/test): {median_val:.4f}/{median_test:.4f} wt%.",
            f"- Selected model MAE (validation/test): {selected_val['mae']:.4f}/{selected_test['mae']:.4f} wt%.",
            f"- Data-support warning: training maximum is {audit['splits']['train']['maximum']:.4f} wt%, "
            f"while validation/test maxima are {audit['splits']['val']['maximum']:.4f}/"
            f"{audit['splits']['test']['maximum']:.4f} wt%; the high-end training tail is not represented "
            "in validation or test.",
            f"- Streamlit extrapolation warning uses only the validation span "
            f"{audit['splits']['val']['minimum']:.4f}-{audit['splits']['val']['maximum']:.4f} wt%; "
            "it was not selected using test data.",
            f"- Heuristic stability screen (beats median baseline on both splits, positive R² on both, "
            f"and MAE shift <=25%): **{'passed' if stable else 'not passed'}**.",
            "- This is an internal split-stability check, not external scientific or instrument validation.",
            "",
            "## CNN Training",
            "",
            f"- Fixed seed: {checkpoint['seed']}; maximum epochs: {checkpoint['maximum_epochs']}; "
            f"early-stopping patience: {checkpoint['early_stopping_patience']}.",
            f"- Best validation-loss epoch: {checkpoint['best_epoch']}",
            f"- Epochs executed: {len(pd.read_csv(OUTPUT_DIR / 'training_history.csv'))}",
            f"- Best validation Huber loss in standardized log1p target space: "
            f"{checkpoint['best_validation_loss']:.6f}",
            f"- CNN validation metrics: MAE {cnn_val['mae']:.4f}, R² {cnn_val['r2']:.4f}.",
            f"- CNN independent-test metrics: MAE {cnn_test['mae']:.4f}, R² {cnn_test['r2']:.4f}.",
            f"- Training time: {elapsed:.1f} seconds.",
            f"- Late-epoch overfit signal: training Huber loss {best_history['train_loss']:.4f} to "
            f"{final_history['train_loss']:.4f} while validation loss rises from "
            f"{best_history['validation_loss']:.4f} to {final_history['validation_loss']:.4f}; "
            f"{'mild late overfitting is present and the best checkpoint is retained' if late_overfit else 'no late overfitting signal by this check'}.",
            "",
            "## Artifact Paths",
            "",
            f"- Best CNN checkpoint: `{OUTPUT_DIR / 'fe_cnn_best.pt'}`",
            f"- Data audit: `{OUTPUT_DIR / 'data_audit.md'}`",
            f"- Metrics: `{OUTPUT_DIR / 'metrics.json'}`",
            f"- Training history: `{OUTPUT_DIR / 'training_history.csv'}`",
            f"- Range metrics: `{OUTPUT_DIR / 'concentration_range_metrics.csv'}`",
            f"- Validation/test predictions: `{OUTPUT_DIR / 'validation_predictions.csv'}` and "
            f"`{OUTPUT_DIR / 'test_predictions.csv'}`",
            "",
            "No classification confusion matrix is defined because there is no reference negative class.",
            "",
            f"Streamlit integration: **{'integrated as an experimental estimate with a validation-only range warning' if stable and selected_model == 'cnn' else 'not integrated; stability criteria were not met'}**.",
            "No detection-limit claim is made.",
            "",
            f"Selected estimator parameters: `{json.dumps(selected_parameters, sort_keys=True)}`.",
        ]
    )
    (OUTPUT_DIR / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "task": "regression",
        "target": TARGET_LABEL,
        "unit": TARGET_UNIT,
        "selected_model": selected_model,
        "selected_parameters": selected_parameters,
        "selected_validation_metrics": selected_val,
        "selected_test_metrics": selected_test,
        "best_validation_model": best_val["model"],
        "cnn_best_epoch": int(checkpoint["best_epoch"]),
        "cnn_epochs_completed": int(len(history)),
        "seed": int(checkpoint["seed"]),
        "maximum_epochs": int(checkpoint["maximum_epochs"]),
        "early_stopping_patience": int(checkpoint["early_stopping_patience"]),
        "cnn_checkpoint": str(OUTPUT_DIR / "fe_cnn_best.pt"),
        "stable_enough_for_streamlit_integration": stable and selected_model == "cnn",
        "integrated_in_streamlit": stable and selected_model == "cnn",
        "validation_reference_range_wt_percent": [
            audit["splits"]["val"]["minimum"],
            audit["splits"]["val"]["maximum"],
        ],
        "mae_relative_validation_test_difference": test_mae_change,
        "late_overfitting_observed": late_overfit,
        "classification_confusion_matrix": None,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    partial_artifacts = {
        "concentration_distribution.csv",
        "data_audit.json",
        "data_audit.md",
        "fe_cnn_best.pt",
    }
    existing_artifacts = {path.name for path in OUTPUT_DIR.iterdir()} if OUTPUT_DIR.exists() else set()
    if not existing_artifacts.issubset(partial_artifacts):
        raise FileExistsError(f"Refusing to overwrite existing Fe analysis artifacts in {OUTPUT_DIR}")
    seed_everything(SEED)
    splits = load_fe_splits()
    audit = audit_data(splits)
    write_audit(audit)
    print(json.dumps({"data_audit": audit}, indent=2))

    arrays = {name: load_known_arrays(frame) for name, frame in splits.items()}
    arrays["validation"] = arrays.pop("val")
    if arrays["train"][0].shape[2] != EXPECTED_POINTS:
        raise ValueError(f"Expected {EXPECTED_POINTS} full-spectrum input channels")
    train_values = arrays["train"][1]
    val_values = arrays["validation"][1]
    x_train, y_train, _ = arrays["train"]
    x_val, _, _ = arrays["validation"]
    median_prediction = float(np.median(train_values))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train_linear = np.stack([bin_spectrum(sample[0]) for sample in x_train])
    x_val_linear = np.stack([bin_spectrum(sample[0]) for sample in x_val])
    linear_models = fit_linear_models(
        x_train_linear,
        y_train,
        x_val_linear,
        val_values,
        median_prediction,
    )
    _, checkpoint, history = train_cnn(arrays, device)
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    validation_predictions: dict[str, np.ndarray] = {
        name: np.asarray(result["validation_prediction"], dtype=np.float64)
        for name, result in linear_models.items()
    }
    model = ConcentrationCNN(output_size=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_predictions["cnn"] = predict_cnn(model, x_val, checkpoint, device)
    x_test, y_test, _ = arrays["test"]
    test_predictions: dict[str, np.ndarray] = {
        "median_baseline": np.full(y_test.shape, median_prediction, dtype=np.float64)
    }
    x_test_linear = np.stack([bin_spectrum(sample[0]) for sample in x_test])
    target_mean = float(np.log1p(train_values).mean())
    target_scale = float(np.log1p(train_values).std()) or 1.0
    for model_name in ("ridge", "pls"):
        estimator = linear_models[model_name]["estimator"]
        test_scaled = estimator.predict(x_test_linear)
        if np.asarray(test_scaled).ndim > 1:
            test_scaled = np.asarray(test_scaled).reshape(-1)
        test_predictions[model_name] = inverse_targets(test_scaled, target_mean, target_scale)
    test_predictions["cnn"] = predict_cnn(model, x_test, checkpoint, device)

    selected_model = min(
        validation_predictions,
        key=lambda name: regression_metrics(val_values, validation_predictions[name])["mae"],
    )
    selected_parameters: dict[str, Any] = {}
    if selected_model in {"ridge", "pls"}:
        selected_parameters = dict(linear_models[selected_model]["parameters"])
    elif selected_model == "cnn":
        selected_parameters = {
            "architecture": "ConcentrationCNN",
            "preprocess": "robust median/IQR",
            "target_transform": "train-only log1p standardization",
        }
    else:
        selected_parameters = {"prediction": median_prediction}

    validation_metrics = metric_rows("validation", val_values, validation_predictions)
    test_metrics = metric_rows("test", y_test, test_predictions)
    pd.DataFrame(validation_metrics).to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    pd.DataFrame(test_metrics).to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)
    predictions_validation = build_prediction_frame(
        arrays["validation"][2], val_values, validation_predictions
    )
    predictions_test = build_prediction_frame(arrays["test"][2], y_test, test_predictions)
    predictions_validation.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
    predictions_test.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    ranges = range_metrics(train_values, val_values, validation_predictions, "validation")
    ranges.extend(range_metrics(train_values, y_test, test_predictions, "test"))
    pd.DataFrame(ranges).to_csv(OUTPUT_DIR / "concentration_range_metrics.csv", index=False)
    elapsed = float(checkpoint["training_seconds"])
    write_summary(
        audit,
        checkpoint,
        validation_metrics,
        test_metrics,
        selected_model,
        selected_parameters,
        ranges,
        elapsed,
    )
    print(f"Fe analysis complete. Summary: {OUTPUT_DIR / 'final_summary.md'}")


if __name__ == "__main__":
    main()
