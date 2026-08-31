"""MgO second quantitative-modeling iteration: median vs Ridge vs PLSR vs existing CNN.

Diagnostic-only comparison. Uses the existing train/validation target-group
splits. Does not use the test split, does not modify the detection model,
does not modify z903_concentration_best.pt, and does not modify the Streamlit
app.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.concentration_inference import load_concentration_checkpoint, predict_concentrations
from src.data import load_regression_split, load_spectrum
from src.preprocessing import robust_scale

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "concentration" / "mgo_iteration2"
BIN_SIZE = 25
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
PLS_COMPONENTS = (2, 4, 8, 12, 16, 24, 32, 48)
PLS_R2_TOLERANCE = 0.01


def bin_spectrum(values: np.ndarray, bin_size: int = BIN_SIZE) -> np.ndarray:
    """Average-pool a spectrum to reduce dimensionality for classical models."""
    usable = (len(values) // bin_size) * bin_size
    return values[:usable].reshape(-1, bin_size).mean(axis=1)


def load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (binned classical features, raw intensity, MgO truth) for known rows."""
    frame = load_regression_split(split)
    truth = pd.to_numeric(frame["Mg"], errors="coerce").to_numpy(dtype=np.float64)
    known = np.isfinite(truth)
    binned, raw = [], []
    for path in frame.loc[known, "spectrum_path"]:
        _, intensity = load_spectrum(path)
        raw.append(intensity)
        binned.append(bin_spectrum(robust_scale(intensity)))
    return np.asarray(binned, dtype=np.float64), np.asarray(raw, dtype=np.float32), truth[known]


def quantile_bands(y: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "bottom_50": y <= np.percentile(y, 50),
        "p50_p90": (y > np.percentile(y, 50)) & (y <= np.percentile(y, 90)),
        "top_10": y > np.percentile(y, 90),
        "top_5": y > np.percentile(y, 95),
    }


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    errors = p - y
    centered = y - y.mean()
    r2 = float(1.0 - np.sum(errors**2) / np.sum(centered**2)) if np.sum(centered**2) > 0 else float("nan")
    pearson = float(pearsonr(y, p).statistic) if len(y) > 1 and np.std(p) > 0 else float("nan")
    spearman = float(spearmanr(y, p).statistic) if len(y) > 1 else float("nan")
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median_absolute_error": float(np.median(np.abs(errors))),
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "bias": float(np.mean(errors)),
    }


def banded_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, dict[str, float | int]]:
    return {name: compute_metrics(y[mask], p[mask]) for name, mask in quantile_bands(y).items() if mask.any()}


def scatter_plot(name: str, y: np.ndarray, p: np.ndarray, limit: float) -> None:
    directory = OUTPUT / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(4.5, 4.5), dpi=140)
    axis.scatter(y, p, s=14, alpha=0.6, color="#176b87")
    axis.plot([0, limit], [0, limit], color="black", linewidth=1)
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("Ground truth MgO (wt%)")
    axis.set_ylabel("Predicted MgO (wt%)")
    axis.set_title(f"Validation: {name}")
    figure.tight_layout()
    figure.savefig(directory / f"validation_{name}.png")
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    x_train, _raw_train, y_train = load_split("train")
    x_val, raw_val, y_val = load_split("val")

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    limit = float(max(y_train.max(), y_val.max())) * 1.05

    results: dict[str, object] = {"n_train": int(len(y_train)), "n_val": int(len(y_val)), "bin_size": BIN_SIZE, "n_features": int(x_train.shape[1])}

    # A. Median baseline.
    median_value = float(np.median(y_train))
    median_pred = np.full_like(y_val, median_value)
    results["median_baseline"] = {"train_median": median_value, "metrics": compute_metrics(y_val, median_pred), "bands": banded_metrics(y_val, median_pred)}
    scatter_plot("median_baseline", y_val, median_pred, limit)

    # B. Ridge regression, alpha tuned on validation only.
    ridge_grid = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(x_train_scaled, y_train)
        prediction = model.predict(x_val_scaled)
        ridge_grid.append({"alpha": alpha, "metrics": compute_metrics(y_val, prediction)})
    best_ridge = max(ridge_grid, key=lambda row: row["metrics"]["r2"])
    ridge_model = Ridge(alpha=best_ridge["alpha"]).fit(x_train_scaled, y_train)
    ridge_pred = ridge_model.predict(x_val_scaled)
    results["ridge"] = {"grid": ridge_grid, "selected_alpha": best_ridge["alpha"], "metrics": compute_metrics(y_val, ridge_pred), "bands": banded_metrics(y_val, ridge_pred)}
    scatter_plot("ridge", y_val, ridge_pred, limit)

    # C. PLSR, component count tuned on validation only, preferring smaller models.
    pls_grid = []
    for components in PLS_COMPONENTS:
        model = PLSRegression(n_components=components)
        model.fit(x_train_scaled, y_train)
        prediction = model.predict(x_val_scaled).ravel()
        pls_grid.append({"components": components, "metrics": compute_metrics(y_val, prediction)})
    best_r2 = max(row["metrics"]["r2"] for row in pls_grid)
    candidates = [row for row in pls_grid if row["metrics"]["r2"] >= best_r2 - PLS_R2_TOLERANCE]
    best_pls = min(candidates, key=lambda row: row["components"])
    pls_model = PLSRegression(n_components=best_pls["components"]).fit(x_train_scaled, y_train)
    pls_pred = pls_model.predict(x_val_scaled).ravel()
    results["plsr"] = {"grid": pls_grid, "selected_components": best_pls["components"], "tolerance": PLS_R2_TOLERANCE, "metrics": compute_metrics(y_val, pls_pred), "bands": banded_metrics(y_val, pls_pred)}
    scatter_plot("plsr", y_val, pls_pred, limit)

    # D. Existing multi-output CNN, unmodified.
    cnn_model, transform, checkpoint = load_concentration_checkpoint()
    cnn_pred = np.asarray([predict_concentrations(intensity, cnn_model, transform)["Mg"] for intensity in raw_val])
    results["existing_cnn"] = {"checkpoint_best_epoch": checkpoint.get("best_epoch"), "metrics": compute_metrics(y_val, cnn_pred), "bands": banded_metrics(y_val, cnn_pred)}
    scatter_plot("existing_cnn", y_val, cnn_pred, limit)

    (OUTPUT / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
