"""Train and evaluate the separate NASA Z-903 concentration CNN baseline."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from src.data import EXPECTED_POINTS, load_regression_split, load_spectrum
from src.labels import TARGETS, UNITS
from src.model import ConcentrationCNN
from src.preprocessing import robust_scale
from src.regression import TargetTransform, masked_huber_loss, regression_metrics
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "z903_concentration_best.pt"
OUTPUT = ROOT / "reports" / "concentration"


def build_arrays(frame: pd.DataFrame, transform: TargetTransform) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectra = []
    for path in frame["spectrum_path"]:
        _, intensity = load_spectrum(path)
        spectra.append(robust_scale(intensity))
    values = frame[list(TARGETS)].to_numpy(dtype=np.float32)
    mask = np.isfinite(values).astype(np.float32)
    labels = transform.transform(np.nan_to_num(values, nan=0.0))
    return np.asarray(spectra, dtype=np.float32)[:, None, :], labels.astype(np.float32), mask


def evaluate(model: ConcentrationCNN, loader: DataLoader, transform: TargetTransform) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    predictions, labels, masks = [], [], []
    model.eval()
    with torch.inference_mode():
        for inputs, target, mask in loader:
            predictions.append(model(inputs).numpy())
            labels.append(target.numpy())
            masks.append(mask.numpy())
    scaled = np.concatenate(predictions)
    actual_scaled = np.concatenate(labels)
    known = np.concatenate(masks)
    actual = transform.inverse(actual_scaled)
    predicted = transform.inverse(scaled)
    return regression_metrics(actual, predicted, known), actual, predicted, known


def scatter_plots(split: str, actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> None:
    directory = OUTPUT / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    for index, element in enumerate(TARGETS):
        known = mask[:, index].astype(bool)
        figure, axis = plt.subplots(figsize=(4.5, 4.0), dpi=140)
        axis.scatter(actual[known, index], predicted[known, index], s=12, alpha=0.65, color="#176b87")
        if known.any():
            low = min(actual[known, index].min(), predicted[known, index].min())
            high = max(actual[known, index].max(), predicted[known, index].max())
            axis.plot([low, high], [low, high], color="black", linewidth=1)
        axis.set_xlabel(f"Ground truth ({UNITS[element]})")
        axis.set_ylabel(f"Predicted ({UNITS[element]})")
        axis.set_title(f"{split}: {element}")
        figure.tight_layout()
        figure.savefig(directory / f"{split}_{element}.png")
        plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = load_regression_split("train")
    validation = load_regression_split("val")
    test = load_regression_split("test")
    train_values = train[list(TARGETS)].to_numpy(dtype=np.float32)
    train_mask = np.isfinite(train_values).astype(np.float32)
    transform = TargetTransform.fit(train_values, train_mask)
    arrays = {name: build_arrays(frame, transform) for name, frame in (("train", train), ("validation", validation), ("test", test))}
    x_train, y_train, m_train = arrays["train"]
    model = ConcentrationCNN(output_size=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(m_train)), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(arrays["validation"][0]), torch.from_numpy(arrays["validation"][1]), torch.from_numpy(arrays["validation"][2])), batch_size=32)
    best_score = float("inf")
    best_epoch = 0
    patience = 0
    for epoch in range(1, 31):
        model.train()
        for inputs, labels, masks in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_huber_loss(model(inputs), labels, masks)
            loss.backward()
            optimizer.step()
        validation_metrics, _, _, _ = evaluate(model, val_loader, transform)
        score = float(np.nanmean([row["log1p_mae"] for row in validation_metrics]))
        if score < best_score:
            best_score = score
            best_epoch = epoch
            patience = 0
            torch.save({"model_state_dict": model.state_dict(), "model_name": "ConcentrationCNN", "targets": TARGETS, "input_shape": [1, 1, EXPECTED_POINTS], "preprocess": "robust", "transform_means": transform.means.tolist(), "transform_scales": transform.scales.tolist(), "best_epoch": epoch}, CHECKPOINT)
        else:
            patience += 1
            if patience >= 5:
                break
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["model_state_dict"])
    reports = {"best_validation_epoch": best_epoch, "splits": {}}
    for name in ("validation", "test"):
        values = arrays[name]
        loader = DataLoader(TensorDataset(torch.from_numpy(values[0]), torch.from_numpy(values[1]), torch.from_numpy(values[2])), batch_size=32)
        metrics, actual, predicted, mask = evaluate(model, loader, transform)
        median_metrics = regression_metrics(actual, np.where(mask > 0, np.nanmedian(train_values, axis=0), 0.0), mask)
        reports["splits"][name] = {"cnn": metrics, "median_baseline": median_metrics}
        pd.DataFrame(metrics).to_csv(OUTPUT / f"{name}_metrics.csv", index=False)
        pd.DataFrame(median_metrics).to_csv(OUTPUT / f"{name}_median_baseline.csv", index=False)
        scatter_plots(name, actual, predicted, mask)
    (OUTPUT / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
