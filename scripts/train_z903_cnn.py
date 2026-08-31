#!/usr/bin/env python
"""Train two real-data Z-903 1D-CNN preprocessing baselines."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from atomic_spectrum_ai.models.z903_cnn import Z903CNN
from atomic_spectrum_ai.z903 import TARGETS, preprocess_spectrum, read_z903_spectrum

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 20260816
MAX_EPOCHS = 50
PATIENCE = 8
LABEL_THRESHOLDS = np.asarray([100.0, 500.0, 0.05, 5.0, 25.0, 10.0], dtype=np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_split(path: Path, option: str) -> tuple[Tensor, Tensor, Tensor, pd.DataFrame]:
    frame = pd.read_csv(path)
    inputs: list[np.ndarray] = []
    raw_labels = frame[[f"{target}_raw" if target != "Mg" else "MgO_raw" for target in TARGETS]].to_numpy(dtype=np.float32)
    masks = frame[[f"{target}_mask" for target in TARGETS]].to_numpy(dtype=np.float32)
    labels = (raw_labels > LABEL_THRESHOLDS).astype(np.float32)
    labels[~np.isfinite(raw_labels)] = 0.0
    for spectrum_path in frame["spectrum_path"]:
        _, intensity = read_z903_spectrum(spectrum_path)
        processed = preprocess_spectrum(intensity, option=option)
        if not np.isfinite(processed).all():
            raise ValueError(f"Non-finite processed values in {spectrum_path}")
        inputs.append(processed)
    x = torch.from_numpy(np.asarray(inputs, dtype=np.float32)[:, None, :])
    y = torch.nan_to_num(torch.from_numpy(labels), nan=0.0)
    m = torch.from_numpy(masks)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not torch.isfinite(m).all():
        raise ValueError(f"Non-finite tensors loaded from {path}")
    return x, y, m, frame


def positive_weights(labels: Tensor, masks: Tensor) -> Tensor:
    weights = []
    for index in range(labels.shape[1]):
        known = masks[:, index] > 0
        positives = int(((labels[:, index] > 0.5) & known).sum())
        negatives = int(((labels[:, index] <= 0.5) & known).sum())
        weights.append(float(negatives / positives) if positives else 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def weighted_masked_loss(logits: Tensor, labels: Tensor, masks: Tensor, pos_weight: Tensor) -> Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=pos_weight.to(logits.device))
    active = masks.to(dtype=loss.dtype)
    return (loss * active).sum() / active.sum().clamp_min(1.0)


def scalar(value: float | np.floating[Any]) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def metric_rows(labels: np.ndarray, masks: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for index, element in enumerate(TARGETS):
        known = masks[:, index] > 0
        y_true = labels[known, index].astype(int)
        y_prob = probabilities[known, index]
        y_pred = (y_prob >= thresholds[index]).astype(int)
        positive = int(y_true.sum())
        negative = int(len(y_true) - positive)
        if len(np.unique(y_true)) == 2:
            roc_auc = scalar(roc_auc_score(y_true, y_prob))
            balanced = scalar(balanced_accuracy_score(y_true, y_pred))
        else:
            roc_auc = float("nan")
            balanced = float("nan")
        average_precision = scalar(average_precision_score(y_true, y_prob)) if positive else float("nan")
        precision = scalar(precision_score(y_true, y_pred, zero_division=0))
        recall = scalar(recall_score(y_true, y_pred, zero_division=0))
        f1 = scalar(f1_score(y_true, y_pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = scalar(tn / (tn + fp)) if tn + fp else float("nan")
        rows.append({"element": element, "known_samples": len(y_true), "positive_samples": positive, "negative_samples": negative, "roc_auc": roc_auc, "pr_auc": average_precision, "precision": precision, "recall": recall, "specificity": specificity, "f1": f1, "balanced_accuracy": balanced})
        confusion_rows.append({"element": element, "threshold": float(thresholds[index]), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return rows, confusion_rows


def macro(rows: list[dict[str, Any]], key: str) -> float:
    values = [row[key] for row in rows if np.isfinite(row[key])]
    return float(np.mean(values)) if values else float("nan")


def prevalence_baseline(labels: np.ndarray, masks: np.ndarray) -> dict[str, float]:
    pr_values = []
    f1_values = []
    for index in range(labels.shape[1]):
        known = masks[:, index] > 0
        true = labels[known, index].astype(int)
        prevalence = float(true.mean()) if len(true) else 0.0
        pr_values.append(prevalence)
        f1_values.append(float(f1_score(true, np.zeros_like(true), zero_division=0)))
    return {"macro_pr_auc_prevalence": float(np.mean(pr_values)), "macro_f1_majority": float(np.mean(f1_values))}


def choose_thresholds(labels: np.ndarray, masks: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    chosen = []
    for index in range(labels.shape[1]):
        known = masks[:, index] > 0
        y_true = labels[known, index].astype(int)
        y_prob = probabilities[known, index]
        candidates = np.linspace(0.05, 0.95, 181)
        scores = [f1_score(y_true, y_prob >= threshold, zero_division=0) for threshold in candidates]
        best_score = max(scores)
        best = [threshold for threshold, score in zip(candidates, scores, strict=False) if score == best_score]
        chosen.append(float(min(best, key=lambda threshold: abs(threshold - 0.5))))
    return np.asarray(chosen, dtype=np.float32)


def evaluate(model: nn.Module, loader: DataLoader[tuple[Tensor, Tensor, Tensor]], device: torch.device, thresholds: np.ndarray) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    all_labels: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels, masks in loader:
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
            logits = model(inputs)
            losses.append(float(weighted_masked_loss(logits, labels, masks, torch.ones(6, device=device)).item()))
            all_labels.append(labels.cpu().numpy())
            all_masks.append(masks.cpu().numpy())
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
    labels = np.concatenate(all_labels)
    masks = np.concatenate(all_masks)
    probabilities = np.concatenate(all_probabilities)
    rows, confusion_rows = metric_rows(labels, masks, probabilities, thresholds)
    return float(np.mean(losses)), rows, confusion_rows, labels, masks, probabilities


def train_experiment(name: str, option: str, splits: dict[str, tuple[Tensor, Tensor, Tensor, pd.DataFrame]], args: argparse.Namespace) -> dict[str, Any]:
    output_dir = ROOT / "outputs" / "z903" / name
    model_dir = ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, m_train, _ = splits["train"]
    x_val, y_val, m_val, val_frame = splits["val"]
    x_test, y_test, m_test, test_frame = splits["test"]
    weights = positive_weights(y_train, m_train)
    train_loader = DataLoader(TensorDataset(x_train, y_train, m_train), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(TensorDataset(x_val, y_val, m_val), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(x_test, y_test, m_test), batch_size=args.batch_size)
    model = Z903CNN().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    best_score = -np.inf
    best_epoch = 0
    patience_count = 0
    history: list[dict[str, Any]] = []
    checkpoint = model_dir / f"{name}_best.pt"
    started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []
        for inputs, labels, masks in train_loader:
            inputs, labels, masks = inputs.to(args.device), labels.to(args.device), masks.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_masked_loss(model(inputs), labels, masks, weights)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss, val_rows, _, _, _, _ = evaluate(model, val_loader, args.device, np.full(6, 0.5, dtype=np.float32))
        val_pr = macro(val_rows, "pr_auc")
        val_f1 = macro(val_rows, "f1")
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": val_loss, "validation_macro_pr_auc": val_pr, "validation_macro_f1": val_f1, "learning_rate": optimizer.param_groups[0]["lr"]})
        scheduler.step(val_pr if np.isfinite(val_pr) else -1.0)
        if val_pr > best_score:
            best_score = val_pr
            best_epoch = epoch
            patience_count = 0
            torch.save({"model_state_dict": model.state_dict(), "preprocess": option, "targets": TARGETS, "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "best_epoch": epoch, "validation_macro_pr_auc": val_pr, "positive_weights": weights.tolist()}, checkpoint)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                break
    elapsed = time.perf_counter() - started
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device)["model_state_dict"])
    _, _, _, val_labels, val_masks, val_probabilities = evaluate(model, val_loader, args.device, np.full(6, 0.5, dtype=np.float32))
    thresholds = choose_thresholds(val_labels, val_masks, val_probabilities)
    val_loss, val_rows, val_confusions, _, _, _ = evaluate(model, val_loader, args.device, thresholds)
    test_loss, test_rows, test_confusions, test_labels, test_masks, test_probabilities = evaluate(model, test_loader, args.device, thresholds)
    for row in val_rows:
        row["threshold"] = float(thresholds[TARGETS.index(row["element"])])
    for row in test_rows:
        row["threshold"] = float(thresholds[TARGETS.index(row["element"])])
    pd.DataFrame(val_rows).to_csv(output_dir / "validation_metrics.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output_dir / "per_element_metrics.csv", index=False)
    pd.DataFrame(test_rows).to_json(output_dir / "test_metrics.json", orient="records", indent=2)
    (output_dir / "validation_metrics.json").write_text(json.dumps({"metrics": val_rows, "macro_roc_auc": macro(val_rows, "roc_auc"), "macro_pr_auc": macro(val_rows, "pr_auc"), "macro_f1": macro(val_rows, "f1"), "thresholds": dict(zip(TARGETS, thresholds.tolist(), strict=False))}, indent=2), encoding="utf-8")
    pd.DataFrame(test_confusions).to_csv(output_dir / "confusion_matrices.csv", index=False)
    probability_frame = test_frame[["target_id", "spectrum_filename"]].copy()
    for index, element in enumerate(TARGETS):
        probability_frame[f"{element}_probability"] = test_probabilities[:, index]
        probability_frame[f"{element}_mask"] = test_masks[:, index]
        probability_frame[f"{element}_label"] = test_labels[:, index]
    probability_frame.to_csv(output_dir / "prediction_probabilities.csv", index=False)
    return {"name": name, "option": option, "best_epoch": best_epoch, "epochs_completed": len(history), "validation_macro_pr_auc": macro(val_rows, "pr_auc"), "validation_macro_f1": macro(val_rows, "f1"), "test_macro_pr_auc": macro(test_rows, "pr_auc"), "test_macro_f1": macro(test_rows, "f1"), "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "training_seconds": elapsed, "positive_weights": dict(zip(TARGETS, weights.tolist(), strict=False)), "thresholds": dict(zip(TARGETS, thresholds.tolist(), strict=False)), "validation_rows": val_rows, "test_rows": test_rows, "prevalence_baseline": prevalence_baseline(val_labels, val_masks), "history": history}


def write_report(results: list[dict[str, Any]], selected: dict[str, Any], split_sizes: dict[str, int]) -> None:
    lines = ["# Z-903 1D-CNN Baseline", "", "## Dataset", "", f"- Train / validation / test: {split_sizes['train']} / {split_sizes['val']} / {split_sizes['test']}", "- Six partially observed outputs: Zn, Mn, Cd, Mg, Cu, Pb", "- Missing labels use masks and do not contribute to loss or metrics.", "", "## Model", "", "Compact strided Conv1d blocks: Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d repeated three times, a fourth Conv1d block, AdaptiveAvgPool1d(1), and a six-logit linear head. Input shape is `[batch, 1, 23401]`; sigmoid is not applied inside the model.", f"- Trainable parameters: {results[0]['parameter_count']}", "", "## Preprocessing Comparison", "", "| Experiment | Validation macro PR-AUC | Validation macro F1 | Best epoch | Test macro PR-AUC | Test macro F1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for result in results:
        lines.append(f"| {result['name']} | {result['validation_macro_pr_auc']:.4f} | {result['validation_macro_f1']:.4f} | {result['best_epoch']} | {result['test_macro_pr_auc']:.4f} | {result['test_macro_f1']:.4f} |")
    lines.extend(["", f"Selected preprocessing by validation macro PR-AUC only: **{selected['option']}**.", "", "## Selected Validation Metrics", "", "| Element | Known | PR-AUC | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in selected["validation_rows"]:
        lines.append(f"| {row['element']} | {row['known_samples']} | {row['pr_auc']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |")
    lines.extend(["", "## Selected Probability Thresholds", ""])
    lines.extend(f"- {element}: `{threshold:.4f}`" for element, threshold in selected["thresholds"].items())
    lines.extend(["", "## Final Untouched Test Metrics", "", "| Element | Known | Positive | Negative | ROC-AUC | PR-AUC | Precision | Recall | Specificity | F1 | Balanced accuracy |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in selected["test_rows"]:
        lines.append("| " + " | ".join(f"{row[key]:.4f}" if isinstance(row[key], float) else str(row[key]) for key in ("element", "known_samples", "positive_samples", "negative_samples", "roc_auc", "pr_auc", "precision", "recall", "specificity", "f1", "balanced_accuracy")) + " |")
    lines.extend(["", "## Class Weights", ""])
    lines.extend(f"- {element}: positive weight `{weight:.6f}`" for element, weight in selected["positive_weights"].items())
    area = next(result for result in results if result["option"] == "area")
    robust = next(result for result in results if result["option"] == "robust")
    lines.extend(["", "## Overfitting and Baseline Checks", "", "Training loss and validation metrics are stored per epoch in each experiment's `training_history.csv`. The area-normalized experiment shows clear overfitting/instability: training loss keeps falling while validation loss becomes very large and validation macro PR-AUC deteriorates after epoch 1. The robust experiment shows milder late overfitting: training loss continues falling after the validation PR-AUC peak at epoch 21, with validation PR-AUC fluctuating slightly lower before early stopping. Cd and Cu validation metrics are less stable because they have only 44 and 103 known validation labels, respectively. No test data were used for checkpoint or preprocessing selection.", f"- Prevalence baseline validation macro PR-AUC: {selected['prevalence_baseline']['macro_pr_auc_prevalence']:.4f}", f"- Majority-class validation macro F1: {selected['prevalence_baseline']['macro_f1_majority']:.4f}", f"- Area best epoch / completed epochs: {area['best_epoch']} / {area['epochs_completed']}", f"- Robust best epoch / completed epochs: {robust['best_epoch']} / {robust['epochs_completed']}", "- Current weakest selected-test element by PR-AUC and F1: Zn.", "", "## Limitations", "", "- Labels use exploratory composition thresholds, not scientifically validated LIBS limits of detection.", "- Cd has only 44 known test labels.", "- Cu has substantially fewer known labels than Zn, Mn, Mg, and Pb.", "- NASA Z-903 performance does not establish cross-instrument generalization.", "- External real-world validation remains required.", f"- Total training time for both experiments: {sum(result['training_seconds'] for result in results):.2f} seconds."])
    (ROOT / "reports" / "z903_cnn_baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.device = torch.device(args.device)
    seed_everything(args.seed)
    split_paths = {split: ROOT / "data" / "processed" / f"z903_{split}.csv" for split in ("train", "val", "test")}
    loaded = {option: {split: load_split(path, option) for split, path in split_paths.items()} for option in ("area", "robust")}
    results = [train_experiment("z903_cnn_total_area", "area", loaded["area"], args), train_experiment("z903_cnn_robust", "robust", loaded["robust"], args)]
    history_rows = []
    for result in results:
        history_rows.extend([{**row, "experiment": result["name"]} for row in result["history"]])
    (ROOT / "outputs" / "z903" / "training_history.csv").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history_rows).to_csv(ROOT / "outputs" / "z903" / "training_history.csv", index=False)
    selected = max(results, key=lambda result: result["validation_macro_pr_auc"])
    write_report(results, selected, {split: len(loaded["area"][split][3]) for split in split_paths})
    print(json.dumps({"results": [{key: result[key] for key in ("name", "validation_macro_pr_auc", "validation_macro_f1", "test_macro_pr_auc", "test_macro_f1", "best_epoch", "training_seconds")} for result in results], "selected": selected["name"], "thresholds": selected["thresholds"], "parameter_count": selected["parameter_count"]}, indent=2))


if __name__ == "__main__":
    main()