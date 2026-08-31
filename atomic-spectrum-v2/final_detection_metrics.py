"""Calculate final frozen six-target detection metrics on the test split only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.data import load_spectrum
from src.inference import load_checkpoint, predict
from src.labels import CNN_THRESHOLDS, RAW_THRESHOLDS, TARGETS

ROOT = Path(__file__).resolve().parent
TEST_PATH = ROOT / "data" / "splits" / "z903_test.csv"
OUTPUT_DIR = ROOT / "reports" / "detection"
RAW_COLUMNS = {"Zn": "Zn_raw", "Mn": "Mn_raw", "Cd": "Cd_raw", "Mg": "MgO_raw", "Cu": "Cu_raw", "Pb": "Pb_raw"}
MASK_COLUMNS = {element: f"{element}_mask" for element in TARGETS}
EXPECTED_KNOWN = {"Zn": 352, "Mn": 368, "Cd": 44, "Mg": 386, "Cu": 114, "Pb": 321}


def evaluate() -> tuple[dict[str, object], list[dict[str, object]]]:
    frame = pd.read_csv(TEST_PATH)
    model, checkpoint = load_checkpoint()
    probabilities = []
    for path in frame["spectrum_path"]:
        _, intensity = load_spectrum(path)
        probabilities.append([predict(intensity, model)[element]["probability"] for element in TARGETS])
    probabilities_array = np.asarray(probabilities, dtype=float)

    rows: list[dict[str, object]] = []
    truth_matrix = np.full((len(frame), len(TARGETS)), np.nan, dtype=float)
    prediction_matrix = probabilities_array >= np.asarray([CNN_THRESHOLDS[element] for element in TARGETS])
    known_matrix = np.zeros_like(prediction_matrix, dtype=bool)
    for index, element in enumerate(TARGETS):
        raw = pd.to_numeric(frame[RAW_COLUMNS[element]], errors="coerce").to_numpy(dtype=float)
        known = pd.to_numeric(frame[MASK_COLUMNS[element]], errors="coerce").fillna(0).to_numpy() > 0
        known &= np.isfinite(raw)
        truth = raw > RAW_THRESHOLDS[element]
        truth_matrix[:, index] = truth
        known_matrix[:, index] = known
        actual = truth[known].astype(int)
        predicted = prediction_matrix[known, index].astype(int)
        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
        known_count = int(len(actual))
        accuracy = float((tp + tn) / known_count) if known_count else float("nan")
        precision = float(precision_score(actual, predicted, zero_division=0))
        recall = float(recall_score(actual, predicted, zero_division=0))
        specificity = float(tn / (tn + fp)) if tn + fp else 0.0
        f1 = float(f1_score(actual, predicted, zero_division=0))
        pr_auc = float(average_precision_score(actual, probabilities_array[known, index])) if known_count else float("nan")
        rows.append({"element": element, "known": known_count, "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "accuracy": accuracy, "precision": precision, "recall": recall, "sensitivity": recall, "specificity": specificity, "balanced_accuracy": (recall + specificity) / 2.0, "f1": f1, "pr_auc": pr_auc, "threshold": CNN_THRESHOLDS[element]})

    if {row["element"]: row["known"] for row in rows} != EXPECTED_KNOWN:
        raise RuntimeError(f"Unexpected known-label counts: {rows}; expected {EXPECTED_KNOWN}")

    known_flat = known_matrix.ravel()
    actual_flat = truth_matrix.ravel()[known_flat].astype(int)
    predicted_flat = prediction_matrix.ravel()[known_flat].astype(int)
    micro_tp = int(np.sum((actual_flat == 1) & (predicted_flat == 1)))
    micro_tn = int(np.sum((actual_flat == 0) & (predicted_flat == 0)))
    micro_fp = int(np.sum((actual_flat == 0) & (predicted_flat == 1)))
    micro_fn = int(np.sum((actual_flat == 1) & (predicted_flat == 0)))
    fully_known = known_matrix.all(axis=1)
    exact_match = float(np.mean(np.all(truth_matrix[fully_known] == prediction_matrix[fully_known], axis=1))) if fully_known.any() else float("nan")
    summary = {"checkpoint": str(ROOT / "models" / "z903_cnn_robust_best.pt"), "checkpoint_preprocess": checkpoint.get("preprocess"), "thresholds": CNN_THRESHOLDS, "known_total": int(known_flat.sum()), "expected_known_total": int(sum(EXPECTED_KNOWN.values())), "per_element": rows, "overall_label_accuracy": float(np.mean(actual_flat == predicted_flat)), "macro_accuracy": float(np.mean([row["accuracy"] for row in rows])), "macro_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in rows])), "micro": {"tp": micro_tp, "tn": micro_tn, "fp": micro_fp, "fn": micro_fn, "precision": float(precision_score(actual_flat, predicted_flat, zero_division=0)), "recall": float(recall_score(actual_flat, predicted_flat, zero_division=0)), "f1": float(f1_score(actual_flat, predicted_flat, zero_division=0))}, "exact_match_accuracy": exact_match, "fully_labelled_test_spectra": int(fully_known.sum())}
    return summary, rows


def write_reports(summary: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "final_accuracy_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Final Frozen Z-903 Detection Accuracy Metrics", "", "Test-only evaluation of the existing frozen checkpoint. Missing labels were excluded from every denominator. No thresholds or model parameters were changed.", "", "| Element | Known | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | PR-AUC |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["per_element"]:
        lines.append(f"| {row['element']} | {row['known']} | {row['tp']} | {row['tn']} | {row['fp']} | {row['fn']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['pr_auc']:.4f} |")
    lines.extend(["", f"Overall label accuracy = {summary['overall_label_accuracy']:.2%}", f"Macro accuracy = {summary['macro_accuracy']:.2%}", f"Macro balanced accuracy = {summary['macro_balanced_accuracy']:.2%}", f"Micro precision = {summary['micro']['precision']:.2%}", f"Micro recall = {summary['micro']['recall']:.2%}", f"Micro F1 = {summary['micro']['f1']:.2%}", f"Exact-match accuracy = {summary['exact_match_accuracy']:.2%} ({summary['fully_labelled_test_spectra']} fully labelled test spectra)", "", "## Balanced Accuracy", "", "| Element | Balanced accuracy | Specificity | Sensitivity |", "|---|---:|---:|---:|"])
    for row in summary["per_element"]:
        lines.append(f"| {row['element']} | {row['balanced_accuracy']:.4f} | {row['specificity']:.4f} | {row['sensitivity']:.4f} |")
    (OUTPUT_DIR / "final_accuracy_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result, _ = evaluate()
    write_reports(result)
    print(json.dumps(result, indent=2))
