#!/usr/bin/env python
"""Evaluate the saved physics-constrained checkpoint on validation data only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score

from atomic_spectrum_ai.models.z903_physics_constrained import PhysicsConstrainedZ903CNN
from atomic_spectrum_ai.physics_windows import PhysicsWindowDataset
from atomic_spectrum_ai.z903 import TARGETS, WAVELENGTHS

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = dict(zip(TARGETS, (0.705, 0.235, 0.5, 0.355, 0.505, 0.405), strict=True))
RAW_THRESHOLDS = dict(zip(TARGETS, (100, 500, 0.05, 5, 25, 10), strict=True))


def collate(batch):
    return [torch.stack([item[0][index] for item in batch]) for index in range(6)], torch.stack([item[1] for item in batch]), torch.stack([item[2] for item in batch])


def predict(model, branches):
    with torch.no_grad():
        return torch.sigmoid(model(branches)).numpy()


def metrics(frame, probabilities):
    rows = []
    for index, target in enumerate(TARGETS):
        raw = pd.to_numeric(frame[f"{target}_raw" if target != "Mg" else "MgO_raw"], errors="coerce")
        known = frame[f"{target}_mask"].to_numpy() > 0
        labels = (raw.to_numpy() > RAW_THRESHOLDS[target]).astype(int)[known]
        probs = probabilities[known, index]
        rows.append({"target": target, "pr_auc": average_precision_score(labels, probs), "f1": f1_score(labels, probs >= THRESHOLDS[target], zero_division=0), "known": len(labels)})
    return rows


def matrix_match(frame, probabilities):
    matrix = pd.read_csv(ROOT / "outputs/z903/matrix_shortcut_audit/matrix_matched_metrics.csv")
    rows = []
    controls = ["CaO", "MgO", "Fe2O3*", "Al2O3", "SiO2"]
    workbook = pd.read_excel(ROOT / "data/metadata/libs_metadata.xlsx", engine="openpyxl")
    workbook.columns = [str(column).strip() for column in workbook.columns]
    workbook["_key"] = workbook["PELLET NAME"].astype(str).str.lower()
    lookup = workbook.set_index("_key")[controls]
    composition = frame["sample_id"].astype(str).str.lower().map(lookup.to_dict(orient="index"))
    complete = np.asarray([isinstance(value, dict) for value in composition])
    if not complete.any():
        return rows
    composition_array = np.asarray([[value[column] for column in controls] if isinstance(value, dict) else [np.nan] * len(controls) for value in composition], dtype=float)
    composition_array = np.nan_to_num(composition_array / (np.nanstd(composition_array, axis=0) + 1e-8), nan=0.0)
    for index, target in enumerate(TARGETS):
        raw = pd.to_numeric(frame[f"{target}_raw" if target != "Mg" else "MgO_raw"], errors="coerce").to_numpy()
        known = frame[f"{target}_mask"].to_numpy() > 0
        positive = np.flatnonzero(complete & known & (raw > RAW_THRESHOLDS[target]))
        negative = set(np.flatnonzero(complete & known & (raw <= RAW_THRESHOLDS[target])).tolist())
        pairs = []
        for positive_index in positive:
            if not negative:
                break
            selected = min(negative, key=lambda candidate: np.linalg.norm(composition_array[positive_index] - composition_array[candidate]))
            negative.remove(selected)
            pairs.extend([positive_index, selected])
        labels = (raw[pairs] > RAW_THRESHOLDS[target]).astype(int)
        rows.append({"target": target, "matched_samples": len(pairs), "matrix_matched_pr_auc": average_precision_score(labels, probabilities[pairs, index]), "matrix_matched_f1": f1_score(labels, probabilities[pairs, index] >= THRESHOLDS[target], zero_division=0), "baseline_matrix_matched_pr_auc": float(matrix.loc[matrix["target"] == target, "matched_pr_auc"].iloc[0]), "baseline_matrix_matched_f1": float(matrix.loc[matrix["target"] == target, "matched_f1"].iloc[0])})
    return rows


def occlusion(model, dataset, loader):
    branches, _, _ = next(iter(loader))
    baseline = predict(model, branches)
    rows = []
    windows = pd.read_csv(ROOT / "data/processed/z903_target_windows.csv")
    for target_index, target in enumerate(TARGETS):
        selected = windows[(windows["target"] == target) & windows["selected"]].reset_index(drop=True)
        for window_index, window in selected.iterrows():
            center = (window.window_start_nm + window.window_end_nm) / 2
            branch = branches[target_index].clone()
            start = int(np.argmin(np.abs(WAVELENGTHS - window.window_start_nm)))
            end = int(np.argmin(np.abs(WAVELENGTHS - window.window_end_nm))) + 1
            branch[:, :, start:end] = 0.0
            changed = list(branches)
            changed[target_index] = branch
            probabilities = predict(model, changed)
            rows.append({"target": target, "window_index": window_index, "center_nm": center, "target_probability_change": float(np.mean(probabilities[:, target_index] - baseline[:, target_index])), "mean_absolute_cross_output_change": float(np.mean(np.abs(probabilities - baseline), axis=0).mean())})
    return rows


def main() -> None:
    dataset = PhysicsWindowDataset(ROOT / "data/processed/z903_val.csv", ROOT / "data/processed/z903_target_windows.csv")
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), collate_fn=collate)
    branches, _, _ = next(iter(loader))
    model = PhysicsConstrainedZ903CNN([len(dataset.window_indices[target]) for target in TARGETS])
    model.load_state_dict(torch.load(ROOT / "models/z903_physics_constrained_best.pt", map_location="cpu")["model_state_dict"])
    model.eval()
    frame = pd.read_csv(ROOT / "data/processed/z903_val.csv")
    probabilities = predict(model, branches)
    constrained_metrics = metrics(frame, probabilities)
    matched = matrix_match(frame, probabilities)
    occluded = occlusion(model, dataset, loader)
    output = ROOT / "outputs/z903/physics_constrained"
    pd.DataFrame(matched).to_csv(output / "matrix_matched_metrics.csv", index=False)
    pd.DataFrame(occluded).to_csv(output / "window_occlusion.csv", index=False)
    baseline = pd.read_csv(ROOT / "outputs/z903/z903_cnn_robust/validation_metrics.csv")
    lines = ["# Z-903 Physics-Constrained CNN", "", "The saved constrained checkpoint was evaluated after training; no retraining or historical test-set tuning was performed.", "", "## Selected Windows", ""]
    windows = pd.read_csv(ROOT / "data/processed/z903_target_windows.csv")
    for target in TARGETS:
        lines.append(f"- {target}: {int(((windows['target'] == target) & windows['selected']).sum())} windows; branch length {len(dataset.window_indices[target])}")
    lines.extend(["", "## Validation Comparison", "", "| Target | Full PR-AUC | Full F1 | Window PR-AUC | Window F1 |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in constrained_metrics:
        base = baseline[baseline["element"] == row["target"]].iloc[0]
        lines.append(f"| {row['target']} | {base['pr_auc']:.4f} | {base['f1']:.4f} | {row['pr_auc']:.4f} | {row['f1']:.4f} |")
    lines.extend(["", "## Matrix-Matched Validation", "", "| Target | Full matched PR-AUC | Window matched PR-AUC | Full matched F1 | Window matched F1 |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in matched:
        lines.append(f"| {row['target']} | {row['baseline_matrix_matched_pr_auc']:.4f} | {row['matrix_matched_pr_auc']:.4f} | {row['baseline_matrix_matched_f1']:.4f} | {row['matrix_matched_f1']:.4f} |")
    lines.extend(["", "## Target-Window Occlusion", "", "| Target | Mean target probability change after window occlusion |", "| --- | ---: |"])
    for target in TARGETS:
        values = [row["target_probability_change"] for row in occluded if row["target"] == target]
        lines.append(f"| {target} | {np.mean(values):.6f} |")
    lines.extend(["", "## Assessment", "", "Window branches enforce target-specific inputs, but they do not prove physical specificity. Compare matrix-matched changes and target-window occlusion against the full-spectrum baseline before scientific interpretation.", "", "- Historical test data were not used for redesign selection or tuning.", "- Labels remain exploratory composition thresholds rather than validated detection limits."])
    (ROOT / "reports/z903_physics_constrained_cnn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"validation": constrained_metrics, "matrix_matched": matched, "occlusion_rows": len(occluded)}, indent=2))


if __name__ == "__main__":
    main()