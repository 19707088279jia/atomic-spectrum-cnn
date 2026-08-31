#!/usr/bin/env python
"""Select target-specific NIST windows using training spectra only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from atomic_spectrum_ai.z903 import TARGETS, WAVELENGTHS, preprocess_spectrum, read_z903_spectrum

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ("Ca", "Fe", "Al", "Si", "Na", "K", "Ti", "Mg")
RAW_THRESHOLDS = dict(zip(TARGETS, (100.0, 500.0, 0.05, 5.0, 25.0, 10.0), strict=True))


def lines(element: str) -> pd.DataFrame:
    tables = [pd.read_csv(ROOT / "data" / "nist_lines" / "processed" / f"{element}_{stage}.csv") for stage in ("I", "II")]
    table = pd.concat(tables, ignore_index=True)
    wavelength_column = "observed_wavelength_nm" if table["observed_wavelength_nm"].notna().any() else "ritz_wavelength_nm"
    table["line_nm"] = pd.to_numeric(table[wavelength_column], errors="coerce")
    table["strength"] = pd.to_numeric(table["relative_intensity"], errors="coerce")
    if not table["strength"].notna().any():
        table["strength"] = pd.to_numeric(table["Aki"], errors="coerce")
    table["strength"] = table["strength"].fillna(0.0)
    return table.dropna(subset=["line_nm"]).sort_values("strength", ascending=False)


def nearest_matrix(line_nm: float, matrix_lines: dict[str, pd.DataFrame]) -> tuple[str, float]:
    candidates = []
    for element, table in matrix_lines.items():
        distance = float(np.min(np.abs(table["line_nm"].to_numpy() - line_nm)))
        nearest = float(table.iloc[int(np.argmin(np.abs(table["line_nm"].to_numpy() - line_nm)))] ["line_nm"])
        candidates.append((distance, element, nearest))
    distance, element, nearest = min(candidates)
    return element, nearest


def contrast(line_nm: float, train: pd.DataFrame, spectra: np.ndarray, target: str) -> tuple[float, float, float, float]:
    index = int(np.argmin(np.abs(WAVELENGTHS - line_nm)))
    window = max(1, int(round(0.5 / float(WAVELENGTHS[1] - WAVELENGTHS[0]))))
    positive, negative = [], []
    raw_column = "MgO_raw" if target == "Mg" else f"{target}_raw"
    values = spectra[:, max(0, index - window) : index + window + 1].mean(axis=1)
    for row_index, (_, row) in enumerate(train.iterrows()):
        value = float(values[row_index])
        if pd.notna(row[raw_column]):
            (positive if float(row[raw_column]) > RAW_THRESHOLDS[target] else negative).append(value)
    p = np.asarray(positive)
    n = np.asarray(negative)
    effect = float((p.mean() - n.mean()) / (np.sqrt((p.var() + n.var()) / 2) + 1e-8)) if len(p) and len(n) else 0.0
    return float(p.mean()) if len(p) else np.nan, float(np.median(p)) if len(p) else np.nan, float(n.mean()) if len(n) else np.nan, effect


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "processed" / "z903_train.csv")
    spectra = []
    for path in train["spectrum_path"]:
        _, intensity = read_z903_spectrum(path)
        spectra.append(preprocess_spectrum(intensity, "robust"))
    spectra_array = np.asarray(spectra, dtype=np.float32)
    matrix_lines = {element: lines(element) for element in MATRIX}
    rows = []
    selected_counts = {}
    for target in TARGETS:
        target_lines = lines(target).drop_duplicates("line_nm")
        scored = []
        for _, line in target_lines.iterrows():
            line_nm = float(line["line_nm"])
            matrix_element, matrix_nm = nearest_matrix(line_nm, matrix_lines)
            pmean, pmedian, nmean, effect = contrast(line_nm, train, spectra_array, target)
            distance = abs(line_nm - matrix_nm)
            scored.append({"target": target, "target_line_nm": line_nm, "ion_stage": line["ionization_stage"], "NIST_intensity": float(line.get("relative_intensity", np.nan)) if pd.notna(line.get("relative_intensity", np.nan)) else np.nan, "Aki": float(line.get("Aki", np.nan)) if pd.notna(line.get("Aki", np.nan)) else np.nan, "window_start_nm": line_nm - 0.5, "window_end_nm": line_nm + 0.5, "nearest_matrix_element": matrix_element, "nearest_matrix_line_nm": matrix_nm, "matrix_distance_nm": distance, "matrix_confounded": distance <= 0.25, "training_positive_mean": pmean, "training_positive_median": pmedian, "training_negative_mean": nmean, "training_positive_contrast": effect, "selected": False})
        scored.sort(key=lambda row: (row["matrix_confounded"], -row["training_positive_contrast"], -row["NIST_intensity"] if np.isfinite(row["NIST_intensity"]) else 0.0))
        chosen = [
            row
            for row in scored
            if not row["matrix_confounded"]
            and row["training_positive_contrast"] >= 0.20
        ]
        for row in chosen:
            row["selected"] = True
        selected_counts[target] = len(chosen)
        rows.extend(scored)
    output = ROOT / "data" / "processed" / "z903_target_windows.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    (ROOT / "data" / "processed" / "z903_target_windows.json").write_text(json.dumps({"selected_counts": selected_counts, "source": "NIST ASD cached I/II tables", "contrast_split": "train only", "window_half_width_nm": 0.5}, indent=2), encoding="utf-8")
    print(json.dumps(selected_counts, indent=2))


if __name__ == "__main__":
    main()