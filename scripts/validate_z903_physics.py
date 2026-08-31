#!/usr/bin/env python
"""Validate robust Z-903 CNN attributions against NIST atomic-line data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from atomic_spectrum_ai.models.z903_cnn import Z903CNN
from atomic_spectrum_ai.z903 import TARGETS, WAVELENGTHS, preprocess_spectrum, read_z903_spectrum

ROOT = Path(__file__).resolve().parents[1]
NIST_URL = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
THRESHOLDS = np.asarray([0.705, 0.235, 0.5, 0.355, 0.505, 0.405], dtype=np.float32)
TARGET_SYMBOLS = {element: element for element in TARGETS}


def load_model(checkpoint: Path) -> Z903CNN:
    model = Z903CNN()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def load_rows(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_inputs(frame: pd.DataFrame) -> tuple[torch.Tensor, np.ndarray]:
    values = []
    for path in frame["spectrum_path"]:
        _, intensity = read_z903_spectrum(path)
        values.append(preprocess_spectrum(intensity, option="robust"))
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Non-finite robust-scaled input")
    return torch.from_numpy(array[:, None, :]), array


def predict(model: Z903CNN, inputs: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(inputs)).numpy()


def integrated_gradients(model: Z903CNN, samples: torch.Tensor, target_index: int, steps: int = 8) -> np.ndarray:
    """Compute IG in alpha batches, retaining one profile per input sample."""
    baseline = torch.zeros_like(samples)
    profiles: list[np.ndarray] = []
    for chunk_start in range(0, len(samples), 32):
        chunk = samples[chunk_start : chunk_start + 32]
        expanded = baseline[chunk_start : chunk_start + len(chunk)] + (
            torch.linspace(0.0, 1.0, steps)[:, None, None, None]
            * (chunk - baseline[chunk_start : chunk_start + len(chunk)])[None, ...]
        )
        expanded = expanded.reshape(-1, 1, samples.shape[-1]).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(expanded)[:, target_index].sum()
        gradients = torch.autograd.grad(logits, expanded)[0].reshape(steps, len(chunk), 1, -1)
        attribution = (chunk - baseline[chunk_start : chunk_start + len(chunk)])[None, ...] * gradients
        profiles.append(attribution.mean(dim=0).detach().numpy()[:, 0, :])
    return np.abs(np.concatenate(profiles, axis=0))


def local_peaks(profile: np.ndarray, count: int = 20, minimum_distance: int = 20) -> list[int]:
    candidates = [index for index in range(1, len(profile) - 1) if profile[index] >= profile[index - 1] and profile[index] >= profile[index + 1]]
    selected: list[int] = []
    for index in sorted(candidates, key=lambda item: profile[item], reverse=True):
        if all(abs(index - other) >= minimum_distance for other in selected):
            selected.append(index)
        if len(selected) == count:
            break
    return selected


def nist_lines(element: str) -> tuple[pd.DataFrame, str]:
    tables = []
    for stage in ("I", "II"):
        path = ROOT / "data" / "nist_lines" / "processed" / f"{element}_{stage}.csv"
        if path.exists():
            tables.append(pd.read_csv(path))
    if not tables:
        return pd.DataFrame(columns=["element", "ionization_stage", "observed_wavelength_nm", "ritz_wavelength_nm", "relative_intensity", "Aki"]), "Cached NIST ASD tables unavailable"
    return pd.concat(tables, ignore_index=True), "NIST ASD cached normalized tables"


def fallback_lines(element: str) -> tuple[pd.DataFrame, str]:
    path = ROOT / "data" / "reference_spectra" / element / f"{element}_reference.csv"
    if not path.exists():
        return pd.DataFrame(columns=["wavelength_nm", "species", "source"]), "No repository reference file"
    frame = pd.read_csv(path)
    if frame.empty or "wavelength_nm" not in frame:
        return pd.DataFrame(columns=["wavelength_nm", "species", "source"]), "Repository reference file is empty"
    frame = frame.rename(columns={"ionization_state": "species"})
    frame["species"] = frame.get("species", "unknown")
    frame["source"] = "Repository reference file; provenance requires independent verification"
    return frame[["wavelength_nm", "species", "source"]], "Repository reference file"


def attribution_rows(model: Z903CNN, frame: pd.DataFrame, inputs: torch.Tensor, probabilities: np.ndarray, split: str) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    profiles: dict[str, np.ndarray] = {}
    labels = frame[[f"{element}_raw" if element != "Mg" else "MgO_raw" for element in TARGETS]].to_numpy(dtype=np.float32)
    masks = frame[[f"{element}_mask" for element in TARGETS]].to_numpy(dtype=np.float32)
    raw_thresholds = np.asarray([100, 500, 0.05, 5, 25, 10], dtype=np.float32)
    for target_index, element in enumerate(TARGETS):
        known_positive = (masks[:, target_index] > 0) & (labels[:, target_index] > raw_thresholds[target_index])
        predicted_positive = probabilities[:, target_index] >= THRESHOLDS[target_index]
        selected = known_positive & predicted_positive
        selected_indices = np.flatnonzero(selected)
        profile = np.zeros(WAVELENGTHS.size, dtype=np.float64)
        if len(selected_indices):
            profile = integrated_gradients(model, inputs[selected_indices], target_index).mean(axis=0)
        maximum = profile.max() if profile.size else 0.0
        normalized = profile / maximum if maximum > 0 else profile
        profiles[element] = normalized.astype(np.float32)
        for rank, index in enumerate(local_peaks(normalized), start=1):
            rows.append({"split": split, "element": element, "rank": rank, "wavelength_nm": float(WAVELENGTHS[index]), "attribution": float(normalized[index]), "selected_samples": int(len(selected_indices))})
        pd.DataFrame({"wavelength_nm": WAVELENGTHS, "attribution": normalized}).to_csv(ROOT / "outputs" / "z903" / "physics_validation" / f"{split}_{element}_attribution.csv", index=False)
    return rows, profiles


def compare_lines(peak_rows: list[dict[str, object]], line_map: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows = []
    for peak in peak_rows:
        lines = line_map[peak["element"]]
        if lines.empty:
            rows.append({**peak, "closest_line_nm": np.nan, "difference_nm": np.nan, "within_0_1_nm": False, "within_0_25_nm": False, "within_0_5_nm": False, "line_species": "", "line_source": "unavailable"})
            continue
        wavelength_column = "observed_wavelength_nm" if lines["observed_wavelength_nm"].notna().any() else "ritz_wavelength_nm"
        line_wavelengths = pd.to_numeric(lines[wavelength_column], errors="coerce")
        valid = line_wavelengths.notna()
        lines = lines.loc[valid].copy()
        line_wavelengths = line_wavelengths[valid]
        differences = np.abs(line_wavelengths.to_numpy() - float(peak["wavelength_nm"]))
        index = int(np.argmin(differences))
        difference = float(differences[index])
        intensity = pd.to_numeric(lines.get("relative_intensity", pd.Series(index=lines.index)), errors="coerce")
        aki = pd.to_numeric(lines.get("Aki", pd.Series(index=lines.index)), errors="coerce")
        strength = intensity if intensity.notna().any() else aki
        strongest_index = int(strength.fillna(-np.inf).idxmax()) if strength.notna().any() else int(lines.index[0])
        rows.append({**peak, "closest_line_nm": float(line_wavelengths.iloc[index]), "difference_nm": difference, "within_0_1_nm": difference <= 0.1, "within_0_25_nm": difference <= 0.25, "within_0_5_nm": difference <= 0.5, "line_species": str(lines.iloc[index].get("ionization_stage", "")), "strongest_reference_line_nm": float(line_wavelengths.loc[strongest_index]), "strongest_reference_strength": float(strength.loc[strongest_index]) if strength.notna().any() else np.nan, "line_source": "NIST ASD"})
    return rows


def occlusion(model: Z903CNN, inputs: torch.Tensor, profiles: dict[str, np.ndarray], probabilities: np.ndarray) -> list[dict[str, object]]:
    rows = []
    width = max(1, int(round(0.5 / float(WAVELENGTHS[1] - WAVELENGTHS[0]))))
    for target_index, element in enumerate(TARGETS):
        peaks = local_peaks(profiles[element], count=10)
        for peak in peaks:
            start = max(0, peak - width)
            end = min(inputs.shape[-1], peak + width + 1)
            changed = inputs.clone()
            changed[:, :, start:end] = 0.0
            masked_probability = predict(model, changed)[:, target_index]
            rows.append({"element": element, "wavelength_nm": float(WAVELENGTHS[peak]), "window_nm": 0.5, "mean_probability_before": float(probabilities[:, target_index].mean()), "mean_probability_after": float(masked_probability.mean()), "mean_probability_change": float(masked_probability.mean() - probabilities[:, target_index].mean())})
    return rows


def error_rows(model: Z903CNN, frame: pd.DataFrame, inputs: torch.Tensor, probabilities: np.ndarray, split: str) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows = []
    profiles: dict[str, np.ndarray] = {}
    raw = frame[[f"{element}_raw" if element != "Mg" else "MgO_raw" for element in TARGETS]].to_numpy(dtype=np.float32)
    masks = frame[[f"{element}_mask" for element in TARGETS]].to_numpy(dtype=np.float32)
    raw_thresholds = np.asarray([100, 500, 0.05, 5, 25, 10], dtype=np.float32)
    for target_index, element in enumerate(TARGETS):
        known = masks[:, target_index] > 0
        truth = raw[:, target_index] > raw_thresholds[target_index]
        pred = probabilities[:, target_index] >= THRESHOLDS[target_index]
        selected_profiles = {"tp": [], "fp": [], "fn": []}
        for kind, selection in (("fp", known & ~truth & pred), ("fn", known & truth & ~pred), ("tp", known & truth & pred)):
            indices = np.flatnonzero(selection)
            profile = np.zeros(WAVELENGTHS.size)
            if len(indices):
                profile = integrated_gradients(model, inputs[indices], target_index).mean(axis=0)
            selected_profiles[kind] = profile
            rows.append({"split": split, "element": element, "error_type": kind, "count": int(len(indices)), "mean_concentration": float(np.nanmean(raw[indices, target_index])) if len(indices) else np.nan, "top_wavelength_nm": float(WAVELENGTHS[int(np.argmax(profile))]) if len(indices) else np.nan})
        profiles[element] = selected_profiles
    return rows, profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/z903_cnn_robust_best.pt")
    args = parser.parse_args()
    output = ROOT / "outputs" / "z903" / "physics_validation"
    output.mkdir(parents=True, exist_ok=True)
    model = load_model(ROOT / args.checkpoint)
    val_frame = load_rows(ROOT / "data" / "processed" / "z903_val.csv")
    test_frame = load_rows(ROOT / "data" / "processed" / "z903_test.csv")
    val_inputs, _ = load_inputs(val_frame)
    test_inputs, _ = load_inputs(test_frame)
    val_probabilities = predict(model, val_inputs)
    test_probabilities = predict(model, test_inputs)
    peak_rows, profiles = attribution_rows(model, val_frame, val_inputs, val_probabilities, "validation")
    line_map: dict[str, pd.DataFrame] = {}
    line_sources: dict[str, str] = {}
    for element in TARGETS:
        lines, source = nist_lines(element)
        if lines.empty:
            raise RuntimeError(f"No cached authoritative NIST lines available for {element}")
        line_map[element] = lines
        line_sources[element] = source
        lines.to_csv(output / f"{element}_reference_lines.csv", index=False)
        plt.figure(figsize=(10, 3))
        plt.plot(WAVELENGTHS, profiles[element], color="black")
        if not lines.empty:
            line_wavelength_column = "observed_wavelength_nm" if lines["observed_wavelength_nm"].notna().any() else "ritz_wavelength_nm"
            plt.vlines(lines[line_wavelength_column].dropna(), 0, 1, color="tab:red", alpha=0.15)
        plt.xlim(180, 960)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Normalized IG attribution")
        plt.title(f"{element}: validation true-positive attribution and reference lines")
        plt.tight_layout()
        plt.savefig(output / f"{element}_attribution.png", dpi=140)
        plt.close()
    comparisons = compare_lines(peak_rows, line_map)
    pd.DataFrame(peak_rows).to_csv(output / "attribution_peaks.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "attribution_line_comparison.csv", index=False)
    occlusion_rows = occlusion(model, val_inputs, profiles, val_probabilities)
    pd.DataFrame(occlusion_rows).to_csv(output / "occlusion_results.csv", index=False)
    validation_errors, _ = error_rows(model, val_frame, val_inputs, val_probabilities, "validation")
    test_errors, _ = error_rows(model, test_frame, test_inputs, test_probabilities, "test")
    pd.DataFrame(validation_errors + test_errors).to_csv(output / "error_analysis.csv", index=False)
    overlap_rows = []
    for element in TARGETS:
        for other in TARGETS:
            if element == other:
                continue
            overlap = float(np.mean(profiles[element] * profiles[other]))
            overlap_rows.append({"target": element, "other_target": other, "attribution_profile_overlap": overlap})
    pd.DataFrame(overlap_rows).to_csv(output / "shortcut_overlap.csv", index=False)
    report = ["# Z-903 Physics Validation", "", "This analysis uses only the selected robust-scaling checkpoint. Attribution profiles and occlusion regions were defined from correctly predicted positive validation samples. The test split was not used to tune attribution, line matching, or thresholds.", "", "## Reference Lines", ""]
    for element in TARGETS:
        report.append(f"- {element}: {line_sources[element]}; retrieved/reference lines in 180-960 nm: {len(line_map[element])}")
    report.extend(["", "## Attribution and Line Agreement", "", "| Element | TP validation samples | Peak regions | Within 0.1 nm | Within 0.25 nm | Within 0.5 nm | Strongest/reference line examples |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for element in TARGETS:
        peaks = [row for row in comparisons if row["element"] == element]
        selected_count = peaks[0]["selected_samples"] if peaks else 0
        examples = ", ".join(f"{row['closest_line_nm']:.4f} nm ({row['line_species']})" for row in peaks[:3] if pd.notna(row.get("closest_line_nm"))) or "none"
        report.append(f"| {element} | {selected_count} | {len(peaks)} | {sum(row['within_0_1_nm'] for row in peaks)} | {sum(row['within_0_25_nm'] for row in peaks)} | {sum(row['within_0_5_nm'] for row in peaks)} | {examples} |")
    report.extend(["", "Line proximity is not proof of physical use: NIST line density, unresolved transitions, arbitrary instrument response, matrix effects, and attribution aggregation can all produce apparent proximity.", "", "## Occlusion", "", "| Element | Mean probability change after masking top regions | Regions with probability decrease |", "| --- | ---: | ---: |"])
    for element in TARGETS:
        rows = [row for row in occlusion_rows if row["element"] == element]
        report.append(f"| {element} | {np.mean([row['mean_probability_change'] for row in rows]):.6f} | {sum(row['mean_probability_change'] < 0 for row in rows)} / {len(rows)} |")
    report.extend(["", "## Shortcut and Error Checks", "", "Attribution profile overlap is reported in `shortcut_overlap.csv`; high overlap with another target's profile is a possible co-occurrence/matrix shortcut, not a causal conclusion.", "", "False-positive and false-negative counts, concentrations, and attribution maxima are in `error_analysis.csv` for validation and test.", "", "## Assessment", ""])
    for element in TARGETS:
        rows = [row for row in comparisons if row["element"] == element]
        occlusion_rows_element = [row for row in occlusion_rows if row["element"] == element]
        match_rate = sum(row["within_0_5_nm"] for row in rows) / len(rows) if rows else 0.0
        decrease_rate = sum(row["mean_probability_change"] < 0 for row in occlusion_rows_element) / len(occlusion_rows_element) if occlusion_rows_element else 0.0
        if line_map[element].empty or match_rate < 0.25 or decrease_rate < 0.5:
            status = "WEAK / POSSIBLE SHORTCUT"
        elif match_rate >= 0.5 and decrease_rate >= 0.7:
            status = "STRONG PHYSICAL EVIDENCE"
        else:
            status = "MODERATE PHYSICAL EVIDENCE"
        caveat = " Cd has only 44 known test labels; conclusions remain uncertain." if element == "Cd" else ""
        report.append(f"- **{element}: {status}**; attribution/reference overlap within 0.5 nm: {match_rate:.1%}; occlusion decrease rate: {decrease_rate:.1%}.{caveat}")
    report.extend(["", "These are interpretability associations, not validated detection claims. External real-world and cross-instrument validation remain required."])
    (ROOT / "reports" / "z903_physics_validation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.checkpoint), "attribution_peaks": len(peak_rows), "occlusion_rows": len(occlusion_rows), "reference_line_counts": {element: len(line_map[element]) for element in TARGETS}, "report": "reports/z903_physics_validation.md"}, indent=2))


if __name__ == "__main__":
    main()