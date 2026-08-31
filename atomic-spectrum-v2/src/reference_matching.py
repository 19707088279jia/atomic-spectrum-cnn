"""Deterministic NIST peak matching limited to the six V2 targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from .labels import TARGETS

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "data" / "nist_lines" / "processed"


def extract_peaks(wavelengths: np.ndarray, intensity: np.ndarray) -> pd.DataFrame:
    """Extract deterministic peaks using the validated matching defaults."""
    scale = float(np.max(intensity) - np.min(intensity))
    normalized = (intensity - np.min(intensity)) / scale if scale > 0 else np.zeros_like(intensity)
    spacing = float(np.median(np.diff(wavelengths)))
    indices, properties = find_peaks(
        intensity,
        prominence=0.05,
        distance=max(1, int(np.ceil(0.25 / spacing))),
        height=np.min(intensity) + 0.05 * scale,
    )
    widths = peak_widths(intensity, indices, rel_height=0.5)[0] * spacing
    return pd.DataFrame({
        "peak_wavelength_nm": wavelengths[indices],
        "peak_intensity": intensity[indices],
        "normalized_intensity": normalized[indices],
        "prominence": properties["prominences"],
        "width": widths,
    })


def load_references(wavelength_range: tuple[float, float]) -> pd.DataFrame:
    """Load cached NIST I/II lines for only the six V2 targets."""
    rows = []
    for element in TARGETS:
        for stage in ("I", "II"):
            path = REFERENCE_DIR / f"{element}_{stage}.csv"
            if not path.is_file():
                continue
            table = pd.read_csv(path)
            wavelength_column = "observed_wavelength_nm" if pd.to_numeric(table["observed_wavelength_nm"], errors="coerce").notna().any() else "ritz_wavelength_nm"
            table["wavelength_nm"] = pd.to_numeric(table[wavelength_column], errors="coerce")
            table["relative_intensity"] = pd.to_numeric(table["relative_intensity"], errors="coerce")
            table = table[table.wavelength_nm.between(*wavelength_range)]
            for _, row in table.dropna(subset=["wavelength_nm"]).iterrows():
                rows.append({"element": element, "ion_stage": stage, "wavelength_nm": float(row.wavelength_nm), "reference_intensity": row.relative_intensity})
    return pd.DataFrame(rows)


def match(wavelengths: np.ndarray, intensity: np.ndarray, tolerance_nm: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return target scores and an explainable matched-peak table."""
    peaks = extract_peaks(wavelengths, intensity)
    references = load_references((float(wavelengths.min()), float(wavelengths.max())))
    rows = []
    for element in TARGETS:
        target = references[references.element == element]
        matched = []
        for _, reference in target.iterrows():
            if peaks.empty:
                continue
            distances = np.abs(peaks.peak_wavelength_nm.to_numpy() - reference.wavelength_nm)
            index = int(np.argmin(distances))
            if float(distances[index]) <= tolerance_nm:
                peak = peaks.iloc[index]
                matched.append({"element": element, "observed_peak_nm": float(peak.peak_wavelength_nm), "reference_line_nm": float(reference.wavelength_nm), "wavelength_error_nm": float(distances[index]), "peak_prominence": float(peak.prominence), "ion_stage": reference.ion_stage})
        rows.extend(matched)
    matches = pd.DataFrame(rows)
    score_rows = []
    for element in TARGETS:
        found = matches[matches.element == element] if not matches.empty else matches
        score_rows.append({"element": element, "score": float(min(len(found) / 3.0, 1.0)), "result": "DETECTED" if len(found) >= 2 else ("POSSIBLE" if len(found) else "NOT DETECTED"), "matched_peaks": len(found)})
    return pd.DataFrame(score_rows), matches


__all__ = ["extract_peaks", "load_references", "match"]
