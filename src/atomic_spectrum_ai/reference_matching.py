"""Deterministic numerical-spectrum reference and peak matching.

This module deliberately has no dependency on the CNN, checkpoints, or image
models.  It uses cached NIST line tables and (when present) the Z-903 empirical
line audit only to choose useful reference candidates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

TARGET_ELEMENTS = ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb")
MATRIX_ELEMENTS = ("Ca", "Fe", "Al", "Si", "Na", "K", "Ti", "Mg")
ALL_ELEMENTS = tuple(dict.fromkeys((*TARGET_ELEMENTS, *MATRIX_ELEMENTS)))
DEFAULT_TOLERANCE_NM = 0.25


@dataclass(frozen=True)
class SpectrumData:
    """Clean spectrum plus the untouched uploaded table."""

    wavelength_nm: np.ndarray
    intensity: np.ndarray
    original: pd.DataFrame


@dataclass(frozen=True)
class MatchingResult:
    peaks: pd.DataFrame
    references: pd.DataFrame
    matches: pd.DataFrame
    scores: pd.DataFrame


def _column_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _find_column(columns: list[object], aliases: set[str]) -> object | None:
    return next((column for column in columns if _column_key(column) in aliases), None)


def load_spectrum_csv(source: str | Path | BinaryIO) -> SpectrumData:
    """Load a CSV, retaining its original frame while cleaning a working copy."""
    original = pd.read_csv(source)
    wavelength_column = _find_column(
        list(original.columns),
        {"wavelength", "wavelengthnm", "wavelengthnanometer", "wavelengthnanometers", "lambda", "wl", "nm"},
    )
    intensity_column = _find_column(
        list(original.columns),
        {"intensity", "signal", "counts", "count", "emission", "radiance", "sum", "value"},
    )
    if wavelength_column is None or intensity_column is None:
        raise ValueError("CSV must contain recognizable wavelength and intensity columns")

    cleaned = pd.DataFrame(
        {
            "wavelength_nm": pd.to_numeric(original[wavelength_column], errors="coerce"),
            "intensity": pd.to_numeric(original[intensity_column], errors="coerce"),
        }
    )
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        raise ValueError("Spectrum contains no finite wavelength/intensity rows")
    cleaned = cleaned.groupby("wavelength_nm", as_index=False)["intensity"].mean()
    cleaned = cleaned.sort_values("wavelength_nm").reset_index(drop=True)
    if len(cleaned) < 2 or np.any(np.diff(cleaned["wavelength_nm"]) <= 0):
        raise ValueError("Spectrum must contain at least two increasing wavelengths")
    return SpectrumData(
        cleaned["wavelength_nm"].to_numpy(float),
        cleaned["intensity"].to_numpy(float),
        original.copy(),
    )


def detect_peaks(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    *,
    minimum_prominence: float = 0.05,
    minimum_normalized_intensity: float = 0.05,
    minimum_peak_distance_nm: float = 0.25,
) -> pd.DataFrame:
    """Extract deterministic peaks and widths (width is reported in nm)."""
    wavelengths = np.asarray(wavelength_nm, dtype=float)
    values = np.asarray(intensity, dtype=float)
    if wavelengths.size != values.size or wavelengths.size < 2:
        raise ValueError("Wavelength and intensity arrays must have equal length >= 2")
    if not np.all(np.isfinite(wavelengths)) or not np.all(np.isfinite(values)):
        raise ValueError("Peak extraction requires finite wavelength and intensity values")
    spacing = float(np.median(np.diff(wavelengths)))
    if spacing <= 0:
        raise ValueError("Wavelengths must be strictly increasing")
    scale = float(np.max(values) - np.min(values))
    normalized = (values - np.min(values)) / scale if scale > 0 else np.zeros_like(values)
    distance = max(1, int(np.ceil(minimum_peak_distance_nm / spacing)))
    indices, properties = find_peaks(
        values,
        prominence=minimum_prominence,
        distance=distance,
        height=minimum_normalized_intensity * scale + np.min(values),
    )
    if not len(indices):
        return pd.DataFrame(columns=["peak_wavelength_nm", "peak_intensity", "normalized_intensity", "prominence", "width"])
    prominences = properties["prominences"]
    widths = peak_widths(values, indices, rel_height=0.5)[0] * spacing
    frame = pd.DataFrame(
        {
            "peak_wavelength_nm": wavelengths[indices],
            "peak_intensity": values[indices],
            "normalized_intensity": normalized[indices],
            "prominence": prominences,
            "width": widths,
        }
    )
    return frame.sort_values("peak_wavelength_nm").reset_index(drop=True)


def _nist_table(element: str, data_dir: Path) -> pd.DataFrame:
    frames = []
    for stage in ("I", "II"):
        path = data_dir / f"{element}_{stage}.csv"
        if not path.is_file():
            continue
        table = pd.read_csv(path)
        wavelength_column = "observed_wavelength_nm" if pd.to_numeric(table.get("observed_wavelength_nm"), errors="coerce").notna().any() else "ritz_wavelength_nm"
        table["wavelength_nm"] = pd.to_numeric(table[wavelength_column], errors="coerce")
        table["relative_intensity"] = pd.to_numeric(table.get("relative_intensity"), errors="coerce")
        table["Aki"] = pd.to_numeric(table.get("Aki"), errors="coerce")
        table["element"] = element
        table["ion_stage"] = table.get("ionization_stage", stage)
        frames.append(table[["element", "ion_stage", "wavelength_nm", "relative_intensity", "Aki"]])
    if not frames:
        return pd.DataFrame(columns=["element", "ion_stage", "wavelength_nm", "relative_intensity", "Aki"])
    return pd.concat(frames, ignore_index=True)


def load_reference_database(
    wavelength_range: tuple[float, float],
    *,
    data_dir: str | Path | None = None,
    empirical_path: str | Path | None = None,
    max_target_lines: int = 30,
    max_matrix_lines: int = 50,
) -> pd.DataFrame:
    """Load useful cached NIST lines in the requested range.

    Empirically selected Z-903 lines are high confidence; remaining candidates
    are moderate confidence and selected by normalized NIST strength.
    """
    root = Path(__file__).resolve().parents[2]
    data_path = Path(data_dir) if data_dir is not None else root / "data" / "nist_lines" / "processed"
    low, high = wavelength_range
    empirical = None
    audit = Path(empirical_path) if empirical_path is not None else root / "data" / "processed" / "z903_empirical_target_lines.csv"
    if audit.is_file():
        empirical = pd.read_csv(audit)
        empirical["wavelength_nm"] = pd.to_numeric(empirical.get("wavelength_nm"), errors="coerce")
    selected = []
    for element in ALL_ELEMENTS:
        table = _nist_table(element, data_path)
        table = table[table["wavelength_nm"].between(low, high)].copy()
        if table.empty:
            continue
        table["reference_intensity"] = table["relative_intensity"]
        strength = table["relative_intensity"].fillna(0.0)
        if not (strength > 0).any():
            strength = np.log1p(table["Aki"].clip(lower=0).fillna(0.0))
        table["strength"] = strength
        max_strength = float(strength.max())
        table["normalized_strength"] = strength / max_strength if max_strength > 0 else 0.0
        table["confidence"] = "moderate"
        if empirical is not None and element in TARGET_ELEMENTS and "selected" in empirical:
            high_lines = empirical[(empirical["target"] == element) & empirical["selected"].astype(bool)]["wavelength_nm"]
            close = table["wavelength_nm"].apply(
                lambda value, selected_lines=high_lines: bool(
                    np.any(np.abs(selected_lines - value) <= 1e-6)
                )
            )
            table.loc[close, "confidence"] = "high"
        table = table.sort_values(["confidence", "normalized_strength"], ascending=[True, False])
        limit = max_target_lines if element in TARGET_ELEMENTS else max_matrix_lines
        selected.append(table.head(limit))
    if not selected:
        return pd.DataFrame(columns=["element", "ion_stage", "wavelength_nm", "reference_intensity", "Aki", "confidence", "normalized_strength"])
    return pd.concat(selected, ignore_index=True).drop_duplicates(["element", "wavelength_nm"])


def analyze_spectrum(
    spectrum: SpectrumData,
    references: pd.DataFrame,
    *,
    tolerance_nm: float = DEFAULT_TOLERANCE_NM,
    minimum_prominence: float = 0.05,
    minimum_normalized_intensity: float = 0.05,
    minimum_peak_distance_nm: float = 0.25,
    detected_score: float = 0.60,
    possible_score: float = 0.15,
    min_detected_lines: int = 2,
    min_strong_detected_lines: int = 1,
) -> MatchingResult:
    """Match peaks, classify interference, and compute interpretable scores."""
    peaks = detect_peaks(
        spectrum.wavelength_nm,
        spectrum.intensity,
        minimum_prominence=minimum_prominence,
        minimum_normalized_intensity=minimum_normalized_intensity,
        minimum_peak_distance_nm=minimum_peak_distance_nm,
    )
    rows = []
    target_refs = references[references["element"].isin(TARGET_ELEMENTS)]
    matrix_refs = references[references["element"].isin(MATRIX_ELEMENTS)]
    for _, line in target_refs.iterrows():
        distances = np.abs(peaks["peak_wavelength_nm"].to_numpy() - float(line.wavelength_nm))
        if not len(distances) or float(distances.min()) > tolerance_nm:
            continue
        peak = peaks.iloc[int(np.argmin(distances))]
        candidates = matrix_refs[matrix_refs["element"] != line.element]
        candidates = candidates[np.abs(candidates["wavelength_nm"] - line.wavelength_nm) <= tolerance_nm]
        strong = candidates[candidates["normalized_strength"] >= 0.5]
        level = "UNIQUE / LOW INTERFERENCE" if strong.empty else ("HIGH INTERFERENCE" if len(strong) > 1 else "POSSIBLE INTERFERENCE")
        interference = strong.iloc[0] if not strong.empty else None
        reference_strength = float(line.normalized_strength)
        weight = float(peak.normalized_intensity * max(reference_strength, 0.1) * (1.0 if strong.empty else 0.35))
        rows.append({"element": line.element, "observed_peak_nm": float(peak.peak_wavelength_nm), "reference_line_nm": float(line.wavelength_nm), "wavelength_error_nm": float(peak.peak_wavelength_nm - line.wavelength_nm), "observed_intensity": float(peak.peak_intensity), "peak_prominence": float(peak.prominence), "ion_stage": line.ion_stage, "reference_intensity": line.reference_intensity, "Aki": line.Aki, "interference_element": interference.element if interference is not None else "", "interference_line_nm": float(interference.wavelength_nm) if interference is not None else np.nan, "interference_level": level, "match_weight": weight, "confidence": line.confidence})
    matches = pd.DataFrame(rows, columns=["element", "observed_peak_nm", "reference_line_nm", "wavelength_error_nm", "observed_intensity", "peak_prominence", "ion_stage", "reference_intensity", "Aki", "interference_element", "interference_line_nm", "interference_level", "match_weight", "confidence"])
    score_rows = []
    for element in TARGET_ELEMENTS:
        evaluated = target_refs[target_refs.element == element]
        found = matches[matches.element == element]
        low_interference = found[found.interference_level == "UNIQUE / LOW INTERFERENCE"]
        strong_found = found[found.confidence == "high"]
        evidence = float(found.match_weight.sum()) / max(float(evaluated.normalized_strength.sum()), 1e-12)
        score = float(np.clip(evidence * (0.5 + 0.5 * min(len(low_interference), 3) / 3), 0.0, 1.0))
        enough_for_possible = score >= possible_score or (len(found) > 0 and len(low_interference) == 0)
        status = "DETECTED" if score >= detected_score and len(low_interference) >= min_detected_lines and len(strong_found) >= min_strong_detected_lines else ("POSSIBLE" if enough_for_possible else "NOT DETECTED")
        score_rows.append({"element": element, "reference_lines_evaluated": len(evaluated), "strong_lines_evaluated": int((evaluated.confidence == "high").sum()), "matched_lines": len(found), "strong_matched_lines": len(strong_found), "low_interference_matches": len(low_interference), "mean_wavelength_error_nm": float(found.wavelength_error_nm.abs().mean()) if len(found) else np.nan, "reference_matching_score": score, "status": status})
    return MatchingResult(peaks, references, matches, pd.DataFrame(score_rows))


def save_peak_table(peaks: pd.DataFrame, path: str | Path) -> None:
    """Save extracted peaks as CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    peaks.to_csv(path, index=False)


def save_match_table(matches: pd.DataFrame, path: str | Path) -> None:
    """Save explainable reference matches as CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(path, index=False)


def plot_matching_result(spectrum: SpectrumData, result: MatchingResult, path: str | Path) -> None:
    """Render the numerical spectrum and matched reference markers."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(spectrum.wavelength_nm, spectrum.intensity, color="black", linewidth=1, label="Unknown spectrum")
    if not result.peaks.empty:
        ax.scatter(result.peaks.peak_wavelength_nm, result.peaks.peak_intensity, color="crimson", s=20, label="Experimental peaks")
    colors = dict(zip(TARGET_ELEMENTS, ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"), strict=True))
    for element in TARGET_ELEMENTS:
        lines = result.matches[result.matches.element == element]
        for index, line in lines.iterrows():
            ax.axvline(line.reference_line_nm, color=colors[element], alpha=0.45, linewidth=1, label=element if index == lines.index[0] else None)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.legend(ncol=3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


__all__ = ["TARGET_ELEMENTS", "MATRIX_ELEMENTS", "SpectrumData", "MatchingResult", "load_spectrum_csv", "detect_peaks", "load_reference_database", "analyze_spectrum", "save_peak_table", "save_match_table", "plot_matching_result"]