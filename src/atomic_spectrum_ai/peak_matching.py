from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from atomic_spectrum_ai.reference_spectra import MIXTURE_ELEMENTS, load_all_references


def detect_peaks(wavelength: np.ndarray, intensity: np.ndarray, height: float = 0.1, distance: int = 5):
    peaks, props = find_peaks(intensity, height=height, distance=distance)
    return wavelength[peaks], props


def match_peaks_to_references(wavelength: np.ndarray, intensity: np.ndarray, tolerance_nm: float = 0.2) -> dict[str, float]:
    refs = load_all_references()
    peak_wl, props = detect_peaks(wavelength, intensity, height=0.05)
    scores = {el: 0.0 for el in MIXTURE_ELEMENTS}
    if len(peak_wl) == 0:
        return scores
    for el, df in refs.items():
        if df.empty:
            continue
        ref_wl = df['wavelength_nm'].values
        # count number of peaks within tolerance
        matches = 0
        for pw in peak_wl:
            if np.any(np.abs(ref_wl - pw) <= tolerance_nm):
                matches += 1
        # simple score: matched peaks / min(total ref peaks, 5)
        scores[el] = matches / max(1, min(len(ref_wl), 5))
    return scores


def load_numeric_spectrum(csv_path: str):
    df = pd.read_csv(csv_path)
    return df["wavelength_nm"].values, df["intensity"].values


__all__ = ["detect_peaks", "match_peaks_to_references", "load_numeric_spectrum"]
