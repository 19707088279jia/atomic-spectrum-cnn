from io import StringIO

import numpy as np
import pandas as pd
import pytest

from atomic_spectrum_ai.reference_matching import (
    SpectrumData,
    analyze_spectrum,
    detect_peaks,
    load_spectrum_csv,
)


def synthetic_spectrum() -> tuple[np.ndarray, np.ndarray]:
    wavelength = np.linspace(400.0, 402.0, 401)
    intensity = 0.01 * np.sin(wavelength * 10.0) + 0.1
    intensity += 1.0 * np.exp(-((wavelength - 400.5) ** 2) / 0.0008)
    intensity += 0.7 * np.exp(-((wavelength - 401.25) ** 2) / 0.0012)
    return wavelength, intensity


def reference_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_csv_variants_sort_and_preserve_original_with_nonfinite_rows() -> None:
    spectrum = load_spectrum_csv(
        StringIO("WL,Signal\n401,2\n400,1\nNaN,3\n402,inf\n")
    )
    assert spectrum.original.shape == (4, 2)
    assert spectrum.wavelength_nm.tolist() == [400.0, 401.0]
    assert spectrum.intensity.tolist() == [1.0, 2.0]


def test_peak_positions_and_configurable_distance() -> None:
    wavelength, intensity = synthetic_spectrum()
    peaks = detect_peaks(
        wavelength,
        intensity,
        minimum_prominence=0.2,
        minimum_normalized_intensity=0.2,
        minimum_peak_distance_nm=0.5,
    )
    assert np.allclose(peaks.peak_wavelength_nm, [400.5, 401.25], atol=0.01)
    assert set(["normalized_intensity", "prominence", "width"]) <= set(peaks.columns)


def test_tolerance_and_multiple_line_evidence() -> None:
    wavelength, intensity = synthetic_spectrum()
    refs = reference_frame(
        [
            {"element": "Zn", "ion_stage": "I", "wavelength_nm": 400.51, "reference_intensity": 100.0, "Aki": 1.0, "confidence": "high", "normalized_strength": 1.0},
            {"element": "Zn", "ion_stage": "I", "wavelength_nm": 401.24, "reference_intensity": 70.0, "Aki": 1.0, "confidence": "high", "normalized_strength": 0.7},
        ]
    )
    result = analyze_spectrum(SpectrumData(wavelength, intensity, pd.DataFrame()), refs, tolerance_nm=0.1, minimum_prominence=0.2, minimum_normalized_intensity=0.2)
    assert len(result.matches) == 2
    assert result.scores.loc[0, "status"] == "DETECTED"


def test_interference_prevents_detected_from_one_peak() -> None:
    wavelength, intensity = synthetic_spectrum()
    refs = reference_frame(
        [
            {"element": "Cu", "ion_stage": "I", "wavelength_nm": 400.5, "reference_intensity": 100.0, "Aki": 1.0, "confidence": "high", "normalized_strength": 1.0},
            {"element": "Fe", "ion_stage": "I", "wavelength_nm": 400.5, "reference_intensity": 100.0, "Aki": 1.0, "confidence": "moderate", "normalized_strength": 1.0},
        ]
    )
    result = analyze_spectrum(SpectrumData(wavelength, intensity, pd.DataFrame()), refs, minimum_prominence=0.2, minimum_normalized_intensity=0.2)
    assert result.matches.iloc[0].interference_level == "POSSIBLE INTERFERENCE"
    assert result.scores.loc[result.scores.element == "Cu", "status"].item() == "POSSIBLE"


def test_empty_and_missing_reference_results_are_deterministic() -> None:
    wavelength, intensity = synthetic_spectrum()
    refs = pd.DataFrame(columns=["element", "ion_stage", "wavelength_nm", "reference_intensity", "Aki", "confidence", "normalized_strength"])
    first = analyze_spectrum(SpectrumData(wavelength, intensity, pd.DataFrame()), refs, minimum_prominence=0.2, minimum_normalized_intensity=0.2)
    second = analyze_spectrum(SpectrumData(wavelength, intensity, pd.DataFrame()), refs, minimum_prominence=0.2, minimum_normalized_intensity=0.2)
    assert first.matches.empty
    assert first.scores.equals(second.scores)


def test_invalid_csv_is_actionable() -> None:
    with pytest.raises(ValueError, match="wavelength and intensity"):
        load_spectrum_csv(StringIO("x,y\n1,2\n"))