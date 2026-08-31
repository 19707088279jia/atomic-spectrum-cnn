from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from src.data import EXPECTED_POINTS, demo_labels, load_demo, load_spectrum
from src.inference import load_checkpoint, predict
from src.labels import CNN_THRESHOLDS, TARGETS
from src.preprocessing import robust_scale
from src.reference_matching import match

ROOT = Path(__file__).resolve().parents[1]


def test_csv_loading_and_demo_loading() -> None:
    wavelengths, intensity, row = load_demo("Clear Demo")
    assert len(wavelengths) == EXPECTED_POINTS
    assert len(intensity) == EXPECTED_POINTS
    assert row["filename"] == "clear_mix349.csv"
    assert set(demo_labels(row)) == set(TARGETS)


def test_invalid_csv_handling(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"x": [1], "y": [2]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_spectrum(path)


def test_only_six_target_elements() -> None:
    assert TARGETS == ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb")
    assert not {"Fe", "Ca", "Na"}.intersection(TARGETS)


def test_robust_preprocessing() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    scaled = robust_scale(values)
    assert np.isclose(np.median(scaled), 0.0)
    assert np.isfinite(scaled).all()


def test_checkpoint_and_six_cnn_outputs() -> None:
    model, checkpoint = load_checkpoint(ROOT / "models" / "z903_cnn_robust_best.pt")
    wavelengths, intensity, _ = load_demo("Clear Demo")
    assert checkpoint["preprocess"] == "robust"
    tensor = torch.from_numpy(robust_scale(intensity)).reshape(1, 1, EXPECTED_POINTS)
    assert tuple(tensor.shape) == (1, 1, 23401)
    outputs = predict(intensity, model)
    assert tuple(outputs) == TARGETS
    assert all(0.0 <= item["probability"] <= 1.0 for item in outputs.values())
    assert wavelengths.shape == (EXPECTED_POINTS,)


def test_frozen_thresholds() -> None:
    assert CNN_THRESHOLDS == {"Zn": 0.7050, "Mn": 0.2350, "Cd": 0.5000, "Mg": 0.3550, "Cu": 0.5050, "Pb": 0.4050}


def test_reference_matching_is_independent() -> None:
    wavelengths, intensity, _ = load_demo("Clear Demo")
    scores, matches = match(wavelengths, intensity)
    assert scores["element"].tolist() == list(TARGETS)
    assert set(scores.columns) == {"element", "score", "result", "matched_peaks"}
    assert isinstance(matches, pd.DataFrame)
