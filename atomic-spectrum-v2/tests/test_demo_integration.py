from pathlib import Path

import numpy as np
import pytest
from src.ag_inference import (
    classify_ag,
    load_ag_classifier,
    load_ag_concentration,
    predict_ag_concentration,
)
from src.cu_inference import (
    SUPPORTED_RANGE_PPM,
    load_cu_concentration,
    predict_cu_concentration,
)
from src.data import load_demo

ROOT = Path(__file__).resolve().parents[1]


def test_ag_two_stage_pipeline_runs_end_to_end() -> None:
    _, intensity, _ = load_demo("Clear Demo")
    classifier = load_ag_classifier()
    classification = classify_ag(intensity, classifier)
    assert isinstance(classification["positive"], (bool, np.bool_))
    assert 0.0 <= classification["probability"] <= 1.0
    model, checkpoint = load_ag_concentration()
    estimate = predict_ag_concentration(intensity, model, checkpoint)
    assert estimate >= 0.0
    assert np.isfinite(estimate)


def test_ag_classifier_checkpoint_missing_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ag_classifier(tmp_path / "missing.joblib")


def test_cu_concentration_pipeline_runs_and_is_not_clipped_to_supported_range() -> None:
    _, intensity, _ = load_demo("Clear Demo")
    model, checkpoint = load_cu_concentration()
    estimate = predict_cu_concentration(intensity, model, checkpoint)
    assert estimate >= 0.0
    assert np.isfinite(estimate)
    assert SUPPORTED_RANGE_PPM == (1.0, 500.0)


def test_cu_concentration_checkpoint_missing_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cu_concentration(tmp_path / "missing.pt")
