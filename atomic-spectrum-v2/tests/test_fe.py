import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_demo  # noqa: E402
from src.fe_inference import (  # noqa: E402
    FE_VALIDATION_RANGE_WT_PERCENT,
    load_fe_checkpoint,
    predict_fe_concentration,
)
from train_fe import audit_data, load_fe_splits, regression_metrics  # noqa: E402


def test_fe_audit_keeps_missing_references_out_of_classification() -> None:
    audit = audit_data(load_fe_splits())

    assert audit["overall"]["total_spectra"] == 2567
    assert audit["overall"]["known_fe_values"] == 2556
    assert audit["overall"]["missing_fe_values"] == 11
    assert audit["overall"]["zero_values"] == 0
    assert audit["overall"]["positive_values"] == 2556
    assert audit["overall"]["classification_feasible_at_zero"] is False
    assert [audit["splits"][name]["known"] for name in ("train", "val", "test")] == [
        1794,
        376,
        386,
    ]


def test_regression_metrics_are_in_original_units() -> None:
    metrics = regression_metrics(np.asarray([1.0, 2.0, 3.0]), np.asarray([2.0, 2.0, 4.0]))

    assert metrics["n"] == 3
    assert metrics["mae"] == pytest.approx(2.0 / 3.0)
    assert metrics["median_absolute_error"] == pytest.approx(1.0)
    assert metrics["bias"] == pytest.approx(2.0 / 3.0)


def test_fe_checkpoint_runs_full_spectrum_inference() -> None:
    model, checkpoint, device = load_fe_checkpoint()
    _, intensity, _ = load_demo("Clear Demo")

    estimate = predict_fe_concentration(intensity, model, checkpoint, device)

    assert checkpoint["target"] == "Fe2O3*"
    assert checkpoint["preprocess"] == "robust"
    assert FE_VALIDATION_RANGE_WT_PERCENT == (0.12, 17.4)
    assert np.isfinite(estimate)
    assert estimate >= 0.0
    with pytest.raises(ValueError, match="Expected 23401 intensity values"):
        predict_fe_concentration(np.zeros(10, dtype=np.float32), model, checkpoint, device)


def test_streamlit_app_displays_fe_as_experimental_regression() -> None:
    app_text = (PROJECT_ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")

    assert "render_fe_section(intensity)" in app_text
    assert "not a presence/absence classification" in app_text
    assert "validation-supported range" in app_text
