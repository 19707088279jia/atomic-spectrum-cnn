from pathlib import Path

import numpy as np
import torch
from src.concentration_inference import load_concentration_checkpoint, predict_concentrations
from src.data import load_demo, load_metadata, load_regression_split
from src.labels import TARGETS, UNITS
from src.mgo_ridge_inference import (
    FEATURE_COUNT,
    TARGET_LABEL,
    TARGET_UNIT,
    bin_spectrum,
    build_display,
    load_pipeline,
    predict_mgo,
)
from src.model import ConcentrationCNN
from src.preprocessing import robust_scale
from src.regression import TargetTransform, masked_huber_loss

ROOT = Path(__file__).resolve().parents[1]


def test_metadata_concentrations_and_units() -> None:
    metadata = load_metadata()
    row = metadata.loc["agv1a"]
    assert row["Zn"] == 88
    assert row["MgO"] == 1.49
    assert UNITS == {"Zn": "ppm", "Mn": "ppm", "Cd": "ppm", "Mg": "oxide wt%", "Cu": "ppm", "Pb": "ppm"}


def test_split_masks_preserve_missing_values() -> None:
    train = load_regression_split("train")
    assert set(TARGETS).issubset(train.columns)
    assert train["Cd"].isna().any()
    assert train["Cu"].isna().any()


def test_train_only_log_transform_standardization_and_inverse() -> None:
    values = np.asarray([[0.0, 10.0, np.nan, 4.0, 2.0, 1.0], [10.0, 20.0, 0.05, 8.0, 4.0, 2.0]], dtype=np.float32)
    mask = np.isfinite(values).astype(np.float32)
    transform = TargetTransform.fit(values, mask)
    transformed = transform.transform(np.nan_to_num(values, nan=0.0))
    restored = transform.inverse(transformed)
    assert np.allclose(restored[mask.astype(bool)], values[mask.astype(bool)], atol=1e-5)
    expected_means = np.array(
        [np.log1p(values[np.isfinite(values[:, index]), index]).mean() for index in range(6)]
    )
    assert np.allclose(transform.means, expected_means)


def test_regression_model_shape_and_masked_loss_is_finite() -> None:
    model = ConcentrationCNN()
    inputs = torch.zeros((2, 1, 23401))
    outputs = model(inputs)
    assert tuple(outputs.shape) == (2, 6)
    labels = torch.zeros_like(outputs)
    mask = torch.tensor([[1, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    assert torch.isfinite(masked_huber_loss(outputs, labels, mask))


def test_regression_checkpoint_and_units_are_not_detection_probabilities() -> None:
    model, transform, checkpoint = load_concentration_checkpoint()
    _, intensity, _ = load_demo("Clear Demo")
    estimates = predict_concentrations(intensity, model, transform)
    assert checkpoint["model_name"] == "ConcentrationCNN"
    assert tuple(estimates) == TARGETS
    assert all(value >= 0.0 for value in estimates.values())
    app_text = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Predicted MgO" in app_text
    assert "Probability" not in app_text


def test_frozen_mgo_ridge_pipeline_and_prediction_contract() -> None:
    pipeline = load_pipeline()
    _, intensity, _ = load_demo("Clear Demo")
    features = bin_spectrum(robust_scale(intensity))
    prediction = predict_mgo(intensity, pipeline)
    assert len(features) == FEATURE_COUNT == 936
    assert np.isfinite(prediction)
    assert TARGET_LABEL == "MgO"
    assert TARGET_UNIT == "wt%"
    assert pipeline["alpha"] == 100.0
    assert pipeline["expected_points"] == 23401


def test_mgo_ui_distinguishes_manual_and_demo_ground_truth() -> None:
    app_text = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Ground Truth MgO" in app_text
    assert "truth_row is not None" in app_text
    assert "Predicted MgO" in app_text
    assert "wt%" in app_text
    assert "probability" not in app_text.lower().split("MgO Quantitative Estimate", 1)[-1]
    manual = build_display(6.85)
    demo = build_display(6.85, 6.6)
    assert "Ground Truth MgO" not in manual
    assert "Absolute Error" not in manual
    assert demo["Ground Truth MgO"] == 6.6
    assert demo["Absolute Error"] == 0.25