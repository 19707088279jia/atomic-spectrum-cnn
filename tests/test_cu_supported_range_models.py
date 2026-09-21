from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from develop_cu_supported_range_models import (  # noqa: E402
    EXISTING_CU_DETECTION_THRESHOLD_PPM,
    HIGH_PPM,
    LOW_PPM,
    apply_target_transform,
    filter_supported_range,
    invert_target_transform,
    load_train_validation_cu,
    out_of_range_counts,
    predict_cnn,
    range_counts,
    select_final_model,
    train_one_cnn,
    warning_classifier_feasible,
)

from atomic_spectrum_ai.models.z903_cnn import Z903CNN  # noqa: E402


def test_existing_threshold_is_unchanged() -> None:
    assert EXISTING_CU_DETECTION_THRESHOLD_PPM == 25.0


def test_filter_supported_range_uses_inclusive_1_to_500_ppm() -> None:
    frame = pd.DataFrame({"cu_ppm": [0.0, 0.5, 1.0, 250.0, 500.0, 500.01, np.nan, 900.0]})

    filtered = filter_supported_range(frame)

    assert sorted(filtered["cu_ppm"].tolist()) == [1.0, 250.0, 500.0]


def test_filter_supported_range_never_treats_missing_as_zero() -> None:
    frame = pd.DataFrame({"cu_ppm": [np.nan, np.nan, 5.0]})

    filtered = filter_supported_range(frame)

    assert filtered["cu_ppm"].tolist() == [5.0]
    assert not (filtered["cu_ppm"] == 0).any()


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_train_validation_cu_never_reads_test_split(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {
            "PELLET NAME": ["UNITS", "target-a", "target-b"],
            "Zn": ["ppm", 1.0, 2.0],
            "Mn": ["ppm", 1.0, 2.0],
            "Cd": ["ppm", 1.0, 2.0],
            "Mg": ["wt %", 1.0, 2.0],
            "Cu": ["ppm", 5.0, None],
            "Pb": ["ppm", 1.0, 2.0],
        }
    )
    write_split(tmp_path / "z903_train.csv", [{"target_id": "target-a", "group_id": "group-a", "spectrum_path": "a.csv"}])
    write_split(tmp_path / "z903_val.csv", [{"target_id": "target-b", "group_id": "group-b", "spectrum_path": "b.csv"}])
    # Deliberately do not create z903_test.csv; loading must succeed without it.

    metadata_path = tmp_path / "metadata.xlsx"
    metadata.to_excel(metadata_path, index=False, sheet_name="Sheet1")

    frame, cu_column = load_train_validation_cu(tmp_path, metadata_path)

    assert cu_column == "Cu"
    assert set(frame["split"]) == {"train", "val"}
    assert frame.loc[frame["target_id"] == "target-b", "cu_ppm"].isna().all()


def test_range_counts_first_range_is_inclusive_on_both_ends() -> None:
    values = pd.Series([1.0, 5.0, 10.0, 10.01, 25.0])
    ranges = (("1-10 ppm", 1.0, 10.0), (">10-25 ppm", 10.0, 25.0))

    counts = range_counts(values, ranges)

    assert counts["1-10 ppm"] == 3
    assert counts[">10-25 ppm"] == 2


def test_target_transform_round_trip() -> None:
    values = np.array([1.0, 31.0, 500.0])

    for mode in ("log1p", "raw"):
        transformed = apply_target_transform(values, mode)
        recovered = invert_target_transform(transformed, mode)
        np.testing.assert_allclose(recovered, values, atol=1e-8)


def test_invert_target_transform_clips_to_nonnegative_ppm() -> None:
    assert invert_target_transform(np.array([-5.0]), "raw")[0] == 0.0
    assert invert_target_transform(np.array([-5.0]), "log1p")[0] >= 0.0


def test_select_final_model_prefers_positive_pearson_and_lowest_mae() -> None:
    candidates = {
        "collapsed": {"mae": 5.0, "r2": -1.0, "pearson": -0.2, "spearman": 0.1, "n": 50},
        "good": {"mae": 10.0, "r2": 0.4, "pearson": 0.6, "spearman": 0.7, "n": 50},
    }

    winner, reason = select_final_model(candidates)

    assert winner == "good"
    assert "good" in reason


def test_warning_classifier_feasible_matches_observed_cu_counts() -> None:
    # Actual Phase 1 audit counts: train >500 ppm = 18, validation >500 ppm = 1.
    assert warning_classifier_feasible(train_over=18, val_over=1) is False
    assert warning_classifier_feasible(train_over=18, val_over=6) is True


def test_out_of_range_counts_excludes_missing_and_low_range() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "train", "val"],
            "cu_ppm": [600.0, 100.0, np.nan, 700.0],
        }
    )

    counts = out_of_range_counts(frame)

    assert counts["train_over_500"] == 1
    assert counts["val_over_500"] == 1


def test_cnn_prediction_is_deterministic_in_eval_mode() -> None:
    torch.manual_seed(0)
    x_train = np.random.default_rng(0).normal(size=(6, 23401)).astype(np.float32)
    y_train = np.array([1.0, 5.0, 20.0, 50.0, 100.0, 300.0])
    x_val = np.random.default_rng(1).normal(size=(3, 23401)).astype(np.float32)
    y_val = np.array([10.0, 40.0, 120.0])

    model, _, _, mean, scale = train_one_cnn(x_train, y_train, x_val, y_val, "log1p", epochs=1)

    first = predict_cnn(model, x_val, "log1p", mean, scale)
    second = predict_cnn(model, x_val, "log1p", mean, scale)

    np.testing.assert_array_equal(first, second)
    assert (first >= 0.0).all()


def test_cnn_checkpoint_serialization_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    x_train = np.random.default_rng(2).normal(size=(6, 23401)).astype(np.float32)
    y_train = np.array([1.0, 5.0, 20.0, 50.0, 100.0, 300.0])
    x_val = np.random.default_rng(3).normal(size=(2, 23401)).astype(np.float32)
    y_val = np.array([10.0, 120.0])

    model, _, _, mean, scale = train_one_cnn(x_train, y_train, x_val, y_val, "raw", epochs=1)
    before = predict_cnn(model, x_val, "raw", mean, scale)

    checkpoint_path = tmp_path / "cu_supported_range_final.pt"
    torch.save({"model_state_dict": model.state_dict(), "target_mean": mean, "target_scale": scale}, checkpoint_path)

    reloaded = Z903CNN(output_size=1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    after = predict_cnn(reloaded, x_val, "raw", checkpoint["target_mean"], checkpoint["target_scale"])

    np.testing.assert_allclose(before, after, atol=1e-6)


def test_low_and_high_ppm_bounds_match_phase1_recommendation() -> None:
    assert (LOW_PPM, HIGH_PPM) == (1.0, 500.0)
