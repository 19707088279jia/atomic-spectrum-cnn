from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_cu_concentration import (  # noqa: E402
    EXISTING_CU_DETECTION_THRESHOLD_PPM,
    candidate_supported_range,
    fixed_range_counts,
    load_frozen_splits,
    recommend_strategy,
)


def write_split(path: Path, target_id: str, group_id: str) -> None:
    pd.DataFrame({"target_id": [target_id], "group_id": [group_id]}).to_csv(path, index=False)


def test_existing_threshold_is_unchanged() -> None:
    assert EXISTING_CU_DETECTION_THRESHOLD_PPM == 25.0


def test_load_frozen_splits_preserves_missing_cu_values(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {"PELLET NAME": ["target-a", "target-b", "target-c"], "Cu": [5.0, np.nan, 30.0]}
    )
    metadata["_sample_key"] = metadata["PELLET NAME"].str.replace("-", "", regex=False)
    write_split(tmp_path / "z903_train.csv", "target-a", "group-a")
    write_split(tmp_path / "z903_val.csv", "target-b", "group-b")
    write_split(tmp_path / "z903_test.csv", "target-c", "group-c")

    frame = load_frozen_splits(tmp_path, metadata, "Cu")

    assert frame.loc[frame["target_id"] == "target-b", "cu_ppm"].isna().all()
    assert frame["cu_ppm"].notna().sum() == 2


def test_load_frozen_splits_rejects_cross_split_group_leakage(tmp_path: Path) -> None:
    metadata = pd.DataFrame({"PELLET NAME": ["target-a", "target-b"], "Cu": [0.0, 30.0]})
    metadata["_sample_key"] = metadata["PELLET NAME"].str.replace("-", "", regex=False)
    write_split(tmp_path / "z903_train.csv", "target-a", "group-a")
    write_split(tmp_path / "z903_val.csv", "target-b", "group-a")
    write_split(tmp_path / "z903_test.csv", "target-b", "group-b")

    with pytest.raises(ValueError, match="leakage"):
        load_frozen_splits(tmp_path, metadata, "Cu")


def test_fixed_range_counts_uses_part_c_bins() -> None:
    values = pd.Series([0.0, 0.5, 5.0, 20.0, 50.0, 300.0, 800.0, 2000.0, 9000.0])

    counts = fixed_range_counts(values)

    assert counts["0 ppm"] == 1
    assert counts[">0-1 ppm"] == 1
    assert counts[">1-10 ppm"] == 1
    assert counts[">10-25 ppm"] == 1
    assert counts[">25-100 ppm"] == 1
    assert counts[">100-500 ppm"] == 1
    assert counts[">500-1000 ppm"] == 1
    assert counts[">1000-5000 ppm"] == 1
    assert counts[">5000 ppm"] == 1


def test_recommend_strategy_flags_insufficient_data_when_labels_are_sparse() -> None:
    known = pd.Series([0.0] * 20 + [5.0] * 10)

    strategy, _ = recommend_strategy(known, {}, None)

    assert strategy.startswith("D.")


def test_candidate_supported_range_excludes_test_split() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train"] * 15 + ["val"] * 8 + ["test"] * 100,
            "cu_ppm": [0.0] * 15 + [0.0] * 8 + [9000.0] * 100,
        }
    )

    supported_range, _ = candidate_supported_range(frame, min_train=10, min_val=5)

    assert supported_range == "0 ppm to 0 ppm"


def test_candidate_supported_range_picks_longest_contiguous_run() -> None:
    frame = pd.DataFrame(
        {
            "split": (
                ["train"] * 3 + ["train"] * 12 + ["train"] * 12
                + ["val"] * 2 + ["val"] * 6 + ["val"] * 6
            ),
            "cu_ppm": (
                [0.0] * 3 + [5.0] * 12 + [15.0] * 12
                + [0.0] * 2 + [5.0] * 6 + [15.0] * 6
            ),
        }
    )

    supported_range, _ = candidate_supported_range(frame, min_train=10, min_val=5)

    assert supported_range == "1 ppm to 25 ppm"
