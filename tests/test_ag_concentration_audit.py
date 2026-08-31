from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_ag_concentration import load_frozen_splits, recommendation  # noqa: E402


def write_split(path: Path, target_id: str, group_id: str) -> None:
    pd.DataFrame({"target_id": [target_id], "group_id": [group_id]}).to_csv(path, index=False)


def test_load_frozen_splits_preserves_missing_ag_values(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {"PELLET NAME": ["target-a", "target-b", "target-c"], "Ag": [0.5, np.nan, 1.0]}
    )
    metadata["_sample_key"] = metadata["PELLET NAME"].str.replace("-", "", regex=False)
    write_split(tmp_path / "z903_train.csv", "target-a", "group-a")
    write_split(tmp_path / "z903_val.csv", "target-b", "group-b")
    write_split(tmp_path / "z903_test.csv", "target-c", "group-c")

    frame = load_frozen_splits(tmp_path, metadata, "Ag")

    assert frame.loc[frame["target_id"] == "target-b", "ag_ppm"].isna().all()
    assert frame["ag_ppm"].notna().sum() == 2


def test_load_frozen_splits_rejects_cross_split_group_leakage(tmp_path: Path) -> None:
    metadata = pd.DataFrame({"PELLET NAME": ["target-a", "target-b"], "Ag": [0.0, 1.0]})
    metadata["_sample_key"] = metadata["PELLET NAME"].str.replace("-", "", regex=False)
    write_split(tmp_path / "z903_train.csv", "target-a", "group-a")
    write_split(tmp_path / "z903_val.csv", "target-b", "group-a")
    write_split(tmp_path / "z903_test.csv", "target-b", "group-b")

    with pytest.raises(ValueError, match="leakage"):
        load_frozen_splits(tmp_path, metadata, "Ag")


def test_recommendation_selects_two_stage_for_skewed_zero_inflated_labels() -> None:
    values = pd.Series([0.0] * 64 + [0.01] * 100 + [0.1] * 100 + [14.0] * 40)

    strategy, _, severe_tail = recommendation(values)

    assert strategy == "B. Two-stage model"
    assert severe_tail