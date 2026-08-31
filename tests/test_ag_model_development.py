from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from develop_ag_models import average_pool, banded_metrics, validation_threshold  # noqa: E402


def test_average_pool_reduces_z903_spectrum_to_936_features() -> None:
    values = np.arange(23401, dtype=np.float32)

    pooled = average_pool(values)

    assert pooled.shape == (936,)
    assert pooled[0] == 12


def test_validation_threshold_optimizes_balanced_accuracy() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    threshold = validation_threshold(y_true, probabilities)

    assert threshold == 0.6


def test_banded_metrics_reports_insufficient_sparse_validation_support() -> None:
    truth = np.array([0.01, 0.02, 0.03, 0.04])
    prediction = np.array([0.01, 0.02, 0.03, 0.04])

    metrics = banded_metrics(truth, prediction)

    assert metrics["0-0.1 ppm"] == {"n": 4, "status": "Insufficient validation support"}