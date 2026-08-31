from pathlib import Path

import torch

from atomic_spectrum_ai.models.z903_cnn import Z903CNN
from atomic_spectrum_ai.z903_inference import Z903_THRESHOLDS, load_z903_cnn

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_checkpoint_has_expected_numerical_contract() -> None:
    model, checkpoint, device = load_z903_cnn(ROOT / "models" / "z903_cnn_robust_best.pt")
    assert isinstance(model, Z903CNN)
    assert device == torch.device("cpu")
    assert checkpoint["preprocess"] == "robust"
    assert tuple(checkpoint["targets"]) == ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb")
    assert Z903_THRESHOLDS["Zn"] == 0.7050


def test_probability_thresholds_are_frozen_six_target_values() -> None:
    assert tuple(Z903_THRESHOLDS) == ("Zn", "Mn", "Cd", "Mg", "Cu", "Pb")
    assert Z903_THRESHOLDS == {
        "Zn": 0.7050,
        "Mn": 0.2350,
        "Cd": 0.5000,
        "Mg": 0.3550,
        "Cu": 0.5050,
        "Pb": 0.4050,
    }