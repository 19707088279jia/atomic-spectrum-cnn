from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from atomic_spectrum_ai.z903 import (
    WAVELENGTHS,
    Z903Dataset,
    masked_binary_cross_entropy,
    preprocess_spectrum,
)


def test_preprocessing_options_are_finite_and_shape_preserving() -> None:
    intensity = np.linspace(0.0, 10.0, WAVELENGTHS.size, dtype=np.float32)
    intensity[10] = np.nan
    intensity[20] = np.inf

    area = preprocess_spectrum(intensity, option="area")
    robust = preprocess_spectrum(intensity, option="robust")

    assert area.shape == (23401,)
    assert robust.shape == (23401,)
    assert np.isfinite(area).all()
    assert np.isfinite(robust).all()


def test_masked_binary_cross_entropy_ignores_unknown_targets() -> None:
    logits = torch.zeros((1, 2))
    labels = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])

    loss = masked_binary_cross_entropy(logits, labels, mask)

    assert torch.isclose(loss, torch.tensor(0.69314718), atol=1e-5)


def test_z903_dataset_returns_one_d_input_labels_and_masks(tmp_path) -> None:
    spectrum_path = tmp_path / "plibs_z903_demo.csv"
    pd.DataFrame({"wavelength": WAVELENGTHS, "intensity": np.ones(WAVELENGTHS.size)}).to_csv(spectrum_path, index=False)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "spectrum_path": str(spectrum_path),
                "Zn_raw": 10.0,
                "Mn_raw": np.nan,
                "Cd_raw": 0.1,
                "MgO_raw": 2.0,
                "Cu_raw": np.nan,
                "Pb_raw": 5.0,
                "Zn_mask": 1,
                "Mn_mask": 0,
                "Cd_mask": 1,
                "Mg_mask": 1,
                "Cu_mask": 0,
                "Pb_mask": 1,
            }
        ]
    ).to_csv(manifest_path, index=False)

    spectrum, labels, mask = Z903Dataset(manifest_path)[0]

    assert spectrum.shape == (1, 23401)
    assert labels.shape == (6,)
    assert mask.shape == (6,)
    assert labels[1].item() == 0.0
    assert mask[1].item() == 0.0