from pathlib import Path

import numpy as np
from PIL import Image

from atomic_spectrum_ai.synthetic import generate_dataset, make_spectrum


def test_make_spectrum_is_normalized() -> None:
    rng = np.random.default_rng(1)
    x, y = make_spectrum("Fe", 200.0, 800.0, 400, rng)
    assert x.shape == y.shape == (400,)
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0


def test_generate_small_dataset(tmp_path: Path) -> None:
    created = generate_dataset(
        root=tmp_path,
        classes=["Fe", "Cu"],
        samples_per_split={"train": 1, "validation": 1, "test": 1},
        wavelength_min=200.0,
        wavelength_max=800.0,
        points=200,
        dpi=50,
        seed=1,
    )
    assert len(created) == 6
    with Image.open(created[0]) as image:
        assert image.width > 0
        assert image.height > 0
