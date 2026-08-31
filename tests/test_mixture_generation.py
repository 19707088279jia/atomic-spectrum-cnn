import pandas as pd

from atomic_spectrum_ai import reference_spectra
from scripts.generate_mixture_data import generate_samples


def test_generate_small_manifest(tmp_path):
    out = tmp_path / "data"
    out.mkdir()
    manifest_path = generate_samples(out, n_samples=5, seed=0, split="test")
    assert manifest_path.exists()
    df = pd.read_csv(manifest_path)
    assert len(df) == 5
    # label columns
    for c in reference_spectra.MIXTURE_ELEMENTS:
        assert c in df.columns
