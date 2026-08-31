import pandas as pd
import pytest

from atomic_spectrum_ai import reference_spectra


def test_elements_list():
    els = reference_spectra.MIXTURE_ELEMENTS
    assert isinstance(els, list)
    assert len(els) == 6


@pytest.mark.parametrize('el', reference_spectra.MIXTURE_ELEMENTS)
def test_load_reference_exists_or_empty(el):
    df = reference_spectra.load_reference_spectrum(el)
    assert isinstance(df, pd.DataFrame)
    # must have required columns (may be empty until real data added)
    assert 'wavelength_nm' in df.columns
    assert 'intensity' in df.columns
