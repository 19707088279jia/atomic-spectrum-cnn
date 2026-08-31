from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2] / "data" / "reference_spectra"

MIXTURE_ELEMENTS = [
    "Zn",
    "Mn",
    "Cd",
    "Mg",
    "Cu",
    "Pb",
]


def load_reference_spectrum(element: str) -> pd.DataFrame:
    element = element.strip()
    if element not in MIXTURE_ELEMENTS:
        raise ValueError(f"Unknown element: {element}")
    p = BASE / element / f"{element}_reference.csv"
    if not p.exists():
        raise FileNotFoundError(f"Reference file not found: {p}")
    df = pd.read_csv(p)
    if 'wavelength_nm' not in df.columns or 'intensity' not in df.columns:
        raise ValueError(f"Reference file {p} missing required columns")
    df = df.sort_values('wavelength_nm').reset_index(drop=True)
    return df


def load_all_references() -> dict[str, pd.DataFrame]:
    refs = {}
    for el in MIXTURE_ELEMENTS:
        try:
            refs[el] = load_reference_spectrum(el)
        except FileNotFoundError:
            refs[el] = pd.DataFrame(columns=['wavelength_nm', 'intensity'])
    return refs


__all__ = ["MIXTURE_ELEMENTS", "load_reference_spectrum", "load_all_references"]
