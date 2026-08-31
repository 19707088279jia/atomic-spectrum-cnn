"""Generate reference spectrum plots for mixture elements.

Reads CSVs from data/reference_spectra/{Element}/{Element}_reference.csv and
writes PNG files to data/reference_plots/. Also writes a combined overlay plot.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from atomic_spectrum_ai.reference_spectra import load_all_references

OUTDIR = Path(__file__).resolve().parents[1] / "data" / "reference_plots"
OUTDIR.mkdir(parents=True, exist_ok=True)


def normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    if intensity.max() == 0:
        return intensity
    return intensity / np.max(intensity)


def plot_reference(element: str, df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    wl = df['wavelength_nm'].values
    inten = df['intensity'].values
    inten = normalize_intensity(inten)
    ax.plot(wl, inten, lw=1)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Normalized intensity')
    ax.set_title(f'{element} reference')
    ax.set_xlim(200, 600)
    ax.grid(True, alpha=0.3)
    return ax


def main():
    refs = load_all_references()
    # individual plots
    for el, df in refs.items():
        fig, ax = plt.subplots(figsize=(8, 3))
        if df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center')
        else:
            plot_reference(el, df, ax=ax)
        out = OUTDIR / f"{el}_reference.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)

    # combined plot
    fig, ax = plt.subplots(figsize=(10, 4))
    for el, df in refs.items():
        if df.empty:
            continue
        wl = df['wavelength_nm'].values
        inten = df['intensity'].values
        inten = normalize_intensity(inten)
        ax.plot(wl, inten, label=el, lw=1)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Normalized intensity')
    ax.set_title('Combined reference spectra')
    ax.set_xlim(200, 600)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'combined_reference.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
