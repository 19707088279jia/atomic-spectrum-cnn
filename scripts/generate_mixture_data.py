"""Generate synthetic mixture spectra from reference spectra.

Saves PNG images and a CSV manifest with multi-hot labels.
"""
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from atomic_spectrum_ai.reference_spectra import MIXTURE_ELEMENTS, load_all_references


def default_wavelength_grid(wmin=200.0, wmax=600.0, npoints=2000):
    return np.linspace(wmin, wmax, npoints)


def interpolate_spectrum(df, grid, shift_nm=0.0):
    if df.empty:
        return np.zeros_like(grid)
    wl = df['wavelength_nm'].values + shift_nm
    inten = df['intensity'].values
    # normalize per-element before combining
    if inten.max() != 0:
        inten = inten / np.max(inten)
    return np.interp(grid, wl, inten, left=0.0, right=0.0)


def add_baseline(spectrum, scale=0.02):
    n = len(spectrum)
    x = np.linspace(-1, 1, n)
    baseline = scale * (0.5 * x + 0.25 * x**2)
    return spectrum + baseline


def broaden(spectrum, sigma_px=2.0):
    # simple Gaussian kernel convolution
    n = int(max(3, sigma_px * 6))
    x = np.linspace(-3 * sigma_px, 3 * sigma_px, n)
    kernel = np.exp(-0.5 * (x / sigma_px) ** 2)
    kernel = kernel / kernel.sum()
    return np.convolve(spectrum, kernel, mode='same')


def generate_samples(out_dir: Path, n_samples: int = 100, seed: int = 42, split: str = 'train'):
    random.seed(seed)
    np.random.seed(seed)
    refs = load_all_references()
    grid = default_wavelength_grid()
    out_dir = Path(out_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n_samples):
        k = random.randint(1, 4)
        elems = random.sample(MIXTURE_ELEMENTS, k)
        weights = np.random.uniform(0.3, 1.0, size=k)
        spectrum = np.zeros_like(grid)
        for el, w in zip(elems, weights, strict=True):
            shift = np.random.normal(0.0, 0.05)  # small wavelength shift in nm
            spec_el = interpolate_spectrum(refs[el], grid, shift_nm=shift)
            spectrum += w * spec_el
        # optional broadening
        spectrum = broaden(spectrum, sigma_px=np.random.uniform(1.0, 3.0))
        # baseline and noise
        spectrum = add_baseline(spectrum, scale=np.random.uniform(0.0, 0.03))
        noise_level = np.random.uniform(0.0, 0.02)
        spectrum += np.random.normal(0.0, noise_level, size=spectrum.shape)
        # normalize final
        if spectrum.max() > 0:
            spectrum = spectrum / np.max(np.abs(spectrum))
        # save PNG
        fname = f"mix_{i:06d}.png"
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(grid, spectrum, lw=1)
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(-0.05, 1.05)
        ax.axis('off')
        fig.savefig(out_dir / fname, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        # save numeric spectrum as CSV for peak-matching/testing
        csv_name = f"mix_{i:06d}.csv"
        pd.DataFrame({'wavelength_nm': grid, 'intensity': spectrum}).to_csv(out_dir / csv_name, index=False)
        # labels
        label = {el: (1 if el in elems else 0) for el in MIXTURE_ELEMENTS}
        record = {'file': f"{split}/{fname}", 'spectrum_csv': f"{split}/{csv_name}"}
        record.update(label)
        records.append(record)
    manifest = pd.DataFrame.from_records(records)
    manifest_path = out_dir.parent / f"{split}_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/mixture', help='output base dir')
    p.add_argument('--n', type=int, default=200, help='number of samples')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split', default='train')
    args = p.parse_args()
    out = Path(args.out)
    manifest = generate_samples(out, n_samples=args.n, seed=args.seed, split=args.split)
    print('Wrote manifest:', manifest)


if __name__ == '__main__':
    main()
