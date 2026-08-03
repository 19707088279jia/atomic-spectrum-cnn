from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from atomic_spectrum_ai.nist_import import (
    assign_split,
    generate_manifest,
    import_nist_dataset,
    read_nist_spectrum,
    render_spectrum_image,
)


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def test_read_valid_nist_csv(tmp_path: Path) -> None:
    path = tmp_path / "Fe_001.csv"
    write_csv(
        path,
        [
            [" Wavelength (nm) ", " Sum ", "Fe I"],
            ["200.0", "1.0", ""],
            ["200.1", "2.0", "1.0"],
            ["200.2", "", "2.0"],
            ["200.3", "3.0", "3.0"],
            ["600.0", "6.0", "6.0"],
        ],
    )

    wavelengths, intensities = read_nist_spectrum(path, element="Fe")

    assert wavelengths[0] == 200.0
    assert wavelengths[-1] == 600.0
    assert intensities.shape == wavelengths.shape
    assert np.isfinite(intensities).all()


def test_header_detection(tmp_path: Path) -> None:
    path = tmp_path / "Cu_002.csv"
    write_csv(
        path,
        [
            ["  wavelength (nm)  ", "   sum   "],
            ["300.0", "1.0"],
            ["301.0", "2.0"],
        ],
    )

    wavelengths, intensities = read_nist_spectrum(path, element="Cu", grid_points=2)

    assert wavelengths.shape[0] == 2
    assert intensities.shape[0] == 2


def test_intensity_normalization(tmp_path: Path) -> None:
    path = tmp_path / "Na_003.csv"
    write_csv(
        path,
        [
            ["Wavelength (nm)", "Sum"],
            ["200.0", "10"],
            ["201.0", "20"],
            ["202.0", "30"],
        ],
    )

    _, intensities = read_nist_spectrum(path, element="Na")

    assert intensities.min() >= 0.0
    assert intensities.max() <= 1.0


def test_wavelength_interpolation(tmp_path: Path) -> None:
    path = tmp_path / "Ca_001.csv"
    write_csv(
        path,
        [
            ["Wavelength (nm)", "Sum"],
            ["200.0", "1.0"],
            ["400.0", "2.0"],
            ["600.0", "3.0"],
        ],
    )

    wavelengths, intensities = read_nist_spectrum(path, element="Ca", grid_points=5)

    assert wavelengths.shape == (5,)
    assert intensities.shape == (5,)
    assert wavelengths[0] == 200.0
    assert wavelengths[-1] == 600.0


def test_invalid_csv_rejection(tmp_path: Path) -> None:
    path = tmp_path / "Mg_001.csv"
    write_csv(
        path,
        [
            ["Wavelength", "Sum"],
            ["200.0", "1.0"],
            ["201.0", "2.0"],
        ],
    )

    try:
        read_nist_spectrum(path, element="Mg")
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid CSV to raise")


def test_source_level_split_assignment() -> None:
    assert assign_split("Fe_001.csv") == "train"
    assert assign_split("Cu_002.csv") == "validation"
    assert assign_split("Na_003.csv") == "test"


def test_manifest_generation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    rows = generate_manifest(manifest_path, [
        {
            "element": "Fe",
            "source_file": "Fe/Fe_001.csv",
            "output_file": "train/Fe/Fe_001_0000.png",
            "split": "train",
            "source_index": 1,
            "Te_eV": 0.7,
            "Ne_cm3": 1e17,
            "resolution": 500,
            "augmentation_seed": 123,
        }
    ])

    assert len(rows) == 1
    assert manifest_path.exists()
    assert rows[0]["output_file"].endswith(".png")


def test_horizontal_flipping_not_used(tmp_path: Path) -> None:
    image_path = tmp_path / "flipping_check.png"
    wavelengths = np.linspace(200.0, 600.0, 20)
    intensities = np.zeros_like(wavelengths)
    intensities[10] = 1.0

    try:
        render_spectrum_image(wavelengths, intensities, image_path, dpi=100)
    except AssertionError as exc:
        raise AssertionError("horizontal flipping was used") from exc

    assert image_path.exists()
    with Image.open(image_path) as image:
        assert image.size[0] > 0
        assert image.size[1] > 0


def test_import_nist_dataset_creates_images(tmp_path: Path) -> None:
    input_dir = tmp_path / "nist_raw"
    output_dir = tmp_path / "nist_images"
    source_dir = input_dir / "Fe"
    source_dir.mkdir(parents=True)
    for name in ["Fe_001.csv", "Fe_002.csv", "Fe_003.csv"]:
        write_csv(
            source_dir / name,
            [
                ["Wavelength (nm)", "Sum"],
                ["200.0", "1.0"],
                ["400.0", "2.0"],
                ["600.0", "3.0"],
            ],
        )

    manifest_rows = import_nist_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        copies_per_source=2,
        seed=7,
        classes=["Fe"],
    )

    assert len(manifest_rows) == 6
    assert (output_dir / "train" / "Fe").exists()
    assert (output_dir / "validation" / "Fe").exists()
    assert (output_dir / "test" / "Fe").exists()
    assert (output_dir / "manifest.csv").exists()
