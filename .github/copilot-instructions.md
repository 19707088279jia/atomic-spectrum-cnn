# Repository-wide GitHub Copilot instructions

## Purpose

This is a Python 3.11/PyTorch computer-vision project for classifying atomic emission spectrum plot images. The initial task is single-label classification among Fe, Cu, Na, Ca, and Mg. It includes synthetic data only to verify the software pipeline; synthetic performance is not scientific validation.

## Architecture

- `src/atomic_spectrum_ai/`: reusable package code.
- `scripts/`: thin CLI entry points.
- `configs/`: YAML experiment configurations.
- `app/`: Streamlit inference UI.
- `tests/`: pytest tests.
- `outputs/`: generated artifacts; never commit.
- `data/`: generated or real data; never commit except empty `.gitkeep` files.

## Build and validation

Use Python 3.11. From the repository root:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
ruff check .
```

Smoke pipeline:

```bash
python scripts/generate_synthetic_data.py --config configs/quickstart.yaml --samples-per-class 4
python scripts/train.py --config configs/smoke_test.yaml
```

## Coding rules

- Use `pathlib.Path`, type hints, docstrings, and explicit exceptions.
- Keep scripts small; put logic in the package.
- Make randomness reproducible through a seed.
- Use configuration values rather than hidden constants.
- Load checkpoints with `map_location`.
- Save `state_dict`, model name, class names, and image size in checkpoints.
- Do not add horizontal image flipping because x-position represents wavelength.
- Avoid aggressive random cropping that removes diagnostic wavelength regions.
- Never render element names, labels, titles, or legends into training images.
- Never let train/test contain augmentations derived from the same original spectrum.
- Add or update tests with every behavior change.
- Do not use a test skip to hide a failure.
- Do not commit model weights or datasets.

## Scientific communication

Separate software functionality from scientific validity. Phrase conclusions as image-classification results unless independent real spectra and scientifically valid splits have been used.
