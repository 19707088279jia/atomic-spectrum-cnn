# Agent Instructions

This repository builds a scientifically cautious CNN classifier for atomic spectrum images.

## Non-negotiable rules

- Run tests before and after changes.
- Never state that training or tests succeeded unless the command was actually executed.
- Preserve wavelength orientation; never use horizontal flip augmentation.
- Do not infer labels from filenames, plot titles, legends, metadata, or directory strings inside model inputs.
- Keep synthetic-data results clearly separated from real-data scientific conclusions.
- Maintain Python 3.11 compatibility.
- Use type hints and actionable error messages.
- Keep CLI behavior backward compatible unless a task explicitly requires a breaking change.
- Save class-name order with every checkpoint.
- Do not commit datasets, model weights, secrets, or generated output folders.

## Validation commands

```bash
pytest -q
ruff check .
python scripts/generate_synthetic_data.py --config configs/quickstart.yaml --samples-per-class 4
python scripts/train.py --config configs/smoke_test.yaml
```
