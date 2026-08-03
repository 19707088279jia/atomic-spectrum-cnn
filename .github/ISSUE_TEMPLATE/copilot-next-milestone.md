---
name: Copilot next milestone
about: Assign the next Atomic Spectrum AI milestone to GitHub Copilot
title: "[Copilot] Complete next spectrum AI milestone"
labels: enhancement
assignees: ""
---

Read `COPILOT_START_HERE.md` and complete only the first unfinished milestone.

Before editing:

- Run the existing tests.
- Inspect the current CLI and configuration behavior.

Requirements:

- Preserve wavelength orientation; never use horizontal flip augmentation.
- Prevent label leakage from text, legends, filenames, metadata, image style, or source.
- Add tests for every behavior change.
- Run `pytest -q` and `ruff check .`.
- Run the smoke training configuration when model or data-pipeline code changes.
- Report exact commands and results.
