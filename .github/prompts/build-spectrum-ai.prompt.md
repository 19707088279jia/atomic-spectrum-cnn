# Build and validate the next Atomic Spectrum AI milestone

Read `COPILOT_START_HERE.md` and identify the first unfinished milestone. Work only on that milestone.

Requirements:

1. Inspect the existing implementation before making changes.
2. Run tests before editing.
3. Preserve Python 3.11 compatibility and existing CLI commands.
4. Add tests for every behavior change.
5. Never use horizontal image flipping.
6. Prevent label leakage from text, legends, filenames, image style, or source metadata.
7. Keep synthetic data claims separate from real scientific validation.
8. Run `pytest -q` and `ruff check .` before completion.
9. If model or data-pipeline code changed, execute the smoke training configuration.
10. Report exact commands and results; do not claim unexecuted success.
