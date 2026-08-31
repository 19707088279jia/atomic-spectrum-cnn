# Atomic Spectrum Identification V2

A clean six-target Streamlit application for numerical NASA Z-903 spectra.

Supported elements are exactly Zn, Mn, Cd, Mg, Cu, and Pb. The application has
no legacy image CNN, model selector, checkpoint field, or unrelated-element
workflow.

## Run

From this directory:

```powershell
..\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

The app accepts a numerical CSV with `wavelength,intensity` columns or a
built-in NASA demo. It displays the spectrum, six independent sigmoid CNN
probabilities, and deterministic NIST reference/peak-matching results.
Probabilities are detections, not concentrations.

The copied checkpoint is `models/z903_cnn_robust_best.pt`. It uses robust
median/IQR preprocessing and requires the tensor shape `[1, 1, 23401]`.
Thresholds are frozen in `src/labels.py` and are not recalculated.

## Quantitative Regression

The separate `train_concentration.py` baseline uses the existing train,
validation, and untouched test target-group splits. It reads `Zn`, `Mn`, `Cd`,
`MgO`, `Cu`, and `Pb` from the copied `libs_metadata.xlsx` workbook. Missing
values remain masked and do not become zeros. Training-only log1p means and
scales are stored in `models/z903_concentration_best.pt`.

The regression output is a physical-unit estimate, not a probability and not a
certified laboratory measurement. The first baseline selected epoch 8 by
validation macro log1p MAE. Results and per-element scatter plots are written
under `reports/concentration/`.

Run the baseline from this directory:

```powershell
..\.venv\Scripts\python.exe train_concentration.py
```

## Tests

```powershell
..\.venv\Scripts\python.exe -m pytest -q
..\.venv\Scripts\python.exe -m ruff check .
```
