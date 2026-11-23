# Atomic Spectrum Identification V2

A Streamlit application for numerical NASA Z-903 spectra.

The frozen detection model covers exactly Zn, Mn, Cd, Mg, Cu, and Pb. Fe is
available separately as an experimental concentration estimate, not as a
detection output. The application has no legacy image CNN, model selector,
checkpoint field, or unrelated-element workflow.

## Run

From this directory:

```powershell
..\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

The app accepts a numerical CSV with `wavelength,intensity` columns or a
built-in NASA demo. It displays the spectrum, six independent sigmoid CNN
probabilities, an experimental Fe2O3-equivalent concentration estimate, and
deterministic NIST reference/peak-matching results. Probabilities are
detections, not concentrations.

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

## Experimental Fe Concentration

The Fe reference is the workbook column `Fe2O3*` in wt% Fe2O3-equivalent. Of
2,567 Z-903 spectra, 2,556 have known Fe values and all 2,556 are positive;
therefore this project does not train a Fe presence/absence classifier. The
Fe model is a separate full-spectrum 1D-CNN regression model, trained only on
known references using the existing frozen train/validation/test splits and
robust median/IQR spectral preprocessing.

The checkpoint is saved at `..\reports\fe_analysis\fe_cnn_best.pt`. Its
validation/test MAE is 1.095/1.203 wt% Fe2O3-equivalent, respectively. The
validation reference span is 0.12-17.40 wt%; estimates outside it are flagged
as unsupported extrapolations. This warning range is fixed from validation
only, not selected using the independent test set. Performance degrades in
the high concentration range, and the 63.9 wt% training maximum is not
represented in the held-out splits. This is an experimental estimate, not a
validated LIBS detection limit or certified analytical measurement.

Run the Fe audit and model comparison from this directory with:

```powershell
..\.venv\Scripts\python.exe train_fe.py
```

Audit, training history, validation/test metrics, range analysis, predictions,
and the final summary are saved under `..\reports\fe_analysis\`.

## Tests

```powershell
..\.venv\Scripts\python.exe -m pytest -q
..\.venv\Scripts\python.exe -m ruff check .
```
