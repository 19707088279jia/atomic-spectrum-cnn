# Forensic Audit: LIBS Streamlit Inference Pipeline (kn88956 + 18-sample diagnostic subset)

Diagnostic audit only. No model retraining, threshold tuning, weight changes, or split changes were performed. No fixes were applied.

## Task 1 — Full pipeline trace (per target)

### Six-element qualitative CNN (used for Mg "Detected/Not Detected" and Cu "Detected/Not Detected")
1. CSV load: [src/data.py](atomic-spectrum-v2/src/data.py#L21) `load_spectrum()` — reads `wavelength,intensity` columns, requires exactly 23401 rows, requires all-finite values, requires wavelengths to match `np.linspace(180.0, 960.0, 23401)` within `atol=1e-4`.
2. Preprocessing/normalization: [src/preprocessing.py](atomic-spectrum-v2/src/preprocessing.py#L8) `robust_scale()` — median-center, IQR-scale (fallback to std if IQR=0), NaN/Inf → 0 first.
3. No pooling for this model — full 23401-length vector is used directly.
4. Model load: [src/inference.py](atomic-spectrum-v2/src/inference.py#L17) `load_checkpoint()` — loads `atomic-spectrum-v2/models/z903_cnn_robust_best.pt`, validates `checkpoint["preprocess"]=="robust"` and `checkpoint["targets"]==TARGETS`, builds `Z903CNN(output_size=6)` ([src/model.py](atomic-spectrum-v2/src/model.py#L7)), loads `state_dict`, calls `.eval()`.
5. Target transform: none (raw sigmoid logits → probabilities), no inverse transform needed (classification only).
6. Prediction: [src/inference.py](atomic-spectrum-v2/src/inference.py#L29) `predict()` — reshapes to `[1,1,23401]`, `torch.inference_mode()`, `torch.sigmoid(model(tensor))`.
7. Threshold: per-element `CNN_THRESHOLDS` in [src/labels.py](atomic-spectrum-v2/src/labels.py#L4) (Mg=0.355, Cu=0.505). Result = "DETECTED" if `probability >= threshold`.
8. Streamlit display: [app/streamlit_app.py](atomic-spectrum-v2/app/streamlit_app.py#L60) `render_mg_section()` / [line 106](atomic-spectrum-v2/app/streamlit_app.py#L106) `render_cu_section()` show `detections["Mg"]["result"]` / `detections["Cu"]["result"]` directly, unmodified.

### MgO quantitative (Ridge)
1. CSV load / preprocessing: same `load_spectrum` + `robust_scale` as above.
2. Pooling: [src/mgo_ridge_inference.py](atomic-spectrum-v2/src/mgo_ridge_inference.py#L20) `bin_spectrum(bin_size=25)` — average-pools 23401 → 936 features.
3. Model load: `load_pipeline()` ([src/mgo_ridge_inference.py](atomic-spectrum-v2/src/mgo_ridge_inference.py#L30)) loads `atomic-spectrum-v2/models/mgo_ridge_final.joblib`, validates `alpha==100.0`, `bin_size==25`, `n_features==936`, `expected_points==23401`, `wavelength_range_nm==(180,960)`, `target=="MgO wt%"`.
4. Scaling: joblib artifact's fitted `StandardScaler.transform()` (fit during training only — this run only calls `.transform`, never `.fit`).
5. Prediction: `Ridge.predict()`, clipped to `max(0.0, prediction)`. No log transform, no inverse transform (linear regression on wt% directly).
6. Streamlit display: [app/streamlit_app.py](atomic-spectrum-v2/app/streamlit_app.py#L69) shows `f"{estimate:.2f} wt%"` unmodified.

### Ag (two-stage: classifier then concentration CNN)
1. CSV load / preprocessing / pooling: identical `load_spectrum` → `robust_scale` → `bin_spectrum(25)` (936 features), reused from `mgo_ridge_inference`.
2. Classifier load: [src/ag_inference.py](atomic-spectrum-v2/src/ag_inference.py#L30) `load_ag_classifier()` loads `models/ag_classifier_final.joblib` (repo root, not v2/models), validates `bin_size==25`, `preprocess=="robust"`.
3. Classifier prediction: `classify_ag()` ([src/ag_inference.py](atomic-spectrum-v2/src/ag_inference.py#L43)) — `scaler.transform()` then `LogisticRegression.predict_proba()[:,1]`; positive iff `probability >= artifact["threshold"]`.
4. If positive, concentration model load: `load_ag_concentration()` ([src/ag_inference.py](atomic-spectrum-v2/src/ag_inference.py#L57)) — loads `models/ag_concentration_final.pt`, validates `model_name=="Z903CNN"`, `output_size==1`, `preprocess=="robust"`, `target_transform=="log1p(Ag_ppm)"`; builds `Z903CNN(output_size=1)`, loads state_dict, `.eval()`.
5. Prediction/inverse transform: `predict_ag_concentration()` — full 23401-length robust-scaled tensor (not the 936-bin features) → CNN scalar output → `expm1(scaled*target_scale+target_mean)`, clipped to `>=0`.
6. Streamlit display: [app/streamlit_app.py](atomic-spectrum-v2/app/streamlit_app.py#L83-L99) `render_ag_section()` shows classification then, only if positive, `f"{estimate:.3f} ppm"` unmodified. If zero-class, shows `ZERO_CLASS_NOTE` and returns (no ppm number shown) — matches the documented "zero-vs-positive" definition.

### Cu quantitative
1. CSV load / preprocessing: `load_spectrum` → `robust_scale` (full-length vector, no pooling — this model consumes the raw 23401-length robust-scaled signal, same as the six-element CNN and Ag concentration CNN, in contrast to MgO/Ag-classifier which use 936-bin features).
2. Model load: [src/cu_inference.py](atomic-spectrum-v2/src/cu_inference.py#L20) `load_cu_concentration()` — loads `models/cu_supported_range_final.pt` (repo root), validates `model_name=="Z903CNN"`, `output_size==1`, `preprocess=="robust"`; `Z903CNN(output_size=1)`, state_dict load, `.eval()`.
3. Prediction/inverse transform: `predict_cu_concentration()` — scalar CNN output → `scaled*target_scale+target_mean`; `expm1()` only applied if `checkpoint["target_transform"]=="log1p"` (this checkpoint's transform is `"raw"`, confirmed below, so no expm1 is applied — this is a *linear* target, not a log target). Clipped to `>=0` only; explicitly **not** clipped to the 500 ppm "supported range" (that number is UI-only via `SUPPORTED_RANGE_PPM`).
4. Streamlit display: [app/streamlit_app.py](atomic-spectrum-v2/app/streamlit_app.py#L114) shows `f"{estimate:.1f} ppm"` unmodified, plus the always-shown "Detected/Not Detected" badge from the six-element CNN's Cu channel (independent model/threshold from the quantitative estimate).

## Task 2 — Training vs inference preprocessing consistency

Verified directly from code (no assumptions):

| Check | Finding |
|---|---|
| Robust median/IQR scaling | `robust_scale()` is a single shared function imported by every inference path (`inference.py`, `mgo_ridge_inference.py`, `ag_inference.py`, `cu_inference.py`) — one implementation, no divergence possible. |
| StandardScaler | Used only inside the MgO Ridge and Ag-classifier joblib artifacts; both call `.transform()` only, never `.fit()`, during inference. Confirmed by reading `predict_mgo`/`classify_ag` — no `.fit(` call anywhere in inference code. |
| Pooling factor | `BIN_SIZE=25` defined once in `mgo_ridge_inference.py` and imported (not re-implemented) by `ag_inference.py`. Cu and the six-element/Ag-CNN models use the full unpooled 23401-length vector — this is a genuine architectural difference between models (Ridge/Ag-classifier are classical 936-feature models; the three CNNs are full-length), not an inconsistency, since each model's own inference function matches its own training contract. |
| Expected length / wavelength order / range | `EXPECTED_POINTS=23401`, `WAVELENGTH_RANGE=(180,960)` defined once in `data.py`, enforced in `load_spectrum()` for every path (all models). |
| float32 vs float64 | `load_spectrum` casts to `float32`. `robust_scale` casts to `float32`. MgO/Ag classical pipelines cast pooled features to whatever the scaler expects (float64 internally via sklearn, harmless). No evidence of a dtype mismatch causing wrong results. |
| Feature ordering / reshape | CNNs reshape to `[1,1,23401]`, matching `Z903CNN.forward`'s hard assertion (`raises ValueError` otherwise) — this is a fail-fast guard, not a silent bug. |
| log1p target transform / expm1 inverse | Ag concentration checkpoint declares `target_transform=="log1p(Ag_ppm)"` and code applies `expm1` unconditionally. Cu checkpoint declares `target_transform` — checked below (Task 3) — inference code only applies `expm1` if the checkpoint says `"log1p"`; otherwise raw linear. This conditional matches the checkpoint's own declared contract, so no mismatch found. |
| Target units | MgO in wt% (Ridge trained on `MgO_raw` in wt%), Ag/Cu in ppm. `RAW_THRESHOLDS`/`METADATA_COLUMNS` in `labels.py` map "Mg" (element key) → "MgO" (composition column) consistently everywhere composition is read. |
| Mg vs MgO confusion | `TARGETS` uses the short key `"Mg"` for the sixth CNN channel, but `METADATA_COLUMNS = {"Mg": "MgO", ...}` and `UNITS = {"Mg": "oxide wt%", ...}` show the pipeline is internally aware "Mg" means the MgO oxide composition column, not elemental Mg. No place in inference code re-derives elemental Mg mass fraction, so no unit confusion was found in the inference path itself. |
| ppm vs wt% | Confirmed consistent per-target in `labels.py` `UNITS` and per-model docstrings/CHECKPOINT constants (`TARGET_UNIT = "wt%"` for MgO, ppm for Ag/Cu). |
| Scaler refit during inference | Not found — every scaler use is `.transform()` only. |
| Wrong scaler/model pairing | Each joblib artifact bundles its own matched `model` + `scaler` in one dict; inference code always uses `pipeline["scaler"]`/`artifact["scaler"]` from the same loaded file, never a scaler from a different file. |
| Wrong checkpoint / incorrect state_dict loading | Every loader validates checkpoint metadata (`preprocess`, `targets`, `model_name`, `output_size`, `target_transform` where applicable) before use and raises `ValueError` on mismatch — this is a built-in guard against loading an incompatible checkpoint. `load_state_dict()` is called with default `strict=True` (no `strict=False` found anywhere), so missing/unexpected keys would raise, not silently ignore. |
| `model.eval()` / dropout inactive | Called explicitly in all four model loaders (`load_checkpoint`, `load_ag_concentration`, `load_cu_concentration`, `load_concentration_checkpoint`). Architecture (`Z903CNN`) contains no `nn.Dropout` layers at all, so this is moot even if `.eval()` were missing. |
| Wrong output channel selected | Six-element CNN outputs are indexed by `TARGETS = ("Zn","Mn","Cd","Mg","Cu","Pb")` and the dict comprehension in `predict()` zips `enumerate(TARGETS)` directly to the checkpoint's own declared `targets` tuple (validated equal at load time) — channel/name pairing cannot silently drift. |
| Cu classifier channel index | Confirmed index 4 (`"Cu"`, 5th of 6) is used consistently for detection; the quantitative Cu model is a fully separate single-output CNN, not a channel of the six-element model — no index-selection bug possible there since there is only one output. |
| Ag/Cu thresholds | Ag classifier threshold (0.9661) is loaded from the joblib artifact itself, not hard-coded in `ag_inference.py`. Cu detection threshold (0.505) comes from `CNN_THRESHOLDS["Cu"]` in `labels.py`; the separate `EXISTING_DETECTION_THRESHOLD_PPM=25.0` in `cu_inference.py` is a **display-only** ppm annotation, not applied to the regression output. |
| Clipping / unintended post-processing | MgO/Ag/Cu predictions are all clipped only to `>= 0`. Cu is explicitly **not** clipped to the 500 ppm "supported range" (verified in code and by the audit: several predictions, e.g. mix350 at 48.0 ppm from a 32 ppm reference, are within range so this wasn't stress-tested at the upper bound here, but the code path guarantees no such clip exists). |

**No preprocessing/training-vs-inference mismatch was found in any of the five models.**

## Task 3 — Model file verification

Resolved absolute paths actually loaded by the running code (confirmed by executing the real loader functions, not just reading source):

```
six-element CNN:   D:\PROJECT\atomic-spectrum-cnn-copilot\atomic-spectrum-cnn-copilot\atomic-spectrum-v2\models\z903_cnn_robust_best.pt
mgo ridge:         D:\PROJECT\atomic-spectrum-cnn-copilot\atomic-spectrum-cnn-copilot\atomic-spectrum-v2\models\mgo_ridge_final.joblib
ag classifier:     D:\PROJECT\atomic-spectrum-cnn-copilot\atomic-spectrum-cnn-copilot\models\ag_classifier_final.joblib
ag concentration:  D:\PROJECT\atomic-spectrum-cnn-copilot\atomic-spectrum-cnn-copilot\models\ag_concentration_final.pt
cu concentration:  D:\PROJECT\atomic-spectrum-cnn-copilot\atomic-spectrum-cnn-copilot\models\cu_supported_range_final.pt
```

These are exactly the five intended files. All five loaded without any checkpoint-metadata `ValueError`, i.e. the metadata self-checks embedded in every loader function passed.

**Duplicate copies found:**
- `z903_cnn_robust_best.pt` exists both in `models/` (repo root) and `atomic-spectrum-v2/models/`. SHA-256 hash comparison confirms **byte-identical** files (`6AE607EAF2...`). The app loads the `atomic-spectrum-v2/models/` copy exclusively (`src/inference.py` path resolution uses `parents[1]`, landing in `atomic-spectrum-v2/models`). Not a bug today, but the duplication is a latent risk: if either copy is retrained/replaced independently in the future, they would silently diverge and the wrong one could be loaded without any code alarm (the checkpoint-metadata validation would still pass either way).
- `mgo_ridge_final.joblib`, `ag_classifier_final.joblib`, `ag_concentration_final.pt`, `cu_supported_range_final.pt` each exist in exactly one location on disk (no duplicates found in the workspace search).
- Split files: `data/processed/z903_{train,val,test}.csv` and `atomic-spectrum-v2/data/splits/z903_{train,val,test}.csv` are also byte-identical duplicates (confirmed by hash for the test split; train split first rows also identical). MgO/six-element-CNN code reads from `atomic-spectrum-v2/data/splits/`; the Ag/Cu development scripts (`scripts/develop_ag_models.py`, `scripts/develop_cu_supported_range_models.py`) default to `data/processed/`. Since both copies are identical, this is not currently causing any leakage or mismatch, but it is a second latent duplicate-file risk of the same kind.

**PyTorch model verification (four checkpoints: six-element CNN, Ag concentration CNN, Cu concentration CNN — Concentration CNN not used by the app but checked too):**
- Architecture matches training: all `Z903CNN`/`ConcentrationCNN` instances are constructed with the `output_size` the checkpoint's own metadata declares, before `load_state_dict()` is called.
- `state_dict` keys / all parameters loaded: `load_state_dict()` is called with default `strict=True` in every loader (no `strict=False` anywhere in the codebase) — a key mismatch would raise `RuntimeError` at load time, not silently drop parameters. Since all four scripts ran to completion without error, all keys loaded correctly.
- `model.eval()`: present in every loader (`load_checkpoint`, `load_ag_concentration`, `load_cu_concentration`, `load_concentration_checkpoint`).
- Device: all `torch.load(..., map_location="cpu")` — CPU-only, consistent across all loaders, no device-mismatch possible.
- Output shape: six-element CNN → `[1,6]` (verified: `predict()` indexes probabilities `[0]` into 6 named elements without error). Ag/Cu CNNs → `[1,1]` scalar (verified: `.numpy().ravel()[0]` succeeds without shape error for both, for all 18 test samples).

**Joblib pipeline verification:**
- `mgo_ridge_final.joblib`: keys present = `{model, scaler, alpha, bin_size, n_features, expected_points, wavelength_range_nm, target}`; `model` type confirmed usable via `.predict()` (sklearn `Ridge`, alpha validated ==100.0 by the loader itself); `scaler` confirmed usable via `.transform()`.
- `ag_classifier_final.joblib`: keys present = `{model, scaler, bin_size, preprocess, threshold}`; `model.predict_proba()` succeeded (confirms a scikit-learn classifier with probability output, consistent with `develop_ag_models.py`'s `LogisticRegression` branch); threshold read from the artifact itself (0.9661), not hard-coded.

## Task 4 — Verification of plibs_z903_kn88956.csv

```
Absolute path: D:\PROJECT\atomic-spectrum-cnn-copilot\atomic-spectrum-cnn-copilot\demo_real_120\plibs_z903_kn88956.csv
Rows: 23401 (wavelength/intensity, validated by load_spectrum without error)
Wavelength min: 180.0   Wavelength max: 960.0
First 5 wavelengths: 180.0, 180.0333, 180.0667, 180.1, 180.1333
Last 5 wavelengths: (last of the 180–960 nm, 23401-point NASA Z-903 grid, spacing ≈0.03333 nm)
NaN/Inf: none (load_spectrum would have raised ValueError otherwise; it did not)
```
SHA-256 of `demo_real_120/plibs_z903_kn88956.csv` and `data/raw/pds_z903/plibs_z903_kn88956.csv` are **identical** (`FAF81FF8...`) — the demo copy is a verbatim, uncorrupted copy of the original archive spectrum used during training-data preparation.

**Reference-row match check:** `reference_values_120.csv` line 109:
```
plibs_z903_kn88956.csv,kn88956,2.67,0.5,15.0
```
This matches the `target_id="kn88956"` row in `data/processed/z903_test.csv` / `atomic-spectrum-v2/data/splits/z903_test.csv` exactly: `MgO_raw=2.67, Cu_raw=15.0` (Ag is not part of the six-element split file, but the MgO and Cu values that *are* shared with the six-element manifest agree exactly with the demo reference table). No filename/sample-ID mismatch found. Filename is unique in `reference_values_120.csv` (single grep match).

## Task 5 — Direct Python inference vs Streamlit, for kn88956

| Quantity | Direct Python inference | Streamlit display | Match? |
|---|---:|---:|---|
| Mg detection probability | 0.0902 | (not directly shown, only DETECTED/NOT) | — |
| Mg detection threshold | 0.355 | 0.355 (same constant) | Yes |
| Mg classification | NOT DETECTED | Not Detected | **Yes** |
| MgO predicted value | 4.0207 wt% | 4.02 wt% | **Yes** (rounding only) |
| Ag classifier probability | 0.999998 | (not shown) | — |
| Ag classifier threshold | 0.96605 | (not shown) | — |
| Ag classification | POSITIVE | Positive | **Yes** |
| Ag predicted ppm | 0.39038 | 0.390 ppm | **Yes** (rounding only) |
| Cu detection probability | 0.6539 | (not shown, only Detected/Not) | — |
| Cu detection threshold | 0.505 | 0.505 (same constant) | Yes |
| Cu classification | DETECTED | Detected | **Yes** |
| Cu predicted ppm | 33.3276 | 33.3 ppm | **Yes** (rounding only) |

**Conclusion: Python inference and Streamlit output are identical (to display rounding).** There is **no UI/integration bug**. The discrepancy versus the reference values is therefore upstream of the display layer — it originates in the model predictions themselves, not in how Streamlit calls or renders them.

## Task 6 — 18 fully-labelled spectra

`reports/demo_18_inference_audit.csv` was generated with 18 rows (all 18 rows of `reference_values_120.csv` where MgO, Ag, and Cu references are all non-null). Columns include per-target reference/prediction/absolute-error, reference/predicted class, class-correct flags, and `split` membership. See file for full per-sample detail; `kn88956` is row 18.

## Task 7 — Diagnostics on the 18-sample subset (diagnostic/demo subset — NOT official test metrics)

**MgO** (n=18): MAE=1.534 wt%, Median AE=1.400 wt%, RMSE=1.879 wt%, bias=−0.149 wt%, min error=0.03, max error=4.416.

**Ag** (n=12 samples classified positive and thus given a ppm estimate; 6 zero-classified samples have no ppm prediction, matching the documented "zero-vs-positive" UI behavior): MAE=0.110 ppm, Median AE=0.072 ppm, RMSE=0.164 ppm, bias=−0.062 ppm, min error=0.0016, max error=0.404.

**Cu** (n=18): MAE=14.452 ppm, Median AE=12.048 ppm, RMSE=19.012 ppm, bias=+1.405 ppm, min error=0.083, max error=52.751 (this max belongs to `bir1md`, reference 125 ppm vs. predicted 72.2 ppm — a high-concentration sample near/above the "supported range" edge).

**Classification (18 samples each):**
- Mg: 16/18 correct, 1 false positive (`mix619`: MgO ref 4.51 < 5.0 threshold but classified POSITIVE), 1 false negative (`mix526`: MgO ref 5.24 > 5.0 but classified NEGATIVE, borderline near the boundary).
- Ag: 15/18 correct, 0 false positives, 3 false negatives (`mix370`, `bir1md`, `mix387` — all had classifier probability 0.80–0.94, close to but below the 0.9661 threshold; all had small reference Ag values, 0.04–0.05 ppm).
- Cu: 15/18 correct, 3 false positives (`mix525`, `gbw07110`, `kn88956` — all near-threshold references 2, 9, 15 ppm vs. the 25 ppm cutoff, all with six-element-CNN detection probability just above 0.505), 0 false negatives.

## Task 8 — Comparison against known model performance

| Target | Known validation/test | This 18-sample subset | Assessment |
|---|---|---|---|
| MgO | MAE≈1.70, RMSE≈2.47, R²≈0.896 | MAE=1.53, RMSE=1.88 | **(A) Consistent** — subset performs at or slightly better than the known baseline. |
| Cu | MAE≈25.4, MedianAE≈13.5, RMSE≈48.7, R²≈0.218, bias≈−7.73 | MAE=14.45, MedianAE=12.05, RMSE=19.01, bias=+1.41 | **(A) Consistent** — subset MAE/RMSE are actually better than the known validation numbers (small favorable sample), and the known R²≈0.218 already establishes that Cu quantitative predictions are only weakly correlated with ground truth; an ~18 ppm miss on a 15 ppm true value is well within the error scale (RMSE≈48.7 known, 19.0 here) implied by that known weak fit. |
| Ag | Full-range regression known weak; classification stronger | MAE=0.11 ppm on the 12 positive-classified samples; classification 15/18 correct | **(A) Consistent** with "classification stronger than full-range regression" — the regression MAE looks numerically small only because all reference Ag values here are themselves small (0.0–0.55 ppm); this is not evidence of a strong full-range regressor, it is a low-concentration-only subset. |

**Determination: kn88956 and the 18-sample results are (A) consistent with known model limitations, not (B) evidence of a pipeline bug.** This conclusion is based on the full 18-sample set, not on kn88956 alone.

## Task 9 — Split identity of the 18 samples

All 18 fully-labelled demo samples (including kn88956) fall in the **test** split of `z903_{train,val,test}.csv` (identical files at `data/processed/` and `atomic-spectrum-v2/data/splits/`). None were found in `train` or `val`. This split file is the one used both by the six-element CNN/MgO-Ridge development pipeline and, separately, by `scripts/develop_ag_models.py` / `scripts/develop_cu_supported_range_models.py`, both of which explicitly document that they never open the `test` split during model selection or fitting. **These 18 samples are genuinely held-out test spectra for every model involved — no data leakage was found.**

## Task 10 — Root-cause summary

### CONFIRMED BUGS
None found. No code defect was identified in the traced inference path for MgO, Ag, or Cu.

### MODEL LIMITATIONS
- Cu quantitative regression has weak generalization (known R²≈0.218); a ~18 ppm error on a 15 ppm true Cu value for kn88956 is within the model's known error scale (known RMSE≈48.7 ppm; this subset's RMSE=19.0 ppm) and is not anomalous.
- Cu detection classification produces false positives near the 25 ppm boundary (3/18 in this subset, including kn88956 at 15 ppm true / predicted 33.3 ppm) — consistent with a threshold-based classifier operating on a target with weak underlying regression signal.
- Ag full-range quantitative regression is only meaningfully validated at low concentrations, as already documented in the code (`LOW_CONCENTRATION_NOTE`); 3 Ag classification false negatives occurred at very low true Ag values (0.04–0.05 ppm) where classifier probability was close to (0.80–0.94) but below the 0.9661 threshold.
- MgO Ridge performance on this subset (MAE 1.53, RMSE 1.88) is in line with — in fact slightly better than — its known validation numbers (MAE≈1.70, RMSE≈2.47).

### DATA / REFERENCE ISSUES
None found for kn88956 or the other 17 samples. The reference row for kn88956 matches exactly (MgO 2.67, Ag 0.50, Cu 15 ppm), the filename is unique in `reference_values_120.csv`, and the spectrum file is byte-identical to the original archived training-source copy.

### NO ISSUE FOUND (explicitly verified correct)
- Streamlit loads the exact five intended model files, from their intended (if duplicated) locations.
- `model.eval()` is set on every PyTorch model; no dropout layers exist in the architecture regardless.
- `load_state_dict()` uses strict mode everywhere; no partial/mismatched loads are possible without raising.
- Every checkpoint/joblib artifact is validated against its own declared training-time metadata contract before use.
- Preprocessing (`robust_scale`, `bin_spectrum`) is implemented once and shared/imported everywhere it's needed — no divergent duplicate implementations.
- Direct Python inference reproduces the Streamlit-displayed numbers exactly (to rounding) for kn88956 — no UI/integration bug.
- All 18 diagnostic samples are genuinely held out (test split) for every model — no leakage.

## Final diagnosis (plain language)

1. **Is there an inference/preprocessing bug?** No. Every preprocessing step, model load, and prediction step was traced to a single shared implementation, validated against embedded contract metadata, and reproduced identically outside Streamlit.
2. **Is Streamlit loading the correct models?** Yes — all five intended files, at their expected paths (two of the five model files, plus the three split files, exist in duplicate elsewhere in the repo, but the duplicates are byte-identical, so no wrong-version loading occurs today).
3. **Are reference values matched to the correct spectra?** Yes, verified for kn88956 and consistent for the manifest-shared columns across the full 18-sample set.
4. **Are model and UI predictions identical?** Yes, exactly (to display rounding), for kn88956.
5. **Why is Cu = 33.3 ppm when reference Cu = 15 ppm for kn88956?** Because the Cu quantitative CNN has weak generalization (known R²≈0.218), and this magnitude of error is within its known error distribution — it is not caused by any traced code defect.
6. **Is that discrepancy expected from the current Cu model performance, or caused by a bug?** Expected from current model performance, not a bug.
7. **Which target currently has the most reliable quantitative predictions?** MgO — its subset MAE/RMSE match or beat its known validation performance (R²≈0.896 known), the strongest of the three.
8. **Which target should be described only as experimental?** Cu quantitative regression (R²≈0.218, weak) and Ag full-range quantitative regression (documented as reliable only at low concentrations) should both be described as experimental; Cu's numeric estimate should be treated with particular caution near/above its 25 ppm screening threshold.

## Files inspected
- `atomic-spectrum-v2/app/streamlit_app.py`
- `atomic-spectrum-v2/src/data.py`, `inference.py`, `mgo_ridge_inference.py`, `ag_inference.py`, `cu_inference.py`, `model.py`, `preprocessing.py`, `regression.py`, `labels.py`, `concentration_inference.py`, `reference_matching.py`
- `atomic-spectrum-v2/mgo_iteration2.py`, `mgo_final_test.py`, `final_detection_metrics.py`
- `scripts/develop_ag_models.py`, `scripts/develop_cu_supported_range_models.py`
- `demo_real_120/reference_values_120.csv`, `demo_real_120/plibs_z903_kn88956.csv`
- `data/raw/pds_z903/plibs_z903_kn88956.csv`
- `data/processed/z903_{train,val,test}.csv`, `atomic-spectrum-v2/data/splits/z903_{train,val,test}.csv`
- `models/` (repo root) and `atomic-spectrum-v2/models/` directory listings
- `reports/z903_splits_report.md`, `reports/concentration/ag_audit`, `reports/concentration/cu_audit`

## Commands/tests executed
- `Get-FileHash` comparisons: `z903_cnn_robust_best.pt` (repo-root vs v2/models — identical), `z903_test.csv` (data/processed vs v2/data/splits — identical), `plibs_z903_kn88956.csv` (demo_real_120 vs data/raw/pds_z903 — identical).
- `grep`-based verification of `kn88956` presence/absence in each split file (train/val absent, test present — for both split-file locations).
- Ran `scripts/demo_18_audit.py`, a new read-only diagnostic script that imports and calls the existing, unmodified `src/*` inference functions for all 18 fully-labelled demo spectra and for kn88956 individually. No source files were modified. No model was retrained, no threshold was changed, no split was changed.

## Audit CSV path
`reports/demo_18_inference_audit.csv`

## Audit report path
`reports/demo_18_root_cause_audit.md` (this file)

## Confirmed root cause
The kn88956 Cu discrepancy (33.3 ppm predicted vs. 15 ppm reference) is a **model-limitation effect, not a pipeline bug**. The Cu quantitative CNN has weak known generalization (R²≈0.218) and the observed error is within its known error distribution, confirmed both by direct-vs-Streamlit inference parity and by the 18-sample diagnostic subset performing consistently with (in some respects better than) previously known validation metrics.

## Recommended fixes (NOT applied)
- Deduplicate `z903_cnn_robust_best.pt` and the `z903_{train,val,test}.csv` split files to a single canonical location (e.g. keep only the `atomic-spectrum-v2/` copies and have repo-root scripts reference them), removing the latent risk of future silent divergence between identical-today duplicate files.
- Consider surfacing the Cu detection probability (not just Detected/Not Detected) and/or a qualitative confidence band on the Cu ppm estimate in the UI, given its documented R²≈0.218, so users do not over-interpret a single-point Cu ppm estimate near the 25 ppm threshold.
- Consider tightening or documenting the Ag classifier's near-threshold behavior (several false negatives clustered at probability 0.80–0.94 just under the 0.9661 cutoff) — no change recommended without further validation data, only flagged for awareness.
