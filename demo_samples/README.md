# NASA Z-903 Demo Samples

These are four real NASA Z-903 spectra copied from the existing training
manifest and source files. No model was retrained and no synthetic spectrum is
included.

| Selector | File | Source target ID | Presentation purpose |
| --- | --- | --- | --- |
| Simple Demo | `simple_barite.csv` | `barite` | One known positive Mg label; all other target labels are missing. |
| Mixed Demo | `mixed_mix801.csv` | `mix801` | Six known positive labels under the frozen Z-903 label thresholds. |
| Clear Demo | `clear_mix349.csv` | `mix349` | Six known labels; four positives and two negatives, with the frozen CNN correct on all six labels and the reference matcher correct on four. |
| Challenging / Near-threshold Demo | `challenging_near_threshold_agv1a.csv` | `agv1a` | Six known labels with several compositions close to screening thresholds; retained as the harder comparison case. |

The accompanying `demo_ground_truth.csv` records the original manifest values.
`UNKNOWN` means the NASA label was unavailable, not that the element was
absent. A zero would only be shown for a label that was explicitly known and
measured as zero.

## Suggested presentation flow

1. Launch the app with `streamlit run app/streamlit_app.py`.
2. Select `Reference / Peak Matching (No AI)` and choose each demo from the
   `Demo Samples` selector.
3. Show the spectrum preview, extracted peaks, match table, interference
   levels, and final comparison table.
4. Switch to `CNN Detection` with the same demo selected to show the frozen
   `NASA Z-903 1D-CNN` numerical output beside the independent reference
   result.

The reference method uses only numerical wavelength/intensity data and cached
NIST lines. The NASA CNN view uses `models/z903_cnn_robust_best.pt`, robust
median/IQR preprocessing, and a `[1, 1, 23401]` tensor directly from the CSV.
The legacy image CNN remains available only for manually uploaded images.
Neither output predicts concentration. The composition values are ground-truth
metadata from the NASA manifest, not model predictions or detection limits.

## Clear Demo Selection

`mix349` is a train-split sample selected without using the historical test
set. All six labels are known. Its frozen-threshold labels are Zn NEGATIVE,
Mn POSITIVE, Cd POSITIVE, Mg POSITIVE, Cu POSITIVE, and Pb NEGATIVE. The
positive multiples are 2.614x for Mn, 1.2x for Cd, 1.32x for Mg, and 1.28x
for Cu; Zn and Pb are 0.25x and 0.21x of their thresholds. The existing
methods are shown exactly as they return: the CNN gets all six labels correct,
and reference matching gets four of six correct. No predictions or thresholds
were changed to obtain this result.