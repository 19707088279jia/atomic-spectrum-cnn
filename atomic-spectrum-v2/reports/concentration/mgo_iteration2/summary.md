# MgO Second Quantitative-Modeling Iteration

Diagnostic experiment comparing classical spectroscopic regression against the
existing multi-output CNN, for MgO wt% only. The existing train and
validation target-group splits were used. The test split was not used for
any part of model selection, tuning, or evaluation in this iteration. No
detection model, no `z903_concentration_best.pt`, and no Streamlit UI code
were modified.

Script: `mgo_iteration2.py`. Raw results: `summary.json`. Scatter plots:
`plots/validation_{method}.png`.

## Setup

- Training rows with known MgO: 1797. Validation rows with known MgO: 376.
- Spectral preprocessing for Ridge/PLSR: the same robust median/IQR scaling
  used by the CNN, then average-pooled into 936 bins of 25 channels each
  (23,400 of 23,401 channels used; this reduction is data-independent and
  introduces no leakage).
- Features were standardized with a `StandardScaler` fit on training rows
  only.
- MgO itself was **not** log-transformed for Ridge/PLSR; both models were fit
  directly on raw wt% values.
- Ridge alpha grid: 0.1, 1, 10, 100, 1000, 10000. Selected by validation R2.
- PLSR component grid: 2, 4, 8, 12, 16, 24, 32, 48. The smallest component
  count within 0.01 R2 of the best grid value was selected, to avoid an
  unnecessarily large model for negligible improvement.
- The existing CNN was loaded unmodified from `models/z903_concentration_best.pt`
  (`best_epoch = 8`) and used only for its MgO output.

## Validation Metrics (Overall)

| Method | MAE | RMSE | Median AE | R² | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| A. Median baseline | 4.787 | 7.493 | 3.440 | -0.066 | n/a | n/a |
| B. Ridge (alpha=100) | **1.747** | **2.693** | 1.200 | **0.862** | 0.929 | 0.906 |
| C. PLSR (12 components) | 1.859 | 2.733 | **1.288** | 0.858 | 0.926 | 0.901 |
| D. Existing CNN | 2.701 | 5.533 | 1.018 | 0.419 | 0.835 | **0.947** |

Both classical methods produce a large, positive R2 improvement over the
existing CNN (0.86 vs 0.42) and over the median baseline (which is negative,
confirming the baseline explains none of the validation variance). The CNN
has the best median absolute error and best Spearman rank correlation, but
its mean-based metrics (MAE, RMSE, R2, Pearson) are substantially worse,
which is explained below by the concentration-band breakdown.

## Validation Metrics by Concentration Band

`Bias` is prediction minus ground truth; negative bias means the model
underpredicts.

### Bottom 50%

| Method | N | MAE | RMSE | R² | Bias |
|---|---:|---:|---:|---:|---:|
| Median baseline | 188 | 2.922 | 3.154 | -6.055 | 2.922 |
| Ridge | 188 | 1.508 | 2.558 | -3.641 | 0.601 |
| PLSR | 188 | 1.549 | 2.324 | -2.830 | 0.621 |
| Existing CNN | 188 | **0.493** | **0.704** | **0.648** | -0.325 |

### 50th–90th Percentile

| Method | N | MAE | RMSE | R² | Bias |
|---|---:|---:|---:|---:|---:|
| Median baseline | 150 | 3.362 | 4.317 | -1.538 | -3.361 |
| Ridge | 150 | **1.376** | **1.741** | **0.587** | 0.260 |
| PLSR | 150 | 1.427 | 1.796 | 0.561 | 0.224 |
| Existing CNN | 150 | 2.587 | 3.141 | -0.343 | -2.553 |

### Top 10%

| Method | N | MAE | RMSE | R² | Bias |
|---|---:|---:|---:|---:|---:|
| Median baseline | 38 | 19.640 | 20.802 | -8.211 | -19.640 |
| Ridge | 38 | **4.390** | **5.239** | **0.416** | **-3.320** |
| PLSR | 38 | 5.099 | 5.871 | 0.266 | -3.709 |
| Existing CNN | 38 | 14.076 | 16.172 | -4.567 | -14.076 |

### Top 5%

| Method | N | MAE | RMSE | R² | Bias |
|---|---:|---:|---:|---:|---:|
| Median baseline | 19 | 25.550 | 25.985 | -29.115 | -25.550 |
| Ridge | 19 | **5.481** | **6.367** | -0.808 | **-3.498** |
| PLSR | 19 | 6.354 | 7.131 | -1.268 | -3.767 |
| Existing CNN | 19 | 20.711 | 21.572 | -19.754 | -20.711 |

## High-Concentration Underprediction

This is the central problem identified in the first-iteration failure
analysis, and it is dramatically reduced by both classical methods:

- Top 10% bias: CNN `-14.08` wt% vs Ridge `-3.32` wt% vs PLSR `-3.71` wt%.
- Top 5% bias: CNN `-20.71` wt% vs Ridge `-3.50` wt% vs PLSR `-3.77` wt%.
- Top 10% R2: CNN `-4.57` (far worse than predicting the mean) vs Ridge
  `+0.42` (still explains real variance) vs PLSR `+0.27`.
- Top 5% R2 remains negative for Ridge (`-0.81`) and PLSR (`-1.27`), so
  neither classical model is reliable at the most extreme 5% of
  concentrations, but the magnitude of the failure is far smaller than the
  CNN's near-total collapse.

Visual confirmation from the scatter plots (`plots/validation_*.png`,
identical 0–47.4 wt% axis limits):

- **Median baseline:** a flat horizontal line at 4.08 wt%, as expected.
- **Existing CNN:** the cloud saturates below roughly 15 predicted wt%
  regardless of ground truth; every point above about 20 wt% ground truth
  falls far below the diagonal. This is the same regression-to-the-mean
  pattern documented in the first-iteration failure analysis.
- **Ridge:** points track the diagonal closely through roughly 30 wt%, with
  under-prediction appearing mainly above about 30 wt% ground truth. The
  overall shape is close to the reference line across most of the range.
- **PLSR:** very similar to Ridge, slightly more scatter in the low range and
  a similar under-prediction pattern above roughly 25–30 wt%.

## Regression-To-The-Mean Check

- Median baseline prediction range: fixed at 4.08 wt% (by construction).
- Existing CNN prediction range on validation: 0 to 14.13 wt%, far narrower
  than the ground-truth range of 0 to 40 wt%. This confirms the
  regression-to-the-mean collapse already identified in the first-iteration
  failure analysis.
- Ridge and PLSR both produce predictions spanning roughly 0 to the low 40s
  wt%, tracking the full ground-truth range far more closely, though both
  still underpredict the single highest validation points.

## Model Selection (Validation Only)

**Selected method: Ridge regression, alpha = 100, on the 936-feature
robust-scaled and binned spectrum.**

Rationale against the stated criteria:

1. **Positive and materially improved R2:** Ridge R2 = 0.862, more than double
   the existing CNN's 0.419, and far above the median baseline's -0.066.
2. **Lower MAE than the median baseline:** Ridge MAE = 1.747 vs baseline
   4.787 (a 63% reduction).
3. **Reduced high-concentration bias:** Ridge top-10% bias (-3.32) and top-5%
   bias (-3.50) are both roughly one-quarter the magnitude of the CNN's
   top-10% (-14.08) and top-5% (-20.71) bias.
4. **Reasonable top-10% performance:** Ridge is the only method with positive
   R2 (0.416) in the top 10% band; the CNN and median baseline are both
   strongly negative there.

PLSR (12 components) is a close second on every metric (R2 = 0.858, MAE =
1.859) and is a reasonable alternative if a lower-variance linear-latent-
variable model is preferred, but it does not exceed Ridge on any of the four
selection criteria above, so it was not selected as the primary
recommendation.

Ridge was not selected merely because its average MAE was slightly better;
it was selected because it improved R2 substantially, beat the baseline MAE,
and specifically reduced the high-concentration bias and top-10% R2 that
caused the original CNN's failure.

## Readiness for Final Independent Evaluation

MgO is ready for a single, final untouched-test-set evaluation of the
selected Ridge model (alpha = 100, same 936-feature robust-scaled/binned
preprocessing, same `StandardScaler` fit on training data). The top-5% band
still shows a negative R2 for Ridge, so any test-set report should continue
to disclose that the most extreme MgO concentrations remain unreliable, and
predictions should keep the existing "Experimental quantitative estimate"
qualification rather than being described as a validated measurement. No
further hyperparameter search or model change should be introduced once the
test set is evaluated.
