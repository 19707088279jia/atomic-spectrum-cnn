# MgO Final Untouched-Test Evaluation

One-time final evaluation of the already-selected MgO quantitative model.
Model selection is finished; this test result was not used to change alpha,
preprocessing, bin width, feature standardization, or to select an
alternative model. No further tuning followed this evaluation.

**Frozen model:** Ridge regression, `alpha = 100`.

**Frozen preprocessing:**

1. Existing validated Z-903 robust median/IQR spectral preprocessing.
2. Average-pooled into 936 bins of 25 channels each (23,400 of 23,401
   channels).
3. Feature standardization (`StandardScaler`) fit on training-set spectra
   only.

The model was fit once on the existing training split (1,797 known-MgO
spectra) using this exact configuration, then evaluated exactly once on the
existing untouched test split. The existing train/validation/test
target-group splits were used unchanged.

## 1. Final Test Metrics

| Metric | Value |
|---|---:|
| Known MgO test samples | 386 |
| MAE | 1.6989 wt% |
| RMSE | 2.4681 wt% |
| Median absolute error | 1.2130 wt% |
| R² | **0.8957** |
| Pearson correlation | 0.9467 |
| Spearman correlation | 0.9188 |
| Overall bias | 0.1501 wt% |

## 2. Test Metrics By Concentration Band

| Band | N | MAE | RMSE | R² | Bias |
|---|---:|---:|---:|---:|---:|
| Bottom 50% | 193 | 1.5172 | 2.1568 | -2.2104 | 0.6944 |
| 50th-90th percentile | 154 | 1.2924 | 1.6512 | 0.7115 | 0.0948 |
| Top 10% | 39 | 4.2031 | 5.1482 | 0.4090 | **-2.3257** |
| Top 5% | 20 | 5.6807 | 6.5765 | -0.5352 | **-2.8986** |

The top-10% and top-5% bias remain negative (underprediction), but both are
far smaller in magnitude than the low-band R² instability would suggest;
the bottom-50% R² is negative only because that band has very little
variance in ground truth, so small absolute errors produce a poor R² there
even though MAE is modest.

## 3. Fixed Comparison

| Method | Known | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| A. Training-set median baseline | 386 | 5.0652 | 7.9498 | -0.0819 |
| B. Existing multi-output CNN (declared reference) | 386 | 2.8023 | 5.4403 | 0.4933 |
| B. Existing multi-output CNN (recomputed, unmodified checkpoint) | 386 | 2.8023 | 5.4403 | 0.4933 |
| **Frozen MgO Ridge (final)** | 386 | **1.6989** | **2.4681** | **0.8957** |

The recomputed existing-CNN test metrics match the declared reference values
exactly, confirming the CNN checkpoint and evaluation path are unchanged.
This comparison was not used to modify the Ridge model; it is reported after
the frozen configuration was already evaluated.

### High-concentration bias comparison

| Band | Ridge bias | Existing CNN bias |
|---|---:|---:|
| Top 10% | -2.3257 | -13.6586 |
| Top 5% | -2.8986 | -18.9802 |

## 4. Predeclared Interpretation

Using the fixed interpretation bands declared before this evaluation:

- R² > 0.80: Strong experimental quantitative performance
- R² 0.70-0.80: Good experimental quantitative performance
- R² 0.50-0.70: Moderate quantitative performance; further external validation needed
- R² 0-0.50: Weak quantitative performance
- R² <= 0: No useful quantitative generalization

Test R² = **0.8957**, which falls in the **R² > 0.80** band:

**Strong experimental quantitative performance.**

Additional predeclared checks:

- Ridge MAE (1.6989 wt%) < existing CNN MAE (2.8023 wt%): **True**.
- High-concentration bias materially reduced versus the existing CNN:
  **True** (top 10% bias magnitude reduced from 13.66 to 2.33 wt%; top 5%
  bias magnitude reduced from 18.98 to 2.90 wt%).

## 5. Plots

- `test_scatter.png`: prediction vs. ground truth, `y = x` reference line,
  identical physical units (wt%) on both axes, full untruncated axis range,
  R² and MAE annotated on the plot.
- `test_residuals.png`: residual (`prediction - ground truth`) vs. ground
  truth. Residuals are scattered near zero through roughly 15 wt% and become
  increasingly negative (underestimated) from about 20 to 35 wt%, confirming
  a real but much smaller high-MgO underestimation tendency than the
  existing CNN. One very high-MgO test sample (~47 wt%) was overestimated,
  which is why the top-5% band's mean bias (-2.90) is smaller in magnitude
  than the systematic underestimation visible in the mid-30s wt% region.

## 6. Saved Model

The frozen pipeline needed for inference was saved only after this
evaluation completed:

```text
models/mgo_ridge_final.joblib
```

The saved artifact contains:

- Fitted Ridge model (coefficients, intercept, `alpha = 100`).
- Fitted `StandardScaler` (training-only mean/scale).
- Bin size (25 channels per bin, 936 total features).
- Expected input dimensions (23,401 channels, 180-960 nm).
- Target label ("MgO wt%") and preprocessing description.

No CNN checkpoint (`z903_cnn_robust_best.pt` or
`z903_concentration_best.pt`) was modified or overwritten.

## Summary

1. Test known count: **386**
2. Test MAE: **1.6989 wt%**
3. Test RMSE: **2.4681 wt%**
4. Test median AE: **1.2130 wt%**
5. Test R²: **0.8957**
6. Pearson: **0.9467**
7. Spearman: **0.9188**
8. Top-10% bias: **-2.3257 wt%**
9. Top-5% bias: **-2.8986 wt%**
10. Comparison with CNN: Ridge MAE 1.6989 vs CNN 2.8023 wt% (39% lower);
    Ridge R² 0.8957 vs CNN 0.4933; Ridge top-10%/top-5% bias magnitude is
    roughly 6x smaller than the CNN's.
11. Comparison with median baseline: Ridge MAE 1.6989 vs baseline 5.0652
    wt% (66% lower); Ridge R² 0.8957 vs baseline -0.0819.
12. Predefined success criteria met: **Yes** - R² > 0.80 ("Strong
    experimental quantitative performance"), Ridge MAE below the existing
    CNN's reference MAE, and materially reduced high-concentration bias
    versus the existing CNN.
13. Saved model path: `models/mgo_ridge_final.joblib`

No additional tuning was performed using this test result, and none should
follow. MgO concentration estimates from this model remain experimental
quantitative estimates, not certified or laboratory measurements.
