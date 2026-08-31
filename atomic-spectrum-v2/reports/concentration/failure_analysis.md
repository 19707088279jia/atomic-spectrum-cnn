# Concentration Regression Failure Analysis

Diagnostic analysis of the existing `models/z903_concentration_best.pt`. No
training, hyperparameter tuning, checkpoint modification, or detection-model
change was performed. The existing train/validation/test target-group splits
were reused. The test split is reported only as an untouched diagnostic set;
it was not used for model selection.

The checkpoint metadata reports `best_epoch = 8`, robust median/IQR
preprocessing, six outputs, and train-fitted log1p standardization parameters.
All metrics below are in original physical units unless explicitly marked
`log1p`.

## Executive Findings

- The model is strongly compressed toward a narrow prediction range for Zn,
  Mn, Cd, Cu, and Pb. It retains rank information in many cases, but not
  calibrated magnitude information.
- Pearson correlation is weak or near zero for Zn, Mn, Cd, Cu, and Pb while
  Spearman correlation is materially higher. This is the signature of partial
  ordering without quantitative calibration.
- Extreme concentrations dominate squared error. The top 10% of ground-truth
  concentrations account for 96.1% to 100.0% of validation SSE and 82.6% to
  100.0% of test SSE, depending on element.
- MgO is the only element with a useful positive R2 on both validation and
  test (`0.419` and `0.493`). Its scatter has a clear diagonal trend, but high
  MgO values are still systematically underpredicted.
- Cd is especially unreliable because only 44 validation and 44 test labels
  are known. Its test MAE (`0.19896 ppm`) is slightly worse than the median
  baseline (`0.19727 ppm`) and its R2 is negative.
- Recommended second-iteration priorities: MgO first; then Mn, Pb, and Zn with
  explicit tail-aware evaluation and calibration. Cu and Cd require more
  labeled data and/or a revised modeling strategy before a second CNN is
  justified. The current model should remain an experimental quantitative
  estimate, not a measurement claim.

## Distribution Statistics

Quantiles are ordered as: `min, Q1, median, Q3, P90, P95, P99, max`.
Prediction quantiles use only rows with a known ground-truth concentration for
the corresponding element.

### Validation

| Element | Ground-truth quantiles | Prediction quantiles | Pearson | Spearman |
|---|---|---|---:|---:|
| Zn ppm | 1, 47, 87, 111, 139, 152, 396, 82187 | 7.739, 49.93, 84.34, 107.3, 117.7, 122.7, 127.4, 136.5 | -0.0570 | 0.7686 |
| Mn ppm | 0, 465, 1162, 1421, 1704, 2011, 16120, 64015 | 149.1, 496.8, 839.9, 1222, 1433, 1513, 1644, 1678 | 0.1424 | 0.7531 |
| Cd ppm | 0, 0.05, 0.06, 0.1225, 0.34, 1.123, 2.749, 3.17 | 0.02713, 0.07686, 0.09546, 0.1107, 0.1203, 0.1226, 0.1297, 0.1322 | 0.3822 | 0.5795 |
| MgO wt% | 0, 0.65, 4.015, 7.513, 15.11, 19.9, 32.84, 40 | 0, 0.4618, 2.768, 4.992, 7.398, 10.24, 12.89, 14.13 | 0.8352 | 0.9470 |
| Cu ppm | 0, 5, 34, 63, 87.2, 155.6, 299.9, 896 | 2.357, 14.28, 32.22, 49.39, 64.61, 80.25, 91.31, 93.58 | 0.1693 | 0.6169 |
| Pb ppm | 0, 3, 7, 18, 32, 42.35, 404.6, 858.1 | 1.485, 5.237, 9.99, 22.06, 31.39, 35.44, 44.03, 53.86 | 0.1490 | 0.7830 |

### Test

| Element | Ground-truth quantiles | Prediction quantiles | Pearson | Spearman |
|---|---|---|---:|---:|
| Zn ppm | 0, 51.75, 89, 110, 149, 165.8, 1041, 40871 | 4.222, 53.46, 85.97, 104, 116.6, 122.1, 131, 148.1 | 0.0260 | 0.6728 |
| Mn ppm | 0, 465, 1158, 1396, 1781, 1987, 4120, 62366 | 166.2, 492.4, 835.9, 1211, 1398, 1538, 1647, 1753 | 0.0633 | 0.7159 |
| Cd ppm | 0, 0.04, 0.06, 0.14, 0.24, 0.5575, 3.136, 3.17 | 0.03129, 0.08255, 0.1005, 0.1172, 0.1261, 0.1289, 0.1336, 0.1356 | 0.3225 | 0.4893 |
| MgO wt% | 0, 0.705, 4.065, 7.805, 16.48, 22.57, 32.64, 47.37 | 0, 0.508, 2.796, 4.863, 8.927, 11.19, 16.04, 21.58 | 0.8790 | 0.9451 |
| Cu ppm | 0, 5, 30, 60.75, 88.1, 119.8, 3406, 4893 | 0.6974, 11.66, 34.72, 49.67, 62.12, 77.94, 100.5, 122.3 | -0.0881 | 0.6480 |
| Pb ppm | 0, 2.1, 7, 20, 31.7, 44.5, 91.42, 4642 | 1.301, 5.04, 9.614, 19.76, 29.43, 35.53, 42.64, 51.14 | 0.0045 | 0.7800 |

## Error By Concentration Band

Bands are defined separately within each split and element using the
corresponding ground-truth quantiles. `B50` is the bottom 50%, `P50-90` is
above the median through P90, `T10` is above P90, and `T5` is above P95.
`Bias` is prediction minus ground truth.

### Validation

| Element | Band | N | MAE | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| Zn | B50 / P50-90 / T10 / T5 | 171 / 137 / 33 / 15 | 16.04 / 17.13 / 4010 / 8787 | 20.57 / 25.06 / 15990 / 23720 | 9.914 / -10.48 / -4010 / -8787 |
| Mn | B50 / P50-90 / T10 / T5 | 191 / 136 / 36 / 19 | 206.8 / 301.9 / 5367 / 9566 | 266.9 / 365.7 / 13980 / 19230 | -0.582 / -206.6 / -5367 / -9566 |
| Cd | B50 / P50-90 / T10 / T5 | 24 / 15 / 5 / 3 | 0.04212 / 0.04178 / 1.394 / 2.073 | 0.04717 / 0.05322 / 1.735 / 2.217 | 0.04159 / -0.02091 / -1.394 / -2.073 |
| MgO | B50 / P50-90 / T10 / T5 | 188 / 150 / 38 / 19 | 0.4933 / 2.587 / 14.08 / 20.71 | 0.7041 / 3.141 / 16.17 / 21.57 | -0.3246 / -2.553 / -14.08 / -20.71 |
| Cu | B50 / P50-90 / T10 / T5 | 52 / 40 / 11 / 6 | 12.03 / 21.67 / 196.2 / 299.6 | 16.24 / 25.65 / 300.2 / 401.0 | 11.03 / -6.15 / -196.2 / -299.6 |
| Pb | B50 / P50-90 / T10 / T5 | 161 / 123 / 30 / 16 | 3.868 / 7.879 / 113.1 / 197.9 | 6.128 / 9.943 / 252.2 / 344.9 | 3.656 / 4.284 / -111.3 / -197.8 |

### Test

| Element | Band | N | MAE | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| Zn | B50 / P50-90 / T10 / T5 | 177 / 140 / 35 / 18 | 18.02 / 18.95 / 1762 / 3382 | 24.19 / 25.66 / 7165 / 9991 | 9.251 / -12.7 / -1762 / -3382 |
| Mn | B50 / P50-90 / T10 / T5 | 184 / 151 / 33 / 19 | 197.8 / 313.8 / 3432 / 5437 | 248.7 / 382.2 / 11070 / 14570 | 19.95 / -219.7 / -3432 / -5437 |
| Cd | B50 / P50-90 / T10 / T5 | 25 / 15 / 4 / 3 | 0.05152 / 0.05406 / 1.664 / 2.169 | 0.05794 / 0.06472 / 2.139 / 2.468 | 0.04889 / -0.03612 / -1.664 / -2.169 |
| MgO | B50 / P50-90 / T10 / T5 | 193 / 154 / 39 / 20 | 0.4607 / 2.988 / 13.66 / 18.98 | 0.636 / 3.523 / 15.55 / 20.13 | -0.2605 / -2.958 / -13.66 / -18.98 |
| Cu | B50 / P50-90 / T10 / T5 | 58 / 44 / 12 / 6 | 12.55 / 18.07 / 806.9 / 1560 | 18.68 / 21.93 / 1796 / 2540 | 11.81 / -5.021 / -805.4 / -1560 |
| Pb | B50 / P50-90 / T10 / T5 | 168 / 121 / 32 / 16 | 3.902 / 6.815 / 202.3 / 385.3 | 5.945 / 9.017 / 832.8 / 1178 | 3.757 / 1.937 / -201.3 / -385.3 |

## Extreme-Concentration Dominance

The percentage of total squared error contributed by the upper tails was:

| Element | Validation top 10% | Validation top 5% | Test top 10% | Test top 5% |
|---|---:|---:|---:|---:|
| Zn | 100.00% | 100.00% | 99.99% | 99.99% |
| Mn | 99.55% | 99.44% | 99.18% | 98.99% |
| Cd | 99.37% | 97.37% | 99.20% | 99.09% |
| MgO | 86.34% | 76.81% | 82.58% | 70.95% |
| Cu | 96.12% | 93.57% | 99.89% | 99.84% |
| Pb | 99.06% | 98.76% | 99.93% | 99.89% |

Therefore, extreme concentrations dominate RMSE for every target. They also
strongly influence R2 because the squared-error numerator is dominated by a
small number of large misses. Negative or near-zero R2 for Zn, Mn, Cd, Cu, and
Pb is not evidence that the model has no rank signal; it is evidence that the
magnitude calibration fails badly in the long tail.

## Regression-To-The-Mean Diagnosis

The prediction ranges are much narrower than the ground-truth ranges:

| Element | Validation truth range | Validation prediction range | Test truth range | Test prediction range |
|---|---:|---:|---:|---:|
| Zn | 1 to 82187 | 7.74 to 136.5 | 0 to 40871 | 4.22 to 148.1 |
| Mn | 0 to 64015 | 149.1 to 1678 | 0 to 62366 | 166.2 to 1753 |
| Cd | 0 to 3.17 | 0.0271 to 0.1322 | 0 to 3.17 | 0.0313 to 0.1356 |
| MgO | 0 to 40 | 0 to 14.13 | 0 to 47.37 | 0 to 21.58 |
| Cu | 0 to 896 | 2.36 to 93.58 | 0 to 4893 | 0.697 to 122.3 |
| Pb | 0 to 858.1 | 1.48 to 53.86 | 0 to 4642 | 1.30 to 51.14 |

This is clear regression to the mean. For Zn, Mn, Cd, Cu, and Pb, the model
cannot approach the largest ground-truth values at all. The median predictions
are also near the central data mass: validation medians are 84.34, 839.9,
0.09546, 2.768, 32.22, and 9.99, while train medians are 90, 1162, 0.06,
4.08, 31, and 7 respectively. Mg has a broader response but still saturates
below the highest values.

## Scatter-Plot Inspection

The existing plots inspected were:

- `reports/concentration/plots/validation_{Zn,Mn,Cd,Mg,Cu,Pb}.png`
- `reports/concentration/plots/test_{Zn,Mn,Cd,Mg,Cu,Pb}.png`

Observed patterns:

- **Zn:** Validation and test show a dense low-concentration cloud and a nearly
  horizontal prediction band while extreme ground truths extend to tens of
  thousands. The diagonal reference line is mostly empty at high x. This is
  severe magnitude collapse.
- **Mn:** Both plots show a low cloud with predictions capped around the
  1,000--1,700 ppm region. Large Mn values are visibly far below the diagonal.
  Rank ordering exists in the dense region but high values are not recovered.
- **Cd:** Predictions are compressed into roughly 0.03--0.14 ppm while ground
  truth reaches 3.17 ppm. A few high-Cd points lie far right and almost on the
  x-axis. Sparse labels make this pattern especially unstable.
- **MgO:** Both plots have the clearest diagonal structure and the highest
  correlations. The cloud bends below the diagonal as MgO rises; the model
  predicts the ordering reasonably but underestimates the upper range.
- **Cu:** Most points form a low-range cloud, with high-concentration points
  stranded near low predictions. The test plot includes extreme values near
  3,900--4,900 ppm with predictions near zero.
- **Pb:** The low-range cloud has some rank structure, but high-Pb points are
  almost horizontal near low predictions. The model captures common-range
  ordering more than physical magnitude.

## Ten Largest Absolute Errors

These are diagnostic lists only. Each row includes target ID, ground truth,
prediction, absolute error, and split. They were generated from the existing
checkpoint without changing it.

### Zn

| Split | Target ID | Ground truth | Prediction | Absolute error |
|---|---|---:|---:|---:|
| validation | mix11 | 82187 | 45.4774 | 82141.5 |
| validation | mix71 | 40386 | 51.5083 | 40334.5 |
| validation | mix77 | 8237 | 97.2725 | 8139.73 |
| validation | mix182 | 448 | 33.9604 | 414.04 |
| validation | mix805 | 318 | 105.629 | 212.371 |
| validation | rhmp | 221 | 115.216 | 105.784 |
| validation | c12ma223 | 121 | 29.2648 | 91.7352 |
| validation | mix195 | 215 | 124.65 | 90.3497 |
| validation | mr0601 | 111 | 22.7011 | 88.2989 |
| validation | c12ma227 | 139 | 64.2564 | 74.7436 |
| test | mix41 | 40871 | 101.211 | 40769.8 |
| test | mix83 | 8236 | 40.9393 | 8195.06 |
| test | mix107 | 8102 | 41.1233 | 8060.88 |
| test | umass1832 | 1232 | 113.945 | 1118.05 |
| test | mix179 | 857 | 35.7873 | 821.213 |
| test | e98144 | 403 | 27.5885 | 375.411 |
| test | mix185 | 317 | 30.2914 | 286.709 |
| test | gw911 | 252 | 50.3171 | 201.683 |
| test | ner1482 | 272 | 106.333 | 165.667 |
| test | cvg2bh | 227 | 84.2705 | 142.729 |

### Mn

| Split | Target ID | Ground truth | Prediction | Absolute error |
|---|---|---:|---:|---:|
| validation | mix3 | 64015 | 1045.72 | 62969.3 |
| validation | mix39 | 33154 | 1069.44 | 32084.6 |
| validation | mix51 | 32693 | 1216.6 | 31476.4 |
| validation | mix69 | 32028 | 886.91 | 31141.1 |
| validation | mix81 | 6363 | 306.629 | 6056.37 |
| validation | mix123 | 4808 | 1262.68 | 3545.32 |
| validation | sc1 | 3175 | 630.258 | 2544.74 |
| validation | c12mas051 | 2633 | 741.698 | 1891.3 |
| validation | mix195 | 2646 | 1246.83 | 1399.17 |
| validation | jlk1 | 2060 | 717.997 | 1342 |
| test | mix9 | 62366 | 389.24 | 61976.8 |
| test | b131627 | 10000 | 725.796 | 9274.2 |
| test | zwc | 7513 | 427.722 | 7085.28 |
| test | c10ma009a | 6196 | 744.615 | 5451.38 |
| test | sc4 | 3098 | 641.826 | 2456.17 |
| test | 11kmased006 | 2943 | 836.586 | 2106.41 |
| test | kn18143124 | 2556 | 563.846 | 1992.15 |
| test | e98144 | 1801 | 297.187 | 1503.81 |
| test | r24 | 2099 | 621.468 | 1477.53 |
| test | lp28 | 2630 | 1203.14 | 1426.86 |

### Cd

| Split | Target ID | Ground truth | Prediction | Absolute error |
|---|---|---:|---:|---:|
| validation | bir1a | 3.17 | 0.132181 | 3.03782 |
| validation | mix675 | 2.19 | 0.119975 | 2.07003 |
| validation | mix668 | 1.22 | 0.109354 | 1.11065 |
| validation | jlk1 | 0.57 | 0.0950069 | 0.474993 |
| validation | mix688 | 0.4 | 0.122437 | 0.277563 |
| validation | mag1 | 0.2 | 0.0863366 | 0.113663 |
| validation | mix327 | 0.12 | 0.0271251 | 0.0928749 |
| validation | mix628 | 0 | 0.075584 | 0.075584 |
| validation | mix534 | 0.03 | 0.104813 | 0.0748127 |
| validation | mix656 | 0 | 0.0734036 | 0.0734036 |
| test | bir1md | 3.17 | 0.130929 | 3.03907 |
| test | dnc1 | 3.09 | 0.126436 | 2.96356 |
| test | gbw07110 | 0.61 | 0.105306 | 0.504694 |
| test | mix603 | 0.26 | 0.111942 | 0.148058 |
| test | mix617 | 0.24 | 0.127621 | 0.112379 |
| test | mag1mb | 0.2 | 0.0911404 | 0.10886 |
| test | mix691 | 0.24 | 0.135633 | 0.104367 |
| test | mix298 | 0 | 0.100471 | 0.100471 |
| test | mix625 | 0 | 0.0926313 | 0.0926313 |
| test | mix632 | 0.03 | 0.118052 | 0.0880521 |

### MgO

| Split | Target ID | Ground truth | Prediction | Absolute error |
|---|---|---:|---:|---:|
| validation | kbh9423e | 37.73 | 6.54437 | 31.1856 |
| validation | e131709 | 40 | 13.3108 | 26.6892 |
| validation | mix32 | 29.89 | 3.52161 | 26.3684 |
| validation | kbh9423c | 33.66 | 7.32352 | 26.3365 |
| validation | mix175 | 32.62 | 6.29371 | 26.3263 |
| validation | m6hag | 31.18 | 6.52284 | 24.6572 |
| validation | m31 | 33.49 | 9.24108 | 24.2489 |
| validation | m8 | 30.05 | 7.34012 | 22.7099 |
| validation | m2 | 29.62 | 7.25003 | 22.37 |
| validation | m9a2 | 28.71 | 7.05232 | 21.6577 |
| test | dh4911 | 47.37 | 17.6447 | 29.7253 |
| test | mix106 | 32.59 | 3.48532 | 29.1047 |
| test | mix107 | 32.59 | 4.10152 | 28.4885 |
| test | mix179 | 32.62 | 5.75986 | 26.8601 |
| test | mix176 | 32.62 | 6.98573 | 25.6343 |
| test | m32 | 33.56 | 8.92248 | 24.6375 |
| test | mix139 | 32.73 | 8.85331 | 23.8767 |
| test | m1a | 29.66 | 7.05521 | 22.6048 |
| test | m9a4hag | 26.61 | 7.98344 | 18.6266 |
| test | umass1826a | 27.93 | 9.51732 | 18.4127 |

### Cu

| Split | Target ID | Ground truth | Prediction | Absolute error |
|---|---|---:|---:|---:|
| validation | mix688 | 896 | 19.3179 | 876.682 |
| validation | gl7mt | 262 | 11.4074 | 250.593 |
| validation | mix805 | 300 | 57.3259 | 242.674 |
| validation | mix693 | 296 | 60.1473 | 235.853 |
| validation | jso1 | 169 | 53.8154 | 115.185 |
| validation | mix656 | 89 | 9.56658 | 79.4334 |
| validation | dnc1ma | 100 | 22.1942 | 77.8058 |
| validation | bir1a | 125 | 48.0553 | 76.9447 |
| validation | mix697 | 159 | 82.4699 | 76.5301 |
| validation | mix813 | 117 | 51.8573 | 65.1427 |
| test | mix669 | 4893 | 7.32734 | 4885.67 |
| test | kn181600 | 3840 | 11.6254 | 3828.37 |
| test | mix691 | 505 | 122.259 | 382.741 |
| test | r24 | 140 | 42.4645 | 97.5355 |
| test | bir1md | 125 | 36.9923 | 88.0077 |
| test | kn284253 | 117 | 31.7813 | 85.2187 |
| test | lp28 | 1 | 81.8403 | 80.8403 |
| test | p3mt | 130 | 52.7588 | 77.2412 |
| test | dnc1mc | 100 | 32.7374 | 67.2626 |
| test | dnc1 | 100 | 34.5382 | 65.4618 |

### Pb

| Split | Target ID | Ground truth | Prediction | Absolute error |
|---|---|---:|---:|---:|
| validation | mix398 | 858.1 | 23.8649 | 834.235 |
| validation | mix397 | 839.6 | 9.96994 | 829.63 |
| validation | t11ct113 | 556.1 | 34.0061 | 522.094 |
| validation | mix403 | 428.8 | 6.56176 | 422.238 |
| validation | mix805 | 242.9 | 8.63886 | 234.261 |
| validation | c12ma220 | 84 | 21.6822 | 62.3178 |
| validation | t12ct206 | 95.1 | 42.2512 | 52.8488 |
| validation | mix813 | 53.6 | 8.2408 | 45.3592 |
| validation | c11ma145b | 66 | 25.6191 | 40.3809 |
| validation | m9a1 | 42 | 5.89728 | 36.1027 |
| test | mix387 | 4641.6 | 5.09931 | 4636.5 |
| test | mix400 | 727.6 | 17.6401 | 709.96 |
| test | mix402 | 443.5 | 31.7715 | 411.728 |
| test | gbw07110 | 97.7 | 26.0337 | 71.6663 |
| test | e98144 | 64 | 13.5465 | 50.4535 |
| test | mix815 | 51.3 | 6.71891 | 44.5811 |
| test | t12ct182b | 66.3 | 27.5573 | 38.7427 |
| test | c11ma144a | 64 | 28.4927 | 35.5073 |
| test | iahg14 | 43 | 7.67852 | 35.3215 |
| test | r41 | 61 | 26.5088 | 34.4912 |

## Recommendations

1. **MgO: worth a second iteration now.** It is the only target with strong
   positive R2 on validation and test and high Pearson/Spearman correlation.
   The next iteration should focus on upper-range calibration because the
   current model visibly underpredicts high MgO.
2. **Mn: worth a second iteration, but tail-aware.** Rank correlation is
   substantial and validation/test MAE beats the median baseline, but high Mn
   values collapse toward the center. Use tail-weighted or stratified
   validation, while preserving group splits.
3. **Pb: conditional second iteration.** Rank correlation is high, but
   Pearson/R2 are near zero because several extreme Pb values are missed.
   Consider a tail-aware objective or a two-stage calibration analysis before
   claiming quantitative improvement.
4. **Zn: conditional second iteration.** The model has a meaningful Spearman
   signal but misses extreme Zn by orders of magnitude. A second iteration is
   justified only if validation includes explicit tail metrics and robust
   calibration evaluation.
5. **Cu: defer or redesign.** Spearman is moderate, but Pearson is near zero,
   test has extreme values up to 4,893 ppm, and the model misses them. Cu also
   has only 103 validation and 114 test labels. More labels or a better feature
   strategy is needed.
6. **Cd: defer pending more data.** Only 44 labels are known in each validation
   and test split. Predictions collapse near 0.1 ppm, high values are missed,
   and the test MAE is slightly worse than the median baseline. Do not tune a
   second CNN against this small test sample.

A second iteration should be selected using training/validation only. The
untouched test results above are diagnostic and must not be used to choose
hyperparameters.
