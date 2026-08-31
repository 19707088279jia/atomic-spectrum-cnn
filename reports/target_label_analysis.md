# Target Label Feasibility Analysis

Thresholds are screening candidates derived from the reported units and distributions. They are not LIBS detection limits.

## Metadata Mapping

| Element | Column | Workbook unit | Analysis unit | Elemental-equivalent conversion |
| --- | --- | --- | --- | --- |
| Zn | `Zn` | (ppm) | ppm | not required; metadata column is elemental |
| Mn | `Mn` | (ppm) | ppm | not required; metadata column is elemental |
| Cd | `Cd` | (ppm) | ppm | not required; metadata column is elemental |
| Mg | `MgO` | (wt %) | oxide wt% (raw) + elemental-equivalent wt% | raw MgO multiplied by 0.60304188 to elemental Mg |
| Cu | `Cu` | (ppm) | ppm | not required; metadata column is elemental |
| Pb | `Pb` | (ppm) | ppm | not required; metadata column is elemental |

## Per-Element Statistics

| Element | Metadata | Missing | Zero | >0 | Min non-zero | Max | Median non-zero | P10 | P25 | P50 | P75 | P90 | Equivalent min | Equivalent max | Equivalent median | Classification |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Zn | 2626 | 325 | 7 | 2619 | 1 | 1e+06 | 87 | 17 | 40 | 87 | 111 | 144 | 1 | 1e+06 | 87 | GOOD |
| Mn | 2823 | 128 | 30 | 2793 | 9 | 247143 | 1084 | 232 | 465 | 1084 | 1448 | 1717.4 | 9 | 247143 | 1084 | GOOD |
| Cd | 374 | 2577 | 28 | 346 | 0.02 | 6.02 | 0.06 | 0.03 | 0.05 | 0.06 | 0.12 | 0.215 | 0.02 | 6.02 | 0.06 | GOOD |
| Mg | 2929 | 22 | 75 | 2854 | 0.01 | 49.35 | 3.98 | 0.09 | 0.66 | 3.98 | 7.58 | 14.204 | 0.00603042 | 29.7601 | 2.40011 | GOOD |
| Cu | 929 | 2022 | 26 | 903 | 1 | 1e+06 | 29 | 3 | 6 | 29 | 62 | 102.8 | 1 | 1e+06 | 29 | GOOD |
| Pb | 2429 | 522 | 30 | 2399 | 0.1 | 92832 | 8 | 1 | 3 | 8 | 20 | 33.68 | 0.1 | 92832 | 8 | GOOD |

## Candidate Thresholds

| Element | Threshold | Unit | Positive | Negative | Positive % |
| --- | ---: | --- | ---: | ---: | ---: |
| Zn | 0 | ppm | 2619 | 7 | 99.73% |
| Zn | 1 | ppm | 2576 | 50 | 98.10% |
| Zn | 5 | ppm | 2517 | 109 | 95.85% |
| Zn | 10 | ppm | 2436 | 190 | 92.76% |
| Zn | 25 | ppm | 2217 | 409 | 84.42% |
| Zn | 50 | ppm | 1839 | 787 | 70.03% |
| Zn | 100 | ppm | 1041 | 1585 | 39.64% |
| Zn | 500 | ppm | 40 | 2586 | 1.52% |
| Mn | 0 | ppm | 2793 | 30 | 98.94% |
| Mn | 1 | ppm | 2793 | 30 | 98.94% |
| Mn | 5 | ppm | 2793 | 30 | 98.94% |
| Mn | 10 | ppm | 2790 | 33 | 98.83% |
| Mn | 25 | ppm | 2783 | 40 | 98.58% |
| Mn | 50 | ppm | 2780 | 43 | 98.48% |
| Mn | 100 | ppm | 2655 | 168 | 94.05% |
| Mn | 500 | ppm | 2001 | 822 | 70.88% |
| Cd | 0 | ppm | 346 | 28 | 92.51% |
| Cd | 0.01 | ppm | 346 | 28 | 92.51% |
| Cd | 0.05 | ppm | 203 | 171 | 54.28% |
| Cd | 0.1 | ppm | 97 | 277 | 25.94% |
| Cd | 0.25 | ppm | 31 | 343 | 8.29% |
| Cd | 0.5 | ppm | 25 | 349 | 6.68% |
| Cd | 1 | ppm | 17 | 357 | 4.55% |
| Cd | 2 | ppm | 14 | 360 | 3.74% |
| Mg | 0 | oxide wt% | 2854 | 75 | 97.44% |
| Mg | 0.01 | oxide wt% | 2808 | 121 | 95.87% |
| Mg | 0.1 | oxide wt% | 2551 | 378 | 87.09% |
| Mg | 0.5 | oxide wt% | 2208 | 721 | 75.38% |
| Mg | 1 | oxide wt% | 2044 | 885 | 69.78% |
| Mg | 2 | oxide wt% | 1784 | 1145 | 60.91% |
| Mg | 5 | oxide wt% | 1252 | 1677 | 42.74% |
| Mg | 10 | oxide wt% | 457 | 2472 | 15.60% |
| Cu | 0 | ppm | 903 | 26 | 97.20% |
| Cu | 1 | ppm | 864 | 65 | 93.00% |
| Cu | 5 | ppm | 689 | 240 | 74.17% |
| Cu | 10 | ppm | 593 | 336 | 63.83% |
| Cu | 25 | ppm | 462 | 467 | 49.73% |
| Cu | 50 | ppm | 342 | 587 | 36.81% |
| Cu | 100 | ppm | 94 | 835 | 10.12% |
| Cu | 500 | ppm | 29 | 900 | 3.12% |
| Pb | 0 | ppm | 2399 | 30 | 98.76% |
| Pb | 1 | ppm | 2093 | 336 | 86.17% |
| Pb | 5 | ppm | 1469 | 960 | 60.48% |
| Pb | 10 | ppm | 1015 | 1414 | 41.79% |
| Pb | 25 | ppm | 391 | 2038 | 16.10% |
| Pb | 50 | ppm | 135 | 2294 | 5.56% |
| Pb | 100 | ppm | 54 | 2375 | 2.22% |
| Pb | 500 | ppm | 17 | 2412 | 0.70% |

## Composition Correlations

Pearson correlations use pairwise-available composition values. Oxide values are converted to elemental equivalents before this table when applicable.

|    |     Zn |     Mn |     Cd |     Mg |     Cu |     Pb |
|:---|-------:|-------:|-------:|-------:|-------:|-------:|
| Zn |  1.000 |  0.003 |  0.232 |  0.027 | -0.010 | -0.004 |
| Mn |  0.003 |  1.000 | -0.004 |  0.025 | -0.015 | -0.006 |
| Cd |  0.232 | -0.004 |  1.000 |  0.159 | -0.023 | -0.028 |
| Mg |  0.027 |  0.025 |  0.159 |  1.000 | -0.023 | -0.021 |
| Cu | -0.010 | -0.015 | -0.023 | -0.023 |  1.000 | -0.008 |
| Pb | -0.004 | -0.006 | -0.028 | -0.021 | -0.008 |  1.000 |

## Threshold Combination Analysis

Combinations use the best screening threshold for each element and only samples with all six compositions available. A positive label means concentration strictly greater than its candidate threshold.

| Combination | Samples |
| --- | ---: |
| Mn + Cd + Cu | 35 |
| Zn + Mn + Cd + Cu | 24 |
| Pb | 24 |
| Mn + Cu | 24 |
| Mn | 23 |
| Mn + Pb | 23 |
| Zn + Mn + Cd + Cu + Pb | 22 |
| Mn + Cd + Pb | 20 |
| Mn + Cd + Cu + Pb | 12 |
| Mn + Cu + Pb | 9 |
| Cd | 8 |
| none | 7 |
| Mn + Cd + Mg + Cu + Pb | 7 |
| Zn + Cd + Cu + Pb | 7 |
| Cd + Pb | 4 |
| Zn + Mn + Cd + Pb | 2 |
| Mn + Cd + Mg | 2 |
| Cu + Pb | 2 |
| Mn + Mg | 2 |
| Zn + Cd + Pb | 1 |
| Zn + Mn + Cd + Mg + Cu | 1 |
| Zn + Mn + Cd + Mg | 1 |
| Cu | 1 |
| Zn + Mn + Cu + Pb | 1 |
| Zn + Cu + Pb | 1 |
| Mn + Cd + Mg + Cu | 1 |
| Zn + Cd + Cu | 1 |
| Cd + Cu + Pb | 1 |

## Interpretation

- The limiting target under the selected screening thresholds is Cd, with a minority class of 171 samples.
- There are 266 samples with all six target compositions available for joint correlation and combination analysis.
- All six can be explored together only as a cautious multilabel baseline; class imbalance, missingness, and composition correlations require split-aware validation and should not be interpreted as scientific detection limits.
- Additional real-world data are especially advisable for the limiting target and for independent samples that break common composition associations.
- Recommended next step: define a scientifically justified label policy and leakage-resistant train/validation/test split before any CNN training.
