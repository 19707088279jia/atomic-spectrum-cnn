# Final Frozen Z-903 Detection Accuracy Metrics

Test-only evaluation of the existing frozen checkpoint. Missing labels were excluded from every denominator. No thresholds or model parameters were changed.

| Element | Known | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zn | 352 | 103 | 169 | 40 | 40 | 0.7727 | 0.7203 | 0.7203 | 0.7203 | 0.8232 |
| Mn | 368 | 260 | 49 | 49 | 10 | 0.8397 | 0.8414 | 0.9630 | 0.8981 | 0.9698 |
| Cd | 44 | 23 | 11 | 9 | 1 | 0.7727 | 0.7188 | 0.9583 | 0.8214 | 0.8947 |
| Mg | 386 | 160 | 204 | 11 | 11 | 0.9430 | 0.9357 | 0.9357 | 0.9357 | 0.9872 |
| Cu | 114 | 55 | 31 | 21 | 7 | 0.7544 | 0.7237 | 0.8871 | 0.7971 | 0.8788 |
| Pb | 321 | 117 | 153 | 36 | 15 | 0.8411 | 0.7647 | 0.8864 | 0.8211 | 0.8824 |

Overall label accuracy = 84.23%
Macro accuracy = 82.06%
Macro balanced accuracy = 79.70%
Micro precision = 81.22%
Micro recall = 89.53%
Micro F1 = 85.17%
Exact-match accuracy = 25.93% (27 fully labelled test spectra)

## Balanced Accuracy

| Element | Balanced accuracy | Specificity | Sensitivity |
|---|---:|---:|---:|
| Zn | 0.7644 | 0.8086 | 0.7203 |
| Mn | 0.7315 | 0.5000 | 0.9630 |
| Cd | 0.7542 | 0.5500 | 0.9583 |
| Mg | 0.9423 | 0.9488 | 0.9357 |
| Cu | 0.7416 | 0.5962 | 0.8871 |
| Pb | 0.8479 | 0.8095 | 0.8864 |
