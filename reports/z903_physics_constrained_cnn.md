# Z-903 Physics-Constrained CNN

The saved constrained checkpoint was evaluated after training; no retraining or historical test-set tuning was performed.

## Selected Windows

- Zn: 8 windows; branch length 240
- Mn: 8 windows; branch length 240
- Cd: 8 windows; branch length 240
- Mg: 8 windows; branch length 240
- Cu: 8 windows; branch length 240
- Pb: 8 windows; branch length 240

## Validation Comparison

| Target | Full PR-AUC | Full F1 | Window PR-AUC | Window F1 |
| --- | ---: | ---: | ---: | ---: |
| Zn | 0.9132 | 0.8256 | 0.8363 | 0.5644 |
| Mn | 0.9655 | 0.8998 | 0.9632 | 0.8815 |
| Cd | 0.9030 | 0.8364 | 0.8748 | 0.7500 |
| Mg | 0.9902 | 0.9578 | 0.9251 | 0.8378 |
| Cu | 0.9129 | 0.8814 | 0.6660 | 0.7006 |
| Pb | 0.9227 | 0.8440 | 0.7773 | 0.7481 |

## Matrix-Matched Validation

| Target | Full matched PR-AUC | Window matched PR-AUC | Full matched F1 | Window matched F1 |
| --- | ---: | ---: | ---: | ---: |
| Zn | 0.9174 | 0.8523 | 0.8286 | 0.5644 |
| Mn | 0.9292 | 0.9318 | 0.8017 | 0.7946 |
| Cd | 0.8702 | 0.8611 | 0.7907 | 0.7027 |
| Mg | 0.9904 | 0.9257 | 0.9576 | 0.8378 |
| Cu | 0.9134 | 0.6556 | 0.8654 | 0.6713 |
| Pb | 0.9228 | 0.7893 | 0.8459 | 0.7510 |

## Target-Window Occlusion

| Target | Mean target probability change after window occlusion |
| --- | ---: |
| Zn | 0.000000 |
| Mn | 0.000000 |
| Cd | -0.006778 |
| Mg | 0.000000 |
| Cu | 0.000000 |
| Pb | 0.000000 |

## Assessment

Window branches enforce target-specific inputs, but they do not prove physical specificity. Compare matrix-matched changes and target-window occlusion against the full-spectrum baseline before scientific interpretation.

- Historical test data were not used for redesign selection or tuning.
- Labels remain exploratory composition thresholds rather than validated detection limits.
