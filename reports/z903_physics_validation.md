# Z-903 Physics Validation

This analysis uses only the selected robust-scaling checkpoint. Attribution profiles and occlusion regions were defined from correctly predicted positive validation samples. The test split was not used to tune attribution, line matching, or thresholds.

## Reference Lines

- Zn: NIST ASD cached normalized tables; retrieved/reference lines in 180-960 nm: 301
- Mn: NIST ASD cached normalized tables; retrieved/reference lines in 180-960 nm: 3739
- Cd: NIST ASD cached normalized tables; retrieved/reference lines in 180-960 nm: 171
- Mg: NIST ASD cached normalized tables; retrieved/reference lines in 180-960 nm: 540
- Cu: NIST ASD cached normalized tables; retrieved/reference lines in 180-960 nm: 2540
- Pb: NIST ASD cached normalized tables; retrieved/reference lines in 180-960 nm: 152

## Attribution and Line Agreement

| Element | TP validation samples | Peak regions | Within 0.1 nm | Within 0.25 nm | Within 0.5 nm | Strongest/reference line examples |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Zn | 116 | 20 | 2 | 4 | 5 | 429.2884 nm (I), 455.3270 nm (I), 518.1982 nm (I) |
| Mn | 256 | 20 | 10 | 15 | 17 | 422.3610 nm (II), 382.9680 nm (I), 393.0970 nm (II) |
| Cd | 23 | 20 | 3 | 4 | 6 | 428.5078 nm (II), 263.2190 nm (I), 515.4660 nm (I) |
| Mg | 159 | 20 | 5 | 6 | 6 | 405.4689 nm (I), 424.2445 nm (II), 383.1680 nm (I) |
| Cu | 52 | 20 | 12 | 16 | 18 | 422.5196 nm (II), 393.3268 nm (II), 518.3366 nm (II) |
| Pb | 119 | 20 | 1 | 4 | 4 | 401.9632 nm (I), 424.2140 nm (II), 288.7300 nm (II) |

Line proximity is not proof of physical use: NIST line density, unresolved transitions, arbitrary instrument response, matrix effects, and attribution aggregation can all produce apparent proximity.

## Occlusion

| Element | Mean probability change after masking top regions | Regions with probability decrease |
| --- | ---: | ---: |
| Zn | 0.003991 | 2 / 10 |
| Mn | -0.003767 | 6 / 10 |
| Cd | 0.001170 | 2 / 10 |
| Mg | -0.011973 | 8 / 10 |
| Cu | -0.000176 | 3 / 10 |
| Pb | -0.004096 | 7 / 10 |

## Shortcut and Error Checks

Attribution profile overlap is reported in `shortcut_overlap.csv`; high overlap with another target's profile is a possible co-occurrence/matrix shortcut, not a causal conclusion.

False-positive and false-negative counts, concentrations, and attribution maxima are in `error_analysis.csv` for validation and test.

## Assessment

- **Zn: WEAK / POSSIBLE SHORTCUT**; attribution/reference overlap within 0.5 nm: 25.0%; occlusion decrease rate: 20.0%.
- **Mn: MODERATE PHYSICAL EVIDENCE**; attribution/reference overlap within 0.5 nm: 85.0%; occlusion decrease rate: 60.0%.
- **Cd: WEAK / POSSIBLE SHORTCUT**; attribution/reference overlap within 0.5 nm: 30.0%; occlusion decrease rate: 20.0%. Cd has only 44 known test labels; conclusions remain uncertain.
- **Mg: MODERATE PHYSICAL EVIDENCE**; attribution/reference overlap within 0.5 nm: 30.0%; occlusion decrease rate: 80.0%.
- **Cu: WEAK / POSSIBLE SHORTCUT**; attribution/reference overlap within 0.5 nm: 90.0%; occlusion decrease rate: 30.0%.
- **Pb: WEAK / POSSIBLE SHORTCUT**; attribution/reference overlap within 0.5 nm: 20.0%; occlusion decrease rate: 70.0%.

These are interpretability associations, not validated detection claims. External real-world and cross-instrument validation remain required.
