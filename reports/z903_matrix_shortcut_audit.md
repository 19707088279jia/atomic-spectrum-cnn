# Z-903 Matrix Shortcut Audit

This audit uses only the robust checkpoint and validation data for matching, occlusion, null testing, and methodology decisions. No retraining, threshold changes, or split changes were performed.

## Authoritative Matrix Line References

- Ca: 223 cached NIST I/II lines
- Mg: 319 cached NIST I/II lines
- Fe: 14447 cached NIST I/II lines
- Al: 585 cached NIST I/II lines
- Si: 291 cached NIST I/II lines
- Na: 1064 cached NIST I/II lines
- K: 255 cached NIST I/II lines
- Ti: 4100 cached NIST I/II lines

## Explicit Shared-Wavelength Controls

The nearest cached NIST lines are reported in `attribution_target_matrix_lines.csv`. The three controls are Ca I near 422.67 nm, Ca II near 393.37 nm, and Mg I near 518.36 nm; exact cached values are used rather than guessed constants.

## Matrix Composition Associations

Strongest absolute presence correlations by target:
- Zn: Fe presence correlation 0.617; concentration correlation -0.059
- Mn: Fe presence correlation 0.599; concentration correlation 0.181
- Cd: Fe presence correlation 0.093; concentration correlation -0.055
- Mg: Si presence correlation -0.708; concentration correlation -0.631
- Cu: Fe presence correlation 0.145; concentration correlation 0.087
- Pb: Fe presence correlation -0.601; concentration correlation -0.104

## Matrix-Matched Validation

| Target | Original PR-AUC | Matched PR-AUC | Change | Original F1 | Matched F1 | Change | Matched samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zn | 0.9132 | 0.9174 | 0.0042 | 0.8256 | 0.8286 | 0.0029 | 282 |
| Mn | 0.9655 | 0.9292 | -0.0363 | 0.8998 | 0.8017 | -0.0981 | 192 |
| Cd | 0.9030 | 0.8702 | -0.0328 | 0.8364 | 0.7907 | -0.0457 | 38 |
| Mg | 0.9902 | 0.9904 | 0.0002 | 0.9578 | 0.9576 | -0.0003 | 332 |
| Cu | 0.9129 | 0.9134 | 0.0006 | 0.8814 | 0.8654 | -0.0160 | 96 |
| Pb | 0.9227 | 0.9228 | 0.0001 | 0.8440 | 0.8459 | 0.0019 | 252 |

## Target Versus Matrix Occlusion

| Target | Target-line probability change | Matrix-line probability change |
| --- | ---: | ---: |
| Zn | 0.001367 | 0.004605 |
| Mn | -0.006949 | -0.005468 |
| Cd | -0.002681 | 0.001710 |
| Mg | -0.006263 | -0.012577 |
| Cu | -0.000200 | -0.004230 |
| Pb | -0.000817 | -0.001326 |

## Fixed Ca/Mg Controls

The exact 3x6 control matrix is in `fixed_matrix_controls_3x6.csv`. The cached NIST lines, not the approximate labels in the request, determine the occlusion centers.

| Occluded region | Reference nm | Zn | Mn | Cd | Mg | Cu | Pb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ca I | 422.673000 | 0.003260 | 0.000227 | 0.001738 | -0.004056 | 0.000005 | -0.004605 |
| Ca II | 393.366000 | 0.000394 | -0.002591 | -0.004112 | -0.008433 | -0.004526 | -0.004612 |
| Mg I | 518.360420 | 0.011288 | -0.003120 | 0.004787 | -0.013902 | 0.002844 | 0.007560 |

## Cross-Output and Random Null Results

The full cross-output matrix effects are in `cross_output_matrix_effects.csv`. Random-window enrichment uses 100 fixed-seed windows per target. Because the combined matrix NIST line set is dense, the null overlap can saturate at 100%; in that case the null test is non-discriminating rather than evidence of physical agreement.

| Target | Observed line overlap | Random overlap | Enrichment |
| --- | ---: | ---: | ---: |
| Zn | 1.000 | 1.000 | 1.00x |
| Mn | 1.000 | 1.000 | 1.00x |
| Cd | 1.000 | 1.000 | 1.00x |
| Mg | 1.000 | 1.000 | 1.00x |
| Cu | 1.000 | 1.000 | 1.00x |
| Pb | 1.000 | 1.000 | 1.00x |

## Explicit Shared Wavelength Conclusions

- 422.6000 nm is not primarily explained by Ca in the nearest-line analysis; the nearest matrix reference is Fe I at approximately 422.5955 nm, while Ca is a farther candidate in this local neighborhood.
- 393.3333 nm is not treated as proof of Ca II solely from proximity; the nearest matrix/target context and occlusion effects are reported in the machine-readable outputs.
- 518.3333 nm is not treated as proof of Mg I solely from proximity; Mg I and Fe references are both considered, and target-versus-matrix occlusion is required.

## Assessment

- **Zn: LIKELY MATRIX SHORTCUT**; matrix-matched PR-AUC change 0.0042; target/matrix occlusion changes 0.001367/0.004605; random overlap enrichment 1.00x.
- **Mn: MIXED EVIDENCE**; matrix-matched PR-AUC change -0.0363; target/matrix occlusion changes -0.006949/-0.005468; random overlap enrichment 1.00x.
- **Cd: MIXED EVIDENCE**; matrix-matched PR-AUC change -0.0328; target/matrix occlusion changes -0.002681/0.001710; random overlap enrichment 1.00x. Cd has only 44 known test labels; this audit does not remove that uncertainty.
- **Mg: LIKELY MATRIX SHORTCUT**; matrix-matched PR-AUC change 0.0002; target/matrix occlusion changes -0.006263/-0.012577; random overlap enrichment 1.00x.
- **Cu: LIKELY MATRIX SHORTCUT**; matrix-matched PR-AUC change 0.0006; target/matrix occlusion changes -0.000200/-0.004230; random overlap enrichment 1.00x.
- **Pb: LIKELY MATRIX SHORTCUT**; matrix-matched PR-AUC change 0.0001; target/matrix occlusion changes -0.000817/-0.001326; random overlap enrichment 1.00x.
