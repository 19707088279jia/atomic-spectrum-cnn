# Final Z-903 Physics-Constrained Round

This was the final requested model-development round. The historical test set was not used for line selection, region merging, architecture selection, threshold tuning, or checkpoint selection. No source NASA CSV files were modified.

## Region Reduction

| Target | Original selected lines | Merged independent regions |
| --- | ---: | ---: |
| Zn | 25 | 14 |
| Mn | 252 | 52 |
| Cd | 11 | 9 |
| Mg | 26 | 13 |
| Cu | 238 | 47 |
| Pb | 23 | 16 |

All merged regions preserve their contained NIST transitions in `data/processed/z903_target_regions.csv`.

## Validation Comparison

| Target | Full-spectrum PR-AUC | Full-spectrum F1 | Old hard-window PR-AUC | Old hard-window F1 | Previous hybrid PR-AUC | Previous hybrid F1 | Final corrected hybrid PR-AUC | Final corrected hybrid F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zn | 0.9132 | 0.8256 | 0.8363 | 0.5644 | 0.8275 | 0.6833 | 0.7710 | 0.6820 |
| Mn | 0.9655 | 0.8998 | 0.9632 | 0.8815 | 0.9606 | 0.8905 | 0.9596 | 0.8840 |
| Cd | 0.9030 | 0.8364 | 0.8748 | 0.7500 | 0.7494 | 0.6939 | 0.7995 | 0.7083 |
| Mg | 0.9902 | 0.9578 | 0.9251 | 0.8378 | 0.9430 | 0.8281 | 0.9466 | 0.8474 |
| Cu | 0.9129 | 0.8814 | 0.6660 | 0.7006 | 0.7311 | 0.7107 | 0.8015 | 0.6789 |
| Pb | 0.9227 | 0.8440 | 0.7773 | 0.7481 | 0.7860 | 0.7510 | 0.7870 | 0.7868 |

Final corrected hybrid macro validation PR-AUC: **0.8442**. Best epoch: **18**. Training time: **50.26 seconds**.

## Final Assessment

- **Zn: SHORTCUT RISK.** The corrected regions did not recover full-spectrum performance.
- **Mn: MIXED.** Performance remains close to the full-spectrum and hard-window baselines.
- **Cd: MIXED.** The validation sample is small and the constrained result remains uncertain.
- **Mg: MIXED.** The corrected regions improve over the old hard-window result but remain below full-spectrum performance.
- **Cu: SHORTCUT RISK.** The corrected regions improve materially over the old hard-window and previous hybrid results, but remain below the full-spectrum baseline.
- **Pb: MIXED.** F1 improves over both previous constrained variants, while PR-AUC remains below the full-spectrum result.

## Limitations

The matrix-matched and branch-ablation artifacts from the previous validation-only hybrid audit remain in `outputs/z903/hybrid_eval/` and `outputs/z903/matrix_shortcut_audit/`. The corrected-region model itself was not evaluated on the historical test split and no new test claim is made. The remaining Zn/Cu shortcut risk and the limited Cd labels mean additional independent real LIBS data are required rather than another NASA-only tuning cycle.