# Z-903 Hybrid Physics-Informed CNN

All hybrid development metrics below use validation data only. The historical test split was not used for architecture, window, consistency-weight, or threshold selection.

## Empirical Lines

- Zn: 8 selected empirically ranked lines/windows
- Mn: 8 selected empirically ranked lines/windows
- Cd: 8 selected empirically ranked lines/windows
- Mg: 8 selected empirically ranked lines/windows
- Cu: 8 selected empirically ranked lines/windows
- Pb: 8 selected empirically ranked lines/windows

## Validation Comparison

| Variant | Macro PR-AUC | Macro F1 |
| --- | ---: | ---: |
| full-spectrum | 0.9346 | 0.8742 |
| hard-window | 0.8404 | 0.7471 |
| z903_hybrid | 0.8329 | 0.7596 |
| z903_hybrid_physics | 0.8008 | 0.7697 |

## Hybrid Per-Target Validation

| Target | Hybrid PR-AUC | Hybrid F1 | Physics-loss PR-AUC | Physics-loss F1 |
| --- | ---: | ---: | ---: | ---: |
| Zn | 0.8275 | 0.6833 | 0.8244 | 0.7209 |
| Mn | 0.9606 | 0.8905 | 0.9526 | 0.8861 |
| Cd | 0.7494 | 0.6939 | 0.6206 | 0.7200 |
| Mg | 0.9430 | 0.8281 | 0.9380 | 0.8564 |
| Cu | 0.7311 | 0.7107 | 0.6924 | 0.7048 |
| Pb | 0.7860 | 0.7510 | 0.7770 | 0.7302 |

## Branch Ablations

Global-disabled and line-disabled PR-AUC/F1 are in `outputs/z903/hybrid_eval/ablation_metrics.csv`.

## Target-Branch Occlusion

For each hybrid variant, zeroing one target branch was evaluated against all six outputs. Full results are in the variant occlusion CSVs.

## Matrix-Matched Context

The full-spectrum matrix-matched validation PR-AUC/F1 are retained from the prior audit; the hybrid matrix-matched subset should be interpreted from the same validation-only composition matching procedure before claiming reduced matrix dependence.

## Assessment

The hybrid preserves full-spectrum context while limiting each line branch to its own empirical windows. The first validation result is below the full-spectrum baseline, so target-line constraints have not yet improved predictive performance. Zn and Cu remain the most concerning targets; Cd remains uncertain because of its small known-label count.
