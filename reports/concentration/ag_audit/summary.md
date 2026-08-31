# Ag Concentration Audit

## Data Source and Matching

- Metadata workbook: `data\metadata\libs_metadata.xlsx`
- Metadata sheet: `Metadata_COMPS_PDS_230124`
- Ag metadata column: `Ag`
- Ag unit: ppm (explicitly recorded in the workbook units row).
- Matching: normalized complete `target_id` to normalized `PELLET NAME`, using the same helpers as the Z-903 training manifest.
- Labels are direct values from the Ag composition column; no filename, directory, plot, or model inference was used.
- The workbook supplies composition/reference metadata, but this audit alone cannot establish the laboratory measurement method or uncertainty; consult the source documentation before scientific interpretation.

## Label Coverage

- Total Z-903 spectra: 2567
- Known Ag labels: 328
- Missing Ag labels: 2239
- Ag = 0 ppm: 64
- Ag > 0 ppm: 264

## Frozen Split Counts

| Split | Spectra | Known Ag labels |
| --- | ---: | ---: |
| train | 1803 | 232 |
| val | 377 | 46 |
| test | 387 | 50 |

- Target/sample leakage across frozen splits: False

## All Known Ag Labels (ppm)

| Statistic | Value |
| --- | ---: |
| count | 328 |
| minimum | 0.0 |
| maximum | 14.0 |
| mean | 0.19118902439024388 |
| median | 0.045 |
| standard deviation | 1.0424131558082446 |
| percentile 25 | 0.01 |
| percentile 75 | 0.0625 |
| percentile 90 | 0.4720000000000005 |
| percentile 95 | 0.5 |
| percentile 99 | 1.1 |

## Ag > 0 Labels Only (ppm)

| Statistic | Value |
| --- | ---: |
| count | 264 |
| minimum | 0.01 |
| maximum | 14.0 |
| mean | 0.2375378787878788 |
| median | 0.05 |
| standard deviation | 1.157583525423373 |
| percentile 25 | 0.02 |
| percentile 75 | 0.1 |
| percentile 90 | 0.5 |
| percentile 95 | 0.5 |
| percentile 99 | 1.91400000000001 |

## Fixed Concentration Ranges

| Range | Samples |
| --- | ---: |
| 0 ppm | 64 |
| >0-1 ppm | 255 |
| >1-10 ppm | 7 |
| >10-100 ppm | 2 |
| >100-1000 ppm | 0 |
| >1000 ppm | 0 |

## Positive-Only Quantile Ranges

| Range | Samples |
| --- | ---: |
| 0.01-0.02 ppm | 20 |
| 0.02-0.05 ppm | 80 |
| 0.05-0.1 ppm | 97 |
| 0.1-14 ppm | 67 |

## Recommendation

- Severe long-tail distribution: True
- Recommended strategy: B. Two-stage model
- Reason: Known labels include both zero and positive Ag values, and the observed distribution is strongly right-skewed; separate presence and positive-concentration tasks should be validated without touching the frozen test set.

This is a label-distribution audit only. The frozen test split was used only for counts and descriptive distribution reporting; it was not used for model selection, tuning, or training.
