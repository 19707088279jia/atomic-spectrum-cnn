# Cu Concentration Audit (Phase 1)

This is a label-distribution and diagnostic-reuse audit only. No Cu model was trained or retrained, the six-element detection CNN was not modified, and the existing 25 ppm exploratory Cu detection threshold was not changed.

## Data Source and Matching

- Metadata workbook: `data\metadata\libs_metadata.xlsx`
- Metadata sheet: `Metadata_COMPS_PDS_230124`
- Cu metadata column: `Cu`
- Cu unit: ppm.
- Matching: normalized complete `target_id` to normalized `PELLET NAME`, reusing the Z-903 training manifest helpers.
- Labels are direct values from the Cu composition metadata field; not inferred from filenames, paths, plots, or model outputs.

## Part A: Label Coverage

- Total Z-903 spectra: 2567
- Known Cu labels: 733
- Missing Cu labels: 1834
- Cu = 0 ppm: 17
- Cu > 0 ppm: 716

### All Known Cu Labels (ppm)

| Statistic | Value |
| --- | ---: |
| count | 733 |
| minimum | 0.0 |
| maximum | 10000.0 |
| mean | 161.04229195088678 |
| median | 31.0 |
| standard deviation | 927.4682205453273 |
| percentile 25 | 6.0 |
| percentile 75 | 64.0 |
| percentile 90 | 101.80000000000007 |
| percentile 95 | 185.39999999999918 |
| percentile 99 | 4881.479999999998 |

### Cu > 0 Labels Only (ppm)

| Statistic | Value |
| --- | ---: |
| count | 716 |
| minimum | 1.0 |
| maximum | 10000.0 |
| mean | 164.8659217877095 |
| median | 31.5 |
| standard deviation | 938.0928928971715 |
| percentile 25 | 6.0 |
| percentile 75 | 65.0 |
| percentile 90 | 103.5 |
| percentile 95 | 211.5 |
| percentile 99 | 4887.6 |

## Part B: Frozen Split Support

| Split | Total spectra | Known | Zero | Positive |
| --- | ---: | ---: | ---: | ---: |
| train | 1803 | 516 | 11 | 505 |
| val | 377 | 103 | 3 | 100 |
| test | 387 | 114 | 3 | 111 |

- Target/group leakage across frozen splits: False
- Splits were not modified; test is reported only for descriptive counts and distribution, never for model selection or tuning.

## Part C: Cu Concentration Distribution

| Range | All known | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| 0 ppm | 17 | 11 | 3 | 3 |
| >0-1 ppm | 26 | 16 | 3 | 7 |
| >1-10 ppm | 199 | 138 | 32 | 29 |
| >10-25 ppm | 101 | 78 | 10 | 13 |
| >25-100 ppm | 314 | 214 | 46 | 54 |
| >100-500 ppm | 54 | 41 | 8 | 5 |
| >500-1000 ppm | 9 | 7 | 1 | 1 |
| >1000-5000 ppm | 7 | 5 | 0 | 2 |
| >5000 ppm | 6 | 6 | 0 | 0 |

### Positive-Only Quantile Ranges (all known)

| Range | Samples |
| --- | ---: |
| 1-6 ppm | 164 |
| 6-31.5 ppm | 194 |
| 31.5-65 ppm | 178 |
| 65-10000 ppm | 180 |

### High-Concentration Tail Inspection

- Known Cu labels above the existing 25 ppm threshold: 390 of 733.
- Maximum observed known Cu concentration: 10000.0 ppm.
- See `cu_tail_distribution.png` and `cu_sorted_distribution.png` for the shape of the tail.

## Part D: Existing Cu Regression Failure (Reused Diagnostics, Not Retrained)

- Source: `atomic-spectrum-v2/reports/concentration/failure_analysis.md`
- Checkpoint examined (not modified): `atomic-spectrum-v2/models/z903_concentration_best.pt (not modified, not retrained)`
- Validation truth range: [0, 896] ppm
- Validation prediction range: [2.357, 93.58] ppm
- Test truth range: [0, 4893] ppm
- Test prediction range: [0.6974, 122.3] ppm
- Validation Pearson / Spearman: 0.1693 / 0.6169
- Test Pearson / Spearman: -0.0881 / 0.648
- Validation top-10%/top-5% share of SSE: 96.12% / 93.57%
- Test top-10%/top-5% share of SSE: 99.89% / 99.84%
- Diagnosis: Extreme Cu concentrations dominate squared error (96-100% of validation SSE, 99.9% of test SSE from the top decile). Prediction ranges are far narrower than truth ranges (regression-to-the-mean). Pearson correlation is weak or negative while Spearman is moderate, indicating partial rank ordering without magnitude calibration. The prior report recommended deferring a full-range Cu regression model or redesigning it, citing sparse high-concentration labels and weak linear calibration as the primary observed issues; it did not attribute the failure to matrix shortcuts.
- Primary associated issue: sparse high-concentration data combined with regression-to-the-mean

## Part E: Recommended Cu Strategy

- Recommended strategy: C. Dedicated supported-range Cu regression + out-of-range warning
- Reason: Train/validation labels give adequate support (at least 10 train and 5 validation known labels per bin) only within the 1 ppm to 500 ppm range, but the overall concentration distribution has a long, sparse high-concentration tail (P99/median ratio 157.5) that the existing regression CNN failed to calibrate (weak/negative Pearson, moderate Spearman, regression-to-the-mean, and >90% of squared error from the top decile in both validation and test). A model restricted to the supported range with an explicit out-of-range warning is justified by this evidence; a single full-range regression is not.
- Candidate supported concentration range (TRAIN/VALIDATION only, TEST excluded): 1 ppm to 500 ppm

## Test Set Status

TEST WAS NOT USED FOR MODEL PERFORMANCE OR MODEL SELECTION. It was used only for descriptive label counts and distribution reporting, and only train/validation define the candidate supported range.
