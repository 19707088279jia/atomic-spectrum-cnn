# Z-903 Training Manifest Report

NASA User Guide grouping rule: each exported Z-903 CSV is one averaged 4x3-raster target-level measurement. The complete filename target ID is the primary sample and leakage group identity. Filename-family information is diagnostic only and is not used as the group ID.

## Manifest Integrity

- Total spectrum rows: 2567
- Unique spectrum files: 2567
- Matched metadata rows: 2567
- Unique target IDs: 2567
- Unique group IDs: 2567
- Possible filename families (diagnostic only): 147

## Label Counts Restricted to Matched Z-903 Spectra

| Element | Known labels | Positive | Negative | Missing | Positive % of known |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zn | 2325 | 958 | 1367 | 242 | 41.20% |
| Mn | 2459 | 1793 | 666 | 108 | 72.92% |
| Cd | 302 | 164 | 138 | 2265 | 54.30% |
| Mg | 2559 | 1141 | 1418 | 8 | 44.59% |
| Cu | 733 | 390 | 343 | 1834 | 53.21% |
| Pb | 2131 | 873 | 1258 | 436 | 40.97% |

Exploratory thresholds only; they are not validated LIBS detection limits.

## Exploratory Threshold Partitions

| Element | Threshold | Unit | Known | Positive | Negative | Missing |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Zn | 100 | ppm | 2325 | 958 | 1367 | 242 |
| Mn | 500 | ppm | 2459 | 1793 | 666 | 108 |
| Cd | 0.05 | ppm | 302 | 164 | 138 | 2265 |
| Mg | 5 | wt% MgO | 2559 | 1141 | 1418 | 8 |
| Cu | 25 | ppm | 733 | 390 | 343 | 1834 |
| Pb | 10 | ppm | 2131 | 873 | 1258 | 436 |

## Concentration Ranges

Bins are exploratory distribution bins. ppm elements use logarithmic-style bins; Mg uses MgO wt% bins.

### Zn (ppm)

| Range | Samples |
| --- | ---: |
| 0 | 4 |
| >0-0.01 | 0 |
| 0.01-0.1 | 0 |
| 0.1-1 | 0 |
| 1-10 | 122 |
| 10-100 | 1209 |
| 100-1000 | 966 |
| 1000-10000 | 15 |
| 10000-100000 | 9 |
| >=100000 | 0 |
| missing | 242 |

### Mn (ppm)

| Range | Samples |
| --- | ---: |
| 0 | 21 |
| >0-0.01 | 0 |
| 0.01-0.1 | 0 |
| 0.1-1 | 0 |
| 1-10 | 1 |
| 10-100 | 96 |
| 100-1000 | 959 |
| 1000-10000 | 1365 |
| 10000-100000 | 16 |
| >=100000 | 1 |
| missing | 108 |

### Cd (ppm)

| Range | Samples |
| --- | ---: |
| 0 | 25 |
| >0-0.01 | 0 |
| 0.01-0.1 | 189 |
| 0.1-1 | 72 |
| 1-10 | 16 |
| 10-100 | 0 |
| 100-1000 | 0 |
| 1000-10000 | 0 |
| 10000-100000 | 0 |
| >=100000 | 0 |
| missing | 2265 |

### Mg (wt% MgO)

| Range | Samples |
| --- | ---: |
| 0 | 64 |
| >0-0.01 | 0 |
| 0.01-0.1 | 231 |
| 0.1-0.5 | 294 |
| 0.5-1 | 132 |
| 1-2 | 223 |
| 2-5 | 474 |
| 5-10 | 730 |
| 10-20 | 284 |
| 20-50 | 127 |
| >=50 | 0 |
| missing | 8 |

### Cu (ppm)

| Range | Samples |
| --- | ---: |
| 0 | 17 |
| >0-0.01 | 0 |
| 0.01-0.1 | 0 |
| 0.1-1 | 0 |
| 1-10 | 213 |
| 10-100 | 419 |
| 100-1000 | 71 |
| 1000-10000 | 11 |
| 10000-100000 | 2 |
| >=100000 | 0 |
| missing | 1834 |

### Pb (ppm)

| Range | Samples |
| --- | ---: |
| 0 | 28 |
| >0-0.01 | 0 |
| 0.01-0.1 | 0 |
| 0.1-1 | 19 |
| 1-10 | 1174 |
| 10-100 | 870 |
| 100-1000 | 37 |
| 1000-10000 | 3 |
| 10000-100000 | 0 |
| >=100000 | 0 |
| missing | 436 |

## Actual Spectrum Structure

| Representative file | Columns | Rows | Wavelength range | Spacing median | Intensity columns | Replicates in file | Averaged status |
| --- | --- | ---: | --- | ---: | --- | --- | --- |
| plibs_z903_0201h.csv | wavelength, intensity | 23401 | 180-960 nm | 0.03333333 nm | intensity | no | not determinable from the two-column CSV |
| plibs_z903_metavoltine.csv | wavelength, intensity | 23401 | 180-960 nm | 0.03333333 nm | intensity | no | not determinable from the two-column CSV |
| plibs_z903_zwc.csv | wavelength, intensity | 23401 | 180-960 nm | 0.03333333 nm | intensity | no | not determinable from the two-column CSV |

All inspected CSVs contain one two-column wavelength/intensity vector with 23,401 channels and approximately 0.03333333 nm spacing. The NASA User Guide states that 12 datapoints from a 4x3 raster were averaged before export, so one CSV is treated as one averaged target-level spectrum.

## Leakage Groups

Metadata sheet: `Metadata_COMPS_PDS_230124`

| Metadata column | Role |
| --- | --- |
| PELLET NAME | metadata join key for target_id |
| Zn | Zn composition source |
| Mn | Mn composition source |
| Cd | Cd composition source |
| MgO | Mg composition source |
| Cu | Cu composition source |
| Pb | Pb composition source |

No separate material, aliquot, replicate, or group column is available in the workbook. NASA states that reference target names are unique, so trailing-letter filename families are not treated as physical replicates. `group_id` equals the complete target ID; `possible_family_id` is retained only for later investigation.

| Group size | Number of groups | Spectra |
| ---: | ---: | ---: |
| 1 | 2567 | 2567 |

## Manifest Columns

The six `_mask` columns are 1 when the corresponding composition is known and 0 when it is missing. Missing labels are not converted to negative labels; future masked binary cross entropy should ignore rows with mask 0 for that target.

## Verification

- Every manifest spectrum path exists: True
- Positive + negative + missing equals 2567 for every target: True
