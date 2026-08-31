# Z-903 Archive Reconciliation and Splits

## NASA PDS Archive Reconciliation

- Official inventory URL: https://pds-geosciences.wustl.edu/speclib/urn-nasa-pds-libs_reference_database/data_plibs_z903/collection_data_plibs_z903_inventory.csv
- Inventory rows: 2567
- Primary Z-903 products / represented spectral CSV target IDs: 2567
- Downloaded local spectral CSVs: 2567
- Inventory target IDs missing locally: 0
- Local target IDs absent from inventory: 0

The current machine-readable inventory is the authoritative archive count used here. The User Guide's 2,686 figure is not silently substituted for the current inventory count. A difference is therefore reported as a version/archive-accounting discrepancy unless the inventory exposes a more specific reason.


## Split Method

- Seed: 20260816
- Split fractions: approximately 70% train, 15% validation, 15% test
- Assignment unit: complete target_id / group_id
- Missing labels remain missing; exploratory positive/negative counts below use only known values.
- Signature-stratified deterministic assignment preserves the joint positive/negative/missing patterns as far as bucket sizes allow.

## train

Rows: 1803

| Element | Known | Positive | Negative | Missing |
| --- | ---: | ---: | ---: | ---: |
| Zn | 1632 | 674 | 958 | 171 |
| Mn | 1728 | 1258 | 470 | 75 |
| Cd | 214 | 115 | 99 | 1589 |
| Mg | 1797 | 803 | 994 | 6 |
| Cu | 516 | 273 | 243 | 1287 |
| Pb | 1496 | 614 | 882 | 307 |

## val

Rows: 377

| Element | Known | Positive | Negative | Missing |
| --- | ---: | ---: | ---: | ---: |
| Zn | 341 | 141 | 200 | 36 |
| Mn | 363 | 265 | 98 | 14 |
| Cd | 44 | 25 | 19 | 333 |
| Mg | 376 | 167 | 209 | 1 |
| Cu | 103 | 55 | 48 | 274 |
| Pb | 314 | 127 | 187 | 63 |

## test

Rows: 387

| Element | Known | Positive | Negative | Missing |
| --- | ---: | ---: | ---: | ---: |
| Zn | 352 | 143 | 209 | 35 |
| Mn | 368 | 270 | 98 | 19 |
| Cd | 44 | 24 | 20 | 343 |
| Mg | 386 | 171 | 215 | 1 |
| Cu | 114 | 62 | 52 | 273 |
| Pb | 321 | 132 | 189 | 66 |

## Group Integrity

Unique groups across splits: 2567
Groups appearing in more than one split: 0
