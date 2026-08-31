# Z-903 Empirical Target-Line Audit

Selection is evidence-threshold based and uses training spectra only. There is no fixed top-k, head(8), slice [:8], nlargest(8), or max_windows cap.

## Exact Rules

- Minimum known positive and negative training samples: 20 each
- Detectable local peak: local maximum within +/-1.0 nm above median + 3.0 x IQR
- Positive peak prevalence: at least 0.10 and greater than negative prevalence
- Standardized effect size: at least 0.20
- Positive/negative median ratio: at least 1.10
- Matrix interference: if nearest matrix line is within 0.25 nm, relative matrix strength must be below 0.25

## Selection Counts

| Target | Total NIST | In range | Usable measurements | Min data | Peak prevalence | Effect | Interference | Selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zn | 242 | 242 | 242 | 242 | 89 | 27 | 25 | 25 |
| Mn | 3157 | 3157 | 3157 | 3157 | 1460 | 618 | 252 | 252 |
| Cd | 164 | 164 | 164 | 164 | 59 | 26 | 11 | 11 |
| Mg | 286 | 286 | 286 | 286 | 116 | 82 | 26 | 26 |
| Cu | 2456 | 2456 | 2456 | 2456 | 882 | 367 | 238 | 238 |
| Pb | 152 | 152 | 152 | 152 | 42 | 38 | 23 | 23 |

## Top 30

Top-30 tables and score distributions are saved under `outputs/z903/empirical_line_audit/`.

## Zn/Cu/Pb Special Audit

- Zn: 25 lines selected; high-confidence candidates are not forced to eight and rejected/interfered candidates remain in the full table.
- Cu: 238 lines selected; high-confidence candidates are not forced to eight and rejected/interfered candidates remain in the full table.
- Pb: 23 lines selected; high-confidence candidates are not forced to eight and rejected/interfered candidates remain in the full table.
