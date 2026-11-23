# Fe Reference Data Audit

Reference source: the Z-903 workbook column `Fe2O3*`, reported as wt% Fe2O3-equivalent. This analysis does not reinterpret that oxide-equivalent value as elemental Fe ppm.

## Overall

| Measure | Result |
| --- | ---: |
| Total spectra | 2567 |
| Known Fe references | 2556 |
| Missing Fe references | 11 |
| Minimum | 0.0300 wt% |
| Median | 8.5300 wt% |
| Mean | 8.1473 wt% |
| 95th percentile | 15.4500 wt% |
| Maximum | 63.9000 wt% |
| Zero values | 0 |
| Positive values | 2556 |

## Frozen Split Distributions

| Split | Spectra | Known | Missing | Min | Median | Mean | P95 | Max | Zero | Positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1803 | 1794 | 9 | 0.0300 | 8.6350 | 8.1506 | 15.4435 | 63.9000 | 0 | 1794 |
| val | 377 | 376 | 1 | 0.1200 | 8.2400 | 8.0922 | 15.3800 | 17.4000 | 0 | 376 |
| test | 387 | 386 | 1 | 0.1000 | 8.1900 | 8.1857 | 15.5100 | 18.8900 | 0 | 386 |

Classification is not supported by this reference set: all known values are positive and there are no zero-reference spectra. Any positive cutoff would be an arbitrary concentration bin, not presence/absence or a validated detection limit. Modeling therefore proceeds as quantitative regression only.

Concentration-bin counts are saved in `concentration_distribution.csv`.
