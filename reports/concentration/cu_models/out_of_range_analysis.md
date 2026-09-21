# Cu Out-of-Range Analysis

- TRAIN known Cu labels > 500 ppm: 18
- VALIDATION known Cu labels > 500 ppm: 1
- Warning classifier feasibility threshold: at least 10 TRAIN and 5 VALIDATION known labels above 500 ppm.
- Warning model feasible: False

**Conclusion:** Insufficient data to reliably identify out-of-range high-Cu samples.


Predictions from the supported-range regression model must not be trusted for spectra flagged as out-of-range (or, absent a feasible classifier, for any spectrum suspected to exceed 500 ppm); a true high-Cu sample could otherwise be silently regressed into the 1-500 ppm supported range.

TEST was not used anywhere in this analysis.
