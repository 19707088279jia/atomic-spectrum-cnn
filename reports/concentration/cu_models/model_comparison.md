# Cu Supported-Range (1-500 ppm) Model Comparison

Train/validation only. TEST was never opened by this script. See `model_comparison.json` for full grids and per-model band metrics.

## Winner

- Model: cnn
- Reason: Selected 'cnn' by lowest validation MAE (25.4039 ppm) among candidates with a positive validation Pearson correlation. R2=0.2184, Pearson=0.5127, Spearman=0.6683.

## Summary

| Model | MAE | RMSE | Median AE | R2 | Pearson | Spearman | Bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| median_baseline | 36.3737 | 57.3250 | 25.0000 | -0.0837 | nan | nan | -15.9293 |
| ridge | 30.0857 | 53.9406 | 19.9253 | 0.0405 | 0.3166 | 0.6425 | 1.3120 |
| pls | 27.8775 | 53.1520 | 13.4639 | 0.0684 | 0.3904 | 0.5527 | -15.7856 |
| cnn | 25.4039 | 48.6850 | 13.4865 | 0.2184 | 0.5127 | 0.6683 | -7.7263 |

See `range_performance.csv` for per-concentration-band metrics for every model.
