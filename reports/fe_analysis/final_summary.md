# Fe Concentration Model — Final Summary

## Decision

- Chosen task: **quantitative regression only**.
- Classification is not scientifically identifiable from these labels: known values have no zeros.
- Best validation model among the compared candidates: **cnn** (selected evaluation model: **cnn**).
- Prediction target is the workbook’s `Fe2O3*` wt% reference, an Fe2O3-equivalent value; it is not reported as elemental Fe ppm.
- Ridge and PLS use the existing 936-feature pooled full-wavelength representation; the CNN uses all 23,401 channels.

## Data

- Spectra: 2567
- Usable known Fe labels: 2556
- Missing Fe labels excluded from model fitting: 11
- Fixed split train/validation/test known labels: 1794/376/386
- Training-only low/medium/high boundaries: 4.6667 and 12.0033 wt%.

## Regression Metrics

| Split | Model | N | MAE | Median AE | RMSE | R² | Bias | Pearson r | Spearman ρ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | median_baseline | 376 | 4.4919 | 4.3800 | 4.9597 | -0.0121 | 0.5428 | nan | nan |
| validation | ridge | 376 | 1.2837 | 0.9530 | 1.7512 | 0.8738 | -0.0681 | 0.9350 | 0.9117 |
| validation | pls | 376 | 1.4550 | 1.0348 | 1.9941 | 0.8364 | -0.3407 | 0.9195 | 0.8908 |
| validation | cnn | 376 | 1.0949 | 0.8377 | 1.5233 | 0.9045 | -0.0122 | 0.9511 | 0.9300 |
| test | median_baseline | 386 | 4.4235 | 4.1800 | 4.9288 | -0.0084 | 0.4493 | nan | nan |
| test | ridge | 386 | 1.4376 | 0.9606 | 2.1458 | 0.8089 | -0.1807 | 0.9011 | 0.8930 |
| test | pls | 386 | 1.5485 | 1.1347 | 2.1660 | 0.8053 | -0.4506 | 0.9036 | 0.8918 |
| test | cnn | 386 | 1.2032 | 0.9069 | 1.6823 | 0.8825 | -0.0027 | 0.9395 | 0.9316 |

## Selected-Model Concentration-Range Performance

| Split | Range | N | MAE | Median AE | RMSE | R² | Bias | Pearson r | Spearman ρ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | low | 128 | 0.8422 | 0.5532 | 1.2783 | -0.2290 | 0.4490 | 0.6216 | 0.7018 |
| validation | medium | 115 | 1.2019 | 0.9573 | 1.6282 | 0.4612 | -0.0846 | 0.8040 | 0.8142 |
| validation | high | 133 | 1.2457 | 1.0705 | 1.6417 | -0.7728 | -0.3935 | 0.3589 | 0.3706 |
| test | low | 121 | 0.9439 | 0.6385 | 1.4062 | -0.4576 | 0.6297 | 0.5987 | 0.6875 |
| test | medium | 137 | 1.2525 | 0.9398 | 1.6787 | 0.4729 | -0.1290 | 0.7836 | 0.7716 |
| test | high | 128 | 1.3955 | 1.0926 | 1.9104 | -0.7784 | -0.4652 | 0.4870 | 0.5074 |

## Error Patterns

- Selected-model low/medium/high MAE (validation): 0.8422/1.2019/1.2457 wt%.
- Selected-model low/medium/high MAE (test): 0.9439/1.2525/1.3955 wt%.
- Low-range bias is positive (0.4490/0.6297 wt%), while high-range bias is negative (-0.3935/-0.4652 wt%), indicating predictions regress toward the middle of the concentration range.
- High-range within-bin R2 is negative on validation/test (-0.7728/-0.7784); concentration-dependent performance is weaker at the high end. False-positive/false-negative counts do not apply to this regression task.

## Stability and Generalization

- Validation/test selected-model MAE relative difference: 9.9%.
- Median baseline MAE (validation/test): 4.4919/4.4235 wt%.
- Selected model MAE (validation/test): 1.0949/1.2032 wt%.
- Data-support warning: training maximum is 63.9000 wt%, while validation/test maxima are 17.4000/18.8900 wt%; the high-end training tail is not represented in validation or test.
- Streamlit extrapolation warning uses only the validation span 0.1200-17.4000 wt%; it was not selected using test data.
- Heuristic stability screen (beats median baseline on both splits, positive R² on both, and MAE shift <=25%): **passed**.
- This is an internal split-stability check, not external scientific or instrument validation.

## CNN Training

- Fixed seed: 20260816; maximum epochs: 300; early-stopping patience: 30.
- Best validation-loss epoch: 21
- Epochs executed: 51
- Best validation Huber loss in standardized log1p target space: 0.046200
- CNN validation metrics: MAE 1.0949, R² 0.9045.
- CNN independent-test metrics: MAE 1.2032, R² 0.8825.
- Training time: 458.8 seconds.
- Late-epoch overfit signal: training Huber loss 0.0624 to 0.0477 while validation loss rises from 0.0462 to 0.0585; mild late overfitting is present and the best checkpoint is retained.

## Artifact Paths

- Best CNN checkpoint: `D:\PROJECT\atomic-spectrum-cnn-copilot\atomic-spectrum-cnn-copilot\reports\fe_analysis\fe_cnn_best.pt`
- Data audit: `D:\PROJECT\atomic-spectrum-cnn-copilot\atomic-spectrum-cnn-copilot\reports\fe_analysis\data_audit.md`
- Metrics: `D:\PROJECT\atomic-spectrum-cnn-copilot\atomic-spectrum-cnn-copilot\reports\fe_analysis\metrics.json`
- Training history: `D:\PROJECT\atomic-spectrum-cnn-copilot\atomic-spectrum-cnn-copilot\reports\fe_analysis\training_history.csv`
- Range metrics: `D:\PROJECT\atomic-spectrum-cnn-copilot\atomic-spectrum-cnn-copilot\reports\fe_analysis\concentration_range_metrics.csv`
- Validation/test predictions: `D:\PROJECT\atomic-spectrum-cnn-copilot\atomic-spectrum-cnn-copilot\reports\fe_analysis\validation_predictions.csv` and `D:\PROJECT\atomic-spectrum-cnn-copilot\atomic-spectrum-cnn-copilot\reports\fe_analysis\test_predictions.csv`

No classification confusion matrix is defined because there is no reference negative class.

Streamlit integration: **integrated as an experimental estimate with a validation-only range warning**.
No detection-limit claim is made.

Selected estimator parameters: `{"architecture": "ConcentrationCNN", "preprocess": "robust median/IQR", "target_transform": "train-only log1p standardization"}`.
