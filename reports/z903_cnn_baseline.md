# Z-903 1D-CNN Baseline

## Dataset

- Train / validation / test: 1803 / 377 / 387
- Six partially observed outputs: Zn, Mn, Cd, Mg, Cu, Pb
- Missing labels use masks and do not contribute to loss or metrics.

## Model

Compact strided Conv1d blocks: Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d repeated three times, a fourth Conv1d block, AdaptiveAvgPool1d(1), and a six-logit linear head. Input shape is `[batch, 1, 23401]`; sigmoid is not applied inside the model.
- Trainable parameters: 40518

## Preprocessing Comparison

| Experiment | Validation macro PR-AUC | Validation macro F1 | Best epoch | Test macro PR-AUC | Test macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| z903_cnn_total_area | 0.6779 | 0.6816 | 1 | 0.6806 | 0.6815 |
| z903_cnn_robust | 0.9346 | 0.8742 | 21 | 0.9060 | 0.8323 |

Selected preprocessing by validation macro PR-AUC only: **robust**.

## Selected Validation Metrics

| Element | Known | ROC-AUC | PR-AUC | Precision | Recall | Specificity | F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zn | 341 | 0.9230 | 0.9132 | 0.8286 | 0.8227 | 0.8800 | 0.8256 | 0.8513 |
| Mn | 363 | 0.9104 | 0.9655 | 0.8421 | 0.9660 | 0.5102 | 0.8998 | 0.7381 |
| Cd | 44 | 0.8463 | 0.9030 | 0.7667 | 0.9200 | 0.6316 | 0.8364 | 0.7758 |
| Mg | 376 | 0.9915 | 0.9902 | 0.9636 | 0.9521 | 0.9713 | 0.9578 | 0.9617 |
| Cu | 103 | 0.8985 | 0.9129 | 0.8254 | 0.9455 | 0.7708 | 0.8814 | 0.8581 |
| Pb | 314 | 0.9416 | 0.9227 | 0.7677 | 0.9370 | 0.8075 | 0.8440 | 0.8722 |

## Selected Probability Thresholds

- Zn: `0.7050`
- Mn: `0.2350`
- Cd: `0.5000`
- Mg: `0.3550`
- Cu: `0.5050`
- Pb: `0.4050`

## Final Untouched Test Metrics

| Element | Known | Positive | Negative | ROC-AUC | PR-AUC | Precision | Recall | Specificity | F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zn | 352 | 143 | 209 | 0.8678 | 0.8232 | 0.7203 | 0.7203 | 0.8086 | 0.7203 | 0.7644 |
| Mn | 368 | 270 | 98 | 0.9195 | 0.9698 | 0.8414 | 0.9630 | 0.5000 | 0.8981 | 0.7315 |
| Cd | 44 | 24 | 20 | 0.8688 | 0.8947 | 0.7188 | 0.9583 | 0.5500 | 0.8214 | 0.7542 |
| Mg | 386 | 171 | 215 | 0.9890 | 0.9872 | 0.9357 | 0.9357 | 0.9488 | 0.9357 | 0.9423 |
| Cu | 114 | 62 | 52 | 0.8632 | 0.8788 | 0.7237 | 0.8871 | 0.5962 | 0.7971 | 0.7416 |
| Pb | 321 | 132 | 189 | 0.9202 | 0.8824 | 0.7647 | 0.8864 | 0.8095 | 0.8211 | 0.8479 |

## Class Weights

- Zn: positive weight `1.421365`
- Mn: positive weight `0.373609`
- Cd: positive weight `0.860870`
- Mg: positive weight `1.237858`
- Cu: positive weight `0.890110`
- Pb: positive weight `1.436482`

## Overfitting and Baseline Checks

Training loss and validation metrics are stored per epoch in each experiment's `training_history.csv`. The area-normalized experiment shows clear overfitting/instability: training loss keeps falling while validation loss becomes very large and validation macro PR-AUC deteriorates after epoch 1. The robust experiment shows milder late overfitting: training loss continues falling after the validation PR-AUC peak at epoch 21, with validation PR-AUC fluctuating slightly lower before early stopping. Cd and Cu validation metrics are less stable because they have only 44 and 103 known validation labels, respectively. No test data were used for checkpoint or preprocessing selection.
- Prevalence baseline validation macro PR-AUC: 0.5157
- Majority-class validation macro F1: 0.0000
- Area best epoch / completed epochs: 1 / 9
- Robust best epoch / completed epochs: 21 / 29
- Current weakest selected-test element by PR-AUC and F1: Zn.

## Limitations

- Labels use exploratory composition thresholds, not scientifically validated LIBS limits of detection.
- Cd has only 44 known test labels.
- Cu has substantially fewer known labels than Zn, Mn, Mg, and Pb.
- NASA Z-903 performance does not establish cross-instrument generalization.
- External real-world validation remains required.
- Total training time for both experiments: 228.59 seconds.
