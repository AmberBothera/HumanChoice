# Reproduced results

These values were generated locally from the supplied `all_data.csv` by the included scripts. They replace, rather than silently imitate, the undocumented metrics in the legacy packages.

## Dataset audit

- Raw trials: 1,172,644
- Raw participants: 11,852
- Studies: 100
- Eligible trials after exclusions and log-feature requirements: 1,020,231
- Eligible participants: 11,184
- Larger-later choice rate in eligible data: 55.6%

## Held-out-study evaluation

The test set contains 75,000 sampled trials from 20 studies excluded completely from training.

| Model | Accuracy |
|---|---:|
| Logistic Regression | 0.658 |
| Decision Tree | 0.701 |

The decision tree obtained the higher held-out-study accuracy.

## Participant-grouped cross-validation

On a separate 100,000-trial sample, logistic regression obtained mean accuracy 0.655 and the decision tree obtained mean accuracy 0.704. Each participant remained within one fold.

## Mathematical benchmark

Training-only calibration selected `k=0.01`. The benchmark agreed with 68.6% of held-out choices, leaving 31.4% disagreement. This is agreement with a specified mathematical rule, not a measure of rationality.

## Response time

For the 75,000 test trials, median response time was 1.91 seconds. The benchmark-closeness difficulty proxy had Spearman correlation -0.029 and Pearson correlation -0.044 with response time. These are very small associations and do not support a simple claim that more benchmark-ambiguous choices consistently took longer in this holdout sample.

## Interpretation
