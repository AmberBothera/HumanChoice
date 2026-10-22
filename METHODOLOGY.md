# HumanChoice methodology

## Scope

HumanChoice is a secondary-data analysis of intertemporal choice. It is not the originally proposed student-collected experiment involving ten everyday scenarios. It asks how well transparent decision rules and machine-learning models predict choices between smaller-sooner and larger-later rewards.

## Outcome and features

The outcome is `choice`, where 1 represents the larger-later option. The three predictive features are the log reward ratio, log delay ratio using `log1p`, and log combined reward magnitude. Objective score, benchmark agreement, response time, participant identity, and study identity are never predictive features.

## Validation

The primary evaluation holds out 20% of complete studies with `GroupShuffleSplit(random_state=42)`. Training and test samples are drawn only after that split. A separate five-fold `GroupKFold` experiment groups by participant. `results/tables/split_manifest.json` records the exact held-out studies.

## Mathematical benchmark

The benchmark uses hyperbolic discounted utility, `U=V/(1+k*t)`. The value of `k` is selected on training data only from a documented grid. Agreement means only that the observed choice matches this rule; disagreement is not evidence of irrationality.

Response-time difficulty is represented as one minus the absolute benchmark-utility separation divided by the combined absolute utility. Values nearer one indicate alternatives that the benchmark treats as more similar. This is only a proxy for difficulty.

## Interpretation limits

The analysis is predictive and observational. Feature importance and coefficients do not establish causation. The combined database includes heterogeneous populations, currencies, procedures, and experimental settings. Response time may reflect many processes besides decision difficulty.
