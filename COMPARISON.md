# Package comparison and reconstruction record

## Supplied packages

`HumanChoice_Phase3A_Final` contains a README, a Word report, and six figures. `HumanChoice_GitHub_Ready` contains identical copies of that report and those figures plus a Streamlit shell, a data note, requirements, and one structure-only test.

SHA-256 comparison showed that all six figures and the Word report are byte-for-byte identical between the supplied packages. Neither package contains the analysis source, result CSV files, model artifacts, split manifest, or executable preprocessing logic referenced by its documentation.

## Reconstruction policy

This repository rebuilds the analysis from the supplied raw CSV. The legacy figures were not treated as evidence and were not copied into the reconstructed results. Generated outputs come only from `scripts/run_analysis.py`. Because the legacy packages did not record their random seed or complete hyperparameters, reconstructed metrics may differ and are not tuned to reproduce undocumented values.

## Legacy claims compared with reproduced results

| Quantity | Legacy package | Reconstructed pipeline |
|---|---:|---:|
| Raw trials | 1,172,644 | 1,172,644 |
| Raw participants | 11,852 | 11,852 |
| Studies | 100 | 100 |
| Eligible trials | 1,020,231 | 1,020,231 |
| Eligible participants | 11,184 | 11,184 |
| Benchmark k | approximately 0.015 | 0.010 |
| Benchmark agreement | approximately 0.666 | 0.686 |
| RT Spearman correlation | 0.088 | -0.029 |

The dataset audit counts reproduce exactly. The predictive and response-time results do not. These differences are expected because the supplied packages omitted the split seed, feature code, model hyperparameters, RT filtering, and complete benchmark-calibration procedure. The reconstructed values are not evidence that the legacy values are false; they show that those values cannot be reproduced from the supplied materials alone.

The reconstructed response-time proxy is explicitly defined so that larger values mean the two benchmark utilities are closer and therefore potentially harder to distinguish. The legacy report described absolute utility separation as “difficulty,” even though larger separation ordinarily suggests an easier benchmark comparison.
