# HumanChoice

HumanChoice is a reproducible secondary-data analysis of choices between smaller-sooner and larger-later rewards. It compares an explicit hyperbolic decision rule with interpretable machine-learning models and evaluates whether models generalize to completely unseen studies.

This is a reconstruction of two incomplete supplied packages. See [COMPARISON.md](COMPARISON.md) for what was missing and how legacy claims were handled.

## Research question

How well can mathematical and machine-learning models predict human intertemporal choices, and which observed decision and experimental features are associated with those predictions?

## Data

The project uses the ITC Choice-RT database. The raw CSV is not committed. Dataset citation, license, and setup details are in [data/README.md](data/README.md).

## Reproduce the analysis

Use Python 3.11 or 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_analysis.py --data data/raw/all_data.csv
python scripts/render_results.py
python -m unittest discover -s tests
streamlit run app/streamlit_app.py
```

The analysis writes tables, figures, and a split manifest. The exact study split and random seed are saved in `results/tables/split_manifest.json`.

## Repository structure

```text
app/                    Streamlit dashboard
artifacts/              Generated model artifacts, excluded from Git
data/                   Dataset documentation; raw CSV excluded
results/figures/        Generated figures
results/tables/         Generated metrics and manifests
scripts/run_analysis.py End-to-end analysis entry point
src/humanchoice/        Features, models, and benchmark implementation
tests/                  Unit tests
```

## Methods

The source exclusion flags are applied before modeling. Both models use three transparent features describing reward, delay, and reward magnitude. The primary test set consists of complete held-out studies. Participant-grouped cross-validation provides a separate within-database check. See [METHODOLOGY.md](METHODOLOGY.md) for the complete design.

## Results

The completed local run verified 1,020,231 eligible trials from 11,184 participants across 100 studies. On 20 held-out studies, the decision tree achieved the highest accuracy (0.701). The training-calibrated hyperbolic benchmark selected `k=0.01` and agreed with 68.6% of held-out choices. See [RESULTS.md](RESULTS.md) and the generated tables for precise values.

The dashboard never embeds these metrics in source code; it loads only generated result files.

## Ethics and limitations

This is an observational secondary analysis. Benchmark disagreement is not irrationality, model importance is not causality, and results may not generalize beyond the studies in the database. Respect the dataset's CC BY-NC-SA terms and the licenses of its contributing studies.

## What the student should understand

Before presenting this project, the student should be able to explain the row-level data, exclusion rules, three engineered core features, study versus participant grouping, hyperbolic utility and calibration, each reported metric, overfitting controls, and why predictive associations do not establish causes.

## License note

No license has been selected for this reconstructed source code. Add one before public release only after the project owner chooses the intended terms. The source dataset has its own CC BY-NC-SA license.
