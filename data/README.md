# Data

This project uses the ITC Choice-RT database described by Pongratz and Schoemann, *A large-scale dataset of choice and response-time data in intertemporal choice* (Scientific Data, 2026), DOI `10.1038/s41597-026-06947-4`.

The static dataset is hosted at <https://osf.io/3wsae/> and is licensed CC BY-NC-SA. Review the source license before redistribution. Raw data are deliberately excluded from this repository.

Place the file at `data/raw/all_data.csv`, or supply its location explicitly:

```bash
python scripts/run_analysis.py --data /path/to/all_data.csv
```

One row represents one trial. `choice=0` denotes the smaller-sooner option and `choice=1` the larger-later option. The published data dictionary defines `rt` in seconds; `rt_in_full_seconds` instead records whether times were originally recorded at whole-second resolution. The pipeline applies the source `subj_excl` and `trial_excl` flags, requires valid binary choices and nonnegative delays, and requires positive rewards for logarithmic features.
