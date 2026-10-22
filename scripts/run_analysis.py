#!/usr/bin/env python3
"""Run the complete HumanChoice analysis from the source CSV."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.tree import export_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanchoice.features import CORE_FEATURES, RAW_COLUMNS, add_features, eligible_rows, response_time_seconds  # noqa: E402
from humanchoice.models import metrics, pipeline  # noqa: E402
from humanchoice.normative import calibrate, difficulty, predict  # noqa: E402

SEED = 42


def load_csv(path: Path) -> tuple[pd.DataFrame, dict]:
    chunks = []
    raw_rows = 0
    raw_participants: set[str] = set()
    raw_studies: set[str] = set()
    for chunk in pd.read_csv(path, usecols=RAW_COLUMNS, chunksize=150_000, low_memory=False):
        raw_rows += len(chunk)
        raw_studies.update(chunk["paper"].dropna().astype(str))
        raw_participants.update((chunk["paper"].astype(str) + "::" + chunk["subj_ident"].astype(str)).dropna())
        chunks.append(eligible_rows(chunk))
    data = add_features(pd.concat(chunks, ignore_index=True))
    audit = {
        "raw_trials": raw_rows,
        "raw_participants": len(raw_participants),
        "raw_studies": len(raw_studies),
        "eligible_trials": len(data),
        "eligible_participants": int(data["participant_id"].nunique()),
        "eligible_studies": int(data["paper"].nunique()),
        "choice_ll_rate": float(data["choice"].mean()),
    }
    return data, audit


def sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return frame.sample(n=min(n, len(frame)), random_state=seed).reset_index(drop=True)


def feature_table(fitted, attribute: str) -> pd.DataFrame:
    names = fitted.named_steps["preprocess"].get_feature_names_out()
    values = getattr(fitted.named_steps["model"], attribute)
    if getattr(values, "ndim", 1) > 1:
        values = values[0]
    return pd.DataFrame({"feature": names, "value": values}).sort_values("value", key=abs, ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=250_000)
    parser.add_argument("--test-size", type=int, default=75_000)
    parser.add_argument("--cv-size", type=int, default=100_000)
    args = parser.parse_args()

    tables = ROOT / "results" / "tables"
    figures = ROOT / "results" / "figures"
    artifacts = ROOT / "artifacts"
    for directory in (tables, figures, artifacts):
        directory.mkdir(parents=True, exist_ok=True)

    data, audit = load_csv(args.data)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, test_idx = next(splitter.split(data, groups=data["paper"]))
    train_full, test_full = data.iloc[train_idx], data.iloc[test_idx]
    train, test = sample(train_full, args.train_size, SEED), sample(test_full, args.test_size, SEED + 1)
    cv_data = sample(data, args.cv_size, SEED + 2)
    split = {
        "random_seed": SEED,
        "training_studies": sorted(train_full["paper"].unique().tolist()),
        "test_studies": sorted(test_full["paper"].unique().tolist()),
        "training_trials_available": len(train_full),
        "test_trials_available": len(test_full),
        "training_trials_sampled": len(train),
        "test_trials_sampled": len(test),
    }
    (tables / "audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    (tables / "split_manifest.json").write_text(json.dumps(split, indent=2) + "\n")
    del data, train_full, test_full, train_idx, test_idx
    gc.collect()

    definitions = [
        ("Logistic Regression", "logistic", CORE_FEATURES, []),
        ("Decision Tree", "tree", CORE_FEATURES, []),
    ]
    rows = []
    for label, kind, numeric, categorical in definitions:
        model = pipeline(kind, numeric, categorical, SEED)
        columns = numeric + categorical
        model.fit(train[columns], train["choice"])
        probability = model.predict_proba(test[columns])[:, 1]
        rows.append({"model": label, **metrics(test["choice"], probability)})
        if label == "Logistic Regression":
            feature_table(model, "coef_").rename(columns={"value": "coefficient"}).to_csv(
                tables / "logistic_coefficients.csv", index=False
            )
        elif label == "Decision Tree":
            tree_names = model.named_steps["preprocess"].get_feature_names_out().tolist()
            tree_text = export_text(model.named_steps["model"], feature_names=tree_names, max_depth=4)
            (tables / "decision_tree_paths.txt").write_text(tree_text)
        del model
        gc.collect()
    results = pd.DataFrame(rows)
    results.to_csv(tables / "study_level_holdout_results.csv", index=False)

    grid = np.unique(np.r_[np.logspace(-4, 0, 101), 0.015, 0.0164577074175946])
    k, calibration = calibrate(train, grid)
    sensitivity = calibration.rename(columns={"training_accuracy": "accuracy"})
    sensitivity["split"] = "training"
    test_sensitivity = pd.DataFrame({
        "k": grid,
        "accuracy": [float(np.mean(predict(test, float(x)) == test["choice"].to_numpy())) for x in grid],
        "split": "test",
    })
    sensitivity = pd.concat([sensitivity, test_sensitivity], ignore_index=True)
    sensitivity.to_csv(tables / "normative_sensitivity.csv", index=False)
    benchmark = {"k": k, "test_accuracy": float(np.mean(predict(test, k) == test["choice"].to_numpy()))}
    (tables / "normative_params.json").write_text(json.dumps(benchmark, indent=2) + "\n")

    rt = response_time_seconds(test)
    rt_frame = test.loc[rt.notna(), ["choice"]].copy()
    rt_frame["rt_seconds"] = rt.dropna()
    rt_frame["difficulty"] = difficulty(test.loc[rt.notna()], k)
    rt_frame["difficulty_quintile"] = pd.qcut(rt_frame["difficulty"], 5, labels=False, duplicates="drop") + 1
    rt_summary = rt_frame.groupby("difficulty_quintile", as_index=False).agg(
        trials=("rt_seconds", "size"), median_rt_seconds=("rt_seconds", "median"), mean_difficulty=("difficulty", "mean")
    )
    rt_summary.to_csv(tables / "rt_difficulty_summary.csv", index=False)
    correlations = {
        "n": len(rt_frame), "median_rt_seconds": float(rt_frame["rt_seconds"].median()),
        "spearman_rho": float(spearmanr(rt_frame["difficulty"], rt_frame["rt_seconds"]).statistic),
        "pearson_r": float(pearsonr(rt_frame["difficulty"], rt_frame["rt_seconds"]).statistic),
    }
    (tables / "rt_correlations.json").write_text(json.dumps(correlations, indent=2) + "\n")

    cv_rows = []
    folds = GroupKFold(n_splits=5)
    for label, kind in [("Logistic Regression", "logistic"), ("Decision Tree", "tree")]:
        model = pipeline(kind, CORE_FEATURES, [], SEED)
        scores = cross_validate(
            model, cv_data[CORE_FEATURES], cv_data["choice"],
            groups=cv_data["participant_id"], cv=folds,
            scoring="accuracy", n_jobs=-1,
        )
        cv_rows.append({
            "model": label, "accuracy_mean": scores["test_score"].mean(),
            "accuracy_std": scores["test_score"].std(),
        })
    pd.DataFrame(cv_rows).to_csv(tables / "participant_grouped_cv.csv", index=False)

    disagreement = (predict(test, k) != test["choice"].to_numpy()).astype(int)
    pd.DataFrame([{"benchmark_k": k, "disagreement_rate": disagreement.mean(), "trials": len(test)}]).to_csv(
        tables / "benchmark_disagreement.csv", index=False
    )
    print(json.dumps({"audit": audit, "benchmark": benchmark, "models": rows}, indent=2))


if __name__ == "__main__":
    main()
