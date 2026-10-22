#!/usr/bin/env python3
"""Render figures from generated result tables."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def model_plot(metric: str, filename: str, title: str) -> None:
    table = pd.read_csv(TABLES / "study_level_holdout_results.csv")
    table = table.sort_values(metric)
    ax = table.plot.barh(x="model", y=metric, legend=False, color="#315f86", figsize=(8, 5))
    ax.set_title(title); ax.set_xlabel(metric.replace("_", " ").title()); ax.set_xlim(0, 1); ax.set_ylabel("")
    plt.tight_layout(); plt.savefig(FIGURES / filename, dpi=180); plt.close()


model_plot("accuracy", "01_study_holdout_accuracy.png", "Study-level holdout accuracy")
sensitivity = pd.read_csv(TABLES / "normative_sensitivity.csv")
for split_name, part in sensitivity.groupby("split"):
    plt.plot(part["k"], part["accuracy"], label=split_name)
plt.xscale("log"); plt.xlabel("k"); plt.ylabel("Accuracy"); plt.title("Hyperbolic benchmark sensitivity")
plt.legend(); plt.tight_layout(); plt.savefig(FIGURES / "03_normative_sensitivity.png", dpi=180); plt.close()

rt = pd.read_csv(TABLES / "rt_difficulty_summary.csv")
ax = rt.plot.bar(x="difficulty_quintile", y="median_rt_seconds", legend=False, color="#687f4f")
ax.set_xlabel("Difficulty quintile"); ax.set_ylabel("Median RT (seconds)")
plt.tight_layout(); plt.savefig(FIGURES / "04_rt_difficulty.png", dpi=180); plt.close()

for source, column, filename, title in [
    ("logistic_coefficients.csv", "coefficient", "05_logistic_coefficients.png", "Largest logistic coefficients"),
]:
    table = pd.read_csv(TABLES / source).head(15).sort_values(column)
    table["feature"] = (table["feature"].str.replace(r"^(num__|cat__)", "", regex=True)
                        .str.replace("_", " ").str.title())
    ax = table.plot.barh(x="feature", y=column, legend=False, color="#315f86", figsize=(8, 6))
    ax.set_title(title); plt.tight_layout(); plt.savefig(FIGURES / filename, dpi=180); plt.close()

print(f"Rendered 4 figures in {FIGURES}")
