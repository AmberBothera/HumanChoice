"""Hyperbolic benchmark utilities.

This is a formal comparison rule, not a definition of rationality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def utility(value, delay, k: float):
    return np.asarray(value) / (1.0 + k * np.asarray(delay))


def predict(frame: pd.DataFrame, k: float):
    ss = utility(frame["ss_value"], frame["ss_time"], k)
    ll = utility(frame["ll_value"], frame["ll_time"], k)
    return (ll > ss).astype(int)


def difficulty(frame: pd.DataFrame, k: float):
    ss = utility(frame["ss_value"], frame["ss_time"], k)
    ll = utility(frame["ll_value"], frame["ll_time"], k)
    scale = np.maximum(np.abs(ss) + np.abs(ll), np.finfo(float).eps)
    # Near-equal benchmark utilities imply greater ambiguity/difficulty.
    return 1.0 - (np.abs(ll - ss) / scale)


def calibrate(frame: pd.DataFrame, grid) -> tuple[float, pd.DataFrame]:
    rows = []
    actual = frame["choice"].to_numpy()
    for k in grid:
        acc = float(np.mean(predict(frame, float(k)) == actual))
        rows.append({"k": float(k), "training_accuracy": acc})
    table = pd.DataFrame(rows)
    best = float(table.loc[table["training_accuracy"].idxmax(), "k"])
    return best, table
