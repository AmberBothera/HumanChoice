"""Eligibility rules and leakage-safe feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "paper", "subj_ident", "ss_value", "ss_time", "ll_value", "ll_time",
    "choice", "rt", "subj_excl", "trial_excl", "rt_in_full_seconds",
    "rt_data_analyzed",
]

CORE_FEATURES = ["log_reward_ratio", "log_delay_ratio", "log_reward_magnitude"]


def as_bool(series: pd.Series) -> pd.Series:
    """Normalize common Boolean encodings without treating missing as True."""
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes"})


def eligible_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply documented exclusions and requirements for logarithmic features."""
    numeric = ["ss_value", "ss_time", "ll_value", "ll_time", "choice", "rt"]
    result = frame.copy()
    for column in numeric:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    mask = (
        ~as_bool(result["subj_excl"])
        & ~as_bool(result["trial_excl"])
        & result["choice"].isin([0, 1])
        & result["ss_value"].gt(0)
        & result["ll_value"].gt(0)
        & result["ss_time"].ge(0)
        & result["ll_time"].ge(0)
    )
    result = result.loc[mask].copy()
    # The source documentation states that subj_ident is globally unique, but
    # including paper makes that assumption explicit and auditable.
    result["participant_id"] = result["paper"].astype(str) + "::" + result["subj_ident"].astype(str)
    return result


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create features using only information available before the choice."""
    result = frame.copy()
    result["log_reward_ratio"] = np.log(result["ll_value"] / result["ss_value"])
    result["log_delay_ratio"] = np.log1p(result["ll_time"]) - np.log1p(result["ss_time"])
    result["log_reward_magnitude"] = np.log1p(result["ss_value"] + result["ll_value"])
    return result


def response_time_seconds(frame: pd.DataFrame) -> pd.Series:
    """Return positive RT values; the published data dictionary defines seconds."""
    rt = pd.to_numeric(frame["rt"], errors="coerce")
    return rt.where(rt.gt(0))
