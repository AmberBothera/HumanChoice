"""Model definitions and metric helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def pipeline(model_name: str, numeric: list[str], categorical: list[str], seed: int) -> Pipeline:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    prep = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])
    if model_name == "logistic":
        estimator = LogisticRegression(max_iter=1000, random_state=seed)
    elif model_name == "tree":
        estimator = DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("preprocess", prep), ("model", estimator)])


def metrics(y_true: pd.Series, probability) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y_true, prediction)}
