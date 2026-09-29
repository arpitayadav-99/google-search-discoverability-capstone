"""
models.py — Rule-based baseline + three ML candidates.

Scaler fit ONLY on training data.
Model selection uses validation metrics only.
Test set is never used for selection.

Built on the FlyRank ML Internship dataset: https://flyrank.ai
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def baseline_decline_score_starter(features: pd.DataFrame) -> pd.Series:
    """Non-ML heuristic for starter CSV features. Higher score → higher decline priority.
    Components:
      - impression_window_ratio < 0.80 → declining momentum (weight 0.50)
      - avg_position > 20 → weak visibility  (weight 0.30)
      - ctr < 1.0%        → low CTR          (weight 0.20)
    """
    ratio_c = (1.0 - features.get("impression_window_ratio",
                pd.Series(1.0, index=features.index)).fillna(1.0)).clip(0, 1)
    pos = features.get("avg_position", pd.Series(0.0, index=features.index)).fillna(0.0)
    pos_c = (pos / 50.0).clip(0, 1)
    ctr = features.get("ctr", pd.Series(0.0, index=features.index)).fillna(0.0)
    ctr_c = (1.0 - (ctr / 5.0).clip(0, 1))
    return (ratio_c * 0.5 + pos_c * 0.3 + ctr_c * 0.2).clip(0.0, 1.0).rename("baseline_score")


def baseline_decline_score_temporal(features: pd.DataFrame) -> pd.Series:
    """Non-ML heuristic for temporal-snapshot features."""
    click_c      = (-features.get("click_trend",      pd.Series(0.0, index=features.index))).clip(-1, 1)
    impression_c = (-features.get("impression_trend", pd.Series(0.0, index=features.index))).clip(-1, 1)
    pos_c        = (features.get("position_trend",    pd.Series(0.0, index=features.index)) / 5.0).clip(-1, 1)
    raw = click_c + impression_c + pos_c
    return ((raw + 3.0) / 6.0).clip(0.0, 1.0).rename("baseline_score")


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    s = StandardScaler()
    s.fit(X_train.fillna(0))
    return s


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(scaler.transform(X.fillna(0)), index=X.index, columns=X.columns)


@dataclass
class TrainedModel:
    name: str
    estimator: object
    scaler: Optional[StandardScaler]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_in = apply_scaler(self.scaler, X) if self.scaler is not None else X.fillna(0)
        return self.estimator.predict_proba(X_in)[:, 1]


def train_logistic_regression(X_train, y_train) -> TrainedModel:
    scaler = fit_scaler(X_train)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(apply_scaler(scaler, X_train), y_train)
    return TrainedModel(name="logistic_regression", estimator=clf, scaler=scaler)


def train_random_forest(X_train, y_train) -> TrainedModel:
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train.fillna(0), y_train)
    return TrainedModel(name="random_forest", estimator=clf, scaler=None)


def train_hist_gradient_boosting(X_train, y_train) -> TrainedModel:
    clf = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.06, max_iter=300, random_state=RANDOM_SEED)
    clf.fit(X_train.fillna(0), y_train)
    return TrainedModel(name="hist_gradient_boosting", estimator=clf, scaler=None)


def train_all_candidates(X_train, y_train) -> list:
    return [
        train_logistic_regression(X_train, y_train),
        train_random_forest(X_train, y_train),
        train_hist_gradient_boosting(X_train, y_train),
    ]


def coefficient_table(model: TrainedModel, feature_names) -> pd.DataFrame:
    est = model.estimator
    if hasattr(est, "coef_"):
        vals, kind = est.coef_[0], "coefficient"
    elif hasattr(est, "feature_importances_"):
        vals, kind = est.feature_importances_, "importance"
    else:
        raise ValueError(f"Model {model.name} has neither coef_ nor feature_importances_")
    df = pd.DataFrame({"feature": list(feature_names), kind: vals})
    df["abs_val"] = df[kind].abs()
    return df.sort_values("abs_val", ascending=False).drop(columns="abs_val")
