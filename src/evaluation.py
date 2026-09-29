"""
evaluation.py — Classification metrics, ranking metrics, and model selection.

Model selection uses VALIDATION metrics only.
Test set is evaluated once for final reporting.

Built on the FlyRank ML Internship dataset: https://flyrank.ai
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def leakage_audit_table() -> pd.DataFrame:
    rows = [
        ("log_impressions_90d",        "90-day pre-export aggregate",      "YES", "Pre-export; no future data"),
        ("log_clicks_90d",             "90-day pre-export aggregate",      "YES", "Pre-export"),
        ("log_sessions_90d",           "90-day pre-export aggregate",      "YES", "Pre-export"),
        ("log_ai_sessions_90d",        "90-day pre-export aggregate",      "YES", "Pre-export"),
        ("days_with_impressions",      "90-day count",                     "YES", "Pre-export"),
        ("days_with_sessions",         "90-day count",                     "YES", "Pre-export"),
        ("content_age_days",           "content metadata",                 "YES", "Static property"),
        ("days_since_last_update",     "content metadata",                 "YES", "Pre-export"),
        ("word_count / char_count",    "content metadata",                 "YES", "Static property"),
        ("ctr",                        "90-day derived rate",              "YES", "clicks/impressions, pre-export"),
        ("avg_position",               "90-day GSC average",               "YES", "Pre-export"),
        ("engagement_rate/scroll_rate","90-day GA4 signals",               "YES", "Pre-export"),
        ("ai_traffic_pct",             "90-day derived",                   "YES", "Pre-export"),
        ("search_volume/competition",  "keyword metadata",                 "YES", "External; not label-derived"),
        ("impressions_last_30d",       "last 30-day sub-window",           "YES", "Pre-export sub-window"),
        ("impressions_prev_30d",       "prev 30-day sub-window",           "YES", "Pre-export sub-window"),
        ("impression_window_ratio",    "last_30d / prev_30d",              "YES", "Both windows pre-export; correlated with label (intentional signal)"),
        ("trend_direction",            "LABEL SOURCE",                     "NO — EXCLUDED", "IS the label. Never in feature matrix."),
        ("trend_pct",                  "LABEL SOURCE magnitude",           "NO — EXCLUDED", "Encodes trend_direction magnitude. Excluded."),
        ("StandardScaler.fit()",       "training split only",              "YES", "transform() only on val/test"),
        ("model selection",            "validation split only",            "YES", "Test never used for selection"),
    ]
    return pd.DataFrame(rows, columns=["Feature/Source","Window","Feature-Safe?","Reason"])


def classification_metrics(y_true, y_score, threshold=0.5) -> dict:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "roc_auc":   float(roc_auc_score(y_true, y_score)),
        "pr_auc":    float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
        "n_positive":int(np.asarray(y_true).sum()),
        "n_total":   int(len(y_true)),
        "base_rate": float(np.asarray(y_true).mean()),
    }


def confusion_matrix_table(y_true, y_score, threshold=0.5) -> pd.DataFrame:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=["actual_0","actual_1"], columns=["pred_0","pred_1"])


def precision_recall_lift_at_k(y_true, y_score, k_values=(10,25,50,100,200)) -> pd.DataFrame:
    order    = np.argsort(-np.asarray(y_score))
    y_sorted = np.asarray(y_true)[order]
    n_pos    = y_sorted.sum()
    base     = y_sorted.mean() if len(y_sorted) else np.nan
    rows = []
    for k in k_values:
        k_eff = min(k, len(y_sorted))
        top   = y_sorted[:k_eff]
        p     = float(top.mean()) if k_eff else np.nan
        r     = float(top.sum()/n_pos) if n_pos else np.nan
        rows.append({"k": k_eff, "precision_at_k": p, "recall_at_k": r,
                     "lift_at_k": p/base if base else np.nan})
    return pd.DataFrame(rows)


def compare_baseline_vs_model(y_true, baseline_score, model_score, k_values=(10,25,50,100,200)) -> pd.DataFrame:
    b = precision_recall_lift_at_k(y_true, baseline_score, k_values).add_prefix("baseline_")
    m = precision_recall_lift_at_k(y_true, model_score,    k_values).add_prefix("model_")
    out = pd.concat([b.drop(columns="baseline_k"), m], axis=1).rename(columns={"model_k":"k"})
    return out[["k"]+[c for c in out.columns if c!="k"]]


def evaluate_all_candidates(candidates, X_val, y_val) -> pd.DataFrame:
    rows = []
    for m in candidates:
        sc = m.predict_proba(X_val)
        mt = classification_metrics(y_val.to_numpy(), sc)
        mt["model"] = m.name
        rows.append(mt)
    df = pd.DataFrame(rows)
    model_col = df.pop("model")
    df.insert(0, "model", model_col)
    return df.sort_values("pr_auc", ascending=False).reset_index(drop=True)


def select_best_model(validation_results: list, primary_metric="pr_auc") -> str:
    """Select best model name from validation results. Test must NOT be used here."""
    df = pd.DataFrame(validation_results)
    if df.empty or primary_metric not in df.columns:
        raise ValueError(f"Empty results or missing metric '{primary_metric}'")
    best = df.sort_values(primary_metric, ascending=False).iloc[0]
    print(f"[selection] Best on validation ({primary_metric}): {best['model']} = {best[primary_metric]:.4f}")
    return str(best["model"])
