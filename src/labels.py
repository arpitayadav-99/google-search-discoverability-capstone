"""
labels.py — Leakage-safe label construction.

Starter CSV mode
-----------------
label = (trend_direction == 'down')
trend_direction == 'down' means impressions fell > 20% in last-30d vs prev-30d.
Both windows are pre-export and non-overlapping.

Temporal snapshot mode
-----------------------
label = did clicks per active day fall ≥ decline_threshold in the future window
relative to the historical window?
  historical: [hist_start, cutoff)   — features window
  future:     [cutoff, future_end)   — label ONLY, never used as feature

Built on the FlyRank ML Internship dataset: https://flyrank.ai
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class LabelConfig:
    metric: str = "clicks"               # "clicks" or "impressions"
    decline_threshold: float = 0.30      # ≥30% relative drop → declining
    min_historical_activity: int = 50    # min impressions_sum to be eligible


def compute_decline_label(historical: pd.DataFrame, future: pd.DataFrame, config: LabelConfig = None) -> pd.DataFrame:
    """Compute decline label for temporal-snapshot mode.

    historical: one row per page_id; output of window aggregation for [hist_start, cutoff).
    future:     one row per page_id; output of window aggregation for [cutoff, future_end).
                Used ONLY here — never merged into features.
    """
    cfg = config or LabelConfig()
    metric_col = f"{cfg.metric}_sum"

    h = historical.set_index("page_id") if "page_id" in historical.columns else historical.copy()
    f = future.set_index("page_id")     if "page_id" in future.columns     else future.copy()
    joined = h.join(f, how="left", lsuffix="_hist", rsuffix="_future")

    hist_days   = joined.get("active_days_hist",   pd.Series(np.nan, index=joined.index))
    future_days = joined.get("active_days_future", pd.Series(np.nan, index=joined.index))
    hist_days   = hist_days.replace(0, np.nan)
    future_days = future_days.replace(0, np.nan)

    hist_metric   = joined.get(f"{metric_col}_hist",   pd.Series(0.0, index=joined.index))
    future_metric = joined.get(f"{metric_col}_future", pd.Series(np.nan, index=joined.index)).fillna(0.0)

    hist_rate   = hist_metric   / hist_days
    future_rate = future_metric / future_days.fillna(1)

    eligible = joined["impressions_sum_hist"] >= cfg.min_historical_activity

    with np.errstate(divide="ignore", invalid="ignore"):
        relative_change = ((future_rate - hist_rate) / hist_rate.replace(0, np.nan)).fillna(0.0)

    label = np.where(
        eligible.to_numpy(),
        (relative_change.to_numpy() <= -cfg.decline_threshold).astype(float),
        np.nan,
    )

    out = pd.DataFrame(
        {"eligible": eligible, "hist_rate": hist_rate, "future_rate": future_rate,
         "relative_change": relative_change, "label": label},
        index=joined.index,
    )
    out.index.name = "page_id"
    return out


def starter_csv_label_distribution(df: pd.DataFrame) -> pd.Series:
    return df["trend_direction"].value_counts()


def label_distribution(labels: pd.DataFrame) -> pd.Series:
    return labels[labels["eligible"]]["label"].value_counts(normalize=False).sort_index()


def edge_case_report(labels: pd.DataFrame) -> dict:
    return {
        "n_pages_total": int(len(labels)),
        "n_excluded_low_history": int((~labels["eligible"]).sum()),
        "n_eligible": int(labels["eligible"].sum()),
        "n_future_zero_activity": int(((labels["future_rate"]==0.0) & labels["eligible"]).sum()),
    }
