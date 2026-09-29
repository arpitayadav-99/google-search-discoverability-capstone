"""
features.py — Leakage-safe feature engineering for the starter CSV.

Key rules:
  - trend_direction and trend_pct are NEVER used as features.
  - All features are derived from the 90-day trailing aggregate (pre-export).
  - The 30-day sub-windows (last/prev) both pre-date the export cutoff.
  - Query count note: the starter CSV has no raw daily query column.
    impressions_90d is used as a volume proxy and named accordingly.

Built on the FlyRank ML Internship dataset: https://flyrank.ai
"""
from __future__ import annotations
import numpy as np
import pandas as pd

LABEL_SOURCE_COLUMNS = ["trend_direction", "trend_pct", "is_declining_label"]

FEATURE_COLUMNS = [
    # Volume (log-transformed)
    "log_impressions_90d", "log_clicks_90d", "log_sessions_90d", "log_ai_sessions_90d",
    # Activity span
    "days_with_impressions", "days_with_sessions",
    # Content properties
    "content_age_days", "days_since_last_update", "word_count", "char_count",
    # Search performance
    "ctr", "avg_position", "engagement_rate", "scroll_rate", "ai_traffic_pct",
    # Keyword context
    "search_volume", "competition", "cpc",
    # Comparison-window features (both windows pre-export)
    "impressions_last_30d", "clicks_last_30d",
    "impressions_prev_30d", "clicks_prev_30d",
    # Derived ratio
    "impression_window_ratio",
]


def build_starter_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build numeric feature table from the prepared starter CSV.

    Input:  output of data.prepare_starter_features() — indexed by content_id.
    Output: DataFrame of numeric features (categoricals handled via get_dummies in
            data.starter_feature_matrix).
    """
    if "content_id" in df.columns:
        df = df.set_index("content_id")

    # Guard against accidental label inclusion
    for col in LABEL_SOURCE_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    out = pd.DataFrame(index=df.index)

    # Volume
    out["log_impressions_90d"] = np.log1p(df.get("impressions_90d", pd.Series(0, index=df.index)).fillna(0))
    out["log_clicks_90d"]      = np.log1p(df.get("clicks_90d",      pd.Series(0, index=df.index)).fillna(0))
    out["log_sessions_90d"]    = np.log1p(df.get("sessions_90d",    pd.Series(0, index=df.index)).fillna(0))
    out["log_ai_sessions_90d"] = np.log1p(df.get("ai_sessions_90d", pd.Series(0, index=df.index)).fillna(0))

    # Activity span
    out["days_with_impressions"] = df.get("days_with_impressions", pd.Series(0, index=df.index)).fillna(0)
    out["days_with_sessions"]    = df.get("days_with_sessions",    pd.Series(0, index=df.index)).fillna(0)

    # Content
    out["content_age_days"]       = df.get("content_age_days",       pd.Series(0, index=df.index)).fillna(0)
    out["days_since_last_update"] = df.get("days_since_last_update", pd.Series(0, index=df.index)).fillna(0)
    out["word_count"] = df.get("word_count", pd.Series(0, index=df.index)).fillna(0)
    out["char_count"] = df.get("char_count", pd.Series(0, index=df.index)).fillna(0)

    # Search performance
    out["ctr"]             = df.get("ctr",             pd.Series(0.0, index=df.index)).fillna(0.0)
    out["avg_position"]    = df.get("avg_position",    pd.Series(0.0, index=df.index)).fillna(0.0)
    out["engagement_rate"] = df.get("engagement_rate", pd.Series(0.0, index=df.index)).fillna(0.0)
    out["scroll_rate"]     = df.get("scroll_rate",     pd.Series(0.0, index=df.index)).fillna(0.0)
    out["ai_traffic_pct"]  = df.get("ai_traffic_pct",  pd.Series(0.0, index=df.index)).fillna(0.0)

    # Keyword context
    out["search_volume"] = df.get("search_volume", pd.Series(0.0, index=df.index)).fillna(0.0)
    out["competition"]   = df.get("competition",   pd.Series(0.0, index=df.index)).fillna(0.0)
    out["cpc"]           = df.get("cpc",           pd.Series(0.0, index=df.index)).fillna(0.0)

    # Comparison windows
    out["impressions_last_30d"] = df.get("impressions_last_30d", pd.Series(0, index=df.index)).fillna(0)
    out["clicks_last_30d"]      = df.get("clicks_last_30d",      pd.Series(0, index=df.index)).fillna(0)
    out["impressions_prev_30d"] = df.get("impressions_prev_30d", pd.Series(0, index=df.index)).fillna(0)
    out["clicks_prev_30d"]      = df.get("clicks_prev_30d",      pd.Series(0, index=df.index)).fillna(0)

    # Impression ratio (clip to [0,10])
    if "impression_window_ratio" in df.columns:
        out["impression_window_ratio"] = df["impression_window_ratio"].fillna(1.0).clip(0.0, 10.0)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                out["impressions_prev_30d"] > 0,
                out["impressions_last_30d"] / out["impressions_prev_30d"].replace(0, np.nan),
                1.0,
            )
        out["impression_window_ratio"] = pd.Series(ratio, index=df.index).fillna(1.0).clip(0.0, 10.0)

    return out.fillna(0.0)


# ── Temporal / warehouse feature builders ─────────────────────────────────

def build_level_features(hist_window: pd.DataFrame) -> pd.DataFrame:
    """Level features from historical-window aggregate (one row per page_id)."""
    df = hist_window.set_index("page_id") if "page_id" in hist_window.columns else hist_window.copy()
    out = pd.DataFrame(index=df.index)
    out["impressions_hist"]      = df.get("impressions_sum",      pd.Series(0, index=df.index)).fillna(0)
    out["clicks_hist"]           = df.get("clicks_sum",           pd.Series(0, index=df.index)).fillna(0)
    out["ctr_hist"]              = df.get("ctr_window",           pd.Series(0.0, index=df.index)).fillna(0.0)
    out["avg_position_hist"]     = df.get("avg_position_mean",    pd.Series(0.0, index=df.index)).fillna(0.0)
    # query_activity_sum = SUM of daily query counts (NOT distinct breadth across window)
    col = "query_activity_sum" if "query_activity_sum" in df.columns else (
          "query_count_sum"    if "query_count_sum"    in df.columns else None)
    out["query_activity_sum"] = df[col].fillna(0) if col else 0.0
    out["active_days_hist"] = df.get("active_days", pd.Series(0, index=df.index)).fillna(0)
    return out


def build_trend_features(hist_daily: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Early vs late halves of the historical window (100% pre-cutoff)."""
    d = hist_daily.copy()
    d["event_date"] = pd.to_datetime(d["event_date"])
    hist = d[d["event_date"] < pd.to_datetime(cutoff)]
    if hist.empty:
        return pd.DataFrame()
    mid = hist["event_date"].min() + (hist["event_date"].max() - hist["event_date"].min()) / 2
    early = hist[hist["event_date"] <= mid]
    late  = hist[hist["event_date"] > mid]

    def agg(part):
        if part.empty:
            return pd.DataFrame(columns=["impressions","clicks","avg_position","ctr"])
        g = part.groupby("page_id").agg(
            impressions=("impressions","sum"), clicks=("clicks","sum"),
            avg_position=("position","mean"))
        g["ctr"] = np.where(g["impressions"]>0, g["clicks"]/g["impressions"], 0.0)
        return g

    ea, la = agg(early), agg(late)
    idx = ea.index.union(la.index)
    ea  = ea.reindex(idx).fillna(0.0)
    la  = la.reindex(idx).fillna(0.0)
    out = pd.DataFrame(index=idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["click_trend"]      = ((la["clicks"]-ea["clicks"])    /ea["clicks"].replace(0,np.nan)).fillna(0.0)
        out["impression_trend"] = ((la["impressions"]-ea["impressions"])/ea["impressions"].replace(0,np.nan)).fillna(0.0)
    out["ctr_trend"]      = la["ctr"] - ea["ctr"]
    out["position_trend"] = la["avg_position"] - ea["avg_position"]
    out.index.name = "page_id"
    return out


def build_volatility_features(hist_daily: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Coefficient of variation of daily clicks within historical window."""
    d = hist_daily.copy()
    d["event_date"] = pd.to_datetime(d["event_date"])
    d = d[d["event_date"] < pd.to_datetime(cutoff)]
    g = d.groupby("page_id")["clicks"]
    m = g.mean(); s = g.std().fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = (s / m.replace(0, np.nan)).fillna(0.0)
    out = pd.DataFrame({"click_volatility": cv, "n_active_days_seen": g.count()}, index=m.index)
    out.index.name = "page_id"
    return out


def build_feature_table(hist_window, hist_daily, cutoff):
    """Assemble all temporal-snapshot feature blocks."""
    lv = build_level_features(hist_window)
    tr = build_trend_features(hist_daily, cutoff)
    vo = build_volatility_features(hist_daily, cutoff)
    return lv.join(tr, how="left").join(vo, how="left").fillna(0.0)
