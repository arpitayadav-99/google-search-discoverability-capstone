"""DuckDB helpers for the gated FlyRank daily-performance warehouse.

The functions in this module intentionally keep raw URLs/queries out of outputs.
They aggregate the daily fact table inside DuckDB and return only page-level
numeric aggregates needed for the capstone.
"""
from __future__ import annotations
import numpy as np
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List
import pandas as pd

from . import data as DM

DATE_CANDIDATES = ["event_date", "date", "day", "performance_date", "ds"]
PAGE_CANDIDATES = ["page_id", "content_id", "content_key", "url_hash"]
IMP_CANDIDATES = ["impressions", "impression_count"]
CLICK_CANDIDATES = ["clicks", "click_count"]
SESSION_CANDIDATES = ["sessions", "session_count"]
POSITION_CANDIDATES = ["position", "avg_position", "average_position"]
CLIENT_CANDIDATES = ["client_id", "client_key"]

@dataclass(frozen=True)
class WarehouseSchema:
    date_col: str
    page_col: str
    impressions_col: str
    clicks_col: str
    sessions_col: str | None
    position_col: str | None
    client_col: str | None


def _pick(columns: Iterable[str], candidates: List[str], required: bool = True):
    lookup = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    if required:
        raise ValueError(f"Could not find required warehouse column. Tried: {candidates}. Available: {list(columns)}")
    return None


def discover_schema(con, uri: str | None = None) -> WarehouseSchema:
    uri = uri or DM.FLYRANK_WAREHOUSE_URI
    safe_uri = uri.replace("'", "''")
    desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{safe_uri}') LIMIT 0").fetchdf()
    cols = desc["column_name"].tolist()
    return WarehouseSchema(
        date_col=_pick(cols, DATE_CANDIDATES),
        page_col=_pick(cols, PAGE_CANDIDATES),
        impressions_col=_pick(cols, IMP_CANDIDATES),
        clicks_col=_pick(cols, CLICK_CANDIDATES),
        sessions_col=_pick(cols, SESSION_CANDIDATES, False),
        position_col=_pick(cols, POSITION_CANDIDATES, False),
        client_col=_pick(cols, CLIENT_CANDIDATES, False),
    )


def _quote(c: str) -> str:
    return '"' + c.replace('"', '""') + '"'


def aggregate_window(con, schema: WarehouseSchema, start: str, end: str, uri: str | None = None) -> pd.DataFrame:
    """Aggregate one half-open [start, end) window by page."""
    uri = uri or DM.FLYRANK_WAREHOUSE_URI
    safe_uri = uri.replace("'", "''")
    d, p, imp, clk = map(_quote, [schema.date_col, schema.page_col, schema.impressions_col, schema.clicks_col])
    sess = _quote(schema.sessions_col) if schema.sessions_col else None
    pos = _quote(schema.position_col) if schema.position_col else None
    sess_expr = f", SUM(COALESCE({sess},0)) AS sessions_sum" if sess else ", CAST(0 AS DOUBLE) AS sessions_sum"
    if pos:
        pos_expr = f", CASE WHEN SUM(COALESCE({imp},0)) > 0 THEN SUM(COALESCE({pos},0) * COALESCE({imp},0)) / SUM(COALESCE({imp},0)) ELSE 0 END AS avg_position_mean"
    else:
        pos_expr = ", CAST(0 AS DOUBLE) AS avg_position_mean"
    sql = f"""
    SELECT
      CAST({p} AS VARCHAR) AS page_id,
      SUM(COALESCE({imp},0))::DOUBLE AS impressions_sum,
      SUM(COALESCE({clk},0))::DOUBLE AS clicks_sum
      {sess_expr}
      {pos_expr},
      COUNT(DISTINCT CAST({d} AS DATE))::INTEGER AS active_days
    FROM read_parquet('{safe_uri}')
    WHERE CAST({d} AS DATE) >= DATE '{start}'
      AND CAST({d} AS DATE) < DATE '{end}'
    GROUP BY 1
    """
    return con.execute(sql).fetchdf()


def build_snapshot(con, schema: WarehouseSchema, cutoff: str, hist_days: int = 90, future_days: int = 30, uri: str | None = None) -> pd.DataFrame:
    cutoff_ts = pd.Timestamp(cutoff)
    hist_start = (cutoff_ts - pd.Timedelta(days=hist_days)).strftime("%Y-%m-%d")
    future_end = (cutoff_ts + pd.Timedelta(days=future_days)).strftime("%Y-%m-%d")
    hist = aggregate_window(con, schema, hist_start, cutoff_ts.strftime("%Y-%m-%d"), uri)
    fut = aggregate_window(con, schema, cutoff_ts.strftime("%Y-%m-%d"), future_end, uri)
    return hist, fut, hist_start, future_end


def build_temporal_snapshot(con, schema: WarehouseSchema, cutoff: str, hist_days: int = 90, future_days: int = 30, min_impressions: int = 50, uri: str | None = None) -> pd.DataFrame:
    """Return one leakage-safe page snapshot with features + future decline label."""
    from .labels import LabelConfig, compute_decline_label
    hist, fut, hist_start, future_end = build_snapshot(con, schema, cutoff, hist_days, future_days, uri)
    if hist.empty:
        return pd.DataFrame()
    # Build trend features from two non-overlapping historical sub-windows, both strictly
    # before the cutoff. This is deliberately independent of the future label window.
    cutoff_ts = pd.Timestamp(cutoff)
    mid = cutoff_ts - pd.Timedelta(days=hist_days // 2)
    early = aggregate_window(con, schema, hist_start, mid.strftime("%Y-%m-%d"), uri).set_index("page_id")
    late = aggregate_window(con, schema, mid.strftime("%Y-%m-%d"), cutoff, uri).set_index("page_id")
    h = hist.set_index("page_id")
    idx = h.index
    early = early.reindex(idx).fillna(0.0)
    late = late.reindex(idx).fillna(0.0)
    feat = pd.DataFrame(index=idx)
    feat["impressions_hist"] = h["impressions_sum"].fillna(0)
    feat["clicks_hist"] = h["clicks_sum"].fillna(0)
    feat["sessions_hist"] = h["sessions_sum"].fillna(0)
    feat["active_days_hist"] = h["active_days"].fillna(0)
    feat["ctr_hist"] = np.where(feat["impressions_hist"] > 0, feat["clicks_hist"] / feat["impressions_hist"], 0.0)
    feat["avg_position_hist"] = h["avg_position_mean"].fillna(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        feat["click_trend"] = ((late["clicks_sum"] - early["clicks_sum"]) / early["clicks_sum"].replace(0, np.nan)).fillna(0.0).clip(-1, 10)
        feat["impression_trend"] = ((late["impressions_sum"] - early["impressions_sum"]) / early["impressions_sum"].replace(0, np.nan)).fillna(0.0).clip(-1, 10)
    feat["position_trend"] = late["avg_position_mean"].fillna(0) - early["avg_position_mean"].fillna(0)
    feat["click_volatility"] = 0.0
    labels = compute_decline_label(hist, fut, LabelConfig(metric="clicks", decline_threshold=0.30, min_historical_activity=min_impressions))
    out = feat.join(labels[["eligible", "relative_change", "label"]], how="inner")
    out = out[out["eligible"] & out["label"].notna()].copy()
    out["cutoff"] = pd.Timestamp(cutoff)
    out["hist_start"] = pd.Timestamp(hist_start)
    out["future_end"] = pd.Timestamp(future_end)
    return out.drop(columns=["eligible"])
