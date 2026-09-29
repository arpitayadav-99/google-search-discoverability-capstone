"""
data.py — Data loading, schema validation, and splitting.

Two data paths:
  1. Starter CSV  (data/raw/content_refresh_anonymized.csv) — 30,000 rows × 44 cols, no credentials.
  2. Full warehouse (hf://datasets/FlyRank/internship-warehouse/…) — requires HF_TOKEN.

Real schema (verified from content_refresh_anonymized.csv, 44 columns):
  content_id, client_id, search_volume, competition, competition_level, cpc,
  content_type, main_intent, word_count, char_count, provider_used, model_used,
  impressions_90d, clicks_90d, pageviews_90d, sessions_90d, users_90d,
  engaged_sessions_90d, ai_sessions_90d, scroll_events_90d,
  days_with_impressions, days_with_sessions,
  impressions_last_30d, clicks_last_30d, sessions_last_30d,
  impressions_prev_30d, clicks_prev_30d, sessions_prev_30d,
  content_age_days, age_tier, age_tier_order, days_since_last_update,
  freshness_tier, word_count_tier, char_count_tier,
  ctr, avg_position, engagement_rate, scroll_rate, ai_traffic_pct,
  impression_tier, position_tier,
  trend_direction [LABEL SOURCE — never a feature],
  trend_pct       [LABEL SOURCE — never a feature]

Public-safety: no raw URLs, client names, private queries, or credentials in this file.
Built on the FlyRank ML Internship dataset: https://flyrank.ai
"""
from __future__ import annotations
import hashlib, os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STARTER_CSV  = PROJECT_ROOT / "data" / "raw" / "content_refresh_anonymized.csv"

FLYRANK_WAREHOUSE_URI = os.environ.get(
    "FLYRANK_WAREHOUSE_URI",
    "hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance/**/*.parquet",
)

# Label-source columns — NEVER include in any feature matrix
LABEL_SOURCE_COLUMNS = ["trend_direction", "trend_pct"]

REQUIRED_COLUMNS = [
    "content_id", "client_id", "impressions_90d", "clicks_90d",
    "sessions_90d", "content_age_days", "trend_direction",
]

# Numeric columns present in the real CSV (fill with 0 if missing)
NUMERIC_FILL_ZERO = [
    "search_volume", "competition", "cpc", "word_count", "char_count",
    "impressions_90d", "clicks_90d", "pageviews_90d", "sessions_90d",
    "users_90d", "engaged_sessions_90d", "ai_sessions_90d", "scroll_events_90d",
    "days_with_impressions", "days_with_sessions",
    "impressions_last_30d", "clicks_last_30d", "sessions_last_30d",
    "impressions_prev_30d", "clicks_prev_30d", "sessions_prev_30d",
    "content_age_days", "age_tier_order", "days_since_last_update",
    "ctr", "avg_position", "engagement_rate", "scroll_rate", "ai_traffic_pct",
    "trend_pct",
]

CATEGORICAL_FILL_UNKNOWN = [
    "competition_level", "content_type", "main_intent",
    "age_tier", "freshness_tier", "word_count_tier", "char_count_tier",
    "impression_tier", "position_tier",
]

# Model-safe feature columns (NO trend_direction, NO trend_pct)
MODEL_NUMERIC_FEATURES = [
    "search_volume", "competition", "cpc", "word_count", "char_count",
    "log_impressions_90d", "log_clicks_90d", "log_sessions_90d", "log_ai_sessions_90d",
    "days_with_impressions", "days_with_sessions",
    "content_age_days", "days_since_last_update",
    "ctr", "avg_position", "engagement_rate", "scroll_rate", "ai_traffic_pct",
    "impressions_last_30d", "clicks_last_30d",
    "impressions_prev_30d", "clicks_prev_30d",
    "impression_window_ratio",
]

MODEL_CATEGORICAL_FEATURES = [
    "competition_level", "content_type", "main_intent",
    "age_tier", "freshness_tier", "word_count_tier",
    "impression_tier", "position_tier",
]


def load_starter_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """Load and validate the starter CSV. Raises FileNotFoundError with instructions."""
    csv_path = Path(path) if path else STARTER_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Starter CSV not found at: {csv_path}\n"
            "Obtain it from the FlyRank ML Internship starter repo and place it at:\n"
            "  data/raw/content_refresh_anonymized.csv\n"
            "See README.md → Data Setup for full instructions."
        )
    df = pd.read_csv(csv_path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame, required: list = None) -> None:
    """Raise ValueError if required columns are missing."""
    cols = required or REQUIRED_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Schema validation failed — missing columns: {missing}")
    print(f"[schema] Validated {df.shape[0]:,} rows × {df.shape[1]} columns ✓")


def prepare_starter_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply official FlyRank starter preparation:
    - Fill numerics → 0, categoricals → 'unknown'
    - Exclude pages with 0 impressions or age < 90 days
    - Add log-transforms, derived columns, binary label

    IMPORTANT: trend_direction / trend_pct are NEVER passed to models as features.
    """
    df = df.copy()

    for col in NUMERIC_FILL_ZERO:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            df[col] = 0.0

    for col in CATEGORICAL_FILL_UNKNOWN:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).replace({"": "unknown", "nan": "unknown"})
        else:
            df[col] = "unknown"

    n_before = len(df)
    df = df[(df["impressions_90d"] > 0) & (df["content_age_days"] >= 90)].copy()
    df = df.drop_duplicates(subset=["content_id"]).reset_index(drop=True)
    print(f"[data] Kept {len(df):,}/{n_before:,} rows (excluded {n_before-len(df):,} zero-impression / young pages)")

    # Log-transforms
    df["log_impressions_90d"] = np.log1p(df["impressions_90d"])
    df["log_clicks_90d"]      = np.log1p(df["clicks_90d"])
    df["log_sessions_90d"]    = np.log1p(df["sessions_90d"])
    df["log_ai_sessions_90d"] = np.log1p(df["ai_sessions_90d"])

    # Impression window ratio (both windows pre-export → leakage-safe)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["impression_window_ratio"] = np.where(
            df["impressions_prev_30d"] > 0,
            df["impressions_last_30d"] / df["impressions_prev_30d"],
            1.0,
        )
    df["impression_window_ratio"] = df["impression_window_ratio"].clip(0.0, 10.0)

    # Binary decline label (analysis / EDA only — excluded from feature matrix)
    df["is_declining_label"] = df["trend_direction"].str.lower().eq("down").astype(int)

    return df


def starter_feature_matrix(df: pd.DataFrame) -> tuple:
    """Return (X, y): X = model-safe feature matrix, y = decline label.
    content_id is the index, never a feature.
    Categoricals are one-hot-encoded.
    Label-source columns are never included.
    """
    safe_df = df.copy()
    for col in LABEL_SOURCE_COLUMNS + ["is_declining_label"]:
        if col in safe_df.columns:
            safe_df = safe_df.drop(columns=[col])

    safe_df = safe_df.set_index("content_id")
    num_cols = [c for c in MODEL_NUMERIC_FEATURES if c in safe_df.columns]
    cat_cols = [c for c in MODEL_CATEGORICAL_FEATURES if c in safe_df.columns]

    X_num = safe_df[num_cols].fillna(0.0)
    X_cat = pd.get_dummies(safe_df[cat_cols].fillna("unknown"), drop_first=False, dtype=float)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df.set_index("content_id")["is_declining_label"].loc[X.index]
    return X, y


def three_way_client_split(df: pd.DataFrame, val_fraction=0.15, test_fraction=0.15, seed=42):
    """Three-way split by client_id. No client appears in more than one partition."""
    rng = np.random.default_rng(seed)
    clients = list(df["client_id"].unique())
    rng.shuffle(clients)
    n = len(clients)
    n_test = max(1, int(n * test_fraction))
    n_val  = max(1, int(n * val_fraction))
    test_c  = set(clients[:n_test])
    val_c   = set(clients[n_test: n_test + n_val])
    train_c = set(clients[n_test + n_val:])
    train = df[df["client_id"].isin(train_c)].copy()
    val   = df[df["client_id"].isin(val_c)].copy()
    test  = df[df["client_id"].isin(test_c)].copy()
    print(f"[split] clients — train:{len(train_c)} val:{len(val_c)} test:{len(test_c)}")
    print(f"[split] rows   — train:{len(train):,} val:{len(val):,} test:{len(test):,}")
    return train, val, test


def assert_no_label_leakage_starter(X: pd.DataFrame) -> None:
    """Hard assertion: label-source columns must not appear in X."""
    leaked = [c for c in LABEL_SOURCE_COLUMNS + ["is_declining_label"] if c in X.columns]
    if leaked:
        raise AssertionError(
            f"LEAKAGE DETECTED: {leaked} found in feature matrix. "
            "Remove them before training."
        )
    print("[leakage] No label-source columns in feature matrix ✓")


def get_warehouse_connection():
    """DuckDB connection for the FlyRank gated warehouse. Requires HF_TOKEN."""
    if duckdb is None:
        raise RuntimeError("pip install duckdb")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set. Copy .env.example → .env, fill HF_TOKEN, then load_dotenv()."
        )
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    try:
        con.execute("CREATE OR REPLACE SECRET flyrank_hf (TYPE HUGGINGFACE, TOKEN ?)", [token])
    except Exception:
        pass
    return con


def anonymize_id(raw_id: str, salt: str = "flyrank-capstone") -> str:
    """One-way hash for public-safe page identifiers."""
    h = hashlib.sha256(f"{salt}:{raw_id}".encode()).hexdigest()
    return f"PAGE_{h[:12].upper()}"
