"""
test_pipeline.py — Unit tests: leakage boundaries, label logic, feature safety.
Run: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src import data as data_mod
from src import features, labels, models, evaluation, recommendations


# ── Leakage: label columns not in MODEL_NUMERIC_FEATURES ──────────────────
def test_label_sources_not_in_numeric_features():
    for col in data_mod.LABEL_SOURCE_COLUMNS:
        assert col not in data_mod.MODEL_NUMERIC_FEATURES, (
            f"{col} is a label source but appears in MODEL_NUMERIC_FEATURES — leakage!")


# ── Temporal boundary: half-open intervals ─────────────────────────────────
def test_temporal_boundary_half_open():
    cutoff = pd.Timestamp("2025-09-01")
    dates  = pd.date_range("2025-06-01", "2025-11-30", freq="D")
    hist   = dates[dates <  cutoff]
    fut    = dates[dates >= cutoff]
    assert hist.max() <  cutoff, "Historical window includes cutoff — leakage!"
    assert fut.min()  >= cutoff, "Future window is before cutoff — leakage!"
    assert hist.max() != fut.min(), "Cutoff appears in both windows — leakage!"


def test_cutoff_not_in_both_windows():
    cutoff = pd.Timestamp("2025-09-01")
    for d in pd.date_range("2025-08-28","2025-09-04",freq="D"):
        in_hist = d < cutoff
        in_fut  = d >= cutoff
        assert not (in_hist and in_fut), f"{d} in both windows — leakage!"


# ── Feature safety: starter CSV assert ────────────────────────────────────
def test_assert_no_label_leakage_passes_clean_X():
    X = pd.DataFrame({"log_impressions_90d":[1.0],"ctr":[0.5]}, index=["page1"])
    data_mod.assert_no_label_leakage_starter(X)  # should not raise


def test_assert_no_label_leakage_fails_with_trend_direction():
    X = pd.DataFrame({"log_impressions_90d":[1.0],"trend_direction":["down"]}, index=["p1"])
    with pytest.raises(AssertionError, match="LEAKAGE"):
        data_mod.assert_no_label_leakage_starter(X)


def test_assert_no_label_leakage_fails_with_trend_pct():
    X = pd.DataFrame({"log_impressions_90d":[1.0],"trend_pct":[-0.3]}, index=["p1"])
    with pytest.raises(AssertionError, match="LEAKAGE"):
        data_mod.assert_no_label_leakage_starter(X)


# ── Label: temporal decline label ─────────────────────────────────────────
def test_decline_label_declining_page():
    hist = pd.DataFrame({"page_id":["A"],"impressions_sum":[500],"clicks_sum":[50],"active_days":[30],"ctr_window":[0.1],"avg_position_mean":[10.0],"query_activity_sum":[200]})
    fut  = pd.DataFrame({"page_id":["A"],"impressions_sum":[300],"clicks_sum":[10],"active_days":[30],"ctr_window":[0.03],"avg_position_mean":[15.0],"query_activity_sum":[100]})
    cfg  = labels.LabelConfig(metric="clicks", decline_threshold=0.30, min_historical_activity=50)
    lt   = labels.compute_decline_label(hist, fut, cfg)
    assert lt.loc["A","eligible"], "Page A should be eligible (≥50 impressions)"
    assert lt.loc["A","label"] == 1.0, "Page A should be declining (clicks dropped >30%)"


def test_decline_label_stable_page():
    hist = pd.DataFrame({"page_id":["B"],"impressions_sum":[500],"clicks_sum":[50],"active_days":[30],"ctr_window":[0.1],"avg_position_mean":[10.0],"query_activity_sum":[200]})
    fut  = pd.DataFrame({"page_id":["B"],"impressions_sum":[480],"clicks_sum":[48],"active_days":[30],"ctr_window":[0.10],"avg_position_mean":[10.0],"query_activity_sum":[190]})
    cfg  = labels.LabelConfig(metric="clicks", decline_threshold=0.30, min_historical_activity=50)
    lt   = labels.compute_decline_label(hist, fut, cfg)
    assert lt.loc["B","label"] == 0.0, "Page B should not be declining (<30% drop)"


def test_decline_label_ineligible_low_history():
    hist = pd.DataFrame({"page_id":["C"],"impressions_sum":[10],"clicks_sum":[1],"active_days":[5],"ctr_window":[0.1],"avg_position_mean":[20.0],"query_activity_sum":[5]})
    fut  = pd.DataFrame({"page_id":["C"],"impressions_sum":[2], "clicks_sum":[0],"active_days":[5],"ctr_window":[0.0],"avg_position_mean":[25.0],"query_activity_sum":[2]})
    cfg  = labels.LabelConfig(min_historical_activity=50)
    lt   = labels.compute_decline_label(hist, fut, cfg)
    assert not lt.loc["C","eligible"], "Page C should be ineligible (low history)"
    assert pd.isna(lt.loc["C","label"]), "Ineligible page label should be NaN"


# ── Models: scaler fit only on train ──────────────────────────────────────
def test_scaler_fit_only_on_train():
    rng  = np.random.default_rng(0)
    Xtr  = pd.DataFrame(rng.normal(0,1,(100,5)), columns=[f"f{i}" for i in range(5)])
    Xte  = pd.DataFrame(rng.normal(10,1,(20,5)), columns=[f"f{i}" for i in range(5)])
    ytr  = pd.Series(rng.integers(0,2,100))
    model = models.train_logistic_regression(Xtr, ytr)
    scaler= model.scaler
    assert scaler is not None
    # Scaler mean should be near 0 (train mean), not 10 (test mean)
    assert abs(scaler.mean_[0]) < 1.0, "Scaler was not fit on training data only"


# ── Recommendations: deterministic reason codes ────────────────────────────
def test_reason_codes_deterministic():
    feat = pd.DataFrame({
        "impression_window_ratio":[0.50],
        "avg_position":[25.0],
        "ctr":[0.5],
        "log_impressions_90d":[3.0],
        "content_age_days":[400],
        "days_since_last_update":[200],
    }, index=pd.Index(["page1"], name="content_id"))
    scores = pd.Series([0.80], index=pd.Index(["page1"], name="content_id"), name="score")
    r1 = recommendations.generate_recommendations(feat, scores, mode="starter")
    r2 = recommendations.generate_recommendations(feat, scores, mode="starter")
    assert r1.iloc[0]["reason_codes"] == r2.iloc[0]["reason_codes"], "Reason codes not deterministic"
    assert "STRONG_IMPRESSION_DECLINE" in r1.iloc[0]["reason_codes"], "Expected decline code"
    assert r1.iloc[0]["action"] == "Refresh", "Score 0.80 should map to Refresh"


# ── Model selection uses validation only ──────────────────────────────────
def test_select_best_model_uses_pr_auc():
    vr = [
        {"model":"logistic_regression","roc_auc":0.70,"pr_auc":0.40,"f1":0.50},
        {"model":"random_forest",       "roc_auc":0.65,"pr_auc":0.60,"f1":0.55},
        {"model":"hist_gradient_boosting","roc_auc":0.72,"pr_auc":0.55,"f1":0.53},
    ]
    best = evaluation.select_best_model(vr, "pr_auc")
    assert best == "random_forest", f"Expected random_forest (highest pr_auc 0.60), got {best}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
