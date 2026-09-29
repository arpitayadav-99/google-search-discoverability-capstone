"""
synthetic_smoke_test.py
=======================
*** SYNTHETIC DATA ONLY — NOT REAL FLYRANK RESULTS ***

Validates pipeline wiring end-to-end on seeded synthetic data.
DO NOT copy outputs from this script into the capstone paper.
All output files are prefixed  synthetic_
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import features, labels, models, evaluation, recommendations

RNG = np.random.default_rng(42)
OUT = os.path.join(ROOT, "outputs", "tables")
os.makedirs(OUT, exist_ok=True)

BANNER = "\n*** SYNTHETIC SMOKE TEST — NOT REAL RESULTS ***\n"


def _daily(n_pages=500, n_days=180):
    dates = pd.date_range("2025-06-01", periods=n_days, freq="D")
    pids  = [f"SYN_{i:05d}" for i in range(n_pages)]
    dec   = set(RNG.choice(pids, size=int(n_pages*0.30), replace=False))
    rows  = []
    for pid in pids:
        bi, bc, bp = RNG.integers(30,1500), RNG.uniform(.01,.10), RNG.uniform(4,22)
        for idx, d in enumerate(dates):
            frac = idx/n_days
            df_  = 1.0 - 0.55*max(0,(frac-.5)/.5) if pid in dec else 1.0
            ps   = 4.0 *max(0,(frac-.5)/.5)        if pid in dec else 0.0
            im   = max(0,int(RNG.poisson(bi*df_)))
            ct   = max(0.0, bc*df_+RNG.normal(0,.004))
            cl   = int(RNG.binomial(im, min(ct,1.0))) if im else 0
            rows.append({"page_id":pid,"event_date":d,
                         "impressions":im,"clicks":cl,
                         "position":max(1.0,bp+ps+RNG.normal(0,.8)),
                         "query_count":max(1,im//25)})
    return pd.DataFrame(rows)


def _agg(daily, start, end):
    d = daily.copy(); d["event_date"]=pd.to_datetime(d["event_date"])
    w = d[(d.event_date>=pd.to_datetime(start)) & (d.event_date<pd.to_datetime(end))]
    g = w.groupby("page_id").agg(
        impressions_sum=("impressions","sum"), clicks_sum=("clicks","sum"),
        avg_position_mean=("position","mean"), query_activity_sum=("query_count","sum"),
        active_days=("event_date","nunique")).reset_index()
    g["ctr_window"] = np.where(g["impressions_sum"]>0, g["clicks_sum"]/g["impressions_sum"],0.0)
    return g


def run():
    print(BANNER)
    daily   = _daily()
    cutoff  = "2025-09-27"
    h_start = "2025-06-01"
    f_end   = "2025-11-23"

    # ── boundary checks ────────────────────────────────────────────────────
    d = daily.copy(); d["event_date"] = pd.to_datetime(d["event_date"])
    hist = d[d.event_date < pd.to_datetime(cutoff)]
    fut  = d[d.event_date >= pd.to_datetime(cutoff)]
    assert hist["event_date"].max() < pd.to_datetime(cutoff), "LEAKAGE: hist overlaps cutoff"
    assert fut["event_date"].min()  >= pd.to_datetime(cutoff), "LEAKAGE: future before cutoff"
    print(f"[boundary] hist_max={hist.event_date.max().date()} < cutoff={cutoff}  v")
    print(f"[boundary] fut_min={fut.event_date.min().date()} >= cutoff={cutoff}  v")

    hw = _agg(daily, h_start, cutoff)
    fw = _agg(daily, cutoff,  f_end)

    cfg = labels.LabelConfig(metric="clicks", decline_threshold=0.30, min_historical_activity=50)
    ft  = features.build_feature_table(hw, hist, cutoff)
    lt  = labels.compute_decline_label(hw, fw, cfg)
    eli = lt[lt["eligible"]]
    idx = ft.index.intersection(eli.index)
    X   = ft.loc[idx]
    y   = eli.loc[idx, "label"].astype(int)
    print(f"[features] X={X.shape}, positives={y.sum()}/{len(y)}")
    print(f"[labels]   {labels.edge_case_report(lt)}")

    n  = len(X)
    t1 = int(n*.60); t2 = int(n*.80)
    Xtr,ytr = X.iloc[:t1], y.iloc[:t1]
    Xva,yva = X.iloc[t1:t2], y.iloc[t1:t2]
    Xte,yte = X.iloc[t2:], y.iloc[t2:]
    if 0 in (len(Xtr),len(Xva),len(Xte)):
        print("[smoke] Skip model tests — insufficient pages"); return

    bs = models.baseline_decline_score_temporal(Xte)
    cs = models.train_all_candidates(Xtr, ytr)
    vr = []
    for m in cs:
        mt = evaluation.classification_metrics(yva.to_numpy(), m.predict_proba(Xva))
        mt["model"] = m.name; vr.append(mt)
    bname = evaluation.select_best_model(vr, "pr_auc")
    best  = next(m for m in cs if m.name == bname)
    ts    = best.predict_proba(Xte)
    tm    = evaluation.classification_metrics(yte.to_numpy(), ts)
    cmp   = evaluation.compare_baseline_vs_model(yte.to_numpy(), bs.to_numpy(), ts, [10,25,50])
    print(f"[test]  best={bname}  roc_auc={tm['roc_auc']:.3f}  pr_auc={tm['pr_auc']:.3f}")
    print(cmp.round(3).to_string(index=False))

    sc  = pd.Series(ts, index=Xte.index, name="score")
    rec = recommendations.generate_recommendations(Xte, sc, top_n=20, mode="temporal")
    print(f"[recs]  {len(rec)} recs  actions={rec['action'].value_counts().to_dict()}")

    audit = evaluation.leakage_audit_table()
    audit.to_csv(os.path.join(OUT,"synthetic_leakage_audit.csv"), index=False)
    cmp.to_csv(os.path.join(OUT,"synthetic_baseline_vs_model.csv"), index=False)
    pd.DataFrame([tm]).to_csv(os.path.join(OUT,"synthetic_classification_report.csv"), index=False)
    rec.to_csv(os.path.join(OUT,"synthetic_sample_ranked_recommendations.csv"), index=False)

    print(BANNER)
    print("Smoke test PASSED — outputs in outputs/tables/synthetic_*")
    print("Do NOT use these results in the capstone paper.")


if __name__ == "__main__":
    run()
