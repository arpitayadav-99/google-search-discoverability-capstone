"""Populate paper/index.html from a completed FULL-WAREHOUSE experiment.

Starter CSV results are intentionally rejected because they are cross-sectional and
cannot support the future-window temporal claim required by the capstone.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RESULTS_F=ROOT/"outputs/tables/experiment_results.json"
PAPER_HTML=ROOT/"paper/index.html"
RECS=ROOT/"outputs/tables/ranked_recommendations.csv"

def main():
    if not RESULTS_F.exists():
        raise SystemExit("No experiment_results.json. Run capstone.ipynb in DATA_MODE='warehouse' first.")
    r=json.loads(RESULTS_F.read_text())
    if r.get("data_mode") != "warehouse":
        raise SystemExit("Refusing to publish starter/synthetic results. The paper requires a real warehouse experiment.")
    if not RECS.exists():
        raise SystemExit("ranked_recommendations.csv is missing. Re-run the warehouse notebook.")
    html=PAPER_HTML.read_text()
    tm=r.get("test_metrics_model",{}); bm=r.get("test_metrics_baseline",{})
    replacements={
        "{{N_SNAPSHOTS}}":str(r.get("n_page_snapshots","?")),
        "{{N_TRAIN}}":str(r.get("n_train","?")),"{{N_VAL}}":str(r.get("n_val","?")),"{{N_TEST}}":str(r.get("n_test","?")),
        "{{BEST_MODEL}}":str(r.get("best_model","?")),"{{MODEL_ROC_AUC}}":f"{tm.get('roc_auc',float('nan')):.4f}",
        "{{MODEL_PR_AUC}}":f"{tm.get('pr_auc',float('nan')):.4f}","{{MODEL_PRECISION}}":f"{tm.get('precision',float('nan')):.4f}",
        "{{MODEL_RECALL}}":f"{tm.get('recall',float('nan')):.4f}","{{MODEL_F1}}":f"{tm.get('f1',float('nan')):.4f}",
        "{{BASE_ROC_AUC}}":f"{bm.get('roc_auc',float('nan')):.4f}","{{BASE_PR_AUC}}":f"{bm.get('pr_auc',float('nan')):.4f}",
        "{{N_RECS}}":str(r.get("n_recommendations","?")),"{{TRAIN_CUTOFFS}}":", ".join(r.get("train_cutoffs",[])),
        "{{VAL_CUTOFF}}":str(r.get("validation_cutoff","?")),"{{TEST_CUTOFF}}":str(r.get("test_cutoff","?")),
    }
    for k,v in replacements.items(): html=html.replace(k,v)
    PAPER_HTML.write_text(html)
    print(f"Updated {PAPER_HTML} from warehouse results.")
if __name__=="__main__": main()
