# Google Search Ranking & Discoverability Capstone

**Track:** Machine Learning | **Lane:** Refresh / Content Opportunity Scoring

> Built on the [FlyRank ML Internship dataset](https://flyrank.ai)

---

## Project Question

Which content pages are most associated with declining search visibility (impressions),
and how should they be prioritised for editorial refresh?

---

## Architecture

```
google-search-discoverability-capstone/
├── src/                      # Core pipeline modules
│   ├── data.py               # Data loading, schema validation, splitting
│   ├── features.py           # Leakage-safe feature engineering
│   ├── labels.py             # Label construction (decline signal)
│   ├── models.py             # Baseline + 3 ML candidates
│   ├── evaluation.py         # Classification + ranking metrics, model selection
│   └── recommendations.py   # Deterministic ranked recommendations
├── work/capstone/
│   └── capstone.ipynb        # Main end-to-end notebook (15 sections)
├── scripts/
│   ├── synthetic_smoke_test.py  # Pipeline wiring test (synthetic data only)
│   └── update_paper.py          # Inject real results into paper/index.html
├── tests/
│   └── test_pipeline.py      # Unit tests: leakage, boundaries, labels, models
├── paper/
│   └── index.html            # Deployed research paper (GitHub Pages)
├── outputs/
│   ├── tables/               # CSV / JSON experiment outputs
│   └── figures/              # PNG charts
├── data/
│   └── raw/                  # Starter CSV (gitignored; place here manually)
├── submission/
│   └── paper_url.txt         # Set after GitHub Pages deployment
├── .env.example              # Credential template (never commit .env)
└── .github/workflows/pages.yml  # GitHub Pages deployment workflow
```

---

## Data Setup

### Option A — Starter CSV (development only; no credentials needed)

1. Obtain `content_refresh_anonymized.csv` from the FlyRank ML Internship starter repo.
2. Place it at: `data/raw/content_refresh_anonymized.csv`
3. The file is gitignored (never committed to the public repo).

Schema: 30,000 rows × 44 columns — one row per content item, 90-day trailing aggregate. **This snapshot is for development/validation only; it cannot support a true future-window temporal claim.**
Key columns: content_id, client_id, impressions_90d, clicks_90d, sessions_90d,
trend_direction (label source), trend_pct (label source), ctr, avg_position, …

### Option B — Full FlyRank Warehouse (required for the published capstone)

1. Request access at: https://huggingface.co/datasets/FlyRank/internship-warehouse
2. `cp .env.example .env` then set `HF_TOKEN=<your_token>`
3. In notebook §1 set `DATA_MODE = "warehouse"`

**NEVER commit `.env` or any token to git.**

---

## Setup

```bash
# Clone
git clone <YOUR_GITHUB_REPO_URL>
cd google-search-discoverability-capstone

# Install
pip install -r requirements.txt
```

---

## Running

### 1 — Synthetic smoke test (pipeline wiring, no real data needed)

```bash
python scripts/synthetic_smoke_test.py
```

Output files are prefixed `synthetic_` — **never use them as capstone results**.

### 2 — Unit tests

```bash
python -m pytest tests/ -v
```

Tests cover: leakage boundaries, label logic, model scaler isolation, deterministic
reason codes, validation-only model selection.

### 3 — Main notebook (real data)

```bash
jupyter lab work/capstone/capstone.ipynb
```

Run all cells in order (§1 Setup → §15 Paper Update).
DATA_MODE defaults to `"starter"` (starter CSV). No credentials needed.

### 4 — Generate paper outputs

After running the notebook in `DATA_MODE = "warehouse"`:

```bash
python scripts/update_paper.py
```

This reads only a warehouse-generated `experiment_results.json` and injects its real metrics into `paper/index.html`. It deliberately refuses starter/synthetic results.

---

## Deploying the Paper (GitHub Pages)

1. Push to `main` on GitHub.
2. In repository Settings → Pages → Source: **GitHub Actions**.
3. The `.github/workflows/pages.yml` workflow deploys `paper/` automatically.
4. After deployment, edit `submission/paper_url.txt` and replace the placeholder
   with your live URL (exactly one line, the full https:// URL).

---

## Public-Safety Rules

The public repository **must not** contain:
- Raw URLs or private domains
- Client names or private search queries
- Credentials, API tokens, or HF tokens
- Raw private data exports

All content identifiers are pseudonymous (`content_id`).
Credentials use `.env.example` placeholders only.
The starter CSV is gitignored.

---

## Honest Framing

Results are **observed associations** and **directional signals** for prioritisation.
This analysis does **not** reverse-engineer Google's ranking algorithm and does **not**
prove that refreshing content will causally improve rankings.

---

## Acknowledgments

Built on the **FlyRank ML Internship dataset** — https://flyrank.ai
