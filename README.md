# AlleyLoop — NBA Analytics Platform

End-to-end basketball analytics system covering play-by-play ingestion,
player performance prediction, lineup optimisation, and game outcome modelling.
Built on NBA CDN play-by-play CSV data from seasons 2020–2024.

---

## Architecture

```
data/raw/*.csv  (5 seasons, ~680k rows each)
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  src/pipeline.py  —  Phase 1-2                          │
│  • Schema unification across 5 seasons                  │
│  • Box score aggregation (34 stats per player-game)     │
│  • Lineup stint tracking (score-tracked)                │
└────────────┬──────────────────────────┬─────────────────┘
             │                          │
             ▼                          ▼
   box_scores.parquet        lineup_ratings.parquet
   lineup_stats.parquet      (net/off/def rating +
                              player embeddings)
             │
             ▼
┌────────────────────────────────────────────────────────┐
│  src/features.py  —  Phase 3                           │
│  • Rolling windows: 3 / 5 / 10 game trailing means     │
│  • EWMA (α=0.30, shift-1 to prevent leakage)           │
│  • Season-to-date expanding mean (resets each year)    │
│  • Context: rest days, home/away, opp def rating       │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
     features.parquet  (127,651 rows × 125 cols)
             │
    ┌────────┴──────────┐
    ▼                   ▼
┌──────────────┐  ┌─────────────────────────────────────┐
│ src/         │  │  src/predictor.py  —  Phase 5        │
│ optimizer.py │  │  • RandomForest + XGBoost            │
│ Phase 4      │  │  • Targets: pts, reb, ast, min       │
│              │  │  • SHAP feature importance           │
│ ILP lineup   │  │  • Train 2020-23 / Test 2024         │
│ optimiser    │  └──────────────────┬──────────────────┘
│ (PuLP/CBC)   │                     │
└──────┬───────┘            predictor_results.parquet
       │                             │
       ▼                             ▼
optimizer_eval             ┌─────────────────────────────┐
_2024.parquet              │  src/game_model.py  Phase 7  │
                           │  • Team EWMA aggregation     │
                           │  • LogReg + XGBoost          │
                           │  • Calibration plot          │
                           └──────────────┬──────────────┘
                                          │
                                game_model_results.parquet
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  app.py — Streamlit   │
                              │  • Player Stats        │
                              │  • Lineup Optimizer    │
                              │  • Win Predictor       │
                              └───────────────────────┘
```

---

## Phase Completion Status

| # | Phase | Module | Status | Key Metric |
|---|-------|--------|--------|------------|
| 1 | PBP ingestion & schema unification | `pipeline.py` | ✅ Complete | 5 seasons, ~680k rows/season |
| 2 | Box score aggregation & lineup stints | `pipeline.py` | ✅ Complete | 127,651 player-games, 88,539 lineups |
| 3 | Feature engineering | `features.py` | ✅ Complete | 125 features, shift(1) leakage-free |
| 4 | Lineup optimiser | `optimizer.py` | ✅ Complete | 2.91/5 avg overlap, 57.7% win rate on ≥4/5 |
| 5 | Player performance prediction | `predictor.py` | ✅ Complete | pts MAE 4.60 (↓34.6%), min MAE 4.90 (↓45.4%) |
| 6 | Lineup-level ratings & embeddings | `pipeline.py` | ✅ Complete | net/off/def rating + 10-dim player embedding |
| 7 | Game outcome prediction | `game_model.py` | ✅ Complete | 62.1% accuracy, AUC 0.669 (LogReg) |
| 8 | Injury risk modelling | `injury_risk.py` | 🔲 Stub | Awaiting injury log data |
| 9 | Monte Carlo season simulation | `monte_carlo.py` | 🔲 Stub | Depends on Phase 7 ✅ |
| 10 | Trade impact analysis | `trade_analysis.py` | 🔲 Stub | Depends on Phases 4 & 7 ✅ |
| 11 | Full dashboard expansion | `app.py` | 🔲 Planned | Add tabs for Phases 8-10 |

---

## Model Results (2024 Season Test Set)

### Player Performance Prediction (Phase 5)

| Target | Best Model | MAE | RMSE | vs Baseline |
|--------|-----------|-----|------|-------------|
| Points | XGBoost | 4.60 | 6.01 | ↓ 34.6% |
| Rebounds | XGBoost | 1.92 | 2.54 | ↓ 28.0% |
| Assists | XGBoost | 1.34 | 1.83 | ↓ 32.4% |
| Minutes | XGBoost | 4.90 | 6.40 | ↓ 45.4% |

### Game Outcome Prediction (Phase 7)

| Model | Accuracy | ROC-AUC | Brier Score |
|-------|----------|---------|-------------|
| Logistic Regression | **62.1%** | **0.669** | **0.227** |
| XGBoost | 60.2% | 0.633 | 0.238 |

### Lineup Optimiser (Phase 4 — 2024 Season)

| Metric | Value |
|--------|-------|
| Avg player overlap (out of 5) | 2.91 |
| ≥ 3/5 match rate | 68% |
| Win rate when ≥ 4/5 overlap | 57.7% |
| Win rate when ≤ 2/5 overlap | 39.6% |

---

## Setup

```bash
git clone <repo>
cd alleyloop

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Place raw play-by-play CSV files in `data/raw/`:
```
data/raw/cdnnba_2020.csv
data/raw/cdnnba_2021.csv
data/raw/cdnnba_2022.csv
data/raw/cdnnba_2023.csv
data/raw/cdnnba_2024.csv
```

---

## Running the Pipeline

```bash
# Full run: raw CSV → all processed outputs → models
python run_pipeline.py

# Skip Phase 1-2 if parquets already exist
python run_pipeline.py --skip-raw

# Skip Phase 1-3 and models (fastest re-run)
python run_pipeline.py --skip-raw --skip-models
```

### Running individual modules

```bash
# Phase 2 — box scores only
python -m src.pipeline

# Phase 3 — feature engineering
python -m src.features

# Phase 4 — lineup optimiser evaluation
python -m src.optimizer

# Phase 5 — predictor
python -m src.predictor

# Phase 7 — game model
python -m src.game_model

# Validation — crosscheck 5% of games against nba_api
python -m src.crosscheck
```

---

## Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with three tabs:

| Tab | Description |
|-----|-------------|
| **📊 Player Stats** | Current-form EWMA table for any team + player search |
| **🏆 Lineup Optimizer** | ILP-optimised starting 5 with composite scores |
| **🎯 Win Predictor** | Select home + away team → win probability + key differentials |

---

## Data Outputs

| File | Description |
|------|-------------|
| `data/processed/box_scores.parquet` | 127,651 player-game rows, 34 stats |
| `data/processed/lineup_stats.parquet` | 88,539 unique 5-player lineups |
| `data/processed/lineup_ratings.parquet` | Net/off/def rating + player embeddings |
| `data/processed/game_schedule.parquet` | 5,995 games with home/away team IDs |
| `data/processed/player_positions.parquet` | 557 players with G/F/C positions |
| `data/features/features.parquet` | 127,651 × 125 feature matrix |
| `data/processed/predictor_results.parquet` | MAE/RMSE per model/target |
| `data/processed/game_features.parquet` | 5,979 game-level feature rows |
| `data/processed/game_model_results.parquet` | Accuracy/AUC/Brier per model |
| `data/processed/optimizer_eval_2024.parquet` | Per-game lineup overlap metrics |
| `figures/shap_*.png` | SHAP beeswarm plots for each prediction target |
| `figures/win_prob_calibration.png` | Calibration curve for win probability models |

---

## Key Design Decisions

**No data leakage** — all rolling/EWMA features use `shift(1)` before windowing,
so each game row only contains statistics from prior games.

**Temporal split** — train on seasons 2020–2023, test on 2024. No shuffle splits
on time-series data.

**Active roster filtering** — the lineup optimiser restricts candidates to players
appearing in the last 20 games, preventing traded/injured players from polluting
the recommendation pool.

**Position inference fallback** — when nba_api PlayerIndex lacks a position entry,
a heuristic classifies players: reb ≥ 9 AND blk ≥ 0.8 → C; ast ≥ 5 → G; else → F.

---

## Stack

`pandas` · `pyarrow` · `scikit-learn` · `xgboost` · `shap` · `pulp` · `streamlit` · `nba_api` · `matplotlib`
