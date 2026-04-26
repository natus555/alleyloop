# AlleyLoop — NBA Analytics Platform

## Project Overview
Basketball analytics platform using NBA play-by-play CSV data 
from seasons 2020–2024. End goal: player performance prediction, 
lineup optimization, and win probability modeling.

## Data
- Raw CSVs in data/raw/ — one file per season (2020–2024)
- Processed parquet files in data/processed/
- Feature-engineered datasets in data/features/

## Stack
- Python, pandas, pyarrow, scikit-learn, xgboost, shap, pulp, streamlit

## Conventions
- Always save intermediate outputs as .parquet (not CSV)
- Use season 2020–2023 for training, 2024 for testing (time-series split)
- Never use random shuffle splits on temporal data
- All model evaluation reports MAE, RMSE, and feature importance

## Modules
- src/pipeline.py    → PBP schema unification, box score aggregation, lineup stints
- src/features.py    → rolling windows, EWMA, contextual features (rest days, home/away)
- src/optimizer.py   → starting lineup optimizer (PuLP ILP)
- src/predictor.py   → player performance prediction (RF, XGBoost, LSTM)
- src/game_model.py  → game outcome prediction
- src/crosscheck.py  → validation against official NBA stats (nba_api)
- app.py             → Streamlit dashboard

## Data Flow
data/raw/*.csv → pipeline.run() → data/processed/cdnnba_*.parquet
data/processed/cdnnba_*.parquet → pipeline.build_box_scores() → data/processed/box_scores.parquet
data/processed/cdnnba_*.parquet → pipeline.build_lineup_stats() → data/processed/lineup_stats.parquet
data/processed/box_scores.parquet → features.build() → data/features/features.parquet