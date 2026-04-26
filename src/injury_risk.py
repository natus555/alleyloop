"""
Phase 8 — Injury Risk Modeling  [STUB]

Planned approach
----------------
1. Load signal   : player-level rolling/EWMA minutes from features.parquet
2. Load history  : injured/missed-game flags from a future injury log (not yet collected)
3. Risk signals  :
     - minutes spike  : recent_min / season_avg_min  > 1.25
     - back-to-back streak : is_back_to_back rolling 5-game sum
     - fatigue index  : rolling 10-game minutes z-score
4. Model         : LogisticRegression / Gradient Boosted trees on risk signals → P(injured next game)
5. Output        : per-player risk score table saved to data/processed/injury_risk.parquet
6. Dashboard     : amber/red flags shown on player-stats tab in app.py

Status: waiting on injury log data (not in CDN CSV source).
        Heuristic load-monitoring (steps 1-3) is implementable today.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

FEATURES_DIR  = Path("data/features")
PROCESSED_DIR = Path("data/processed")

# Thresholds for heuristic load flags (no model needed)
MIN_SPIKE_RATIO   = 1.25   # recent_min / season_avg > this → load spike
BTB_WINDOW        = 5      # rolling window for back-to-back count
BTB_FLAG_THRESH   = 2      # ≥ 2 B2Bs in last 5 games → high fatigue


def compute_load_signals(feats: pd.DataFrame) -> pd.DataFrame:
    """
    Add heuristic load-monitoring columns to a player-feature DataFrame.

    Columns added
    -------------
    min_spike      : bool  — recent 5-game avg minutes > 1.25× season avg
    btb_count_5    : int   — back-to-backs in last 5 games
    high_load_flag : bool  — any risk threshold exceeded

    Parameters
    ----------
    feats : features.parquet DataFrame (must have min_roll5, min_season_avg,
            is_back_to_back, personId, game_date)

    Returns
    -------
    DataFrame with extra load columns (same row count as input).
    """
    raise NotImplementedError(
        "Phase 8 stub — implement load signal computation here.\n"
        "Hint: feats already has min_roll5, min_season_avg, is_back_to_back."
    )


def score_players(season: int = 2024) -> pd.DataFrame:
    """
    Compute an injury-risk score for every active player in `season`.

    Returns a DataFrame with columns:
        personId, playerName, teamTricode, game_date,
        min_spike, btb_count_5, high_load_flag, risk_score [0-1]

    Risk score is currently heuristic (no trained model).
    When injury log data is available, replace with model probability.
    """
    raise NotImplementedError(
        "Phase 8 stub — load features.parquet, call compute_load_signals(), "
        "aggregate risk_score per player, save to injury_risk.parquet."
    )


def run():
    """Entry point — score all 2024 season players and print top risks."""
    raise NotImplementedError("Phase 8 stub.")


if __name__ == "__main__":
    run()
