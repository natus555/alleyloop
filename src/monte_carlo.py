"""
Phase 9 — Monte Carlo Season Simulation  [STUB]

Planned approach
----------------
1. Load remaining schedule : game_schedule.parquet filtered to future games
2. Load win model          : LogisticRegression from game_model.py
3. Per simulation (N=1000):
     a. For each remaining game, draw a Bernoulli(p_home) result
     b. Accumulate team wins across the schedule
4. Aggregate across simulations:
     - E[wins]          per team
     - P(≥ 50 wins)     per team
     - P(top-6 seed)    per conference — direct playoff berth
     - P(7-10 seed)     — play-in tournament
     - Seed distribution histogram per team
5. Output : monte_carlo_results.parquet (N × 30 win-total matrix)
            playoff_odds.parquet (one row per team with probabilities)

Dependencies: game_model.py must be run first (produces game_features.parquet
              and the trained model coefficients).

Status: win-probability model (Phase 7) is complete.
        Remaining-schedule extraction and simulation loop are stubs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
N_SIMULATIONS = 1_000
RANDOM_SEED   = 42

# Standings cutoffs (2023-24 format: 30 teams, 82-game season)
PLAYOFF_DIRECT  = 6   # top-6 per conference → direct playoff
PLAYOFF_PLAYIN  = 10  # 7-10 → play-in tournament


def load_remaining_schedule(current_date: str) -> pd.DataFrame:
    """
    Return scheduled games with game_date > current_date that have
    not yet been played (no entry in box_scores.parquet).

    Parameters
    ----------
    current_date : ISO date string, e.g. "2024-03-01"

    Returns
    -------
    DataFrame with columns: gameId, game_date, home_team_id, away_team_id, season
    """
    raise NotImplementedError(
        "Phase 9 stub — filter game_schedule.parquet to games after current_date "
        "that have no matching rows in box_scores.parquet."
    )


def simulate_season(remaining: pd.DataFrame,
                    win_prob_fn,
                    standings_to_date: pd.DataFrame,
                    n: int = N_SIMULATIONS,
                    seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Run `n` Monte Carlo simulations over `remaining` games.

    Parameters
    ----------
    remaining      : output of load_remaining_schedule()
    win_prob_fn    : callable(home_team_id, away_team_id) → float in [0, 1]
                     (wrap game_model.py's LogReg predict_proba)
    standings_to_date : DataFrame [teamId, wins, losses] — current standings
    n              : number of simulations
    seed           : random seed for reproducibility

    Returns
    -------
    DataFrame of shape (n, 30) — each column is a team, each row is a sim,
    values are total wins at season end.
    """
    raise NotImplementedError(
        "Phase 9 stub — vectorise Bernoulli draws over the schedule matrix "
        "and accumulate per-team win totals."
    )


def compute_playoff_odds(sim_matrix: pd.DataFrame,
                         conference_map: dict) -> pd.DataFrame:
    """
    From the simulation win-total matrix, compute playoff odds per team.

    Parameters
    ----------
    sim_matrix     : output of simulate_season()
    conference_map : {teamId: 'East'|'West'}

    Returns
    -------
    DataFrame [teamId, expected_wins, p_direct_playoff, p_playin, p_miss_playoffs]
    """
    raise NotImplementedError(
        "Phase 9 stub — rank teams within each conference per simulation, "
        "then aggregate probabilities across simulations."
    )


def run(current_date: str = "2024-03-01") -> pd.DataFrame:
    """
    Entry point — run full Monte Carlo simulation from current_date.
    Prints a playoff odds table and saves results.
    """
    raise NotImplementedError("Phase 9 stub.")


if __name__ == "__main__":
    run()
