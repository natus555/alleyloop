"""
Phase 10 — Trade Impact Analysis  [STUB]

Planned approach
----------------
Given a proposed trade (Team A sends players P1, P2 → Team B; receives Q1, Q2):

1. Pre-trade baseline
   a. For each team, run optimizer.recommend_lineup() → composite score
   b. Look up lineup_ratings.parquet for the current starting-5 net rating
   c. Compute team-level EWMA aggregates for game_model win probability

2. Post-trade roster
   a. Swap player embeddings: remove departing players, add arriving players
   b. Re-run optimizer on modified roster → new recommended lineup
   c. Re-run game_model feature aggregation → new win probability vector

3. Delta metrics
   composite_score_delta : post_score − pre_score
   net_rating_delta      : estimated using arriving vs departing player embeddings
   win_prob_delta        : expected change in win% against league-average opponent

4. Output
   trade_analysis.parquet : one row per team in the trade with all delta metrics
   Print-friendly summary in the Streamlit dashboard (Phase 11 / future)

Limitation: player fit (chemistry, role overlap, positional balance) is captured
            only via positional constraints in the ILP — shot-creation and spacing
            interactions are not modelled.

Status: depends on Phase 4 (optimizer) and Phase 7 (game model), both complete.
        Core delta computation is a stub.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

FEATURES_DIR  = Path("data/features")
PROCESSED_DIR = Path("data/processed")


def build_post_trade_roster(feats: pd.DataFrame,
                             team_id: int,
                             players_out: list[int],
                             players_in: list[int]) -> pd.DataFrame:
    """
    Return a modified version of the latest player-feature rows for `team_id`
    with `players_out` removed and `players_in` added (using their existing
    rows from feats, wherever their last team was).

    Parameters
    ----------
    feats       : features.parquet DataFrame
    team_id     : the team making the acquisition
    players_out : list of personId values leaving the team
    players_in  : list of personId values arriving

    Returns
    -------
    DataFrame of one row per player on the post-trade roster,
    with the same columns as feats.
    """
    raise NotImplementedError(
        "Phase 10 stub — filter to team_id latest rows, drop players_out, "
        "append latest rows for players_in from their current team."
    )


def evaluate_trade(team_a_id: int,
                   team_b_id: int,
                   a_sends: list[int],
                   b_sends: list[int]) -> pd.DataFrame:
    """
    Evaluate a two-team trade.

    Parameters
    ----------
    team_a_id : team A's teamId
    team_b_id : team B's teamId
    a_sends   : list of personId — players Team A sends to Team B
    b_sends   : list of personId — players Team B sends to Team A

    Returns
    -------
    DataFrame with one row per team:
        teamId, pre_score, post_score, score_delta,
        pre_net_rating, post_net_rating, net_rating_delta,
        pre_win_pct_vs_avg, post_win_pct_vs_avg, win_pct_delta
    """
    raise NotImplementedError(
        "Phase 10 stub — call build_post_trade_roster() for each team, "
        "then re-run optimizer and game_model feature aggregation."
    )


def run_example():
    """
    Example: analyse the 2023-24 Rudy Gobert trade impact on MIN and UTA.
    personIds: Gobert=203497
    """
    raise NotImplementedError("Phase 10 stub.")


if __name__ == "__main__":
    run_example()
