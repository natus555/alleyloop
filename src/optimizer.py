"""
Phase 4 — Starting Lineup Optimizer

Composite score formula:
    score = pts + 1.2*reb + 1.5*ast + 2*stl + 2*blk - 1.5*tov + 20*ts_pct + 0.3*min
    (all using EWMA values for current-form weighting)

Positional constraints (nba_api provides G / F / C):
    ≥ 2 guards  (covers PG + SG)
    ≥ 2 forwards (covers SF + PF)
    ≥ 1 center
    exactly 5 starters total

ILP via PuLP/CBC; falls back to greedy if infeasible.
"""

import time
import warnings
import numpy as np
import pandas as pd
import pulp
from pathlib import Path

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR  = Path("data/features")

SCORE_WEIGHTS = {
    "pts_ewma":    1.0,
    "reb_ewma":    1.2,
    "ast_ewma":    1.5,
    "stl_ewma":    2.0,
    "blk_ewma":    2.0,
    "tov_ewma":   -1.5,
    "ts_pct_ewma": 20.0,
    "min_ewma":    0.3,
}

MIN_MINUTES_THRESHOLD = 5.0   # exclude deep bench / injured (avg < 5 min)


# ── Positions ────────────────────────────────────────────────────────────────

def fetch_player_positions() -> pd.DataFrame:
    """
    Fetch player positions from nba_api PlayerIndex for all 5 seasons.
    Caches result to data/processed/player_positions.parquet.
    Returns DataFrame with columns [personId, position].
    """
    cache = PROCESSED_DIR / "player_positions.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    from nba_api.stats.endpoints import PlayerIndex

    SEASON_MAP = {
        2020: "2020-21", 2021: "2021-22", 2022: "2022-23",
        2023: "2023-24", 2024: "2024-25",
    }
    frames = []
    for year, ss in SEASON_MAP.items():
        pi = PlayerIndex(season=ss, timeout=30)
        df = pi.get_data_frames()[0]
        df["season"] = year
        frames.append(df[["PERSON_ID", "POSITION", "season"]])
        print(f"  {ss}: {len(df)} players")
        time.sleep(0.65)

    raw = pd.concat(frames, ignore_index=True)
    raw["PERSON_ID"] = pd.to_numeric(raw["PERSON_ID"], errors="coerce")

    # Keep most recent (highest season) non-null position per player
    pos = (
        raw.dropna(subset=["POSITION"])
        .sort_values("season")
        .drop_duplicates("PERSON_ID", keep="last")
        [["PERSON_ID", "POSITION"]]
        .rename(columns={"PERSON_ID": "personId", "POSITION": "position"})
        .reset_index(drop=True)
    )
    pos.to_parquet(cache, index=False)
    print(f"Saved: {cache}  ({len(pos)} players with positions)")
    return pos


def _infer_position(row: pd.Series) -> str:
    """Stat-based position fallback when nba_api position is missing."""
    pos = row.get("position", None)
    if pos and not pd.isna(pos) and str(pos).strip():
        return str(pos)
    reb = row.get("reb_ewma") or 0
    ast = row.get("ast_ewma") or 0
    blk = row.get("blk_ewma") or 0
    # True center: dominant rebounder AND shot-blocker
    if reb >= 9 and blk >= 0.8:
        return "C"
    # Guard: primary ball-handler
    if ast >= 5 or (ast >= 3 and reb < 5):
        return "G"
    return "F"


def _pos_flags(position: str) -> dict:
    """
    Map nba_api position string (G, F, C, G-F, C-F, …) to boolean flags.
    Returns {"is_guard": bool, "is_forward": bool, "is_center": bool}.
    """
    if pd.isna(position) or position == "":
        return {"is_guard": True, "is_forward": False, "is_center": False}
    p = str(position).upper()
    return {
        "is_guard":   "G" in p,
        "is_forward": "F" in p,
        "is_center":  "C" in p,
    }


# ── Composite Score ───────────────────────────────────────────────────────────

def composite_score(df: pd.DataFrame) -> pd.Series:
    """Weighted EWMA efficiency score. Higher = better player to start."""
    score = pd.Series(0.0, index=df.index)
    for col, w in SCORE_WEIGHTS.items():
        if col in df.columns:
            score += df[col].fillna(0.0) * w
    return score.round(3)


# ── ILP Optimizer ─────────────────────────────────────────────────────────────

def optimize_lineup(candidates: pd.DataFrame, n_starters: int = 5) -> pd.DataFrame:
    """
    ILP: select n_starters from candidates to maximise composite score.
    candidates must have columns: personId, playerName, position, score.

    Positional constraints:
        ≥ 2 guards, ≥ 2 forwards, ≥ 1 center, total = n_starters.

    Returns DataFrame of selected starters (subset of candidates).
    Falls back to greedy top-n if ILP is infeasible.
    """
    df = candidates.copy().reset_index(drop=True)

    flags = df["position"].apply(_pos_flags).apply(pd.Series)
    df = pd.concat([df, flags], axis=1)

    n = len(df)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)]

    prob = pulp.LpProblem("lineup_optimizer", pulp.LpMaximize)
    prob += pulp.lpSum(df.loc[i, "score"] * x[i] for i in range(n))
    prob += pulp.lpSum(x) == n_starters

    guards   = [i for i in range(n) if df.loc[i, "is_guard"]]
    forwards = [i for i in range(n) if df.loc[i, "is_forward"]]
    centers  = [i for i in range(n) if df.loc[i, "is_center"]]

    if len(guards)   >= 2: prob += pulp.lpSum(x[i] for i in guards)   >= 2
    if len(forwards) >= 2: prob += pulp.lpSum(x[i] for i in forwards) >= 2
    if len(centers)  >= 1: prob += pulp.lpSum(x[i] for i in centers)  >= 1

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[status] == "Optimal":
        chosen = [i for i in range(n) if (x[i].value() or 0) > 0.5]
    else:
        chosen = df.nlargest(n_starters, "score").index.tolist()

    return df.loc[chosen, ["personId", "playerName", "position", "score"]].copy()


# ── Recommend for a single game ───────────────────────────────────────────────

def _active_roster(team_history: pd.DataFrame, n_recent: int = 20) -> pd.DataFrame:
    """
    Return the most recent stats per player, restricted to players who
    appeared for this team in the last n_recent games.
    Prevents including traded/ex-players from prior seasons in the pool.
    """
    sorted_games = (
        team_history[["gameId", "game_date"]]
        .dropna(subset=["game_date"])
        .drop_duplicates()
        .sort_values("game_date")
    )
    recent_game_ids = set(sorted_games["gameId"].tail(n_recent).tolist())
    active_pids = team_history[
        team_history["gameId"].isin(recent_game_ids)
    ]["personId"].unique()

    return (
        team_history[team_history["personId"].isin(active_pids)]
        .sort_values("game_date")
        .groupby("personId").last()
        .reset_index()
    )


def recommend_lineup(team_id: int, game_id: int) -> pd.DataFrame:
    """
    Recommend the optimal starting 5 for team_id before game_id.
    Uses EWMA stats from current-season games prior to game_id.
    """
    feats = pd.read_parquet(FEATURES_DIR / "features.parquet")
    pos   = fetch_player_positions()

    team_history = feats[
        (feats["teamId"].astype("int64") == int(team_id)) &
        (feats["gameId"] < game_id)
    ]
    if team_history.empty:
        return pd.DataFrame()

    latest = _active_roster(team_history)
    latest = latest.merge(pos, on="personId", how="left")
    latest["position"] = latest.apply(_infer_position, axis=1)
    latest["score"] = composite_score(latest)

    active = latest[latest["min_ewma"].fillna(0) >= MIN_MINUTES_THRESHOLD]
    if len(active) < 5:
        active = latest.nlargest(5, "score")

    return optimize_lineup(active)


# ── Evaluate: 2024 season ─────────────────────────────────────────────────────

def evaluate(season: int = 2024) -> pd.DataFrame:
    """
    For every game in `season`, recommend the optimal lineup for each team.
    Compare vs actual starters (top 5 by minutes).
    Report: overlap %, game outcome correlation.

    Returns a summary DataFrame with one row per team-game.
    """
    feats = pd.read_parquet(FEATURES_DIR / "features.parquet")
    box   = pd.read_parquet(PROCESSED_DIR / "box_scores.parquet")
    sched = pd.read_parquet(PROCESSED_DIR / "game_schedule.parquet")
    pos   = fetch_player_positions()

    season_sched = sched[sched["season"] == season].sort_values("game_date")
    season_box   = box[box["season"] == season].copy()
    season_box["teamId"] = season_box["teamId"].astype("int64")

    # Compute team pts per game for W/L
    team_game_pts = (
        season_box.groupby(["gameId", "teamId"])["pts"].sum().reset_index()
    )
    game_pts = team_game_pts.merge(
        team_game_pts.rename(columns={"teamId": "opp_team_id", "pts": "opp_pts"}),
        on="gameId",
    )
    game_pts = game_pts[game_pts["teamId"] != game_pts["opp_team_id"]]
    game_pts["won"] = (game_pts["pts"] > game_pts["opp_pts"]).astype(int)

    records = []
    total = len(season_sched)
    print(f"Evaluating {total} games in {season} season...")

    for i, (_, game) in enumerate(season_sched.iterrows(), 1):
        if i % 200 == 0:
            print(f"  {i}/{total}")

        gid  = game["gameId"]
        date = game["game_date"]

        for team_id in [game["home_team_id"], game["away_team_id"]]:
            # ── actual starters: top 5 by minutes in this game ──────────────
            actual_game = season_box[
                (season_box["gameId"] == gid) & (season_box["teamId"] == team_id)
            ]
            if actual_game.empty:
                continue
            actual_starters = set(
                actual_game.nlargest(5, "min")["personId"].tolist()
            )

            # ── recommended lineup: EWMA from all prior games ───────────────
            team_history = feats[
                (feats["teamId"].astype("int64") == int(team_id)) &
                (feats["gameId"] < gid)
            ]
            if team_history.empty:
                continue

            latest = _active_roster(team_history)
            latest = latest.merge(pos, on="personId", how="left")
            latest["position"] = latest.apply(_infer_position, axis=1)
            latest["score"] = composite_score(latest)
            active = latest[latest["min_ewma"].fillna(0) >= MIN_MINUTES_THRESHOLD]
            if len(active) < 5:
                active = latest.nlargest(5, "score")

            try:
                recommended = optimize_lineup(active)
                rec_ids = set(recommended["personId"].tolist())
            except Exception:
                continue

            # ── metrics ─────────────────────────────────────────────────────
            overlap = len(actual_starters & rec_ids)
            won_row = game_pts[
                (game_pts["gameId"] == gid) & (game_pts["teamId"] == team_id)
            ]
            won = int(won_row["won"].iloc[0]) if not won_row.empty else None

            rec_score_total  = recommended["score"].sum()
            actual_score_total = latest[
                latest["personId"].isin(actual_starters)
            ]["score"].sum()

            records.append({
                "gameId":    gid,
                "game_date": date,
                "teamId":    team_id,
                "overlap_5": overlap,
                "won":       won,
                "rec_score": round(rec_score_total, 2),
                "act_score": round(actual_score_total, 2),
                "rec_better": int(rec_score_total > actual_score_total),
            })

    results = pd.DataFrame(records)
    if results.empty:
        print("No results.")
        return results

    print(f"\n{'='*50}")
    print(f"EVALUATION — {season} season  ({len(results):,} team-games)")
    print(f"{'='*50}")
    print(f"  Avg lineup overlap (out of 5) : {results['overlap_5'].mean():.2f}")
    print(f"  Exact 5/5 match               : {(results['overlap_5']==5).mean()*100:.1f}%")
    print(f"  ≥4/5 match                    : {(results['overlap_5']>=4).mean()*100:.1f}%")
    print(f"  ≥3/5 match                    : {(results['overlap_5']>=3).mean()*100:.1f}%")

    valid = results[results["won"].notna()]
    if not valid.empty:
        higher_score_wins = valid[valid["rec_better"] == 0]  # actual lineup already better
        print(f"\n  Rec lineup > actual score     : {results['rec_better'].mean()*100:.1f}% of games")
        win_when_match = valid[valid["overlap_5"] >= 4]["won"].mean()
        win_when_diff  = valid[valid["overlap_5"] <= 2]["won"].mean()
        print(f"  Win rate when ≥4/5 overlap    : {win_when_match*100:.1f}%")
        print(f"  Win rate when ≤2/5 overlap    : {win_when_diff*100:.1f}%")

    out = PROCESSED_DIR / f"optimizer_eval_{season}.parquet"
    results.to_parquet(out, index=False)
    print(f"\nSaved: {out}")
    return results


if __name__ == "__main__":
    print("Fetching player positions...")
    fetch_player_positions()
    print("\nRunning 2024 season evaluation...")
    evaluate(season=2024)
