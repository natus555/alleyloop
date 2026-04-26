"""
Feature engineering for AlleyLoop player performance prediction.

Input  : data/processed/box_scores.parquet
         data/processed/game_schedule.parquet
         data/processed/shot_features.parquet      (optional, from shot_features.py)
         data/processed/matchup_features.parquet   (optional, from matchup_features.py)
Output : data/features/features.parquet

Feature groups
--------------
Context     : game_date, is_home, rest_days, is_back_to_back, games_on_current_team
Rolling     : 3-game, 5-game, 10-game trailing means (shifted to avoid leakage)
EWMA        : exponentially weighted mean (alpha=0.3, shifted)
              - cross-season (career)       : player_grp EWMA
              - trade-aware (team stint)    : stint_grp EWMA (resets at trade)
Season-to-date : expanding mean within season (shifted)
Shot quality: shot zone rates, eFG%, shot distance (from shotdetail if available)
Matchup     : defender FG% faced, switches_on (from matchups if available)
Opponent    : opponent season-to-date pts allowed per game + eFG% allowed
H2H         : head-to-head win%, avg point diff (last 10 meetings)
Pace        : team possessions per game, pace rank, ORtg, DRtg
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR  = Path("data/features")

ROLL_STATS = [
    "pts", "reb", "ast", "stl", "blk", "tov",
    "min", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    "ts_pct", "efg_pct", "usg_pct", "fouls_drawn",
]
WINDOWS    = [3, 5, 10]
EWMA_ALPHA = 0.3


def _rolling(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


def _ewma(series: pd.Series) -> pd.Series:
    return series.shift(1).ewm(alpha=EWMA_ALPHA, adjust=False, min_periods=1).mean()


def _expanding(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding(min_periods=1).mean()


# ── H2H helper ────────────────────────────────────────────────────────────────

def _build_h2h(df: pd.DataFrame, sched: pd.DataFrame, box: pd.DataFrame,
               n_prev: int = 10) -> pd.DataFrame:
    """
    For each game, compute:
      h2h_win_pct_home  — home team's win% in last n_prev meetings vs this opp
      h2h_pts_diff_avg  — avg (home_pts - away_pts) over last n_prev meetings
    Returns per-game DataFrame merged back onto df by gameId.
    """
    box_num = box.copy()
    box_num["teamId"] = pd.to_numeric(box_num["teamId"], errors="coerce").astype("Int64")
    team_pts = (
        box_num.groupby(["gameId", "teamId"])["pts"].sum().reset_index()
    )
    sched_h = sched[["gameId", "game_date", "season",
                     "home_team_id", "away_team_id"]].copy()
    sched_h["home_team_id"] = sched_h["home_team_id"].astype("Int64")
    sched_h["away_team_id"] = sched_h["away_team_id"].astype("Int64")

    home_pts = (
        team_pts.merge(sched_h, on="gameId")
                .query("teamId == home_team_id")
                .rename(columns={"pts": "home_pts"})[["gameId","game_date","home_team_id","away_team_id","home_pts"]]
    )
    away_pts = (
        team_pts.merge(sched_h, on="gameId")
                .query("teamId == away_team_id")
                .rename(columns={"pts": "away_pts"})[["gameId","away_pts"]]
    )
    gpts = home_pts.merge(away_pts, on="gameId", how="inner")
    gpts["home_win"]  = (gpts["home_pts"] > gpts["away_pts"]).astype(int)
    gpts["pts_diff"]  = gpts["home_pts"] - gpts["away_pts"]
    gpts = gpts.sort_values("game_date").reset_index(drop=True)

    # For each game, look at previous n_prev meetings between SAME pair
    records = []
    for _, row in gpts.iterrows():
        prev = gpts[
            (gpts["game_date"] < row["game_date"]) &
            (
                ((gpts["home_team_id"] == row["home_team_id"]) &
                 (gpts["away_team_id"] == row["away_team_id"])) |
                ((gpts["home_team_id"] == row["away_team_id"]) &
                 (gpts["away_team_id"] == row["home_team_id"]))
            )
        ].tail(n_prev)

        if prev.empty:
            records.append({"gameId": row["gameId"],
                            "h2h_win_pct_home": 0.5,
                            "h2h_pts_diff_avg": 0.0,
                            "h2h_games":        0})
            continue

        # Orient so home team = current game's home team
        prev = prev.copy()
        flipped = prev["home_team_id"] == row["away_team_id"]
        prev.loc[flipped, "home_win"] = 1 - prev.loc[flipped, "home_win"]
        prev.loc[flipped, "pts_diff"] = -prev.loc[flipped, "pts_diff"]

        records.append({
            "gameId":           row["gameId"],
            "h2h_win_pct_home": prev["home_win"].mean(),
            "h2h_pts_diff_avg": prev["pts_diff"].mean(),
            "h2h_games":        len(prev),
        })

    return pd.DataFrame(records)


# ── Pace & efficiency ─────────────────────────────────────────────────────────

def _build_team_efficiency(box: pd.DataFrame,
                           sched: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-game team ORtg, DRtg, Pace and rolling season-to-date averages.
    Returns one row per (gameId, teamId) with pre-game rolling features.
    """
    box2 = box.copy()
    box2["teamId"] = pd.to_numeric(box2["teamId"], errors="coerce").astype("Int64")
    sched2 = sched[["gameId", "game_date", "season",
                    "home_team_id", "away_team_id"]].copy()

    # Team totals per game
    team_game = (
        box2.groupby(["gameId", "teamId", "season"]).agg(
            pts=("pts", "sum"),
            poss=("team_poss", "mean"),  # avg poss across players (team possession total)
            fga=("fga", "sum"),
            fg3m=("fg3m", "sum"),
            fgm=("fgm", "sum"),
            fta=("fta", "sum"),
            tov=("tov", "sum"),
            oreb=("oreb", "sum"),
        ).reset_index()
    )
    # Pace = total FGA + 0.44*FTA - OREB + TOV  (simplified)
    team_game["pace_proxy"] = (
        team_game["fga"] + 0.44 * team_game["fta"]
        - team_game["oreb"] + team_game["tov"]
    )
    # eFG% allowed requires the opponent's data
    # Merge with schedule to identify opponents
    team_game = team_game.merge(
        sched2[["gameId", "game_date", "home_team_id", "away_team_id"]],
        on="gameId", how="left"
    )

    # Cross-join to get opponent pts
    opp = team_game[["gameId", "teamId", "pts", "fga", "fg3m", "fgm", "fta", "tov", "oreb"]]
    opp = opp.rename(columns={c: f"opp_{c}" for c in opp.columns if c not in ["gameId", "teamId"]})
    opp = opp.rename(columns={"teamId": "opp_teamId"})

    team_game = team_game.merge(opp, on="gameId", how="left")
    team_game = team_game[team_game["teamId"] != team_game["opp_teamId"]]

    # ORtg = pts / pace_proxy * 100,  DRtg = opp_pts / pace_proxy * 100
    safe_pace = team_game["pace_proxy"].replace(0, np.nan)
    team_game["ortg"]    = (team_game["pts"]     / safe_pace * 100).fillna(110)
    team_game["drtg"]    = (team_game["opp_pts"] / safe_pace * 100).fillna(110)
    team_game["net_rtg"] = team_game["ortg"] - team_game["drtg"]
    team_game["team_efg"] = (
        (team_game["fgm"] + 0.5 * team_game["fg3m"])
        / team_game["fga"].replace(0, np.nan)
    ).fillna(0)
    team_game["opp_efg"] = (
        (team_game["opp_fgm"] + 0.5 * team_game["opp_fg3m"])
        / team_game["opp_fga"].replace(0, np.nan)
    ).fillna(0)

    # Sort and compute rolling pre-game (shifted)
    team_game = team_game.sort_values(["teamId", "game_date", "gameId"])
    tgrp = team_game.groupby(["teamId", "season"])

    EFF_COLS = ["ortg", "drtg", "net_rtg", "pace_proxy", "team_efg", "opp_efg"]
    for col in EFF_COLS:
        for w in [5, 10]:
            team_game[f"{col}_roll{w}"] = tgrp[col].transform(
                lambda x, w=w: _rolling(x, w)
            )
        team_game[f"{col}_szn"] = tgrp[col].transform(_expanding)

    keep = ["gameId", "teamId", "game_date", "season"] + [
        c for c in team_game.columns if any(
            c.endswith(f"_roll{w}") or c.endswith("_szn")
            for w in [5, 10]
        )
    ]
    return team_game[[c for c in keep if c in team_game.columns]].drop_duplicates(
        subset=["gameId", "teamId"]
    )


# ── Main build ────────────────────────────────────────────────────────────────

def build(output_path: str | None = None,
          include_shot: bool = True,
          include_matchup: bool = True) -> pd.DataFrame:

    output_path = Path(output_path) if output_path else FEATURES_DIR / "features.parquet"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    box   = pd.read_parquet(PROCESSED_DIR / "box_scores.parquet")
    sched = pd.read_parquet(PROCESSED_DIR / "game_schedule.parquet")

    # ── Step 1: attach game metadata ─────────────────────────────────────────
    box_gd = "game_date" in box.columns
    if box_gd:
        box = box.rename(columns={"game_date": "_box_game_date"})

    df = box.merge(
        sched[["gameId", "game_date", "home_team_id", "away_team_id"]],
        on="gameId", how="left",
    )

    if box_gd:
        df["game_date"] = pd.to_datetime(df["_box_game_date"]).fillna(
            pd.to_datetime(df["game_date"])
        )
        df.drop(columns=["_box_game_date"], inplace=True)
    else:
        df["game_date"] = pd.to_datetime(df["game_date"])

    df["teamId_i64"] = pd.to_numeric(df["teamId"], errors="coerce").astype("int64")
    df["is_home"] = (df["teamId_i64"] == df["home_team_id"]).astype("int8")
    df["opp_team_id"] = np.where(
        df["teamId_i64"] == df["home_team_id"],
        df["away_team_id"], df["home_team_id"],
    )
    df.drop(columns=["teamId_i64"], inplace=True)
    df = df.dropna(subset=["game_date"])
    df = df.sort_values(["personId", "game_date", "gameId"]).reset_index(drop=True)

    # ── Step 2: rest days, back-to-back, trade context ───────────────────────
    prev_date = df.groupby("personId")["game_date"].shift(1)
    df["rest_days"]      = (df["game_date"] - prev_date).dt.days.fillna(7).astype("int16")
    df["is_back_to_back"]= (df["rest_days"] == 1).astype("int8")

    if "games_on_current_team" not in df.columns:
        df["games_on_current_team"] = 1
    if "team_stint_id" not in df.columns:
        df["team_stint_id"] = 0
    df["games_on_current_team"] = df["games_on_current_team"].fillna(1).astype("int16")
    df["team_stint_id"]         = df["team_stint_id"].fillna(0).astype("int16")

    # ── Step 3a: rolling windows ──────────────────────────────────────────────
    print("Computing rolling features...", flush=True)
    player_grp = df.groupby("personId")
    for stat in ROLL_STATS:
        if stat not in df.columns:
            continue
        s = player_grp[stat].transform
        for w in WINDOWS:
            df[f"{stat}_roll{w}"] = s(lambda x, w=w: _rolling(x, w))
        df[f"{stat}_ewma"] = s(lambda x: _ewma(x))

    # ── Step 3b: trade-aware EWMA ─────────────────────────────────────────────
    print("Computing stint EWMA...", flush=True)
    STINT_STATS = ["pts","reb","ast","stl","blk","tov","min","ts_pct","usg_pct"]
    stint_grp = df.groupby(["personId", "team_stint_id"])
    for stat in STINT_STATS:
        if stat in df.columns:
            df[f"{stat}_stint_ewma"] = stint_grp[stat].transform(lambda x: _ewma(x))

    # ── Step 4: season-to-date ────────────────────────────────────────────────
    print("Computing season-to-date...", flush=True)
    player_season_grp = df.groupby(["personId", "season"])
    for stat in ROLL_STATS:
        if stat in df.columns:
            df[f"{stat}_season_avg"] = player_season_grp[stat].transform(_expanding)

    # ── Step 5: opponent defensive context ───────────────────────────────────
    print("Computing opponent defensive context...", flush=True)
    team_pts = (
        box.groupby(["gameId", "teamId"])["pts"]
        .sum().reset_index(name="team_pts")
    )
    team_pts["teamId"] = pd.to_numeric(team_pts["teamId"], errors="coerce").astype("int64")

    s2 = sched[["gameId","game_date","season","home_team_id","away_team_id"]].copy()
    home_pts = team_pts.rename(columns={"teamId":"home_team_id","team_pts":"home_pts"})
    away_pts = team_pts.rename(columns={"teamId":"away_team_id","team_pts":"away_pts"})
    game_pts = (
        s2.merge(home_pts, on=["gameId","home_team_id"], how="left")
          .merge(away_pts,  on=["gameId","away_team_id"], how="left")
    )
    home_def = game_pts[["gameId","game_date","season","home_team_id","away_pts"]].rename(
        columns={"home_team_id":"teamId","away_pts":"pts_allowed"})
    away_def = game_pts[["gameId","game_date","season","away_team_id","home_pts"]].rename(
        columns={"away_team_id":"teamId","home_pts":"pts_allowed"})
    def_df = pd.concat([home_def, away_def], ignore_index=True).sort_values(
        ["teamId","game_date"])
    def_df["opp_def_rating"] = (
        def_df.groupby(["teamId","season"])["pts_allowed"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        .round(2)
    )
    opp_lookup = def_df[["gameId","teamId","opp_def_rating"]].rename(
        columns={"teamId":"opp_team_id"})
    df = df.merge(opp_lookup, on=["gameId","opp_team_id"], how="left")

    # ── Step 6: team efficiency (ORtg / DRtg / Pace) ─────────────────────────
    print("Computing team efficiency (ORtg/DRtg/Pace)...", flush=True)
    try:
        eff = _build_team_efficiency(box, sched)
        # Normalise join keys to int64 on both sides
        eff["teamId_key"] = pd.to_numeric(eff["teamId"], errors="coerce").astype("int64")
        eff["gameId_key"] = pd.to_numeric(eff["gameId"], errors="coerce").astype("int64")
        df["_teamId_key"] = pd.to_numeric(df["teamId"], errors="coerce").astype("int64")
        df["_gameId_key"] = pd.to_numeric(df["gameId"], errors="coerce").astype("int64")

        eff_roll_cols = [c for c in eff.columns if any(
            c.endswith(f"_roll{w}") or c.endswith("_szn") for w in [5, 10]
        )]
        df = df.merge(
            eff[["gameId_key", "teamId_key"] + eff_roll_cols],
            left_on=["_gameId_key", "_teamId_key"],
            right_on=["gameId_key", "teamId_key"],
            how="left",
        )
        df.drop(columns=["_teamId_key", "_gameId_key",
                          "gameId_key", "teamId_key"], errors="ignore", inplace=True)
        print(f"  Added {len(eff_roll_cols)} efficiency features")
    except Exception as e:
        print(f"  [features] team efficiency failed: {e}")

    # ── Step 7: H2H features ──────────────────────────────────────────────────
    print("Computing H2H features...", flush=True)
    try:
        h2h = _build_h2h(df, sched, box, n_prev=10)
        df = df.merge(h2h, on="gameId", how="left")
        df["h2h_win_pct_home"] = df["h2h_win_pct_home"].fillna(0.5)
        df["h2h_pts_diff_avg"] = df["h2h_pts_diff_avg"].fillna(0.0)
        df["h2h_games"]        = df["h2h_games"].fillna(0).astype("int16")
        # Flip for away players so the feature is always "home team H2H advantage"
    except Exception as e:
        print(f"  [features] H2H failed: {e}")

    # ── Step 8: merge shot quality features ───────────────────────────────────
    shot_path = PROCESSED_DIR / "shot_features.parquet"
    if include_shot and shot_path.exists():
        print("Merging shot quality features...", flush=True)
        try:
            shot = pd.read_parquet(shot_path)
            shot["personId"] = pd.to_numeric(shot["personId"], errors="coerce").astype("Int64")
            shot["game_id"]  = pd.to_numeric(shot["game_id"],  errors="coerce")
            df["_personId_int"] = pd.to_numeric(df["personId"], errors="coerce").astype("Int64")
            shot_roll_cols = [c for c in shot.columns if any(
                c.endswith(f"_roll{w}") for w in WINDOWS)]
            df = df.merge(
                shot[["game_id","personId"] + shot_roll_cols].rename(
                    columns={"game_id": "_s_gid", "personId": "_s_pid"}),
                left_on=["gameId","_personId_int"],
                right_on=["_s_gid","_s_pid"],
                how="left",
            )
            df.drop(columns=["_personId_int","_s_gid","_s_pid"], errors="ignore", inplace=True)
            print(f"  Added {len(shot_roll_cols)} shot quality features")
        except Exception as e:
            print(f"  [features] shot merge failed: {e}")
    else:
        if include_shot:
            print("  Shot features not found — run src/shot_features.py first")

    # ── Step 9: merge matchup features ───────────────────────────────────────
    mu_path = PROCESSED_DIR / "matchup_features.parquet"
    if include_matchup and mu_path.exists():
        print("Merging matchup/defender features...", flush=True)
        try:
            mu = pd.read_parquet(mu_path)
            mu["personId"] = pd.to_numeric(mu["personId"], errors="coerce").astype("Int64")
            mu["game_id"]  = pd.to_numeric(mu["game_id"],  errors="coerce")
            df["_pid_int"] = pd.to_numeric(df["personId"], errors="coerce").astype("Int64")
            mu_roll_cols = [c for c in mu.columns if any(
                c.endswith(f"_roll{w}") for w in WINDOWS)]
            df = df.merge(
                mu[["game_id","personId"] + mu_roll_cols].rename(
                    columns={"game_id": "_m_gid", "personId": "_m_pid"}),
                left_on=["gameId","_pid_int"],
                right_on=["_m_gid","_m_pid"],
                how="left",
            )
            df.drop(columns=["_pid_int","_m_gid","_m_pid"], errors="ignore", inplace=True)
            print(f"  Added {len(mu_roll_cols)} matchup features")
        except Exception as e:
            print(f"  [features] matchup merge failed: {e}")
    else:
        if include_matchup:
            print("  Matchup features not found — run src/matchup_features.py first")

    # ── Step 10: final column order ───────────────────────────────────────────
    id_cols = ["gameId","season","game_date","personId","playerName",
               "teamId","teamTricode","opp_team_id"]
    context_cols = ["is_home","rest_days","is_back_to_back","opp_def_rating",
                    "games_on_current_team","team_stint_id",
                    "h2h_win_pct_home","h2h_pts_diff_avg","h2h_games"]
    raw_stat_cols = [
        "min","pts","fgm","fga","fg3m","fg3a","ftm","fta",
        "reb","oreb","dreb","ast","stl","blk","tov",
        "pf","tf","fouls_drawn",
        "fg_pct","fg3_pct","ft_pct","efg_pct","ts_pct",
        "team_poss","usg_pct","pts_per36","reb_per36","ast_per36",
    ]
    roll_cols = sorted([c for c in df.columns if any(
        c.endswith(f"_roll{w}") or c.endswith("_ewma")
        or c.endswith("_season_avg") or c.endswith("_szn")
        for w in WINDOWS
    )])
    eff_cols = [c for c in df.columns if any(
        c.startswith(x) for x in ["ortg","drtg","net_rtg","pace_proxy","team_efg","opp_efg"]
    ) and c not in roll_cols]

    final_cols = id_cols + context_cols + raw_stat_cols + roll_cols + eff_cols
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols].reset_index(drop=True)

    df.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  {len(df):,} rows  ×  {df.shape[1]} cols")
    print(f"  Memory: {df.memory_usage(deep=True).sum()/1024**2:.1f} MB")
    return df


if __name__ == "__main__":
    build()
