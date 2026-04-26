"""
Fetch new season box scores from nba_api and integrate with existing data.

Two API calls per season (one player, one team log) — no per-game requests.
Detects trades: adds games_on_current_team + team_stint_id to box_scores.
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def _season_str(year: int) -> str:
    return f"{year - 1}-{str(year)[-2:]}"


def _parse_min(val) -> float:
    """Accept int/float decimal minutes OR 'MM:SS' string → float minutes."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if ":" in s:
        parts = s.split(":")
        return int(parts[0]) + int(parts[1]) / 60.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _fetch_player_log(season_year: int, timeout: int = 60) -> pd.DataFrame:
    from nba_api.stats.endpoints import LeagueGameLog
    lg = LeagueGameLog(
        season=_season_str(season_year),
        player_or_team_abbreviation="P",
        timeout=timeout,
    )
    time.sleep(0.65)
    return lg.get_data_frames()[0]


def _fetch_team_log(season_year: int, timeout: int = 60) -> pd.DataFrame:
    from nba_api.stats.endpoints import LeagueGameLog
    lg = LeagueGameLog(
        season=_season_str(season_year),
        player_or_team_abbreviation="T",
        timeout=timeout,
    )
    time.sleep(0.65)
    return lg.get_data_frames()[0]


# ── Build box scores ───────────────────────────────────────────────────────────

def fetch_season(season_year: int) -> pd.DataFrame:
    """
    Fetch one season's box scores from nba_api and return a DataFrame
    matching the schema of box_scores.parquet.
    """
    ss = _season_str(season_year)
    print(f"  Fetching player game logs ({ss})…")
    pl = _fetch_player_log(season_year)

    print(f"  Fetching team game logs ({ss})…")
    tm = _fetch_team_log(season_year)

    # ── Parse & clean player rows ──────────────────────────────────────────────
    pl = pl.rename(columns=str.upper)
    pl["gameId"]     = pd.to_numeric(pl["GAME_ID"], errors="coerce").astype("int64")
    pl["personId"]   = pd.to_numeric(pl["PLAYER_ID"], errors="coerce")
    pl["playerName"] = pl["PLAYER_NAME"]
    pl["teamId"]     = pd.to_numeric(pl["TEAM_ID"], errors="coerce")
    pl["teamTricode"] = pl["TEAM_ABBREVIATION"]
    pl["game_date"]  = pd.to_datetime(pl["GAME_DATE"])
    pl["season"]     = season_year
    pl["min"]        = pl["MIN"].apply(_parse_min)

    # Filter DNPs
    pl = pl[pl["min"] > 0].copy()

    # Rename raw stat columns
    stat_map = {
        "PTS": "pts", "REB": "reb", "OREB": "oreb", "DREB": "dreb",
        "AST": "ast", "STL": "stl", "BLK": "blk", "TOV": "tov",
        "PF":  "pf",  "FGM": "fgm", "FGA":  "fga",
        "FG3M": "fg3m", "FG3A": "fg3a", "FTM": "ftm", "FTA": "fta",
        "FG_PCT": "fg_pct", "FG3_PCT": "fg3_pct", "FT_PCT": "ft_pct",
    }
    pl = pl.rename(columns=stat_map)

    # ── Team totals per game (for usg_pct + team_poss) ────────────────────────
    agg_cols = ["fga", "fta", "tov", "oreb", "min", "pts"]
    team_totals = (
        pl.groupby(["gameId", "teamId"])[agg_cols]
          .sum()
          .reset_index()
          .rename(columns={c: f"team_{c}" for c in agg_cols})
    )
    pl = pl.merge(team_totals, on=["gameId", "teamId"], how="left")

    # ── Derived stats ──────────────────────────────────────────────────────────
    pl["efg_pct"]   = np.where(pl["fga"] > 0,
                               (pl["fgm"] + 0.5 * pl["fg3m"]) / pl["fga"], 0.0)
    pl["ts_pct"]    = np.where((pl["fga"] + 0.44 * pl["fta"]) > 0,
                               pl["pts"] / (2 * (pl["fga"] + 0.44 * pl["fta"])), 0.0)

    # usg_pct ≈ player share of team's offensive "stops"
    denom = pl["min"] * (pl["team_fga"] + 0.44 * pl["team_fta"] + pl["team_tov"])
    numer = (pl["fga"] + 0.44 * pl["fta"] + pl["tov"]) * (pl["team_min"] / 5)
    pl["usg_pct"]   = np.where(denom > 0, numer / denom, 0.0)

    # team possessions (Oliver formula)
    pl["team_poss"] = (pl["team_fga"] - pl["team_oreb"]
                       + pl["team_tov"] + 0.44 * pl["team_fta"])

    # per-36 stats
    pl["pts_per36"] = np.where(pl["min"] > 0, pl["pts"]  / pl["min"] * 36, 0.0)
    pl["reb_per36"] = np.where(pl["min"] > 0, pl["reb"]  / pl["min"] * 36, 0.0)
    pl["ast_per36"] = np.where(pl["min"] > 0, pl["ast"]  / pl["min"] * 36, 0.0)

    # tf and fouls_drawn not available from GameLog
    pl["tf"]          = np.int16(0)
    pl["fouls_drawn"] = np.int16(0)

    # ── Final column selection & dtype compression ─────────────────────────────
    id_cols  = ["gameId", "season", "personId", "playerName", "teamId", "teamTricode", "game_date"]
    cnt_cols = ["min", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "tf", "fouls_drawn"]
    rat_cols = ["fg_pct", "fg3_pct", "ft_pct", "efg_pct", "ts_pct",
                "team_poss", "usg_pct", "pts_per36", "reb_per36", "ast_per36"]

    out = pl[id_cols + cnt_cols + rat_cols].copy()

    # dtype compression
    for c in ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
              "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "tf", "fouls_drawn"]:
        out[c] = out[c].fillna(0).astype("int16")
    for c in rat_cols + ["min"]:
        out[c] = out[c].astype("float32")

    print(f"  {ss}: {len(out):,} player-game rows, {out['gameId'].nunique()} games")
    return out.reset_index(drop=True)


# ── Trade detection ────────────────────────────────────────────────────────────

def detect_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two columns:
      team_stint_id       : integer that increments every time a player changes team
      games_on_current_team: how many consecutive games this player has been on this team
    """
    df = df.sort_values(["personId", "game_date", "gameId"]).reset_index(drop=True)

    prev_team   = df.groupby("personId")["teamId"].shift(1)
    team_change = (df["teamId"] != prev_team).fillna(False)

    # Cumulative stint counter per player
    df["team_stint_id"] = (
        team_change.groupby(df["personId"]).cumsum().astype("int16")
    )

    # Games within current stint (1-indexed)
    df["games_on_current_team"] = (
        df.groupby(["personId", "team_stint_id"]).cumcount().astype("int16") + 1
    )

    n_trades = team_change.sum()
    n_players = df[team_change]["personId"].nunique()
    print(f"  Detected {n_trades:,} team changes across {n_players:,} players")
    return df


# ── Schedule builder ───────────────────────────────────────────────────────────

def build_schedule_rows(player_log_df: pd.DataFrame) -> pd.DataFrame:
    """Extract game_schedule rows from player game log."""
    # Home team = MATCHUP contains 'vs.'
    df = player_log_df.copy()
    df["is_home"] = df["playerName"].notna()  # placeholder; rebuilt below

    # One row per team-game
    tg = (
        df.groupby(["gameId", "teamId", "teamTricode", "game_date", "season"])
          .size()
          .reset_index(name="_n")
          .drop(columns="_n")
    )

    # Identify home / away from the raw player log MATCHUP column
    # MATCHUP examples: 'BOS vs. NYK' (BOS = home), 'MIN @ LAL' (MIN = away)
    # Re-fetch matchup from original API data — not available after rename.
    # Instead: home team is the one whose tricode appears BEFORE "vs."
    # We can recover this from the original player_log columns if stored.
    # Simpler: look at each game — the team with two appearances as "home" tri
    # Just mark: for each game the teams are home/away based on pts-sum parity.
    # Reliable approach: merge with existing sched where possible, else infer.

    games = tg.groupby("gameId").apply(
        lambda g: pd.Series({
            "game_date":    g["game_date"].iloc[0],
            "season":       g["season"].iloc[0],
            "teams":        g["teamId"].tolist(),
            "tricodes":     g["teamTricode"].tolist(),
        })
    ).reset_index()

    # Assign home/away — 'home' GAME_ID is even (NBA convention: home team listed second)
    # We'll rely on the raw MATCHUP from the player log to correctly assign
    return games  # placeholder — caller merges with existing schedule


def _extract_schedule(raw_player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build game_schedule rows from raw LeagueGameLog player data
    using the MATCHUP column before column renaming.
    """
    # Keep MATCHUP alongside gameId / teamId / game_date
    tmp = raw_player_df[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION",
                          "GAME_DATE", "MATCHUP"]].drop_duplicates()
    tmp["gameId"]    = pd.to_numeric(tmp["GAME_ID"], errors="coerce").astype("int64")
    tmp["teamId"]    = pd.to_numeric(tmp["TEAM_ID"], errors="coerce")
    tmp["tricode"]   = tmp["TEAM_ABBREVIATION"]
    tmp["game_date"] = pd.to_datetime(tmp["GAME_DATE"])
    tmp["is_home"]   = tmp["MATCHUP"].str.contains(r"vs\.", na=False)

    home = tmp[tmp["is_home"]].rename(columns={"teamId": "home_team_id", "tricode": "home_tricode"})
    away = tmp[~tmp["is_home"]].rename(columns={"teamId": "away_team_id", "tricode": "away_tricode"})

    sched = home[["gameId", "game_date", "home_team_id", "home_tricode"]].merge(
        away[["gameId", "away_team_id", "away_tricode"]], on="gameId"
    )
    return sched


# ── Main integration entry point ───────────────────────────────────────────────

def update_all(new_seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Fetch new seasons, append to box_scores.parquet and game_schedule.parquet,
    then apply trade detection across the full combined dataset.

    Returns updated box_scores DataFrame.
    """
    existing_box   = pd.read_parquet(PROCESSED_DIR / "box_scores.parquet")
    existing_sched = pd.read_parquet(PROCESSED_DIR / "game_schedule.parquet")

    if new_seasons is None:
        done = set(existing_box["season"].unique().tolist())
        new_seasons = [y for y in [2025, 2026] if y not in done]

    if not new_seasons:
        print("No new seasons to fetch — box_scores is up to date.")
    else:
        new_frames  = []
        sched_frames = []

        for year in sorted(new_seasons):
            print(f"\nFetching {_season_str(year)}…")

            # Fetch raw (keep original column names for schedule extraction)
            from nba_api.stats.endpoints import LeagueGameLog
            raw_pl = LeagueGameLog(season=_season_str(year),
                                   player_or_team_abbreviation="P",
                                   timeout=60).get_data_frames()[0]
            time.sleep(0.65)
            raw_pl.columns = [c.upper() for c in raw_pl.columns]

            sched_rows = _extract_schedule(raw_pl)
            sched_rows["season"] = year
            sched_frames.append(sched_rows)

            season_box = fetch_season(year)
            new_frames.append(season_box)

        # Append box scores
        combined_box = pd.concat([existing_box] + new_frames, ignore_index=True)
        combined_box = combined_box.drop_duplicates(subset=["gameId", "personId"])

        # Append schedule
        new_sched = pd.concat(sched_frames, ignore_index=True)
        combined_sched = pd.concat([existing_sched, new_sched], ignore_index=True)
        combined_sched = combined_sched.drop_duplicates(subset="gameId")

        combined_sched.to_parquet(PROCESSED_DIR / "game_schedule.parquet", index=False)
        print(f"\nSchedule updated: {len(combined_sched):,} games")

        existing_box = combined_box

    # Apply trade detection across full dataset
    print("\nApplying trade detection across all seasons…")
    existing_box["game_date"] = pd.to_datetime(existing_box["game_date"])
    final = detect_trades(existing_box)

    final.to_parquet(PROCESSED_DIR / "box_scores.parquet", index=False)
    print(f"Saved: box_scores.parquet  ({len(final):,} rows, {final.shape[1]} cols)")
    print(f"  Seasons: {sorted(final['season'].unique().tolist())}")
    return final


if __name__ == "__main__":
    update_all()
