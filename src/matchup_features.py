"""
Matchup / defender quality features from shufinskiy/nba_data matchups datasets.

The matchups dataset records, for each game, which offensive player was guarded
by which defender and their combined stats during that matchup time.

Features computed per player per game (pre-game rolling, no leakage):
    matchup_def_fg_pct_roll{3,5,10}   — FG% the player scored on their defenders
    matchup_pts_per_poss_roll{3,5,10} — pts per partial possession (offensive load)
    defended_fg_pct_roll{3,5,10}      — FG% allowed when this player defended
    matchup_3pt_rate_roll{3,5,10}     — fraction of matchup attempts that were 3s
    switches_on_roll{3,5}             — times switched onto bigger/smaller matchup

Output: data/processed/matchup_features.parquet
"""

import io
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
RAW_MU_DIR    = Path("data/raw/matchups")
RAW_MU_DIR.mkdir(parents=True, exist_ok=True)

GITHUB_BASE    = "https://github.com/shufinskiy/nba_data/raw/main/datasets"
MU_SEASONS     = list(range(2020, 2026))
ROLL_WINDOWS   = [3, 5, 10]
MIN_POSS_THRESHOLD = 2.0   # ignore matchup < 2 partial possessions


def _download_matchups(season_year: int) -> pd.DataFrame:
    cache = RAW_MU_DIR / f"matchups_{season_year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    url = f"{GITHUB_BASE}/matchups_{season_year}.tar.xz"
    print(f"  Downloading matchups_{season_year}...", flush=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:xz") as tar:
        member = tar.getnames()[0]
        f = tar.extractfile(member)
        df = pd.read_csv(f, low_memory=False)

    # Normalise columns
    df.columns = df.columns.str.lower()
    keep = [
        "game_id", "person_id", "team_id",
        "matchups_person_id",
        "partial_possessions",
        "matchup_field_goals_made", "matchup_field_goals_attempted",
        "matchup_field_goals_percentage",
        "matchup_three_pointers_made", "matchup_three_pointers_attempted",
        "player_points", "team_points",
        "switches_on",
        "matchup_minutes",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["season"] = season_year
    df.to_parquet(cache, index=False)
    print(f"    Saved raw: {cache}  ({len(df):,} matchup rows)")
    return df


def _rolling(s: pd.Series, w: int) -> pd.Series:
    return s.shift(1).rolling(w, min_periods=1).mean()


def build_matchup_features(seasons: list = None,
                           output_path: str = None) -> pd.DataFrame:
    """
    For each player in each game compute:

    OFFENSIVE view  (player_id is the OFFENSE):
        matchup_def_fg_pct   — their FG% against the defenders they faced
        matchup_pts_per_poss — pts per partial possession (efficiency)
        matchup_3pt_rate     — fraction of matchup FGA that were 3s

    DEFENSIVE view  (matchups_person_id is the OFFENSE they guarded):
        defended_fg_pct      — FG% allowed by this player as a defender
        defended_pts_per_poss

    Returns pre-game rolling features ready to merge on [game_id, personId].
    """
    if seasons is None:
        seasons = MU_SEASONS

    output_path = Path(output_path) if output_path else \
        PROCESSED_DIR / "matchup_features.parquet"

    frames = []
    for yr in seasons:
        try:
            raw = _download_matchups(yr)
            frames.append(raw)
            print(f"    season {yr}: {len(raw):,} matchup rows")
        except Exception as e:
            print(f"  [matchup_features] season {yr} failed: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Filter low-sample matchups
    df = df[df["partial_possessions"] >= MIN_POSS_THRESHOLD].copy()

    # ── OFFENSIVE stats per player per game ───────────────────────────────────
    off_cols = {
        "matchup_field_goals_made":      "sum",
        "matchup_field_goals_attempted": "sum",
        "matchup_three_pointers_made":   "sum",
        "matchup_three_pointers_attempted": "sum",
        "player_points":                 "sum",
        "partial_possessions":           "sum",
        "switches_on":                   "sum",
    }
    off_cols = {k: v for k, v in off_cols.items() if k in df.columns}

    off = (
        df.groupby(["game_id", "person_id", "team_id", "season"])
          .agg(off_cols)
          .reset_index()
          .rename(columns={"person_id": "personId"})
    )

    safe_fga = off["matchup_field_goals_attempted"].replace(0, np.nan)
    safe_pos = off["partial_possessions"].replace(0, np.nan)

    off["matchup_def_fg_pct"]   = (
        off["matchup_field_goals_made"] / safe_fga
    ).fillna(0)
    off["matchup_pts_per_poss"] = (
        off["player_points"] / safe_pos
    ).fillna(0)
    off["matchup_3pt_rate"]     = (
        off["matchup_three_pointers_attempted"] / safe_fga
    ).fillna(0)

    # ── DEFENSIVE stats — how well this player defended ───────────────────────
    # matchups_person_id is the offensive player guarded by person_id
    def_agg_cols = {
        "matchup_field_goals_made":      "sum",
        "matchup_field_goals_attempted": "sum",
        "player_points":                 "sum",
        "partial_possessions":           "sum",
    }
    def_agg_cols = {k: v for k, v in def_agg_cols.items() if k in df.columns}

    if "matchups_person_id" in df.columns:
        defdf = (
            df.groupby(["game_id", "matchups_person_id", "team_id", "season"])
              .agg(def_agg_cols)
              .reset_index()
              .rename(columns={"matchups_person_id": "personId"})
        )
        safe_fa = defdf["matchup_field_goals_attempted"].replace(0, np.nan)
        safe_ps = defdf["partial_possessions"].replace(0, np.nan)
        defdf["defended_fg_pct"]      = (defdf["matchup_field_goals_made"] / safe_fa).fillna(0)
        defdf["defended_pts_per_poss"]= (defdf["player_points"] / safe_ps).fillna(0)

        merged = off.merge(
            defdf[["game_id", "personId", "defended_fg_pct", "defended_pts_per_poss"]],
            on=["game_id", "personId"], how="outer"
        )
    else:
        merged = off.copy()

    merged["personId"] = pd.to_numeric(merged["personId"], errors="coerce").astype("Int64")
    merged["game_id"]  = pd.to_numeric(merged["game_id"],  errors="coerce")

    # Attach game dates from box scores if available
    bs_path = PROCESSED_DIR / "box_scores.parquet"
    if bs_path.exists():
        bs = pd.read_parquet(bs_path, columns=["gameId", "game_date", "personId"])
        bs["personId"] = pd.to_numeric(bs["personId"], errors="coerce").astype("Int64")
        bs["gameId"]   = pd.to_numeric(bs["gameId"],   errors="coerce")
        merged = merged.merge(
            bs.drop_duplicates(["gameId", "personId"]).rename(columns={"gameId": "game_id"}),
            on=["game_id", "personId"], how="left"
        )
    else:
        merged["game_date"] = pd.NaT

    merged = merged.sort_values(["personId", "game_date", "game_id"]).reset_index(drop=True)

    # ── Rolling features ──────────────────────────────────────────────────────
    print("  Computing rolling matchup features...", flush=True)
    METRICS = [
        "matchup_def_fg_pct", "matchup_pts_per_poss", "matchup_3pt_rate",
        "defended_fg_pct", "defended_pts_per_poss", "switches_on",
    ]
    METRICS = [m for m in METRICS if m in merged.columns]

    grp = merged.groupby("personId")
    for m in METRICS:
        for w in ROLL_WINDOWS:
            merged[f"{m}_roll{w}"] = grp[m].transform(
                lambda x, w=w: _rolling(x, w)
            )

    roll_cols = [c for c in merged.columns if any(f"_roll{w}" in c for w in ROLL_WINDOWS)]
    id_cols   = ["game_id", "personId", "game_date", "season"]
    final = merged[[c for c in id_cols + roll_cols if c in merged.columns]].copy()
    final = final.drop_duplicates(subset=["game_id", "personId"])

    final.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  {len(final):,} rows  ×  {final.shape[1]} cols")
    return final


if __name__ == "__main__":
    print("Building matchup/defender quality features...")
    build_matchup_features()
