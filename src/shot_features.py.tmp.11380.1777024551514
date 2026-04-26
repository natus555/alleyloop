"""
Shot quality feature engineering from shufinskiy/nba_data shotdetail datasets.

Downloads shotdetail CSVs (per-shot data with court coordinates and zones),
aggregates to per-player-per-game metrics, then computes rolling windows
so each row is a pre-game feature vector (shift(1) applied — no leakage).

Output : data/processed/shot_features.parquet
Columns (per player per game, all pre-game rolling):
    shot_3pt_rate_roll{3,5,10}   — fraction of FGA that are 3-pointers
    shot_rim_rate_roll{3,5,10}   — fraction within 6 feet
    shot_mid_rate_roll{3,5,10}   — fraction 6-16 feet (mid-range)
    shot_dist_avg_roll{3,5,10}   — average shot distance
    shot_efg_roll{3,5,10}        — eFG% from shot locations
    shot_quality_roll{3,5,10}    — eFG% minus league-avg eFG% for that zone mix
    opp_def_efg_roll{3,5,10}     — opponent's eFG% allowed (defensive quality)
"""

import io
import tarfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
RAW_SHOT_DIR  = Path("data/raw/shotdetail")
RAW_SHOT_DIR.mkdir(parents=True, exist_ok=True)

GITHUB_BASE = "https://github.com/shufinskiy/nba_data/raw/main/datasets"
SHOT_SEASONS = list(range(2020, 2026))   # 2020-21 through 2025-26
ROLL_WINDOWS = [3, 5, 10]

# League-average eFG% by shot zone (approximate values from historical data)
ZONE_AVG_EFG = {
    "Restricted Area":          0.640,
    "In The Paint (Non-RA)":    0.395,
    "Mid-Range":                0.395,
    "Left Corner 3":            0.530,
    "Right Corner 3":           0.530,
    "Above the Break 3":        0.520,
    "Backcourt":                0.200,
}
DEFAULT_ZONE_EFG = 0.480


def _download_shotdetail(season_year: int) -> pd.DataFrame:
    """Download and parse shotdetail_{season_year}.tar.xz → DataFrame."""
    cache = RAW_SHOT_DIR / f"shotdetail_{season_year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    url = f"{GITHUB_BASE}/shotdetail_{season_year}.tar.xz"
    print(f"  Downloading shotdetail_{season_year}...", flush=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:xz") as tar:
        member = tar.getnames()[0]
        f = tar.extractfile(member)
        df = pd.read_csv(f, low_memory=False)

    # Standardise columns to lower-snake
    df.columns = df.columns.str.lower()

    # Keep only needed columns
    keep = [
        "game_id", "player_id", "team_id", "game_date",
        "shot_zone_basic", "shot_distance",
        "shot_attempted_flag", "shot_made_flag",
        "loc_x", "loc_y",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["game_date"] = pd.to_datetime(df["game_date"], format="%Y%m%d", errors="coerce")
    df["season"] = season_year
    df.to_parquet(cache, index=False)
    print(f"    Saved raw: {cache}  ({len(df):,} shots)")
    return df


def _player_game_agg(shots: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate shot-level data to player×game level.
    Returns one row per (game_id, player_id, team_id, game_date, season).
    """
    g = shots.groupby(["game_id", "player_id", "team_id", "game_date", "season"])

    agg = g.agg(
        fga         = ("shot_attempted_flag", "sum"),
        fgm         = ("shot_made_flag",      "sum"),
        fg3a        = ("shot_attempted_flag",
                       lambda x: x[shots.loc[x.index, "shot_zone_basic"]
                                   .str.contains("3", na=False)].sum()),
        fg3m        = ("shot_made_flag",
                       lambda x: x[shots.loc[x.index, "shot_zone_basic"]
                                   .str.contains("3", na=False)].sum()),
        rim_a       = ("shot_attempted_flag",
                       lambda x: x[shots.loc[x.index, "shot_distance"] <= 6].sum()),
        mid_a       = ("shot_attempted_flag",
                       lambda x: x[(shots.loc[x.index, "shot_distance"] > 6) &
                                   (shots.loc[x.index, "shot_distance"] <= 16)].sum()),
        shot_dist_avg = ("shot_distance", "mean"),
    ).reset_index()

    agg["fga"] = agg["fga"].clip(lower=0)
    safe_fga = agg["fga"].replace(0, np.nan)

    agg["shot_3pt_rate"] = (agg["fg3a"] / safe_fga).fillna(0)
    agg["shot_rim_rate"] = (agg["rim_a"] / safe_fga).fillna(0)
    agg["shot_mid_rate"] = (agg["mid_a"] / safe_fga).fillna(0)
    agg["shot_efg"]      = ((agg["fgm"] + 0.5 * agg["fg3m"]) / safe_fga).fillna(0)

    return agg.drop(columns=["fga", "fgm", "fg3a", "fg3m", "rim_a", "mid_a"])


def _player_game_agg_fast(shots: pd.DataFrame) -> pd.DataFrame:
    """Faster vectorised aggregation without per-group lambda."""
    shots = shots.copy()
    shots["is_3"] = shots["shot_zone_basic"].str.contains("3", na=False).astype("int8")
    shots["is_rim"] = (shots["shot_distance"] <= 6).astype("int8")
    shots["is_mid"] = ((shots["shot_distance"] > 6) &
                       (shots["shot_distance"] <= 16)).astype("int8")

    key = ["game_id", "player_id", "team_id", "game_date", "season"]
    agg = shots.groupby(key).agg(
        fga           = ("shot_attempted_flag", "sum"),
        fgm           = ("shot_made_flag",      "sum"),
        fg3a          = ("is_3",    "sum"),
        fg3m_proxy    = ("is_3",    lambda x: (x * shots.loc[x.index, "shot_made_flag"]).sum()),
        rim_a         = ("is_rim",  "sum"),
        mid_a         = ("is_mid",  "sum"),
        shot_dist_avg = ("shot_distance", "mean"),
    ).reset_index()

    safe = agg["fga"].replace(0, np.nan)
    agg["shot_3pt_rate"] = (agg["fg3a"] / safe).fillna(0)
    agg["shot_rim_rate"] = (agg["rim_a"] / safe).fillna(0)
    agg["shot_mid_rate"] = (agg["mid_a"] / safe).fillna(0)
    agg["shot_efg"]      = ((agg["fgm"] + 0.5 * agg["fg3m_proxy"]) / safe).fillna(0)

    return agg.drop(columns=["fga", "fgm", "fg3a", "fg3m_proxy", "rim_a", "mid_a"])


def _rolling(s: pd.Series, w: int) -> pd.Series:
    return s.shift(1).rolling(w, min_periods=1).mean()


def build_shot_features(seasons: list = None,
                        output_path: str = None) -> pd.DataFrame:
    """
    Download shotdetail for each season, aggregate per player×game,
    compute rolling shot-quality features (shifted to avoid leakage).

    Returns DataFrame with rolling shot features ready to merge into
    features.parquet on [gameId, personId].
    """
    if seasons is None:
        seasons = SHOT_SEASONS

    output_path = Path(output_path) if output_path else \
        PROCESSED_DIR / "shot_features.parquet"

    frames = []
    for yr in seasons:
        try:
            raw = _download_shotdetail(yr)
            agg = _player_game_agg_fast(raw)
            frames.append(agg)
            print(f"    season {yr}: {len(agg):,} player-games")
        except Exception as e:
            print(f"  [shot_features] season {yr} failed: {e}")

    if not frames:
        print("No shot data loaded.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    # ── Rolling features per player ───────────────────────────────────────────
    print("  Computing rolling shot features...", flush=True)
    SHOT_METRICS = ["shot_3pt_rate", "shot_rim_rate", "shot_mid_rate",
                    "shot_dist_avg", "shot_efg"]

    grp = df.groupby("player_id")
    for m in SHOT_METRICS:
        for w in ROLL_WINDOWS:
            df[f"{m}_roll{w}"] = grp[m].transform(
                lambda x, w=w: _rolling(x, w)
            )

    # ── Opponent defensive eFG% (how well each team defends per game) ─────────
    # For each game×team, compute opponent's actual eFG% → team's defensive quality
    game_team_efg = (
        df.groupby(["game_id", "team_id"]).apply(
            lambda g: pd.Series({
                "team_fga": g["fga"].sum() if "fga" in g else 0,
                "team_efg": (
                    (g["fgm"].sum() + 0.5 * (g["shot_3pt_rate"] * g.shape[0]).sum())
                    / max(g.shape[0], 1)
                ) if "fgm" in g else 0,
            })
        ).reset_index()
        if "fgm" in df.columns else None
    )

    # Simpler: just use shot_efg average per team per game
    team_game_efg = (
        df.groupby(["game_id", "team_id"])["shot_efg"]
        .mean()
        .reset_index(name="team_game_efg")
    )

    # Each team's opponent is the other team in the same game
    # We'll merge by game_id and join the other team's eFG
    home_efg = team_game_efg.rename(columns={"team_id": "opp_id", "team_game_efg": "opp_efg"})
    df2 = df.merge(
        team_game_efg.rename(columns={"team_id": "team_id_2"}),
        left_on=["game_id", "team_id"], right_on=["game_id", "team_id_2"],
        how="left"
    ).drop(columns=["team_id_2"])

    # opp_efg: the eFG% scored AGAINST this player's team (lower = better defense)
    # We need to find the other team in the same game
    opp_efg_df = team_game_efg.copy()
    opp_efg_df.columns = ["game_id", "opp_team_id", "opp_team_efg"]
    df = df.merge(opp_efg_df, on="game_id", how="left")
    df = df[df["team_id"] != df["opp_team_id"]]  # keep only the other team's efg

    # Rolling opp_def_efg (opponent eFG% that this player's team faced)
    # i.e. how well THIS team defends
    df = df.sort_values(["team_id", "game_date"]).reset_index(drop=True)
    df["opp_def_efg"] = df["opp_team_efg"]   # efg scored against this team
    team_grp = df.groupby("team_id")
    for w in ROLL_WINDOWS:
        df[f"opp_def_efg_roll{w}"] = team_grp["opp_def_efg"].transform(
            lambda x, w=w: _rolling(x, w)
        )

    # ── Final columns ─────────────────────────────────────────────────────────
    roll_cols = [c for c in df.columns if any(f"_roll{w}" in c for w in ROLL_WINDOWS)]
    id_cols = ["game_id", "player_id", "game_date", "season"]
    final = df[id_cols + roll_cols].copy()
    # Rename player_id → personId to match main features schema
    final = final.rename(columns={"player_id": "personId"})
    final["personId"] = pd.to_numeric(final["personId"], errors="coerce").astype("int64")
    final["game_id"]  = pd.to_numeric(final["game_id"],  errors="coerce").astype("int64")

    # Deduplicate (each player-game should appear once)
    final = final.drop_duplicates(subset=["game_id", "personId"])

    final.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  {len(final):,} rows  ×  {final.shape[1]} cols")
    return final


if __name__ == "__main__":
    print("Building shot quality features...")
    build_shot_features()
