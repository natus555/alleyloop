import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# 43 columns kept after removing irrelevant fields.
# Dropped from original 58:
#   jumpBall* (6)     — 100% null on action rows
#   officialId        — referee ID, not used in modeling
#   xLegacy, yLegacy — duplicate of x/y in legacy coordinate system
#   side              — redundant with possession + teamId
#   playerNameI       — abbreviated name, redundant with playerName/personId
#   edited            — internal metadata timestamp
#   description       — free-text, not for ML
#   isTargetScoreLastPeriod — play-in flag, always False in regular data
#   personIdsFilter   — internal filter field, not used in modeling
CANONICAL_COLS = [
    "gameId", "season", "actionNumber", "orderNumber",
    "period", "periodType", "clock", "timeActual",
    "actionType", "subType", "qualifiers", "descriptor",
    "teamId", "teamTricode", "personId", "playerName",
    "possession", "scoreHome", "scoreAway",
    "isFieldGoal", "x", "y", "area", "areaDetail",
    "shotDistance", "shotResult", "shotActionNumber",
    "pointsTotal",
    "assistPlayerNameInitial", "assistPersonId", "assistTotal",
    "reboundTotal", "reboundDefensiveTotal", "reboundOffensiveTotal",
    "blockPlayerName", "blockPersonId",
    "stealPlayerName", "stealPersonId",
    "turnoverTotal",
    "foulPersonalTotal", "foulTechnicalTotal",
    "foulDrawnPlayerName", "foulDrawnPersonId",
]

DROP_COLS = {
    "value", "shortFormattedClock", "jerseyNumber",
    "isTargetScoreLastPeriod",
}

FILL_DEFAULTS = {
    "area": "",
    "areaDetail": "",
}

_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")


def _parse_clock(s) -> float | None:
    """'PT12M34.56S' → seconds remaining in period."""
    if pd.isna(s) or s == "":
        return None
    m = _CLOCK_RE.match(str(s))
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def _period_secs(period: int) -> float:
    return 720.0 if period <= 4 else 300.0



def _compute_lineup_stats(game_df: pd.DataFrame) -> dict:
    """Returns {personId: {"min": float}} for one game."""
    game_df = game_df.sort_values("orderNumber")

    # on_court: personId → entry_clock_secs
    on_court: dict = {}
    carry_over: set = set()
    player_min: dict = defaultdict(float)
    current_period: int = 0
    period_start_clock: dict = {}

    def _exit(pid, clock_s):
        if pid in on_court:
            player_min[pid] += max(0.0, on_court.pop(pid) - clock_s) / 60.0

    def _enter(pid, clock_s):
        on_court[pid] = clock_s

    rows = zip(
        game_df["actionType"].tolist(),
        game_df["subType"].tolist(),
        game_df["period"].tolist(),
        game_df["clock"].tolist(),
        game_df["personId"].tolist(),
    )

    for atype, stype, period, clock, pid in rows:
        clock_s = _parse_clock(clock) or 0.0

        if atype == "period":
            if stype == "start":
                current_period = int(period) if not pd.isna(period) else current_period + 1
                plen = _period_secs(current_period)
                period_start_clock[current_period] = plen
                for pid2 in carry_over:
                    _enter(pid2, plen)
                carry_over.clear()
            elif stype == "end":
                carry_over = set(on_court.keys())
                for pid2 in list(on_court.keys()):
                    _exit(pid2, 0.0)

        elif atype == "substitution":
            if pd.isna(pid) or pid == 0:
                continue
            if stype == "out":
                _exit(pid, clock_s)
                carry_over.discard(pid)
            elif stype == "in":
                _enter(pid, clock_s)
                carry_over.discard(pid)

        else:
            if pd.isna(pid) or pid == 0:
                continue
            if pid not in on_court and current_period > 0:
                _enter(pid, period_start_clock.get(current_period, _period_secs(current_period)))

    for pid2 in list(on_court.keys()):
        _exit(pid2, 0.0)

    return {pid: {"min": round(player_min[pid], 1)} for pid in player_min}


def _compute_lineup_stints(game_df: pd.DataFrame) -> list[dict]:
    """
    Track every 5-man lineup stint for each team in one game.
    Returns list of {gameId, teamId, season, lineup, min}.
    """
    game_df = game_df.sort_values("orderNumber")
    game_id = game_df["gameId"].iloc[0]
    season = game_df["season"].iloc[0]

    player_team: dict = {}
    team_court: dict = defaultdict(set)     # teamId → set of personIds
    stint_start: dict = {}                  # teamId → entry_clock
    stints: list = []
    carry_over: set = set()
    current_period: int = 0
    period_start_clock: dict = {}

    def _flush(team_id, end_clock):
        if team_id not in stint_start:
            return
        entry_clock = stint_start.pop(team_id)
        lineup = frozenset(team_court[team_id])
        if len(lineup) != 5:
            return
        stints.append({
            "gameId":  game_id,
            "season":  season,
            "teamId":  team_id,
            "lineup":  tuple(sorted(lineup)),
            "min":     round(max(0.0, (entry_clock - end_clock) / 60.0), 2),
        })

    def _try_start(team_id, clock_s):
        if len(team_court[team_id]) == 5:
            stint_start[team_id] = clock_s

    rows = zip(
        game_df["actionType"].tolist(),
        game_df["subType"].tolist(),
        game_df["period"].tolist(),
        game_df["clock"].tolist(),
        game_df["personId"].tolist(),
        game_df["teamId"].tolist(),
    )

    for atype, stype, period, clock, pid, tid in rows:
        clock_s = _parse_clock(clock) or 0.0

        if not pd.isna(pid) and pid != 0 and not pd.isna(tid):
            player_team[pid] = tid

        if atype == "period":
            if stype == "start":
                current_period = int(period) if not pd.isna(period) else current_period + 1
                plen = _period_secs(current_period)
                period_start_clock[current_period] = plen
                for pid2 in carry_over:
                    tid2 = player_team.get(pid2)
                    if tid2 is not None:
                        team_court[tid2].add(pid2)
                for tid2 in {player_team.get(p) for p in carry_over if player_team.get(p)}:
                    _try_start(tid2, plen)
                carry_over.clear()

            elif stype == "end":
                carry_over = {p for court in team_court.values() for p in court}
                for tid2 in list(stint_start.keys()):
                    _flush(tid2, 0.0)
                team_court.clear()

        elif atype == "substitution":
            if pd.isna(pid) or pid == 0:
                continue
            if not pd.isna(tid):
                player_team[pid] = tid
            actual_tid = player_team.get(pid)
            if actual_tid is None:
                continue
            if stype == "out":
                _flush(actual_tid, clock_s)
                team_court[actual_tid].discard(pid)
                carry_over.discard(pid)
                _try_start(actual_tid, clock_s)
            elif stype == "in":
                _flush(actual_tid, clock_s)
                team_court[actual_tid].add(pid)
                carry_over.discard(pid)
                _try_start(actual_tid, clock_s)

        else:
            if pd.isna(pid) or pid == 0:
                continue
            if pd.isna(tid):
                tid = player_team.get(pid)
            if tid is None:
                continue
            player_team[pid] = tid
            if pid not in team_court[tid] and current_period > 0:
                plen = period_start_clock.get(current_period, _period_secs(current_period))
                _flush(tid, plen)
                team_court[tid].add(pid)
                _try_start(tid, plen)

    for tid2 in list(stint_start.keys()):
        _flush(tid2, 0.0)

    return stints


def build_lineup_stats() -> pd.DataFrame:
    """
    Aggregate 5-man lineup stints across all seasons into lineup_stats.parquet.

    Columns
    -------
    teamId, season, p1..p5 (sorted personIds), p1_name..p5_name,
    games, total_min
    """
    dfs = [pd.read_parquet(p) for p in sorted(PROCESSED_DIR.glob("cdnnba_*.parquet"))]
    pbp = pd.concat(dfs, ignore_index=True)
    pbp["personId"] = pd.to_numeric(pbp["personId"], errors="coerce")

    all_stints: list[dict] = []
    games = pbp.groupby("gameId")
    total = pbp["gameId"].nunique()
    print("Tracking lineup stints per game...")
    for i, (_, gdf) in enumerate(games, 1):
        if i % 500 == 0:
            print(f"  {i}/{total}")
        all_stints.extend(_compute_lineup_stints(gdf))

    df = pd.DataFrame(all_stints)
    if df.empty:
        print("No stints found.")
        return df

    # Expand lineup tuple into 5 player-id columns
    for i in range(5):
        df[f"p{i+1}"] = df["lineup"].apply(lambda x, i=i: x[i] if i < len(x) else pd.NA)

    agg = (
        df.groupby(["teamId", "season", "p1", "p2", "p3", "p4", "p5"])
        .agg(
            games    =("gameId", "nunique"),
            total_min=("min",    "sum"),
        )
        .reset_index()
    )
    agg["total_min"] = agg["total_min"].round(1)

    # Attach player names (best-known name per personId)
    box = pd.read_parquet(PROCESSED_DIR / "box_scores.parquet")
    name_map = (
        box[["personId", "playerName"]]
        .drop_duplicates("personId")
        .set_index("personId")["playerName"]
        .to_dict()
    )
    for i in range(1, 6):
        agg[f"p{i}_name"] = agg[f"p{i}"].map(name_map)

    col_order = [
        "teamId", "season",
        "p1", "p2", "p3", "p4", "p5",
        "p1_name", "p2_name", "p3_name", "p4_name", "p5_name",
        "games", "total_min",
    ]
    agg = agg[col_order].sort_values("total_min", ascending=False).reset_index(drop=True)

    out = PROCESSED_DIR / "lineup_stats.parquet"
    agg.to_parquet(out, index=False)
    print(f"\nSaved: {out}  ({len(agg):,} unique lineups)")
    return agg


def build_box_scores() -> pd.DataFrame:
    """
    Aggregate processed play-by-play parquets into per-game player box scores.

    Output saved to data/processed/box_scores.parquet.

    Columns
    -------
    Identity : gameId, season, personId, playerName, teamId, teamTricode
    Volume   : min, pts, fgm, fga, fg3m, fg3a, ftm, fta,
               reb, oreb, dreb, ast, stl, blk, tov, pf, tf, fouls_drawn
    Ratios   : fg_pct, fg3_pct, ft_pct, efg_pct, ts_pct
    Possessions: team_poss, usg_pct, pts_per36, reb_per36, ast_per36
    """
    dfs = [pd.read_parquet(p) for p in sorted(PROCESSED_DIR.glob("cdnnba_*.parquet"))]
    pbp = pd.concat(dfs, ignore_index=True)

    pbp["personId"]  = pd.to_numeric(pbp["personId"],  errors="coerce")
    pbp["scoreHome"] = pd.to_numeric(pbp["scoreHome"], errors="coerce")
    pbp["scoreAway"] = pd.to_numeric(pbp["scoreAway"], errors="coerce")
    pbp["isFieldGoal"] = pd.to_numeric(pbp["isFieldGoal"], errors="coerce")

    player_rows = pbp[pbp["personId"].notna() & (pbp["personId"] != 0)].copy()

    # ── Identity (playerName, teamId, teamTricode, season) ───────────────────
    identity = (
        player_rows[player_rows["playerName"].notna() & (player_rows["playerName"] != "")]
        .sort_values("orderNumber")
        .groupby(["gameId", "personId"])
        .agg(
            playerName  =("playerName",   "first"),
            teamId      =("teamId",        "first"),
            teamTricode =("teamTricode",   "first"),
            season      =("season",        "first"),
        )
        .reset_index()
    )

    # ── Running-total stats (take max of cumulative counter per player-game) ─
    cumul = (
        player_rows
        .groupby(["gameId", "personId"])[
            ["pointsTotal", "reboundTotal", "reboundOffensiveTotal",
             "reboundDefensiveTotal", "turnoverTotal", "foulPersonalTotal",
             "foulTechnicalTotal"]
        ]
        .max()
        .rename(columns={
            "pointsTotal":            "pts",
            "reboundTotal":           "reb",
            "reboundOffensiveTotal":  "oreb",
            "reboundDefensiveTotal":  "dreb",
            "turnoverTotal":          "tov",
            "foulPersonalTotal":      "pf",
            "foulTechnicalTotal":     "tf",
        })
        .reset_index()
    )

    # ── Field goals ──────────────────────────────────────────────────────────
    shots = player_rows[player_rows["isFieldGoal"] == 1].copy()
    shots["made"]    = (shots["shotResult"] == "Made").astype(int)
    shots["is3"]     = (shots["actionType"] == "3pt").astype(int)
    shots["fg3_made"]= ((shots["actionType"] == "3pt") & (shots["shotResult"] == "Made")).astype(int)

    shot_stats = (
        shots
        .groupby(["gameId", "personId"])
        .agg(fga=("made", "count"), fgm=("made", "sum"),
             fg3a=("is3", "sum"),   fg3m=("fg3_made", "sum"))
        .reset_index()
    )

    # ── Free throws ──────────────────────────────────────────────────────────
    fts = player_rows[player_rows["actionType"] == "freethrow"].copy()
    fts["made"] = (fts["shotResult"] == "Made").astype(int)
    ft_stats = (
        fts.groupby(["gameId", "personId"])
        .agg(fta=("made", "count"), ftm=("made", "sum"))
        .reset_index()
    )

    # ── Steals (steal rows have personId = the player who made the steal) ────
    stl_stats = (
        player_rows[player_rows["actionType"] == "steal"]
        .groupby(["gameId", "personId"])
        .size()
        .reset_index(name="stl")
    )

    # ── Blocks (block rows have personId = the player who made the block) ────
    blk_stats = (
        player_rows[player_rows["actionType"] == "block"]
        .groupby(["gameId", "personId"])
        .size()
        .reset_index(name="blk")
    )

    # ── Fouls drawn (personal fouls drawn by each player) ────────────────────
    # personId on foul rows = fouler; foulDrawnPersonId = player who drew the foul
    fd_rows = pbp[
        (pbp["actionType"] == "foul")
        & (pbp["subType"] == "personal")
        & pbp["foulDrawnPersonId"].notna()
    ].copy()
    fd_rows["personId"] = pd.to_numeric(fd_rows["foulDrawnPersonId"], errors="coerce")
    fouls_drawn_stats = (
        fd_rows[fd_rows["personId"].notna()]
        .groupby(["gameId", "personId"])
        .size()
        .reset_index(name="fouls_drawn")
    )

    # ── Assists (on shot events; assistPersonId = the assisting player) ──────
    ast_rows = pbp[pbp["assistPersonId"].notna()].copy()
    ast_rows["personId"] = pd.to_numeric(ast_rows["assistPersonId"], errors="coerce")
    ast_stats = (
        ast_rows[ast_rows["personId"].notna()]
        .groupby(["gameId", "personId"])["assistTotal"]
        .max()
        .reset_index(name="ast")
    )

    # ── Minutes (per-game lineup tracking) ───────────────────────────────────
    print("Computing minutes per game...", flush=True)
    lineup_records = []
    games = pbp.groupby("gameId")
    total = pbp["gameId"].nunique()
    for i, (game_id, gdf) in enumerate(games, 1):
        if i % 500 == 0:
            print(f"  {i}/{total} games processed", flush=True)
        stats = _compute_lineup_stats(gdf)
        for pid, s in stats.items():
            lineup_records.append({"gameId": game_id, "personId": float(pid), **s})
    lineup_df = pd.DataFrame(lineup_records)

    # ── Merge all components ─────────────────────────────────────────────────
    box = identity.copy()
    for df in [cumul, shot_stats, ft_stats, ast_stats, stl_stats, blk_stats,
               fouls_drawn_stats, lineup_df]:
        box = box.merge(df, on=["gameId", "personId"], how="left")

    # Fill zeros for count stats
    count_cols = ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                  "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
                  "tf", "fouls_drawn"]
    box[count_cols] = box[count_cols].fillna(0).astype(int)
    box["min"]      = box["min"].fillna(0.0).round(1)

    # ── Shooting efficiency ───────────────────────────────────────────────────
    fga_s  = box["fga"].replace(0, pd.NA)
    fta_s  = box["fta"].replace(0, pd.NA)
    fg3a_s = box["fg3a"].replace(0, pd.NA)
    ts_denom = (2 * (box["fga"] + 0.44 * box["fta"])).replace(0, pd.NA)

    def _pct(num, denom):
        return pd.to_numeric(num / denom, errors="coerce").round(3)

    box["fg_pct"]  = _pct(box["fgm"],                          fga_s)
    box["fg3_pct"] = _pct(box["fg3m"],                         fg3a_s)
    box["ft_pct"]  = _pct(box["ftm"],                          fta_s)
    box["efg_pct"] = _pct(box["fgm"] + 0.5 * box["fg3m"],     fga_s)
    box["ts_pct"]  = _pct(box["pts"],                          ts_denom)

    # ── Possession-based metrics ──────────────────────────────────────────────
    # Team possessions per game: Poss ≈ FGA - OREB + TOV + 0.44*FTA
    team_poss = (
        box.groupby(["gameId", "teamId"])
        .apply(lambda g: (g["fga"] - g["oreb"] + g["tov"] + 0.44 * g["fta"]).sum(),
               include_groups=False)
        .reset_index(name="team_poss")
    )
    box = box.merge(team_poss, on=["gameId", "teamId"], how="left")

    team_poss_s = box["team_poss"].replace(0, pd.NA)
    min_s       = box["min"].replace(0, pd.NA)

    # Usage rate: share of team possessions used by player
    box["usg_pct"] = _pct(box["fga"] + 0.44 * box["fta"] + box["tov"], team_poss_s)

    box["pts_per36"] = pd.to_numeric(box["pts"] / min_s * 36, errors="coerce").round(1)
    box["reb_per36"] = pd.to_numeric(box["reb"] / min_s * 36, errors="coerce").round(1)
    box["ast_per36"] = pd.to_numeric(box["ast"] / min_s * 36, errors="coerce").round(1)

    out = PROCESSED_DIR / "box_scores.parquet"
    box.to_parquet(out, index=False)
    print(f"\nSaved: {out}  ({len(box):,} player-game records, {box.shape[1]} cols)")
    return box


def clean_box_scores() -> pd.DataFrame:
    """
    Post-process box_scores.parquet:
      - Remove true DNPs (0 min, 0 in all counting stats)
      - Downcast numeric types to save memory
      - Reorder columns into logical groups
    Overwrites box_scores.parquet in place.
    """
    box = pd.read_parquet(PROCESSED_DIR / "box_scores.parquet")
    n_before = len(box)

    count_cols = [
        "pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "tf", "fouls_drawn",
        "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    ]
    dnp_mask = (box["min"] == 0) & (box[count_cols].sum(axis=1) == 0)
    box = box[~dnp_mask].reset_index(drop=True)
    print(f"Removed {n_before - len(box):,} true DNP rows → {len(box):,} remaining")

    # Downcast: personId / teamId to int64, small counts to int16, ratios to float32
    box["personId"] = box["personId"].astype("int64")
    box["teamId"]   = pd.to_numeric(box["teamId"], errors="coerce").astype("Int64")
    box["season"]   = box["season"].astype("int16")

    for col in count_cols:
        box[col] = box[col].astype("int16")

    box["min"] = box["min"].astype("float32")

    for col in ["fg_pct", "fg3_pct", "ft_pct", "efg_pct", "ts_pct",
                "team_poss", "usg_pct", "pts_per36", "reb_per36", "ast_per36"]:
        box[col] = box[col].astype("float32")

    # Canonical column order
    col_order = [
        "gameId", "season", "personId", "playerName", "teamId", "teamTricode",
        "min", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
        "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "tf", "fouls_drawn",
        "fg_pct", "fg3_pct", "ft_pct", "efg_pct", "ts_pct",
        "team_poss", "usg_pct", "pts_per36", "reb_per36", "ast_per36",
    ]
    box = box[col_order]

    out = PROCESSED_DIR / "box_scores.parquet"
    box.to_parquet(out, index=False)
    print(f"Saved cleaned: {out}  ({len(box):,} rows, {box.shape[1]} cols)")
    return box


def load_and_unify(path: Path) -> pd.DataFrame:
    season = int(path.stem.split("_")[-1])
    df = pd.read_csv(path, low_memory=False)
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    for col, default in FILL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    df["season"] = season
    return df[CANONICAL_COLS]


def _compute_lineup_stints_scored(game_df: pd.DataFrame, home_team_id) -> list[dict]:
    """
    Like _compute_lineup_stints but records score_delta per stint.
    score_delta = home_score_change - away_score_change during the stint.
    """
    game_df = game_df.sort_values("orderNumber")
    game_id = game_df["gameId"].iloc[0]
    season  = game_df["season"].iloc[0]

    sh = pd.to_numeric(game_df["scoreHome"], errors="coerce").ffill().fillna(0)
    sa = pd.to_numeric(game_df["scoreAway"], errors="coerce").ffill().fillna(0)
    score_diff = (sh - sa).values

    player_team: dict = {}
    team_court:  dict = defaultdict(set)
    stint_start: dict = {}          # teamId → entry_clock
    stint_sdiff: dict = {}          # teamId → score_diff at entry
    stints:      list = []
    carry_over:  set  = set()
    current_period: int = 0
    period_start_clock: dict = {}

    def _flush(team_id, end_clock, end_sdiff):
        if team_id not in stint_start:
            return
        entry_clock = stint_start.pop(team_id)
        entry_sdiff = stint_sdiff.pop(team_id, end_sdiff)
        lineup = frozenset(team_court[team_id])
        if len(lineup) != 5:
            return
        minutes = round(max(0.0, (entry_clock - end_clock) / 60.0), 2)
        sign = 1 if team_id == home_team_id else -1
        stints.append({
            "gameId":      game_id,
            "season":      season,
            "teamId":      team_id,
            "lineup":      tuple(sorted(lineup)),
            "min":         minutes,
            "net_pts":     round(sign * (end_sdiff - entry_sdiff), 1),
        })

    def _try_start(team_id, clock_s, sdiff):
        if len(team_court[team_id]) == 5:
            stint_start[team_id] = clock_s
            stint_sdiff[team_id] = sdiff

    rows = zip(
        game_df["actionType"].tolist(), game_df["subType"].tolist(),
        game_df["period"].tolist(),     game_df["clock"].tolist(),
        game_df["personId"].tolist(),   game_df["teamId"].tolist(),
        score_diff,
    )

    for atype, stype, period, clock, pid, tid, sdiff in rows:
        clock_s = _parse_clock(clock) or 0.0
        if not pd.isna(pid) and pid != 0 and not pd.isna(tid):
            player_team[pid] = tid

        if atype == "period":
            if stype == "start":
                current_period = int(period) if not pd.isna(period) else current_period + 1
                plen = _period_secs(current_period)
                period_start_clock[current_period] = plen
                for pid2 in carry_over:
                    tid2 = player_team.get(pid2)
                    if tid2 is not None:
                        team_court[tid2].add(pid2)
                for tid2 in {player_team.get(p) for p in carry_over if player_team.get(p)}:
                    _try_start(tid2, plen, sdiff)
                carry_over.clear()
            elif stype == "end":
                carry_over = {p for court in team_court.values() for p in court}
                for tid2 in list(stint_start.keys()):
                    _flush(tid2, 0.0, sdiff)
                team_court.clear()

        elif atype == "substitution":
            if pd.isna(pid) or pid == 0:
                continue
            if not pd.isna(tid):
                player_team[pid] = tid
            actual_tid = player_team.get(pid)
            if actual_tid is None:
                continue
            if stype == "out":
                _flush(actual_tid, clock_s, sdiff)
                team_court[actual_tid].discard(pid)
                carry_over.discard(pid)
                _try_start(actual_tid, clock_s, sdiff)
            elif stype == "in":
                _flush(actual_tid, clock_s, sdiff)
                team_court[actual_tid].add(pid)
                carry_over.discard(pid)
                _try_start(actual_tid, clock_s, sdiff)

        else:
            if pd.isna(pid) or pid == 0:
                continue
            if pd.isna(tid):
                tid = player_team.get(pid)
            if tid is None:
                continue
            player_team[pid] = tid
            if pid not in team_court[tid] and current_period > 0:
                plen = period_start_clock.get(current_period, _period_secs(current_period))
                _flush(tid, plen, sdiff)
                team_court[tid].add(pid)
                _try_start(tid, plen, sdiff)

    final_sdiff = score_diff[-1] if len(score_diff) else 0.0
    for tid2 in list(stint_start.keys()):
        _flush(tid2, 0.0, final_sdiff)

    return stints


PACE = 2.083   # NBA average possessions per lineup-minute (100 poss / 48 min)

EMBED_STATS = ["pts", "reb", "ast", "stl", "blk", "tov", "min",
               "ts_pct", "efg_pct", "usg_pct"]


def build_lineup_ratings() -> pd.DataFrame:
    """
    Build lineup_ratings.parquet extending lineup_stats with:

    Ratings  : net_pts, possessions, net_rating, off_rating, def_rating
    Embedding: mean of 5-player season-avg stat vectors (one col per stat)

    Net rating per 100 possessions = net_pts / possessions * 100
    Possessions ≈ total_min × PACE  (NBA avg 2.083 poss/lineup-min)
    """
    dfs = [pd.read_parquet(p) for p in sorted(PROCESSED_DIR.glob("cdnnba_*.parquet"))]
    pbp = pd.concat(dfs, ignore_index=True)
    pbp["personId"]  = pd.to_numeric(pbp["personId"],  errors="coerce")
    pbp["scoreHome"] = pd.to_numeric(pbp["scoreHome"], errors="coerce")
    pbp["scoreAway"] = pd.to_numeric(pbp["scoreAway"], errors="coerce")

    sched = pd.read_parquet(PROCESSED_DIR / "game_schedule.parquet")
    home_map = sched.set_index("gameId")["home_team_id"].to_dict()

    # ── Score-tracked stints ──────────────────────────────────────────────────
    all_stints: list[dict] = []
    total = pbp["gameId"].nunique()
    print("Computing score-tracked lineup stints...")
    for i, (gid, gdf) in enumerate(pbp.groupby("gameId"), 1):
        if i % 500 == 0:
            print(f"  {i}/{total}")
        home_tid = home_map.get(gid)
        all_stints.extend(_compute_lineup_stints_scored(gdf, home_tid))

    stints_df = pd.DataFrame(all_stints)

    # ── Expand lineup tuple → p1..p5 ─────────────────────────────────────────
    for i in range(5):
        stints_df[f"p{i+1}"] = stints_df["lineup"].apply(
            lambda x, i=i: x[i] if i < len(x) else pd.NA
        )

    # ── Aggregate per unique lineup ───────────────────────────────────────────
    grp_cols = ["teamId", "season", "p1", "p2", "p3", "p4", "p5"]
    agg = (
        stints_df.groupby(grp_cols)
        .agg(
            games    =("gameId",   "nunique"),
            total_min=("min",      "sum"),
            net_pts  =("net_pts",  "sum"),
        )
        .reset_index()
    )
    agg["total_min"] = agg["total_min"].round(1)
    agg["net_pts"]   = agg["net_pts"].round(1)
    agg["possessions"]  = (agg["total_min"] * PACE).round(1)
    agg["net_rating"]   = (agg["net_pts"] / agg["possessions"].replace(0, np.nan) * 100).round(2)

    # Prorate team game totals for offensive/defensive ratings
    box = pd.read_parquet(PROCESSED_DIR / "box_scores.parquet")
    box["teamId"] = pd.to_numeric(box["teamId"], errors="coerce").astype("Int64")
    team_game_pts = box.groupby(["gameId", "teamId"])["pts"].sum().reset_index()
    team_game_pts["teamId"] = team_game_pts["teamId"].astype("int64")

    # opponent pts per team-game
    s2 = sched[["gameId", "home_team_id", "away_team_id"]]
    h = team_game_pts.rename(columns={"teamId": "home_team_id", "pts": "home_pts"})
    a = team_game_pts.rename(columns={"teamId": "away_team_id", "pts": "away_pts"})
    gpts = s2.merge(h, on=["gameId", "home_team_id"]).merge(a, on=["gameId", "away_team_id"])

    home_opp = gpts[["gameId", "home_team_id", "home_pts", "away_pts"]].rename(
        columns={"home_team_id": "teamId", "home_pts": "team_pts", "away_pts": "opp_pts"})
    away_opp = gpts[["gameId", "away_team_id", "away_pts", "home_pts"]].rename(
        columns={"away_team_id": "teamId", "away_pts": "team_pts", "home_pts": "opp_pts"})
    game_opp = pd.concat([home_opp, away_opp])

    # team total minutes per game
    team_min = box.groupby(["gameId", "teamId"])["min"].sum().reset_index(name="team_min")
    team_min["teamId"] = team_min["teamId"].astype("int64")
    game_opp = game_opp.merge(team_min, on=["gameId", "teamId"])

    # per-stint proration
    stint_opp = stints_df.merge(game_opp, on=["gameId", "teamId"], how="left")
    stint_opp["stint_team_pts"] = (
        stint_opp["min"] / stint_opp["team_min"].replace(0, pd.NA) * stint_opp["team_pts"]
    )
    stint_opp["stint_opp_pts"] = (
        stint_opp["min"] / stint_opp["team_min"].replace(0, pd.NA) * stint_opp["opp_pts"]
    )

    for i in range(5):
        stint_opp[f"p{i+1}"] = stint_opp["lineup"].apply(
            lambda x, i=i: x[i] if i < len(x) else pd.NA)

    rating_agg = (
        stint_opp.groupby(grp_cols)
        .agg(pts_scored=("stint_team_pts", "sum"),
             pts_allowed=("stint_opp_pts",  "sum"))
        .reset_index()
    )
    agg = agg.merge(rating_agg, on=grp_cols, how="left")
    agg["off_rating"] = (agg["pts_scored"] / agg["possessions"].replace(0, np.nan) * 100).round(2)
    agg["def_rating"] = (agg["pts_allowed"] / agg["possessions"].replace(0, np.nan) * 100).round(2)

    # ── Mean player embeddings ────────────────────────────────────────────────
    # Season-average stats per player (across all games for that season)
    box["personId"] = pd.to_numeric(box["personId"], errors="coerce")
    player_avg = (
        box[box["min"] > 0]
        .groupby(["personId", "season"])[EMBED_STATS]
        .mean()
        .reset_index()
    )

    # For each lineup, look up each player's season-avg embedding and take the mean
    embed_rows = []
    for _, row in agg.iterrows():
        season = row["season"]
        pids   = [row[f"p{i+1}"] for i in range(5)]
        vecs   = []
        for pid in pids:
            match = player_avg[
                (player_avg["personId"] == pid) & (player_avg["season"] == season)
            ]
            if not match.empty:
                vecs.append(match[EMBED_STATS].values[0])
        if vecs:
            mean_vec = np.mean(vecs, axis=0)
        else:
            mean_vec = np.zeros(len(EMBED_STATS))
        embed_rows.append(mean_vec)

    embed_df = pd.DataFrame(embed_rows, columns=[f"emb_{s}" for s in EMBED_STATS])
    agg = pd.concat([agg.reset_index(drop=True), embed_df], axis=1)

    # ── Player names ──────────────────────────────────────────────────────────
    name_map = (
        box[["personId", "playerName"]].drop_duplicates("personId")
        .set_index("personId")["playerName"].to_dict()
    )
    for i in range(1, 6):
        agg[f"p{i}_name"] = agg[f"p{i}"].map(name_map)

    col_order = (
        ["teamId", "season",
         "p1", "p2", "p3", "p4", "p5",
         "p1_name", "p2_name", "p3_name", "p4_name", "p5_name",
         "games", "total_min", "net_pts", "possessions",
         "net_rating", "off_rating", "def_rating",
         "pts_scored", "pts_allowed"]
        + [f"emb_{s}" for s in EMBED_STATS]
    )
    agg = agg[[c for c in col_order if c in agg.columns]]
    agg = agg.sort_values("total_min", ascending=False).reset_index(drop=True)

    out = PROCESSED_DIR / "lineup_ratings.parquet"
    agg.to_parquet(out, index=False)
    print(f"\nSaved: {out}  ({len(agg):,} lineups, {agg.shape[1]} cols)")
    return agg


def run():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for csv_path in sorted(RAW_DIR.glob("*.csv")):
        df = load_and_unify(csv_path)
        out = PROCESSED_DIR / f"{csv_path.stem}.parquet"
        df.to_parquet(out, index=False)
        print(f"{csv_path.name}  ->  {out.name}  ({len(df):,} rows, {df.shape[1]} cols)")
    print("Done.")


if __name__ == "__main__":
    build_box_scores()
    clean_box_scores()
    build_lineup_stats()
