import json
import re as _re
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

LIVE_DIR = Path("data/live")
LIVE_DIR.mkdir(parents=True, exist_ok=True)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
_HEADERS  = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

NBA_ID_TO_TRI = {
    1610612737: "ATL", 1610612738: "BOS", 1610612751: "BKN",
    1610612766: "CHA", 1610612741: "CHI", 1610612739: "CLE",
    1610612742: "DAL", 1610612743: "DEN", 1610612765: "DET",
    1610612744: "GSW", 1610612745: "HOU", 1610612754: "IND",
    1610612746: "LAC", 1610612747: "LAL", 1610612763: "MEM",
    1610612748: "MIA", 1610612749: "MIL", 1610612750: "MIN",
    1610612740: "NOP", 1610612752: "NYK", 1610612760: "OKC",
    1610612753: "ORL", 1610612755: "PHI", 1610612756: "PHX",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS",
    1610612761: "TOR", 1610612762: "UTA", 1610612764: "WAS",
}
TRI_TO_NBA_ID = {v: k for k, v in NBA_ID_TO_TRI.items()}

# ESPN uses different abbreviations in a few places
_ESPN_TO_TRI = {
    "GS": "GSW", "SA": "SAS", "NY": "NYK", "NO": "NOP",
    "UTH": "UTA",
}

_ESPN_TEAM_NAME_TO_TRI = {
    "Atlanta Hawks": "ATL",       "Brooklyn Nets": "BKN",
    "Boston Celtics": "BOS",      "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",       "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",     "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",     "Indiana Pacers": "IND",
    "LA Clippers": "LAC",         "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",   "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",     "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP","New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC","Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",  "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR","Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",   "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",           "Washington Wizards": "WAS",
}

_STATUS_RANK = {
    "out": 4, "injured reserve": 4, "suspension": 4,
    "doubtful": 3,
    "questionable": 2, "game time decision": 2, "game_time_decision": 2,
    "day-to-day": 1, "day to day": 1, "day_to_day": 1, "probable": 1,
    "active": 0,
}

STATUS_COLORS = {
    "Out": "#e74c3c", "Injured Reserve": "#e74c3c",
    "Doubtful": "#e67e22", "Questionable": "#f1c40f",
    "Game Time Decision": "#f1c40f", "Day-To-Day": "#3498db",
    "Day To Day": "#3498db", "Probable": "#2ecc71", "Active": "#2ecc71",
}

_ESPN_POS_MAP = {"G": "G", "F": "F", "C": "C", "": "F"}


# -- cache helpers --

def _cache_path(name: str) -> Path:
    return LIVE_DIR / f"{name}_{date.today()}.json"


def _load_cache(name: str):
    p = _cache_path(name)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _save_cache(name: str, data):
    _cache_path(name).write_text(json.dumps(data))


def _get(url: str, timeout: int = 20) -> dict:
    r = requests.get(url, headers=_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _norm_tri(tri: str) -> str:
    return _ESPN_TO_TRI.get(tri, tri)


def _resolve_status_rank(status_name: str) -> int:
    key  = status_name.lower().strip()
    rank = _STATUS_RANK.get(key)
    if rank is not None:
        return rank
    return next((v for k, v in _STATUS_RANK.items()
                 if key.startswith(k) or k.startswith(key)), 0)


# -- ESPN team map --

def espn_team_map() -> dict:
    cached = _load_cache("espn_teams")
    if cached:
        return cached
    try:
        data = _get(f"{ESPN_BASE}/teams")
        teams = data["sports"][0]["leagues"][0]["teams"]
        m = {t["team"]["id"]: _norm_tri(t["team"]["abbreviation"]) for t in teams}
        _save_cache("espn_teams", m)
        return m
    except Exception as e:
        print(f"  [live] ESPN team map failed: {e}")
        return {}


# -- schedule --

def fetch_todays_schedule(force: bool = False) -> pd.DataFrame:
    cached = _load_cache("schedule")
    if cached and not force:
        return pd.DataFrame(cached)

    try:
        data = _get(f"{ESPN_BASE}/scoreboard")
    except Exception as e:
        print(f"  [live] schedule fetch failed: {e}")
        return pd.DataFrame()

    games = []
    for e in data.get("events", []):
        comp        = e["competitions"][0]
        competitors = comp["competitors"]
        home = next((c for c in competitors if c["homeAway"] == "home"), None)
        away = next((c for c in competitors if c["homeAway"] == "away"), None)
        if not home or not away:
            continue

        home_tri   = _norm_tri(home["team"]["abbreviation"])
        away_tri   = _norm_tri(away["team"]["abbreviation"])
        status_obj = comp.get("status", {})
        status     = status_obj.get("type", {}).get("description", "Scheduled")

        games.append({
            "game_id":     e["id"],
            "start_time":  comp.get("date", ""),
            "home_tri":    home_tri,
            "away_tri":    away_tri,
            "home_nba_id": TRI_TO_NBA_ID.get(home_tri, 0),
            "away_nba_id": TRI_TO_NBA_ID.get(away_tri, 0),
            "status":      status,
            "home_score":  float(home.get("score", 0) or 0),
            "away_score":  float(away.get("score", 0) or 0),
            "home_q":      [s["value"] for s in home.get("linescores", [])],
            "away_q":      [s["value"] for s in away.get("linescores", [])],
            "venue":       comp.get("venue", {}).get("fullName", ""),
        })

    _save_cache("schedule", games)
    return pd.DataFrame(games)


# -- injury report --

def fetch_injury_report(force: bool = False) -> pd.DataFrame:
    cached = _load_cache("injuries")
    if cached and not force:
        return pd.DataFrame(cached)

    try:
        data = _get(f"{ESPN_BASE}/injuries")
    except Exception as e:
        print(f"  [live] ESPN injuries API failed: {e}")
        return pd.DataFrame()

    emap = espn_team_map()
    rows = []
    for team_entry in data.get("injuries", []):
        team_name = team_entry.get("displayName", "")
        team_id   = str(team_entry.get("id", ""))
        tri       = _ESPN_TEAM_NAME_TO_TRI.get(team_name) or emap.get(team_id, "")
        if not tri:
            continue

        for p in team_entry.get("injuries", []):
            athlete    = p.get("athlete", {})
            name       = athlete.get("displayName", "")
            pos_obj    = athlete.get("position", {})
            position   = pos_obj.get("abbreviation", "") if isinstance(pos_obj, dict) else ""
            status_name = p.get("status", "Out")
            rank        = _resolve_status_rank(status_name)
            details     = p.get("details", {}) or {}
            inj_type    = details.get("type", "")
            inj_detail  = details.get("detail", "")
            detail_str  = (f"{inj_type} – {inj_detail}".strip(" –")
                           if (inj_type or inj_detail)
                           else p.get("shortComment", p.get("longComment", ""))[:120])

            rows.append({
                "team_tri":      tri,
                "nba_team_id":   TRI_TO_NBA_ID.get(tri, 0),
                "player_name":   name,
                "position":      position,
                "status":        status_name,
                "status_rank":   rank,
                "injury_detail": detail_str,
            })

    _save_cache("injuries", rows)
    return pd.DataFrame(rows)


def get_injured_players(team_tri: str, min_status_rank: int = 2) -> pd.DataFrame:
    df = fetch_injury_report()
    if df.empty:
        return df
    return df[(df["team_tri"] == team_tri) &
              (df["status_rank"] >= min_status_rank)].copy()


def injury_impact_score(team_tri: str,
                        feats: pd.DataFrame,
                        min_ewma_threshold: float = 20.0) -> float:
    """Fraction of team EWMA minutes unavailable due to injury. Returns [0, 1]."""
    injured = get_injured_players(team_tri)
    if injured.empty:
        return 0.0

    injured_names = set(injured["player_name"].str.lower())
    nba_id        = TRI_TO_NBA_ID.get(team_tri, 0)
    team_feats    = feats[feats["teamId"].astype("int64") == nba_id]
    if team_feats.empty:
        return 0.0

    latest = (team_feats[team_feats["min_ewma"].fillna(0) > min_ewma_threshold]
              .sort_values("game_date")
              .groupby("personId").last()
              .reset_index())
    if latest.empty:
        return 0.0

    total_min = latest["min_ewma"].sum()
    inj_min   = latest[latest["playerName"].str.lower().isin(injured_names)]["min_ewma"].sum()
    return round(float(inj_min / max(total_min, 1)), 3)


# -- roster positions --

def _subclassify_position(primary: str, stats: dict) -> str:
    reb = float(stats.get("reb", 0) or 0)
    ast = float(stats.get("ast", 0) or 0)
    blk = float(stats.get("blk", 0) or 0)
    pts = float(stats.get("pts", 0) or 0)

    if primary == "G":
        return "G-F" if reb >= 5.5 or (reb >= 4.0 and ast < 4.0) else "G"
    if primary == "F":
        if ast >= 5.0 or (ast >= 3.5 and reb < 5.5):
            return "F-G"
        if blk >= 1.3 or (reb >= 8.5 and blk >= 0.8):
            return "F-C"
        return "F"
    if primary == "C":
        return "C-F" if ast >= 3.5 or pts >= 18 else "C"
    return primary or "F"


def fetch_roster_positions(force: bool = False) -> pd.DataFrame:
    cached = _load_cache("roster_positions")
    if cached and not force:
        return pd.DataFrame(cached)

    emap = espn_team_map()
    if not emap:
        return pd.DataFrame()

    rows = []
    for espn_id, tri in emap.items():
        try:
            data = _get(f"{ESPN_BASE}/teams/{espn_id}/roster", timeout=15)
            for a in data.get("athletes", []):
                raw_pos = a.get("position", {}).get("abbreviation", "") or ""
                primary = _ESPN_POS_MAP.get(raw_pos.upper(), "F")
                rows.append({
                    "player_name": a.get("fullName", ""),
                    "team_tri":    tri,
                    "primary_pos": primary,
                })
            time.sleep(0.2)
        except Exception as e:
            print(f"  [live] position fetch failed for {tri}: {e}")

    _save_cache("roster_positions", rows)
    return pd.DataFrame(rows)


def fetch_recent_trades(force: bool = False, days_back: int = 120) -> pd.DataFrame:
    """
    Fetch recent NBA transactions and keep trade-related records.
    Returns rows with at least player_name and to_team_tri when available.
    """
    cached = _load_cache("recent_trades")
    if cached and not force:
        return pd.DataFrame(cached)

    try:
        from datetime import datetime
        from nba_api.stats.endpoints import playertransactions

        end_dt = date.today()
        start_dt = end_dt - timedelta(days=max(7, int(days_back)))
        raw = playertransactions.PlayerTransactions(
            start_date_nullable=start_dt.strftime("%m/%d/%Y"),
            end_date_nullable=end_dt.strftime("%m/%d/%Y"),
        ).get_data_frames()[0]
    except Exception as e:
        print(f"  [live] transactions fetch failed: {e}")
        return pd.DataFrame()

    if raw.empty:
        _save_cache("recent_trades", [])
        return raw

    # Normalize column names for resilient lookup across nba_api versions.
    colmap = {c.lower(): c for c in raw.columns}

    def _col(*names):
        for n in names:
            c = colmap.get(n.lower())
            if c:
                return c
        return None

    player_col = _col("player_name", "player")
    type_col = _col("transaction_type", "transaction")
    desc_col = _col("description", "notes")
    to_team_col = _col("to_team_abbreviation", "to_team", "team_abbreviation")
    date_col = _col("transaction_date", "date")

    if not player_col:
        return pd.DataFrame()

    rows = []
    for _, r in raw.iterrows():
        trans_type = str(r.get(type_col, "") if type_col else "").strip()
        desc = str(r.get(desc_col, "") if desc_col else "").strip()
        blob = f"{trans_type} {desc}".lower()
        if "trade" not in blob:
            continue

        to_team_raw = str(r.get(to_team_col, "") if to_team_col else "").strip().upper()
        to_team_tri = _norm_tri(to_team_raw) if to_team_raw else ""
        if not to_team_tri:
            # If destination team is missing, skip to avoid bad remaps.
            continue

        rows.append({
            "player_name": str(r.get(player_col, "")).strip(),
            "to_team_tri": to_team_tri,
            "transaction_type": trans_type or "Trade",
            "description": desc,
            "transaction_date": str(r.get(date_col, "") if date_col else ""),
        })

    _save_cache("recent_trades", rows)
    return pd.DataFrame(rows)


def fetch_current_rosters(force: bool = False) -> pd.DataFrame:
    """
    Build current rosters from ESPN team rosters and apply recent trades.
    """
    cached = _load_cache("current_rosters")
    if cached and not force:
        return pd.DataFrame(cached)

    base = fetch_roster_positions(force=force)
    if base.empty:
        return pd.DataFrame()

    roster = base.copy()
    roster["player_name"] = roster["player_name"].astype(str).str.strip()
    roster["team_tri"] = roster["team_tri"].astype(str).str.upper().map(_norm_tri)
    roster["source"] = "espn_roster"

    trades = fetch_recent_trades(force=force)
    if not trades.empty:
        # Latest trade destination wins for each player.
        trades = trades.copy()
        trades["player_name"] = trades["player_name"].astype(str).str.strip()
        trades["to_team_tri"] = trades["to_team_tri"].astype(str).str.upper().map(_norm_tri)
        if "transaction_date" in trades.columns:
            trades = trades.sort_values("transaction_date")
        latest = trades.dropna(subset=["player_name", "to_team_tri"]).drop_duplicates("player_name", keep="last")
        move_map = dict(zip(latest["player_name"], latest["to_team_tri"]))
        roster["team_tri"] = roster["player_name"].map(lambda n: move_map.get(n, None)).fillna(roster["team_tri"])
        traded_names = set(move_map.keys())
        roster["source"] = roster["player_name"].map(lambda n: "trade_adjusted" if n in traded_names else "espn_roster")

    records = roster.to_dict("records")
    _save_cache("current_rosters", records)
    return roster


# -- upcoming schedule --

def fetch_schedule_range(days_ahead: int = 7) -> pd.DataFrame:
    frames = []
    today  = date.today()
    for d in range(days_ahead + 1):
        dt     = today + timedelta(days=d)
        dt_str = dt.strftime("%Y%m%d")
        try:
            data   = _get(f"{ESPN_BASE}/scoreboard?dates={dt_str}")
            events = data.get("events", [])
            for e in events:
                comp        = e["competitions"][0]
                competitors = comp["competitors"]
                home = next((c for c in competitors if c["homeAway"] == "home"), None)
                away = next((c for c in competitors if c["homeAway"] == "away"), None)
                if not home or not away:
                    continue
                frames.append({
                    "game_date": dt.isoformat(),
                    "game_id":   e["id"],
                    "home_tri":  _norm_tri(home["team"]["abbreviation"]),
                    "away_tri":  _norm_tri(away["team"]["abbreviation"]),
                })
            time.sleep(0.2)
        except Exception:
            pass
    return pd.DataFrame(frames) if frames else pd.DataFrame()


# -- advanced stats (nba_api) --

def fetch_advanced_stats(season: str = "2025-26", force: bool = False) -> pd.DataFrame:
    cached = _load_cache("advanced_stats")
    if cached and not force:
        return pd.DataFrame(cached)
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        adv = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced",
        )
        df = adv.get_data_frames()[0]
        keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
                "USG_PCT", "TS_PCT", "PIE", "NET_RATING",
                "OFF_RATING", "DEF_RATING", "EFG_PCT"]
        df = df[[c for c in keep if c in df.columns]].copy()
        for c in df.columns:
            if c not in ("PLAYER_NAME", "TEAM_ABBREVIATION"):
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        records = df.to_dict("records")
        _save_cache("advanced_stats", records)
        print(f"  [live] Advanced stats: {len(df)} players")
        return df
    except Exception as e:
        print(f"  [live] Advanced stats fetch failed: {e}")
        return pd.DataFrame()


def advanced_stats_lookup(season: str = "2025-26") -> dict:
    df = fetch_advanced_stats(season)
    if df.empty:
        return {}
    result = {}
    for _, row in df.iterrows():
        pid = int(row.get("PLAYER_ID", 0))
        result[pid] = {
            "usg_pct": float(row.get("USG_PCT", 0.20) or 0.20),
            "ts_pct":  float(row.get("TS_PCT",  0.56) or 0.56),
            "pie":     float(row.get("PIE",     0.10) or 0.10),
            "net_rtg": float(row.get("NET_RATING", 0) or 0),
            "off_rtg": float(row.get("OFF_RATING", 0) or 0),
            "def_rtg": float(row.get("DEF_RATING", 0) or 0),
            "efg_pct": float(row.get("EFG_PCT", 0.53) or 0.53),
        }
    return result


# -- live scores + box scores (nba_api live) --

def _parse_game_clock(clock_str: str) -> float:
    if not clock_str:
        return 0.0
    m = _re.match(r"PT(\d+)M([\d.]+)S", clock_str)
    return int(m.group(1)) * 60 + float(m.group(2)) if m else 0.0


def fetch_live_scores() -> pd.DataFrame:
    try:
        from nba_api.live.nba.endpoints import scoreboard as _sb
        data  = _sb.ScoreBoard().get_dict()
        games = data["scoreboard"]["games"]
        rows  = []
        for g in games:
            home      = g["homeTeam"]
            away      = g["awayTeam"]
            period    = int(g.get("period", 0) or 0)
            clock_sec = _parse_game_clock(g.get("gameClock", ""))
            status    = int(g.get("gameStatus", 1) or 1)

            period_dur       = 300 if period > 4 else 720
            period_elapsed   = period_dur - clock_sec if period > 0 else 0
            reg_periods_done = max(0, min(period - 1, 4))
            ot_elapsed       = sum(300 for _ in range(max(0, period - 5))) if period > 4 else 0
            total_elapsed    = reg_periods_done * 720 + ot_elapsed + period_elapsed

            rows.append({
                "game_id":           g["gameId"],
                "home_tri":          home["teamTricode"],
                "away_tri":          away["teamTricode"],
                "home_score":        int(home.get("score", 0) or 0),
                "away_score":        int(away.get("score", 0) or 0),
                "status":            status,
                "status_text":       g.get("gameStatusText", ""),
                "period":            period,
                "clock_sec":         clock_sec,
                "total_sec_elapsed": total_elapsed,
                "series_text":       g.get("seriesText", ""),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  [live] nba_api scoreboard failed: {e}")
        return pd.DataFrame()


def fetch_live_boxscore(game_id: str) -> tuple:
    try:
        from nba_api.live.nba.endpoints import boxscore as _bs
        data = _bs.BoxScore(game_id).get_dict()
        game = data["game"]

        def _parse_team(team_data):
            rows = []
            for p in team_data.get("players", []):
                s       = p.get("statistics", {})
                mins    = s.get("minutesCalculated", "PT0M")
                m       = _re.match(r"PT(\d+)M", str(mins))
                min_val = int(m.group(1)) if m else 0
                rows.append({
                    "name":       p.get("name", ""),
                    "jersey":     p.get("jerseyNum", ""),
                    "position":   p.get("position", ""),
                    "starter":    p.get("starter", "0") == "1",
                    "on_court":   p.get("oncourt", "0") == "1",
                    "min":        min_val,
                    "pts":        int(s.get("points", 0) or 0),
                    "reb":        int(s.get("reboundsTotal", 0) or 0),
                    "ast":        int(s.get("assists", 0) or 0),
                    "stl":        int(s.get("steals", 0) or 0),
                    "blk":        int(s.get("blocks", 0) or 0),
                    "tov":        int(s.get("turnovers", 0) or 0),
                    "fgm":        int(s.get("fieldGoalsMade", 0) or 0),
                    "fga":        int(s.get("fieldGoalsAttempted", 0) or 0),
                    "fg3m":       int(s.get("threePointersMade", 0) or 0),
                    "fg3a":       int(s.get("threePointersAttempted", 0) or 0),
                    "ftm":        int(s.get("freeThrowsMade", 0) or 0),
                    "fta":        int(s.get("freeThrowsAttempted", 0) or 0),
                    "plus_minus": int(s.get("plusMinusPoints", 0) or 0),
                })
            return pd.DataFrame(rows)

        return _parse_team(game["homeTeam"]), _parse_team(game["awayTeam"])
    except Exception as e:
        print(f"  [live] box score fetch failed for {game_id}: {e}")
        return pd.DataFrame(), pd.DataFrame()


# -- refresh all --

def refresh_all(verbose: bool = True) -> dict:
    if verbose:
        print("Fetching today's schedule...")
    sched = fetch_todays_schedule(force=True)
    if verbose:
        print(f"  {len(sched)} games today")
        print("Fetching injuries + positions...")
    inj = fetch_injury_report(force=True)
    pos = fetch_roster_positions(force=True)
    out_count = int((inj["status_rank"] >= 4).sum()) if not inj.empty else 0
    q_count   = int((inj["status_rank"].isin([2, 3])).sum()) if not inj.empty else 0
    if verbose:
        print(f"  {out_count} OUT  |  {q_count} Questionable/Doubtful")
        print(f"  {len(pos)} players with positions")
    return {"schedule": sched, "injuries": inj, "positions": pos,
            "out_count": out_count, "q_count": q_count}


if __name__ == "__main__":
    result = refresh_all()
    print("\nToday's games:")
    print(result["schedule"][["home_tri", "away_tri", "status"]].to_string(index=False))
    print("\nInjured players (Out/Doubtful):")
    inj = result["injuries"]
    if not inj.empty:
        print(inj[inj["status_rank"] >= 3][
            ["team_tri", "player_name", "status", "injury_detail"]
        ].to_string(index=False))
