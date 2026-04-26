import math
import unicodedata
import warnings
from datetime import date as _date
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: dict = {}

TEAM_NAMES = {
    "ATL": "Atlanta Hawks",      "BOS": "Boston Celtics",     "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",  "CHI": "Chicago Bulls",      "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",   "DEN": "Denver Nuggets",     "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers",        "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",         "MIL": "Milwaukee Bucks",    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks",  "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",      "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",    "UTA": "Utah Jazz",          "WAS": "Washington Wizards",
}

TEAM_LOGOS = {
    "ATL": "https://a.espncdn.com/i/teamlogos/nba/500/1.png",
    "BOS": "https://a.espncdn.com/i/teamlogos/nba/500/2.png",
    "NOP": "https://a.espncdn.com/i/teamlogos/nba/500/3.png",
    "CHI": "https://a.espncdn.com/i/teamlogos/nba/500/4.png",
    "CLE": "https://a.espncdn.com/i/teamlogos/nba/500/5.png",
    "DAL": "https://a.espncdn.com/i/teamlogos/nba/500/6.png",
    "DEN": "https://a.espncdn.com/i/teamlogos/nba/500/7.png",
    "DET": "https://a.espncdn.com/i/teamlogos/nba/500/8.png",
    "GSW": "https://a.espncdn.com/i/teamlogos/nba/500/9.png",
    "HOU": "https://a.espncdn.com/i/teamlogos/nba/500/10.png",
    "IND": "https://a.espncdn.com/i/teamlogos/nba/500/11.png",
    "LAC": "https://a.espncdn.com/i/teamlogos/nba/500/12.png",
    "LAL": "https://a.espncdn.com/i/teamlogos/nba/500/13.png",
    "MIA": "https://a.espncdn.com/i/teamlogos/nba/500/14.png",
    "MIL": "https://a.espncdn.com/i/teamlogos/nba/500/15.png",
    "MIN": "https://a.espncdn.com/i/teamlogos/nba/500/16.png",
    "BKN": "https://a.espncdn.com/i/teamlogos/nba/500/17.png",
    "NYK": "https://a.espncdn.com/i/teamlogos/nba/500/18.png",
    "ORL": "https://a.espncdn.com/i/teamlogos/nba/500/19.png",
    "PHI": "https://a.espncdn.com/i/teamlogos/nba/500/20.png",
    "PHX": "https://a.espncdn.com/i/teamlogos/nba/500/21.png",
    "POR": "https://a.espncdn.com/i/teamlogos/nba/500/22.png",
    "SAC": "https://a.espncdn.com/i/teamlogos/nba/500/23.png",
    "SAS": "https://a.espncdn.com/i/teamlogos/nba/500/24.png",
    "OKC": "https://a.espncdn.com/i/teamlogos/nba/500/25.png",
    "UTA": "https://a.espncdn.com/i/teamlogos/nba/500/26.png",
    "WAS": "https://a.espncdn.com/i/teamlogos/nba/500/27.png",
    "TOR": "https://a.espncdn.com/i/teamlogos/nba/500/28.png",
    "MEM": "https://a.espncdn.com/i/teamlogos/nba/500/29.png",
    "CHA": "https://a.espncdn.com/i/teamlogos/nba/500/30.png",
}

TEAM_COLORS = {
    "ATL": "#C8102E", "BOS": "#007A33", "BKN": "#000000", "CHA": "#1D1160",
    "CHI": "#CE1141", "CLE": "#860038", "DAL": "#00538C", "DEN": "#0E2240",
    "DET": "#C8102E", "GSW": "#1D428A", "HOU": "#CE1141", "IND": "#002D62",
    "LAC": "#C8102E", "LAL": "#552583", "MEM": "#5D76A9", "MIA": "#98002E",
    "MIL": "#00471B", "MIN": "#0C2340", "NOP": "#0C2340", "NYK": "#006BB6",
    "OKC": "#007AC1", "ORL": "#0077C0", "PHI": "#006BB6", "PHX": "#1D1160",
    "POR": "#E03A3E", "SAC": "#5A2D81", "SAS": "#C4CED4", "TOR": "#CE1141",
    "UTA": "#002B5C", "WAS": "#002B5C",
}

_LEAGUE_AVG_PTS = 113.5
_LEAGUE_AVG_RTG = 113.5
_HOME_COURT_PTS = 2.5


def _norm_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    return nfkd.encode("ascii", "ignore").decode("ascii").lower().strip()


def _to_py(v):
    if v is None:
        return None
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (float, np.floating)) and math.isnan(float(v)):
        return None
    return v


def _safe_int(v, default: int = -1) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, float) and math.isnan(v):
            return default
        if hasattr(v, "item"):
            v = v.item()
        return int(v)
    except Exception:
        return default


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if hasattr(v, "item"):
            v = v.item()
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def _opp_matchup_multiplier(opp_def_rating: float | None) -> float:
    if opp_def_rating is None:
        return 1.0
    rel = (opp_def_rating - _LEAGUE_AVG_RTG) / 10.0
    return max(0.92, min(1.08, 1.0 + rel * 0.20))


def _df_records(df: pd.DataFrame) -> list[dict]:
    return [{k: _to_py(v) for k, v in row.items()} for row in df.to_dict("records")]


def _load():
    feats = pd.read_parquet("data/features/features.parquet")
    box   = pd.read_parquet("data/processed/box_scores.parquet")
    sched = pd.read_parquet("data/processed/game_schedule.parquet")
    gf    = pd.read_parquet("data/processed/game_features.parquet")

    from src.advanced_models import add_elo_ratings, add_workload_features
    feats = add_elo_ratings(feats, sched, box)
    feats = add_workload_features(feats)

    player_models = joblib.load("data/processed/models/player_models.joblib")
    player_fc     = joblib.load("data/processed/models/player_feat_cols.joblib")
    win_bundle    = joblib.load("data/processed/models/win_model.joblib")
    score_bundle  = joblib.load("data/processed/models/score_model.joblib")

    gf_ext = gf.merge(
        sched[["gameId", "home_tricode", "away_tricode"]],
        on="gameId", how="left"
    )

    pos_lookup: dict[str, str] = {}
    live_roster_names_by_team: dict[str, set[str]] = {}
    live_roster_rows_by_team: dict[str, list[dict]] = {}
    recent_trades: list[dict] = []
    try:
        from src.live_data import fetch_current_rosters, fetch_recent_trades
        roster_df = fetch_current_rosters()
        if not roster_df.empty:
            for _, r in roster_df.iterrows():
                pos_lookup[_norm_name(r["player_name"])] = r.get("primary_pos", "F")
                tri = str(r.get("team_tri", "")).upper()
                if tri:
                    norm = _norm_name(r.get("player_name", ""))
                    live_roster_names_by_team.setdefault(tri, set()).add(norm)
                    live_roster_rows_by_team.setdefault(tri, []).append({
                        "player_name": str(r.get("player_name", "")),
                        "norm_name":   norm,
                        "position":    str(r.get("primary_pos", "F")),
                        "source":      str(r.get("source", "espn_roster")),
                    })
        trades_df = fetch_recent_trades()
        if not trades_df.empty:
            recent_trades = _df_records(trades_df)
    except Exception:
        pass

    _cache.update(
        feats=feats, box=box, sched=sched, gf=gf, gf_ext=gf_ext,
        player_models=player_models, player_fc=player_fc,
        win_bundle=win_bundle, score_bundle=score_bundle,
        pos_lookup=pos_lookup,
        live_roster_names_by_team=live_roster_names_by_team,
        live_roster_rows_by_team=live_roster_rows_by_team,
        recent_trades=recent_trades,
    )


@app.on_event("startup")
async def _startup():
    _load()


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(_cache)}


@app.get("/teams")
def teams_list():
    return [
        {"tri": t, "name": n, "color": TEAM_COLORS.get(t, "#6366f1"), "logo": TEAM_LOGOS.get(t, "")}
        for t, n in sorted(TEAM_NAMES.items(), key=lambda x: x[1])
    ]


@app.get("/rosters")
def rosters(team: Optional[str] = None):
    rows_by_team = _cache.get("live_roster_rows_by_team", {})
    if not rows_by_team:
        return {"rosters": []}

    team_filter = team.upper() if team else None
    out = []
    for tri, rows in sorted(rows_by_team.items()):
        if team_filter and tri != team_filter:
            continue
        out.append({
            "team_tri":   tri,
            "team_name":  TEAM_NAMES.get(tri, tri),
            "team_color": TEAM_COLORS.get(tri, "#6366f1"),
            "team_logo":  TEAM_LOGOS.get(tri, ""),
            "players":    sorted(rows, key=lambda p: p.get("player_name", "")),
        })
    return {"rosters": out}


@app.get("/trades/recent")
def trades_recent():
    return {"trades": _cache.get("recent_trades", [])}


@app.get("/predict/players")
def predict_players(
    team: str,
    top_n: int = 8,
    opp: Optional[str] = None,
    injured_out: Optional[str] = None,
    is_home: Optional[bool] = None,
):
    feats  = _cache["feats"]
    models = _cache["player_models"]
    fc     = _cache["player_fc"]
    gf_ext = _cache["gf_ext"]

    team = team.upper()
    cur_season = int(feats["season"].max())
    active = feats[(feats["teamTricode"] == team) & (feats["season"] == cur_season)].copy()
    if active.empty:
        return {"team": team, "players": [], "context": {}}

    live_names = _cache.get("live_roster_names_by_team", {}).get(team, set())
    if live_names:
        active["_norm_player"] = active["playerName"].map(_norm_name)
        active = active[active["_norm_player"].isin(live_names)].copy()
        if active.empty:
            return {"team": team, "players": [], "context": {}}

    all_latest = (
        active.sort_values("game_date")
        .groupby("personId")
        .last()
        .reset_index()
    )
    ewma_col = all_latest["min_ewma"].fillna(0)
    szn_col  = all_latest["min_season_avg"].fillna(0)
    all_latest["_sel_score"] = 0.60 * ewma_col + 0.40 * szn_col
    latest = all_latest.sort_values("_sel_score", ascending=False).head(top_n).copy()

    opp_ctx: dict = {}
    if opp:
        opp = opp.upper()
        opp_rows = gf_ext[
            (gf_ext["home_tricode"] == opp) | (gf_ext["away_tricode"] == opp)
        ] if "home_tricode" in gf_ext.columns else pd.DataFrame()

        if opp_rows.empty:
            opp_feats = feats[feats["teamTricode"] == opp]
            if not opp_feats.empty:
                last_opp = opp_feats.sort_values("game_date").groupby("personId").last()
                opp_ctx = {
                    "opp_def_rating": float(last_opp["drtg_roll5"].mean()),
                    "opp_elo": float(last_opp["opp_elo"].mean()) if "opp_elo" in last_opp.columns else 1500.0,
                }
        else:
            last_opp_game = opp_rows.sort_values("game_date").iloc[-1]
            is_home_opp = str(last_opp_game.get("home_tricode", "")) == opp
            opp_ctx = {
                "opp_def_rating": _to_py(last_opp_game.get("home_drtg_roll5" if is_home_opp else "away_drtg_roll5")),
                "opp_elo":        _to_py(last_opp_game.get("home_team_elo" if is_home_opp else "away_team_elo")),
            }

        if not opp_ctx.get("opp_def_rating"):
            opp_feats = feats[feats["teamTricode"] == opp]
            if not opp_feats.empty:
                last_opp = opp_feats.sort_values("game_date").groupby("personId").last()
                opp_ctx["opp_def_rating"] = float(last_opp["drtg_roll5"].mean())
                if "opp_elo" not in opp_ctx or not opp_ctx["opp_elo"]:
                    opp_ctx["opp_elo"] = 1500.0

    usage_boost = min_boost = 0.0
    if injured_out:
        tokens = [t.strip() for t in injured_out.split(",") if t.strip()]
        out_ids: set[int] = set()
        out_frags: list[str] = []
        for t in tokens:
            if t.isdigit():
                out_ids.add(int(t))
            else:
                out_frags.append(_norm_name(t))

        all_pool = active.sort_values("game_date").groupby("personId").last().reset_index()

        def _is_out(row) -> bool:
            if _safe_int(row.get("personId", -1), -1) in out_ids:
                return True
            nm = _norm_name(row.get("playerName", ""))
            return any(f in nm for f in out_frags)

        for _, r in all_pool.iterrows():
            if _is_out(r):
                usage_boost += _safe_float(r.get("usg_pct_ewma", 0), 0.0)
                min_boost   += _safe_float(r.get("min_ewma", 0), 0.0)

        out_mask = latest.apply(_is_out, axis=1)
        latest = latest[~out_mask].copy()

        n = max(1, len(latest))
        usg_pp = usage_boost / n
        min_pp = min_boost  / n
        for col in ["usg_pct_roll5", "usg_pct_ewma"]:
            if col in latest.columns:
                latest[col] = (latest[col] + usg_pp).clip(upper=0.45)
        for col in ["min_roll5", "min_ewma", "min_stint_ewma"]:
            if col in latest.columns:
                latest[col] = (latest[col] + min_pp).clip(upper=44.0)

    ha_mult = 1.02 if is_home is True else (0.98 if is_home is False else 1.0)
    opp_def_rating = _safe_float(opp_ctx.get("opp_def_rating"), _LEAGUE_AVG_RTG) if opp_ctx else None
    matchup_mult = _opp_matchup_multiplier(opp_def_rating)

    results = []
    for _, row in latest.iterrows():
        row_d = row.to_dict()

        if is_home is not None:
            row_d["is_home"] = 1 if is_home else 0

        if opp_ctx:
            row_d.update(opp_ctx)
            team_elo = float(row_d.get("team_elo") or 1500)
            row_d["elo_diff"] = team_elo - float(row_d.get("opp_elo") or 1500)

        for c in fc:
            if c not in row_d:
                row_d[c] = 0.0

        X = pd.DataFrame([row_d])[fc].fillna(0)
        preds: dict = {}
        for target in ["pts", "reb", "ast", "stl", "blk", "tov", "min"]:
            ewma    = _safe_float(row.get(f"{target}_ewma"), 0.0)
            szn_val = _safe_float(row.get(f"{target}_season_avg"), 0.0)
            if target == "min" and min_boost > 0:
                n = max(1, len(latest))
                ewma = min(44.0, ewma + min_boost / n)
            mult     = 1.0 if target == "min" else ha_mult
            baseline = (0.70 * ewma + 0.30 * szn_val) * mult
            if target in models:
                try:
                    raw = float(models[target]["model"].predict(X)[0])
                    w = 0.05 if target == "min" else 0.10
                    preds[target] = round(max(0.0, w * raw + (1 - w) * baseline), 1)
                except Exception:
                    preds[target] = round(max(0.0, baseline), 1)
            else:
                preds[target] = round(max(0.0, baseline), 1)

        if opp_ctx:
            for stat in ["pts", "reb", "ast", "stl", "blk"]:
                preds[stat] = round(max(0.0, preds[stat] * matchup_mult), 1)
            tov_mult = max(0.92, min(1.08, 2.0 - matchup_mult))
            preds["tov"] = round(max(0.0, preds["tov"] * tov_mult), 1)

        szn_avg = {
            stat: round(_safe_float(row.get(f"{stat}_season_avg"), 0.0), 1)
            for stat in ["pts", "reb", "ast", "stl", "blk", "tov", "min"]
        }

        name = row.get("playerName", "")
        results.append({
            "personId":    _safe_int(row.get("personId"), -1),
            "name":        name,
            "team":        team,
            "position":    _cache.get("pos_lookup", {}).get(_norm_name(name), "F"),
            "predictions": preds,
            "season_avg":  szn_avg,
        })

    results = [p for p in results if p["personId"] >= 0]

    results_sorted = sorted(results, key=lambda p: p["predictions"]["min"], reverse=True)
    for i, p in enumerate(results_sorted):
        p["role"] = "starter" if i < 5 else "bench"
    results = results_sorted

    MAX_MIN = {"starter": 42.0, "bench": 35.0}
    for p in results:
        cap = MAX_MIN[p["role"]]
        if p["predictions"]["min"] > cap:
            p["predictions"]["min"] = cap

    total_min = sum(p["predictions"]["min"] for p in results)
    if total_min > 0 and results:
        scale = 240.0 / total_min
        for p in results:
            old_min = p["predictions"]["min"]
            new_min = round(old_min * scale, 1)
            if old_min > 0:
                stat_scale = new_min / old_min
                for stat in ["pts", "reb", "ast", "stl", "blk", "tov"]:
                    p["predictions"][stat] = round(p["predictions"][stat] * stat_scale, 1)
            p["predictions"]["min"] = new_min

    return {
        "team":    team,
        "players": results,
        "context": {
            "opp":                 opp,
            "opp_def_rating":      _to_py(opp_ctx.get("opp_def_rating")),
            "injured_out":         injured_out,
            "is_home":             is_home,
            "usage_redistributed": round(usage_boost, 3),
            "min_redistributed":   round(min_boost, 1),
        },
    }


def _team_profile(tri: str, out_names: set[str] | None = None) -> dict:
    feats = _cache["feats"]
    cur_season = int(feats["season"].max())
    tf = feats[(feats["teamTricode"] == tri) & (feats["season"] == cur_season)]
    if tf.empty:
        return {}

    live_names = _cache.get("live_roster_names_by_team", {}).get(tri, set())
    if live_names:
        tf = tf[tf["playerName"].map(_norm_name).isin(live_names)]
        if tf.empty:
            return {}

    latest = tf.sort_values("game_date").groupby("personId").last().reset_index()

    pts_lost = 0.0
    if out_names:
        def _out(row) -> bool:
            nm = str(row.get("playerName", "")).lower()
            return any(f in nm for f in out_names)
        out_mask = latest.apply(_out, axis=1)
        for _, r in latest[out_mask].iterrows():
            ewma = float(r.get("pts_ewma") or 0)
            szn  = float(r.get("pts_season_avg") or 0)
            pts_lost += 0.70 * ewma + 0.30 * szn
        latest = latest[~out_mask]

    if latest.empty:
        return {}

    def _w(col_roll5: str, col_szn: str) -> float:
        r5  = float(latest[col_roll5].mean()) if col_roll5 in latest.columns else 0.0
        szn = float(latest[col_szn].mean())   if col_szn  in latest.columns else 0.0
        return 0.70 * r5 + 0.30 * szn

    return {
        "ortg":     _w("ortg_roll5", "ortg_szn") or _LEAGUE_AVG_RTG,
        "drtg":     _w("drtg_roll5", "drtg_szn") or _LEAGUE_AVG_RTG,
        "pace":     _w("pace_proxy_roll5", "pace_proxy_szn") or 98.0,
        "elo":      float(latest["team_elo"].mean()) if "team_elo" in latest.columns else 1500.0,
        "pts_lost": pts_lost,
        "n_active": len(latest),
    }


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _model_game_pred(h_profile: dict, a_profile: dict) -> tuple[float, float, float]:
    try:
        feats        = _cache["feats"]
        win_bundle   = _cache["win_bundle"]
        score_bundle = _cache["score_bundle"]
        win_fc   = win_bundle["feat_cols"]
        score_fc = score_bundle["feat_cols"]

        row: dict = {
            "home_ortg_roll5": h_profile["ortg"], "home_ortg_szn": h_profile["ortg"],
            "home_drtg_roll5": h_profile["drtg"], "home_drtg_szn": h_profile["drtg"],
            "home_pace_proxy_roll5": h_profile["pace"],
            "home_team_elo":   h_profile["elo"],
            "away_ortg_roll5": a_profile["ortg"], "away_ortg_szn": a_profile["ortg"],
            "away_drtg_roll5": a_profile["drtg"], "away_drtg_szn": a_profile["drtg"],
            "away_pace_proxy_roll5": a_profile["pace"],
            "away_team_elo":   a_profile["elo"],
            "elo_diff":        h_profile["elo"] - a_profile["elo"],
            "rest_diff":       0,
            "roster_diff":     h_profile["n_active"] - a_profile["n_active"],
            "home_active_players": h_profile["n_active"],
            "away_active_players": a_profile["n_active"],
            "h2h_win_pct_home": 0.5, "h2h_pts_diff_avg": 0.0, "h2h_games": 0,
        }
        for col in win_fc:
            if col.startswith("diff_"):
                base = col[5:]
                row[col] = row.get(f"home_{base}", 0) - row.get(f"away_{base}", 0)

        Xw = pd.DataFrame([{c: row.get(c, 0) for c in win_fc}]).fillna(0)
        Xs = pd.DataFrame([{c: row.get(c, 0) for c in score_fc}]).fillna(0)
        return (
            float(win_bundle["model"].predict_proba(Xw)[0][1]),
            float(score_bundle["home_pts"].predict(Xs)[0]),
            float(score_bundle["away_pts"].predict(Xs)[0]),
        )
    except Exception:
        return 0.5, _LEAGUE_AVG_PTS, _LEAGUE_AVG_PTS


@app.get("/predict/game")
def predict_game(home: str, away: str):
    home, away = home.upper(), away.upper()

    try:
        home_out: set[str] = set()
        away_out: set[str] = set()
        try:
            from src.live_data import fetch_injury_report
            inj = fetch_injury_report()
            if not inj.empty:
                home_out = set(
                    inj[(inj["team_tri"] == home) & (inj["status_rank"] >= 4)]["player_name"].str.lower()
                )
                away_out = set(
                    inj[(inj["team_tri"] == away) & (inj["status_rank"] >= 4)]["player_name"].str.lower()
                )
        except Exception:
            pass

        h = _team_profile(home, home_out)
        a = _team_profile(away, away_out)
        if not h or not a:
            return {"error": "team not found"}

        h_pts_data = _LEAGUE_AVG_PTS * (h["ortg"] / _LEAGUE_AVG_RTG) * (a["drtg"] / _LEAGUE_AVG_RTG)
        a_pts_data = _LEAGUE_AVG_PTS * (a["ortg"] / _LEAGUE_AVG_RTG) * (h["drtg"] / _LEAGUE_AVG_RTG)
        h_pts_data += _HOME_COURT_PTS
        a_pts_data -= _HOME_COURT_PTS
        h_pts_data -= h["pts_lost"] * 0.60
        a_pts_data -= a["pts_lost"] * 0.60

        win_prob_model, h_pts_model, a_pts_model = _model_game_pred(h, a)

        home_score = round(0.80 * h_pts_data + 0.20 * h_pts_model, 1)
        away_score = round(0.80 * a_pts_data + 0.20 * a_pts_model, 1)

        score_diff    = home_score - away_score
        elo_diff      = h["elo"] - a["elo"]
        win_prob_data = _sigmoid(0.11 * score_diff + 0.0015 * elo_diff)
        win_prob      = max(0.05, min(0.95, round(0.75 * win_prob_data + 0.25 * win_prob_model, 4)))

        if (win_prob > 0.5) != (home_score > away_score):
            home_score, away_score = away_score, home_score

        win_bundle = _cache["win_bundle"]
        return {
            "home": home, "away": away,
            "home_name":  TEAM_NAMES.get(home, home),
            "away_name":  TEAM_NAMES.get(away, away),
            "home_color": TEAM_COLORS.get(home, "#6366f1"),
            "away_color": TEAM_COLORS.get(away, "#8b5cf6"),
            "home_logo":  TEAM_LOGOS.get(home, ""),
            "away_logo":  TEAM_LOGOS.get(away, ""),
            "win_prob_home": win_prob,
            "win_prob_away": round(1 - win_prob, 4),
            "home_score": home_score,
            "away_score": away_score,
            "home_ortg": round(h["ortg"], 1),
            "away_ortg": round(a["ortg"], 1),
            "home_drtg": round(h["drtg"], 1),
            "away_drtg": round(a["drtg"], 1),
            "home_pts_lost": round(h["pts_lost"], 1),
            "away_pts_lost": round(a["pts_lost"], 1),
            "model_acc": round(win_bundle.get("acc", 0), 4),
            "model_auc": round(win_bundle.get("auc", 0), 4),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/schedule/today")
def schedule_today():
    try:
        from src.live_data import fetch_todays_schedule
        df = fetch_todays_schedule()
        if df.empty:
            return {"games": []}
        games = []
        for r in _df_records(df):
            ht, at = r.get("home_tri", ""), r.get("away_tri", "")
            games.append({
                "game_id":    r.get("game_id", ""),
                "home_tri":   ht,   "away_tri":   at,
                "home_name":  TEAM_NAMES.get(ht, ht),
                "away_name":  TEAM_NAMES.get(at, at),
                "home_color": TEAM_COLORS.get(ht, "#6366f1"),
                "away_color": TEAM_COLORS.get(at, "#8b5cf6"),
                "home_logo":  TEAM_LOGOS.get(ht, ""),
                "away_logo":  TEAM_LOGOS.get(at, ""),
                "status":     r.get("status", ""),
                "start_time": r.get("start_time", ""),
                "home_score": r.get("home_score"),
                "away_score": r.get("away_score"),
                "venue":      r.get("venue", ""),
            })
        return {"games": games}
    except Exception as e:
        return {"games": [], "error": str(e)}


@app.get("/schedule/upcoming")
def schedule_upcoming():
    try:
        from src.live_data import fetch_schedule_range
        df = fetch_schedule_range(days_ahead=7)
        if df.empty:
            return {"games": []}
        games = []
        for r in _df_records(df):
            ht, at = r.get("home_tri", ""), r.get("away_tri", "")
            games.append({
                "game_id":    r.get("game_id", ""),
                "start_time": r.get("start_time", ""),
                "home_tri":   ht,   "away_tri":   at,
                "home_name":  TEAM_NAMES.get(ht, ht),
                "away_name":  TEAM_NAMES.get(at, at),
                "home_color": TEAM_COLORS.get(ht, "#6366f1"),
                "away_color": TEAM_COLORS.get(at, "#8b5cf6"),
                "home_logo":  TEAM_LOGOS.get(ht, ""),
                "away_logo":  TEAM_LOGOS.get(at, ""),
            })
        return {"games": games}
    except Exception as e:
        return {"games": [], "error": str(e)}


@app.get("/injuries")
def injuries(team: Optional[str] = None):
    try:
        from src.live_data import fetch_injury_report
        df = fetch_injury_report()
        if df.empty:
            return {"injuries": []}
        if team:
            df = df[df["team_tri"] == team.upper()]
        return {"injuries": [{
            "team_tri":      r.get("team_tri", ""),
            "player_name":   r.get("player_name", ""),
            "position":      r.get("position", ""),
            "status":        r.get("status", ""),
            "status_rank":   r.get("status_rank", 0),
            "injury_detail": r.get("injury_detail", ""),
        } for r in _df_records(df)]}
    except Exception as e:
        return {"injuries": [], "error": str(e)}


@app.get("/injuries/risk")
def injury_risk(team: Optional[str] = None):
    try:
        from datetime import datetime, timezone as _tz, timedelta as _td
        from src.live_data import fetch_injury_report

        feats = _cache["feats"]
        cur_season = int(feats["season"].max())
        tf = feats[feats["season"] == cur_season].copy()
        if team:
            tf = tf[tf["teamTricode"] == team.upper()]

        latest = tf.sort_values("game_date").groupby("personId").last().reset_index()
        cutoff = (datetime.now(_tz.utc) - _td(days=10)).strftime("%Y-%m-%d")
        latest = latest[latest["game_date"] >= cutoff]
        latest = latest[latest["min_season_avg"] >= 10]

        inj_df = fetch_injury_report()
        injured_names: dict[str, dict] = {}
        if not inj_df.empty:
            for r in _df_records(inj_df):
                nm = _norm_name(str(r.get("player_name", "")))
                injured_names[nm] = r

        results = []
        for _, row in latest.iterrows():
            min_avg  = float(row.get("min_season_avg") or 0)
            min_ewma = float(row.get("min_ewma") or 0)
            min_r5   = float(row.get("min_roll5") or 0)
            usg_avg  = float(row.get("usg_pct_season_avg") or 0.01)
            usg_ewma = float(row.get("usg_pct_ewma") or 0)
            rest     = float(row.get("rest_days") or 3)
            b2b      = bool(row.get("is_back_to_back") or False)
            name     = str(row.get("playerName", ""))
            tri      = str(row.get("teamTricode", ""))

            load_ratio  = min_ewma / max(min_avg, 5)
            load_factor = max(0.0, (load_ratio - 0.85) / 0.35)
            rest_factor = max(0.0, 1.0 - rest / 4.0)
            usg_ratio   = usg_ewma / max(usg_avg, 0.01)
            usg_factor  = max(0.0, (usg_ratio - 0.90) / 0.35)
            b2b_factor  = 0.25 if b2b else 0.0

            inj_rec    = injured_names.get(_norm_name(name))
            inj_factor = 0.0
            inj_status = ""
            inj_detail = ""
            if inj_rec:
                rank       = int(inj_rec.get("status_rank") or 0)
                inj_factor = min(1.0, rank / 5.0)
                inj_status = str(inj_rec.get("status", ""))
                inj_detail = str(inj_rec.get("injury_detail", ""))

            raw = (
                0.30 * min(1.0, load_factor) +
                0.25 * rest_factor +
                0.15 * min(1.0, usg_factor) +
                0.10 * b2b_factor +
                0.20 * inj_factor
            )
            risk_pct = round(min(100, raw * 100))

            drivers = []
            if load_factor > 0.3: drivers.append(f"High load ({round(load_ratio*100)}% of avg)")
            if rest_factor > 0.5: drivers.append(f"Low rest ({int(rest)}d)")
            if usg_factor  > 0.3: drivers.append(f"Usage spike ({round(usg_ratio*100)}%)")
            if b2b:               drivers.append("Back-to-back")
            if inj_factor  > 0:   drivers.append(inj_status or "On injury report")

            results.append({
                "player_name": name,
                "team_tri":    tri,
                "team_name":   TEAM_NAMES.get(tri, tri),
                "team_color":  TEAM_COLORS.get(tri, "#6366f1"),
                "risk_pct":    risk_pct,
                "risk_level":  "High" if risk_pct >= 55 else "Medium" if risk_pct >= 30 else "Low",
                "min_avg":     round(min_avg, 1),
                "min_recent":  round(min_r5, 1),
                "rest_days":   int(rest),
                "is_b2b":      b2b,
                "inj_status":  inj_status,
                "inj_detail":  inj_detail,
                "drivers":     drivers,
            })

        results.sort(key=lambda r: -r["risk_pct"])
        return {"players": results[:50] if not team else results}
    except Exception as e:
        return {"players": [], "error": str(e)}


@app.get("/live")
def live_scores():
    try:
        from src.live_data import fetch_live_scores
        df = fetch_live_scores()
        if df.empty:
            return {"games": []}
        games = []
        for r in _df_records(df):
            st = r.get("status", 1)
            status_str = "live" if st == 2 else "final" if st == 3 else "scheduled"
            cs = int(r.get("clock_sec") or 0)
            mins, secs = divmod(cs, 60)
            ht, at = r.get("home_tri", ""), r.get("away_tri", "")
            games.append({
                "homeTeam":   ht,   "awayTeam":   at,
                "home_name":  TEAM_NAMES.get(ht, ht),
                "away_name":  TEAM_NAMES.get(at, at),
                "home_color": TEAM_COLORS.get(ht, "#6366f1"),
                "away_color": TEAM_COLORS.get(at, "#8b5cf6"),
                "home_logo":  TEAM_LOGOS.get(ht, ""),
                "away_logo":  TEAM_LOGOS.get(at, ""),
                "homeScore":  r.get("home_score", 0),
                "awayScore":  r.get("away_score", 0),
                "period":     r.get("period", 0),
                "gameClock":  f"{mins}:{secs:02d}",
                "status":     status_str,
            })
        return {"games": games}
    except Exception as e:
        return {"games": [], "error": str(e)}


@app.get("/boxscore/final")
def final_boxscore_compare(home: str, away: str):
    try:
        from src.live_data import fetch_live_scores, fetch_live_boxscore

        home = home.upper()
        away = away.upper()

        live_df = fetch_live_scores()
        if live_df.empty:
            return {"is_final": False, "home": home, "away": away, "players": {}}

        target = live_df[
            (live_df["home_tri"] == home) &
            (live_df["away_tri"] == away) &
            (live_df["status"] == 3)
        ]
        if target.empty:
            return {"is_final": False, "home": home, "away": away, "players": {}}

        game_id           = str(target.iloc[-1].get("game_id", ""))
        home_score_actual = _safe_int(target.iloc[-1].get("home_score"), 0)
        away_score_actual = _safe_int(target.iloc[-1].get("away_score"), 0)

        home_box, away_box = fetch_live_boxscore(game_id)
        if home_box.empty and away_box.empty:
            return {"is_final": False, "home": home, "away": away, "players": {}}

        pred_game  = predict_game(home, away)
        home_preds = predict_players(team=home, top_n=15, opp=away, is_home=True).get("players", [])
        away_preds = predict_players(team=away, top_n=15, opp=home, is_home=False).get("players", [])

        def _index_preds(players: list[dict]) -> dict[str, dict]:
            return {_norm_name(p.get("name", "")): p for p in players}

        home_idx = _index_preds(home_preds)
        away_idx = _index_preds(away_preds)

        def _rows(df: pd.DataFrame, pred_idx: dict[str, dict]) -> list[dict]:
            rows = []
            for r in _df_records(df):
                name   = str(r.get("name", ""))
                pred   = pred_idx.get(_norm_name(name), {})
                pstats = pred.get("predictions", {}) if isinstance(pred, dict) else {}
                row = {
                    "name":     name,
                    "position": r.get("position", ""),
                    "actual": {
                        "pts": _safe_float(r.get("pts"), 0.0),
                        "reb": _safe_float(r.get("reb"), 0.0),
                        "ast": _safe_float(r.get("ast"), 0.0),
                        "stl": _safe_float(r.get("stl"), 0.0),
                        "blk": _safe_float(r.get("blk"), 0.0),
                        "tov": _safe_float(r.get("tov"), 0.0),
                        "min": _safe_float(r.get("min"), 0.0),
                    },
                    "predicted": {
                        "pts": _safe_float(pstats.get("pts"), 0.0),
                        "reb": _safe_float(pstats.get("reb"), 0.0),
                        "ast": _safe_float(pstats.get("ast"), 0.0),
                        "stl": _safe_float(pstats.get("stl"), 0.0),
                        "blk": _safe_float(pstats.get("blk"), 0.0),
                        "tov": _safe_float(pstats.get("tov"), 0.0),
                        "min": _safe_float(pstats.get("min"), 0.0),
                    },
                    "starter":            bool(r.get("starter", False)),
                    "matched_prediction": bool(pred),
                }
                row["diff"] = {
                    k: round(row["actual"][k] - row["predicted"][k], 1)
                    for k in ["pts", "reb", "ast", "stl", "blk", "tov", "min"]
                }
                rows.append(row)
            rows.sort(key=lambda x: x["actual"]["min"], reverse=True)
            return rows

        return {
            "is_final": True,
            "home": home, "away": away,
            "game_id": game_id,
            "score_actual":    {"home": home_score_actual, "away": away_score_actual},
            "score_predicted": {
                "home": _safe_float(pred_game.get("home_score"), 0.0) if isinstance(pred_game, dict) else 0.0,
                "away": _safe_float(pred_game.get("away_score"), 0.0) if isinstance(pred_game, dict) else 0.0,
            },
            "players": {
                "home": _rows(home_box, home_idx),
                "away": _rows(away_box, away_idx),
            },
        }
    except Exception as e:
        return {"is_final": False, "error": str(e), "players": {}}
