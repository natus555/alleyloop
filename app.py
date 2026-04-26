"""AlleyLoop — NBA Analytics Dashboard"""

import math
import time
import warnings
import joblib
import pandas as pd
import streamlit as st
import unicodedata as _ud
from pathlib import Path
from datetime import date, timedelta

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AlleyLoop",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROCESSED = Path("data/processed")
FEATURES  = Path("data/features")

TEAM_NAMES = {
    "ATL": "Atlanta Hawks",         "BKN": "Brooklyn Nets",
    "BOS": "Boston Celtics",        "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",         "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",      "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",       "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",       "IND": "Indiana Pacers",
    "LAC": "LA Clippers",           "LAL": "LA Lakers",
    "MEM": "Memphis Grizzlies",     "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",       "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",  "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers","SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",     "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",             "WAS": "Washington Wizards",
}

TRI_TO_NBA_ID = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751,
    "CHA": 1610612766, "CHI": 1610612741, "CLE": 1610612739,
    "DAL": 1610612742, "DEN": 1610612743, "DET": 1610612765,
    "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763,
    "MIA": 1610612748, "MIL": 1610612749, "MIN": 1610612750,
    "NOP": 1610612740, "NYK": 1610612752, "OKC": 1610612760,
    "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759,
    "TOR": 1610612761, "UTA": 1610612762, "WAS": 1610612764,
}

STATUS_COLORS = {
    "Out": "#e74c3c", "Injured Reserve": "#e74c3c",
    "Doubtful": "#e67e22", "Questionable": "#f1c40f",
    "Game Time Decision": "#f1c40f", "Day To Day": "#3498db",
    "Probable": "#27ae60", "Active": "#27ae60",
}

TEAM_COLORS = {
    "ATL": "#E03A3E", "BKN": "#000000", "BOS": "#007A33",
    "CHA": "#1D1160", "CHI": "#CE1141", "CLE": "#6F263D",
    "DAL": "#00538C", "DEN": "#FEC524", "DET": "#C8102E",
    "GSW": "#1D428A", "HOU": "#CE1141", "IND": "#FDBB30",
    "LAC": "#C8102E", "LAL": "#552583", "MEM": "#5D76A9",
    "MIA": "#98002E", "MIL": "#00471B", "MIN": "#0C2340",
    "NOP": "#0C2340", "NYK": "#006BB6", "OKC": "#007AC1",
    "ORL": "#0077C0", "PHI": "#006BB6", "PHX": "#1D1160",
    "POR": "#E03A3E", "SAC": "#5A2D81", "SAS": "#8A8D8F",
    "TOR": "#CE1141", "UTA": "#002B5C", "WAS": "#002B5C",
}

_LIGHT_COLOR_TEAMS = {"DEN", "IND"}

_LEAGUE_AVG_ORtg  = 113.0
_LEAGUE_AVG_DRtg  = 113.0
_LEAGUE_AVG_PACE  = 101.3
_LEAGUE_AVG_USG   = 0.20
_LEAGUE_AVG_TS    = 0.570
_REG_GAME_SEC     = 2880   # 4 × 720s
_SCORE_SIGMA      = 6.0    # logit scale: 70% win_prob ≈ 5.1 pt margin

_MODE_PARAMS = {
    "Regular Season": {
        "ortg": 113.0, "drtg": 113.0, "pace": 101.3,
        "score_scale": 1.0, "away_discount": 1.0,
        "rotation": 11, "min_thresh": 10,
    },
    "Playoffs": {
        "ortg": 109.0, "drtg": 109.0, "pace": 98.0,
        "score_scale": 0.945, "away_discount": 0.975,
        "rotation": 10, "min_thresh": 12,
    },
}


# -- data loaders --

@st.cache_data(show_spinner="Loading player features…")
def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES / "features.parquet")
    df["teamId"] = pd.to_numeric(df["teamId"], errors="coerce").astype("int64")
    return df


@st.cache_data(show_spinner="Loading schedule…")
def load_schedule() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "game_schedule.parquet")


@st.cache_data(show_spinner=False)
def load_team_efficiency() -> dict:
    gf    = pd.read_parquet(PROCESSED / "game_features.parquet")
    sched = load_schedule()
    gfs   = gf.merge(sched[["gameId", "home_team_id", "away_team_id"]],
                     on="gameId", how="inner")
    eff = {}
    all_ids = set(sched["home_team_id"].tolist() + sched["away_team_id"].tolist())
    for tid in all_ids:
        h = (gfs[gfs["home_team_id"] == tid]
             [["game_date", "home_ortg_roll5", "home_drtg_roll5", "home_pace_proxy_roll5"]]
             .rename(columns=lambda c: c.replace("home_", "")))
        a = (gfs[gfs["away_team_id"] == tid]
             [["game_date", "away_ortg_roll5", "away_drtg_roll5", "away_pace_proxy_roll5"]]
             .rename(columns=lambda c: c.replace("away_", "")))
        combined = pd.concat([h, a]).sort_values("game_date").tail(20)
        if combined.empty:
            continue
        m = combined.mean(numeric_only=True)
        eff[int(tid)] = {
            "ortg": float(m.get("ortg_roll5",  _LEAGUE_AVG_ORtg) or _LEAGUE_AVG_ORtg),
            "drtg": float(m.get("drtg_roll5",  _LEAGUE_AVG_DRtg) or _LEAGUE_AVG_DRtg),
            "pace": float(m.get("pace_proxy_roll5", _LEAGUE_AVG_PACE) or _LEAGUE_AVG_PACE),
        }
    return eff


@st.cache_data(show_spinner="Loading positions…")
def load_positions() -> pd.DataFrame:
    p = PROCESSED / "player_positions.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_resource(show_spinner="Loading win model…")
def load_win_model():
    p = PROCESSED / "models" / "win_model.joblib"
    if p.exists():
        import src.advanced_models  # noqa
        b = joblib.load(p)
        return b["model"], b["feat_cols"], b
    return None, [], {}


@st.cache_resource(show_spinner="Loading player models…")
def load_player_models():
    p  = PROCESSED / "models" / "player_models.joblib"
    fc = PROCESSED / "models" / "player_feat_cols.joblib"
    if p.exists() and fc.exists():
        import src.advanced_models  # noqa
        return joblib.load(p), joblib.load(fc)
    return {}, []


@st.cache_data(ttl=1800, show_spinner="Fetching live data…")
def get_live_data():
    try:
        from src.live_data import (fetch_todays_schedule, fetch_injury_report,
                                   fetch_schedule_range, fetch_roster_positions)
        today    = fetch_todays_schedule()
        inj      = fetch_injury_report()
        upcoming = fetch_schedule_range(days_ahead=7)
        pos      = fetch_roster_positions()
        return today, inj, upcoming, pos
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_advanced_stats() -> dict:
    try:
        from src.live_data import advanced_stats_lookup
        return advanced_stats_lookup()
    except Exception:
        return {}


def _get_fresh_injuries() -> pd.DataFrame:
    cache_file = Path("data/live") / f"injuries_{date.today().isoformat()}.json"
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) > 600:
        cache_file.unlink(missing_ok=True)
    try:
        from src.live_data import fetch_injury_report
        return fetch_injury_report()
    except Exception:
        return pd.DataFrame()


# -- team helpers --

@st.cache_data
def _tri_id_map() -> dict:
    sched = load_schedule()
    h = sched[["home_team_id","home_tricode"]].rename(columns={"home_team_id":"tid","home_tricode":"tri"})
    a = sched[["away_team_id","away_tricode"]].rename(columns={"away_team_id":"tid","away_tricode":"tri"})
    m = pd.concat([h, a]).drop_duplicates("tid")
    return dict(zip(m["tid"], m["tri"]))


def sorted_tricodes() -> list:
    return sorted(_tri_id_map().values())


def tri_to_id(tri: str) -> int:
    inv = {v: k for k, v in _tri_id_map().items()}
    return int(inv.get(tri, TRI_TO_NBA_ID.get(tri, 0)))


def _team_efficiency(team_id: int) -> tuple:
    eff = load_team_efficiency()
    t   = eff.get(int(team_id), {})
    return (
        t.get("ortg", _LEAGUE_AVG_ORtg),
        t.get("drtg", _LEAGUE_AVG_DRtg),
        t.get("pace", _LEAGUE_AVG_PACE),
    )


# -- prediction helpers --

_ID_COLS = {"personId", "gameId", "teamId", "opp_team_id", "season",
            "game_date", "is_home", "home_win"}


def team_agg(feats: pd.DataFrame, team_id: int, active_min: float = 5.0):
    th = feats[feats["teamId"] == int(team_id)]
    if th.empty:
        return None
    recent_dates = th["game_date"].dropna().drop_duplicates().sort_values().iloc[-30:]
    th = th[th["game_date"].isin(recent_dates)]
    active = th[th["min_ewma"].fillna(0) > active_min]
    if active.empty:
        return None
    latest = active.sort_values("game_date").groupby("personId").last()
    feat_cols = [c for c in latest.columns
                 if c not in _ID_COLS
                 and latest[c].dtype.kind in ("f", "i", "u")
                 and not c.startswith("Unnamed")]
    agg = latest[feat_cols].mean()
    agg["active_players"] = len(latest)
    return agg


def build_game_row(home: pd.Series, away: pd.Series, feat_cols: list,
                   home_inj: float = 0.0, away_inj: float = 0.0) -> pd.DataFrame:
    _OFF_STATS = {"pts_ewma", "ast_ewma", "fgm_ewma", "fg3m_ewma", "fta_ewma",
                  "ts_pct_ewma", "usg_pct_ewma", "ortg_roll5", "ortg_roll10", "ortg_szn"}
    row = {}
    for c in home.index:
        hv = float(home.get(c, 0) or 0)
        av = float(away.get(c, 0) or 0)
        if c in _OFF_STATS:
            hv *= (1 - home_inj * 0.7)
            av *= (1 - away_inj * 0.7)
        row[f"home_{c}"] = hv
        row[f"away_{c}"] = av
        row[f"diff_{c}"] = hv - av
    row["rest_diff"]           = row.get("home_rest_days", 0) - row.get("away_rest_days", 0)
    row["home_active_players"] = row.get("home_active_players", 8) * (1 - home_inj)
    row["away_active_players"] = row.get("away_active_players", 8) * (1 - away_inj)
    row["roster_diff"]         = row["home_active_players"] - row["away_active_players"]
    df = pd.DataFrame([row])
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[feat_cols].fillna(0)


def predict_game(home_tri: str, away_tri: str, feats: pd.DataFrame,
                 win_model, win_feat_cols: list,
                 home_inj: float = 0.0, away_inj: float = 0.0,
                 mode_params: dict = None):
    mp = mode_params or _MODE_PARAMS["Regular Season"]
    h_id = tri_to_id(home_tri)
    a_id = tri_to_id(away_tri)
    ha = team_agg(feats, h_id)
    aa = team_agg(feats, a_id)
    if ha is None or aa is None:
        return None

    win_row  = build_game_row(ha, aa, win_feat_cols, home_inj, away_inj)
    win_prob = float(win_model.predict_proba(win_row)[0, 1])

    h_ortg, h_drtg, h_pace = _team_efficiency(h_id)
    a_ortg, a_drtg, a_pace = _team_efficiency(a_id)

    avg_pace      = (h_pace + a_pace) / 2 * (mp["pace"] / _LEAGUE_AVG_PACE)
    home_adj_ortg = h_ortg * (a_drtg / mp["drtg"])
    away_adj_ortg = a_ortg * (h_drtg / mp["ortg"])
    home_pts_raw  = home_adj_ortg * avg_pace / 100.0 * mp["score_scale"]
    away_pts_raw  = away_adj_ortg * avg_pace / 100.0 * mp["score_scale"] * mp["away_discount"]
    home_pts      = home_pts_raw * (1 - home_inj * 0.7)
    away_pts      = away_pts_raw * (1 - away_inj * 0.7)

    # Split from win_prob so score direction always matches the probability bar
    total_pts      = home_pts + away_pts
    safe_p         = max(0.01, min(0.99, win_prob))
    implied_margin = _SCORE_SIGMA * math.log(safe_p / (1.0 - safe_p))
    home_pts = round(total_pts / 2.0 + implied_margin / 2.0, 1)
    away_pts = round(total_pts / 2.0 - implied_margin / 2.0, 1)

    return win_prob, home_pts, away_pts


def _prob_bar(home_tri, away_tri, p_home):
    """Win-probability bar. Away on left, home on right."""
    p_away = 1 - p_home
    ac = TEAM_COLORS.get(away_tri, "#555555")
    hc = TEAM_COLORS.get(home_tri, "#1d6fa4")
    at = "#111" if away_tri in _LIGHT_COLOR_TEAMS else "#fff"
    ht = "#111" if home_tri in _LIGHT_COLOR_TEAMS else "#fff"
    return f"""
    <div style="display:flex;height:44px;border-radius:8px;overflow:hidden;
                font-size:14px;font-weight:700;margin:8px 0;box-shadow:0 2px 6px rgba(0,0,0,.4);">
      <div style="width:{p_away*100:.1f}%;background:{ac};
                  display:flex;align-items:center;justify-content:center;color:{at};white-space:nowrap;padding:0 6px;">
        {away_tri}&nbsp;{p_away*100:.1f}%
      </div>
      <div style="width:{p_home*100:.1f}%;background:{hc};
                  display:flex;align-items:center;justify-content:center;color:{ht};white-space:nowrap;padding:0 6px;">
        {home_tri}&nbsp;{p_home*100:.1f}%
      </div>
    </div>"""


# -- live game math --

def _live_win_prob(home_score: int, away_score: int,
                   total_sec_elapsed: float,
                   pregame_home_prob: float = 0.5) -> float:
    sec_remaining  = max(_REG_GAME_SEC - total_sec_elapsed, 1)
    min_remaining  = sec_remaining / 60.0
    hca_pts        = 2.5 * (sec_remaining / _REG_GAME_SEC)
    adj_diff       = (home_score - away_score) + hca_pts
    z              = adj_diff / (11.0 * math.sqrt(max(min_remaining / 48.0, 0.005)))
    in_game_p      = 1.0 / (1.0 + math.exp(-z * 1.3))
    w_live         = max(0.0, 1.0 - sec_remaining / _REG_GAME_SEC)
    return round(w_live * in_game_p + (1.0 - w_live) * pregame_home_prob, 4)


def _projected_final(home_score: int, away_score: int,
                     total_sec_elapsed: float,
                     pregame_home_pts: float, pregame_away_pts: float):
    # Pregame prediction + regressed surplus. Early surplus heavily discounted.
    if total_sec_elapsed < 60:
        return pregame_home_pts, pregame_away_pts
    frac     = min(total_sec_elapsed / _REG_GAME_SEC, 0.99)
    sur_h    = (home_score - pregame_home_pts * frac) * frac
    sur_a    = (away_score - pregame_away_pts * frac) * frac
    return round(pregame_home_pts + sur_h, 1), round(pregame_away_pts + sur_a, 1)


# -- sidebar --

st.sidebar.title("🏀 AlleyLoop")
st.sidebar.caption("NBA Prediction Engine · 2020–2026")

tricodes   = sorted_tricodes()
team_label = [f"{t}  —  {TEAM_NAMES.get(t,t)}" for t in tricodes]

sel_idx = st.sidebar.selectbox(
    "Focus team",
    range(len(tricodes)),
    format_func=lambda i: team_label[i],
    index=tricodes.index("BOS") if "BOS" in tricodes else 0,
)
selected_tri  = tricodes[sel_idx]
selected_id   = tri_to_id(selected_tri)
selected_name = TEAM_NAMES.get(selected_tri, selected_tri)

st.sidebar.markdown("---")
_mode = st.sidebar.radio(
    "Mode", list(_MODE_PARAMS.keys()), index=1,
    help="Playoff: slower pace, tighter rotation (10). Regular: 11 players."
)
_mp = _MODE_PARAMS[_mode]

if st.sidebar.button("🔄 Refresh live data"):
    st.cache_data.clear()
    st.rerun()

try:
    _, _, wb = load_win_model()
    if wb.get("auc"):
        st.sidebar.caption(
            f"**Win model**  Acc {wb['acc']*100:.1f}%  AUC {wb['auc']:.3f}\n\n"
            f"**Stats (±3 abs tolerance)**\n\n"
            f"AST ✅ 82%  ·  REB 71%  ·  PTS 32%*\n\n"
            f"_*PTS ceiling ~35% (NBA scoring σ≈7 pts)_"
        )
except Exception:
    pass

tab_live, tab_games, tab_props, tab_team, tab_lineup, tab_injuries = st.tabs([
    "🔴  Live Games",
    "🎯  Game Predictions",
    "📈  Player Props",
    "📊  Team Stats",
    "🏆  Lineup Optimizer",
    "🏥  Injuries & Risk",
])


# -- tab: live games --

with tab_live:
    st.subheader("🔴 Live NBA Games")

    if st.button("⟳ Refresh", key="live_refresh"):
        st.cache_data.clear()
        st.rerun()

    try:
        from src.live_data import fetch_live_scores, fetch_live_boxscore

        @st.cache_data(ttl=30, show_spinner=False)
        def _get_live_scores():
            return fetch_live_scores()

        live_scores = _get_live_scores()
    except Exception as _e:
        live_scores = pd.DataFrame()
        st.warning(f"Live feed unavailable: {_e}")

    _feats_live  = load_features()
    _wm_l, _wfc_l, _ = load_win_model()

    if live_scores.empty:
        st.info("No live games right now. Come back during game time.")
    else:
        live_games = live_scores[live_scores["status"] == 2]
        done_games = live_scores[live_scores["status"] == 3]
        pre_games  = live_scores[live_scores["status"] == 1]

        if not live_games.empty:
            st.markdown("### 🏀 In Progress")

        for _, lg in live_games.iterrows():
            ht, at  = lg["home_tri"], lg["away_tri"]
            hs, as_ = int(lg["home_score"]), int(lg["away_score"])
            period  = int(lg.get("period", 0) or 0)
            clock_s = float(lg.get("clock_sec", 0) or 0)
            elapsed = float(lg.get("total_sec_elapsed", 0) or 0)
            ser_txt = str(lg.get("series_text", "") or "")
            gid     = str(lg["game_id"])

            pregame_result = None
            if _wm_l is not None:
                try:
                    pregame_result = predict_game(ht, at, _feats_live, _wm_l, _wfc_l,
                                                  mode_params=_mp)
                except Exception:
                    pass

            pre_prob  = pregame_result[0] if pregame_result else 0.5
            pre_h_pts = pregame_result[1] if pregame_result else None
            pre_a_pts = pregame_result[2] if pregame_result else None

            live_prob    = _live_win_prob(hs, as_, elapsed, pre_prob)
            _ph, _pa     = _projected_final(hs, as_, elapsed,
                                            pre_h_pts or hs, pre_a_pts or as_)
            _total       = _ph + _pa
            _safe_p      = max(0.01, min(0.99, live_prob))
            _imp_margin  = _SCORE_SIGMA * math.log(_safe_p / (1.0 - _safe_p))
            proj_h       = round(_total / 2.0 + _imp_margin / 2.0, 1)
            proj_a       = round(_total / 2.0 - _imp_margin / 2.0, 1)

            q_label   = f"Q{period}" if period <= 4 else f"OT{period-4}"
            clock_str = f"{int(clock_s//60)}:{int(clock_s%60):02d}" if clock_s > 0 else "End"
            label_str = f"{q_label} {clock_str}  {ser_txt}".strip()

            with st.expander(f"**{at} {as_}  —  {ht} {hs}**  |  {label_str}", expanded=True):
                c_a, c_mid, c_h = st.columns([2, 4, 2])
                hn = TEAM_NAMES.get(ht, ht)
                an = TEAM_NAMES.get(at, at)

                with c_a:
                    st.metric(f"✈️ {an}", f"{as_}", delta=f"proj {proj_a}")
                with c_h:
                    st.metric(f"🏠 {hn}", f"{hs}", delta=f"proj {proj_h}")
                with c_mid:
                    st.markdown(_prob_bar(ht, at, live_prob), unsafe_allow_html=True)
                    total_proj  = proj_h + proj_a
                    fav_tri     = ht if proj_h >= proj_a else at
                    margin_proj = abs(proj_h - proj_a)
                    st.caption(
                        f"Proj total {total_proj:.0f}  |  {fav_tri} by {margin_proj:.1f}  "
                        f"(pregame: {pre_a_pts or '—'} – {pre_h_pts or '—'})"
                    )

                try:
                    @st.cache_data(ttl=30, show_spinner=False)
                    def _live_box(gid_key):
                        return fetch_live_boxscore(gid_key)

                    h_box, a_box = _live_box(gid)
                    if not h_box.empty or not a_box.empty:
                        bc_a, bc_h = st.columns(2)
                        for ctx, bdf, tname in [(bc_h, h_box, hn), (bc_a, a_box, an)]:
                            with ctx:
                                st.markdown(f"**{tname}**")
                                if not bdf.empty:
                                    top = (bdf[bdf["min"] > 0]
                                           .sort_values("pts", ascending=False)
                                           .head(6)[["name","pts","reb","ast","min"]]
                                           .rename(columns={"name":"Player","pts":"PTS",
                                                            "reb":"REB","ast":"AST","min":"MIN"}))
                                    st.dataframe(top, hide_index=True, use_container_width=True)
                except Exception:
                    pass

        if not done_games.empty:
            st.markdown("### ✅ Final")
            for _, dg in done_games.iterrows():
                ht, at  = dg["home_tri"], dg["away_tri"]
                hs, as_ = int(dg["home_score"]), int(dg["away_score"])
                win_t   = ht if hs > as_ else at
                ser_txt = str(dg.get("series_text","") or "")
                st.markdown(
                    f"**{TEAM_NAMES.get(at,at)} {as_}  —  "
                    f"{TEAM_NAMES.get(ht,ht)} {hs}**  "
                    f"· {win_t} wins  {('  ' + ser_txt) if ser_txt else ''}"
                )

        if not pre_games.empty:
            st.markdown("### 🕐 Upcoming Today")
            for _, pg in pre_games.iterrows():
                ht, at  = pg["home_tri"], pg["away_tri"]
                st_txt  = str(pg.get("status_text","") or "")
                ser_txt = str(pg.get("series_text","") or "")

                if _wm_l is not None:
                    try:
                        res = predict_game(ht, at, _feats_live, _wm_l, _wfc_l,
                                           mode_params=_mp)
                        if res:
                            wp, hp, ap = res
                            st.markdown(
                                f"**{TEAM_NAMES.get(at,at)} @ {TEAM_NAMES.get(ht,ht)}**  "
                                f"—  {st_txt}  {ser_txt}",
                            )
                            st.markdown(_prob_bar(ht, at, wp), unsafe_allow_html=True)
                            st.caption(f"Predicted: {at} {ap} — {ht} {hp}  | Total {hp+ap:.0f}")
                            continue
                    except Exception:
                        pass
                st.markdown(f"**{TEAM_NAMES.get(at,at)} @ {TEAM_NAMES.get(ht,ht)}**  —  {st_txt}")

        if not live_games.empty:
            st.info("Live data refreshes every 30 seconds. Use ⟳ Refresh for an instant update.")


# -- tab: game predictions --

with tab_games:
    st.subheader("Game Predictions — Next 7 Days")

    today_sched, inj_report, upcoming, _ = get_live_data()
    feats = load_features()
    win_model, win_feat_cols, win_bundle = load_win_model()

    inj_impact = {}
    if not inj_report.empty:
        from src.live_data import injury_impact_score
        all_tris = set(upcoming["home_tri"].tolist() + upcoming["away_tri"].tolist()) if not upcoming.empty else set()
        for tri in all_tris:
            try:
                inj_impact[tri] = injury_impact_score(tri, feats)
            except Exception:
                inj_impact[tri] = 0.0

    if not today_sched.empty and not upcoming.empty:
        today_ids = set(today_sched["game_id"].astype(str))
        upcoming["_id"] = upcoming["game_id"].astype(str)
        upcoming = upcoming[~upcoming["_id"].isin(today_ids)].drop(columns=["_id"])

    all_games = pd.concat([
        today_sched.assign(game_date=date.today().isoformat()),
        upcoming,
    ], ignore_index=True) if not today_sched.empty else upcoming

    if all_games.empty:
        st.warning("No upcoming games found. ESPN may be in offseason.")
    elif win_model is None:
        st.warning("Win model not loaded. Run `python train_advanced.py --no-tune`.")
    else:
        all_games["game_date"] = pd.to_datetime(all_games["game_date"], errors="coerce").dt.date
        all_games = all_games[all_games["game_date"].notna()]
        all_games = all_games.drop_duplicates(subset=["game_date","home_tri","away_tri"])

        for gdate, day_games in all_games.groupby("game_date"):
            label = ("Today" if gdate == date.today()
                     else "Tomorrow" if gdate == date.today() + timedelta(days=1)
                     else gdate.strftime("%A %b %d"))
            st.markdown(f"### {label}")

            for _, g in day_games.iterrows():
                home_tri  = g["home_tri"]
                away_tri  = g["away_tri"]
                home_name = TEAM_NAMES.get(home_tri, home_tri)
                away_name = TEAM_NAMES.get(away_tri, away_tri)
                status    = g.get("status", "")
                h_inj     = inj_impact.get(home_tri, 0.0)
                a_inj     = inj_impact.get(away_tri, 0.0)

                result = predict_game(home_tri, away_tri, feats,
                                      win_model, win_feat_cols,
                                      home_inj=h_inj, away_inj=a_inj,
                                      mode_params=_mp)

                status_label = str(status) if status and str(status) not in ("Scheduled", "nan", "", "None") else ""
                with st.expander(
                    f"{away_name}  @  {home_name}" + (f"  —  {status_label}" if status_label else ""),
                    expanded=(gdate == date.today())
                ):
                    if result is None:
                        st.info("Not enough team data to predict this game.")
                        continue

                    win_prob, home_pts, away_pts = result
                    col_a, col_mid, col_h = st.columns([2, 4, 2])

                    with col_a:
                        st.markdown(f"**✈️ {away_name}**")
                        if a_inj > 0.05:
                            st.markdown(f"🏥 {a_inj*100:.0f}% min lost to injury")
                        st.metric("Predicted score", f"{away_pts}")

                    with col_h:
                        st.markdown(f"**🏠 {home_name}**")
                        if h_inj > 0.05:
                            st.markdown(f"🏥 {h_inj*100:.0f}% min lost to injury")
                        st.metric("Predicted score", f"{home_pts}")

                    with col_mid:
                        st.markdown(_prob_bar(home_tri, away_tri, win_prob), unsafe_allow_html=True)
                        total  = home_pts + away_pts
                        margin = abs(home_pts - away_pts)
                        fav    = home_tri if home_pts > away_pts else away_tri
                        st.markdown(
                            f"**Total:** {total:.0f} pts &nbsp;|&nbsp; "
                            f"**Spread:** {fav} by {margin:.1f}"
                        )
                        if not inj_report.empty:
                            out_players = inj_report[
                                (inj_report["team_tri"].isin([home_tri, away_tri])) &
                                (inj_report["status_rank"] >= 3)
                            ][["team_tri","player_name","status"]]
                            if not out_players.empty:
                                lines = "  ".join(
                                    f"{r['team_tri']} {r['player_name']} ({r['status']})"
                                    for _, r in out_players.iterrows()
                                )
                                st.caption(f"⚠️ Out/Doubtful: {lines}")

                    if str(status).lower() in ("final","halftime","in progress"):
                        hs = g.get("home_score",""); as_ = g.get("away_score","")
                        st.info(f"**Result:** {away_name} {as_} — {home_name} {hs}  *({status})*")


# -- player props helpers --

def _opp_team_context(feats: pd.DataFrame, opp_id: int) -> dict:
    odf = feats[feats["teamId"] == int(opp_id)]
    if odf.empty:
        return {}
    recent_dates = odf["game_date"].dropna().drop_duplicates().sort_values().iloc[-30:]
    odf = odf[odf["game_date"].isin(recent_dates)]
    latest = (odf[odf["min_ewma"].fillna(0) > 5]
              .sort_values("game_date").groupby("personId").last())
    if latest.empty:
        return {}
    t = latest.select_dtypes("number").mean()
    return {
        "opp_def_rating":     float(t.get("drtg_roll5",  t.get("drtg_szn",  0)) or 0),
        "opp_def_efg_roll3":  float(t.get("opp_def_efg_roll3",  0) or 0),
        "opp_def_efg_roll5":  float(t.get("opp_def_efg_roll5",  0) or 0),
        "opp_def_efg_roll10": float(t.get("opp_def_efg_roll10", 0) or 0),
        "opp_efg_roll5":      float(t.get("shot_efg_roll5",  t.get("opp_efg_roll5",  0)) or 0),
        "opp_efg_roll10":     float(t.get("shot_efg_roll10", t.get("opp_efg_roll10", 0)) or 0),
        "opp_efg_szn":        float(t.get("opp_efg_szn", 0) or 0),
        "opp_elo":            float(t.get("team_elo", t.get("opp_elo", 0)) or 0),
    }


def _ascii_name(name: str) -> str:
    # Strip diacritics so Dončić matches Doncic in injury reports
    return _ud.normalize("NFD", name.lower().strip()).encode("ascii", "ignore").decode()


def _name_match(name: str, name_set: set) -> bool:
    nl       = _ascii_name(name)
    norm_set = {_ascii_name(n) for n in name_set}
    if nl in norm_set:
        return True
    last = nl.split()[-1] if nl.split() else nl
    return any(last == n.split()[-1] for n in norm_set)


def _build_positions(live_pos: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    if live_pos.empty:
        p = PROCESSED / "player_positions.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    recent = (feats[feats["min_ewma"].fillna(0) > 5]
              .sort_values("game_date")
              .groupby("personId").last()
              .reset_index()[["personId", "playerName", "reb_ewma", "ast_ewma", "blk_ewma"]])

    rows = []
    for _, p in live_pos.iterrows():
        primary = str(p.get("primary_pos", "F") or "F").strip()
        name    = str(p.get("player_name", "") or "").strip()

        match = recent[recent["playerName"].str.lower() == name.lower()]
        if match.empty:
            last = name.split()[-1].lower() if name.split() else ""
            match = recent[recent["playerName"].str.lower().str.endswith(last)]

        if not match.empty:
            s   = match.iloc[0]
            reb = float(s.get("reb_ewma") or 0)
            ast = float(s.get("ast_ewma") or 0)
            blk = float(s.get("blk_ewma") or 0)
            pid = int(s["personId"])
        else:
            reb = ast = blk = 0.0
            pid = 0

        if primary == "G":
            pos = "G-F" if reb >= 5.5 or (reb >= 4.0 and ast < 4.0) else "G"
        elif primary == "F":
            if ast >= 5.0 or (ast >= 3.5 and reb < 5.5):
                pos = "F-G"
            elif blk >= 1.3 or (reb >= 8.5 and blk >= 0.8):
                pos = "F-C"
            else:
                pos = "F"
        elif primary == "C":
            pos = "C-F" if ast >= 3.5 else "C"
        else:
            pos = primary or "F"

        rows.append({"personId": pid, "player_name": name,
                     "team_tri": p.get("team_tri", ""),
                     "primary_pos": primary, "position": pos})

    df = pd.DataFrame(rows)
    static_p = PROCESSED / "player_positions.parquet"
    if static_p.exists():
        static = pd.read_parquet(static_p)
        covered = set(df[df["personId"] > 0]["personId"])
        extra   = static[~static["personId"].isin(covered)]
        df = pd.concat([df[df["personId"] > 0], extra], ignore_index=True)

    return df.drop_duplicates("personId")


def _infer_pos(row: pd.Series) -> str:
    pos = str(row.get("position", "") or "").strip()
    if pos and pos not in ("nan", "None"):
        return pos
    reb = float(row.get("reb_ewma") or 0)
    ast = float(row.get("ast_ewma") or 0)
    blk = float(row.get("blk_ewma") or 0)
    if reb >= 8.5 and blk >= 0.8:
        return "C"
    if ast >= 5 or (ast >= 3.5 and reb < 5):
        return "G"
    if reb >= 6:
        return "F-C"
    return "F"


def _blend_stat(row: pd.Series, stat: str) -> float:
    ewma = float(row.get(f"{stat}_ewma", 0) or 0)
    szn  = float(row.get(f"{stat}_season_avg", 0) or 0)
    if szn > 0:
        ewma_w = 0.10 if stat == "pts" else 0.35
        return ewma * ewma_w + szn * (1.0 - ewma_w)
    return ewma


_FALLBACK_SLOTS = [36, 34, 32, 29, 26, 22, 18, 14, 10, 7]


def _team_minute_slots(tdf: pd.DataFrame, n: int = 10) -> list:
    recent = tdf[tdf["min"].fillna(0) > 0].copy()
    if recent.empty:
        return _FALLBACK_SLOTS[:n]
    recent["_rank"] = (recent.groupby("gameId")["min"]
                       .rank(ascending=False, method="first").astype(int))
    dist = recent.groupby("_rank")["min"].mean()
    # +2 min empirical offset (backtest showed slots underpredict by ~3 min)
    return [float(dist.get(i + 1, _FALLBACK_SLOTS[min(i, 9)])) + 2.0 for i in range(n)]


def predict_props_for_game(home_tri: str, away_tri: str,
                           feats: pd.DataFrame,
                           inj_report: pd.DataFrame,
                           positions: pd.DataFrame,
                           home_team_pts: float = None,
                           away_team_pts: float = None,
                           mode_params: dict = None,
                           adv_stats: dict = None,
                           player_models=None,
                           player_feat_cols: list = None) -> pd.DataFrame:
    mp              = mode_params or _MODE_PARAMS["Regular Season"]
    n_rot           = mp["rotation"]
    min_thr         = mp["min_thresh"]
    adv             = adv_stats or {}
    rows            = []
    team_pts_target = {home_tri: home_team_pts, away_tri: away_team_pts}

    live_inj = _get_fresh_injuries()
    if live_inj.empty:
        live_inj = inj_report

    for tri, opp_tri, is_home_flag in [(home_tri, away_tri, 1), (away_tri, home_tri, 0)]:
        tid    = tri_to_id(tri)
        opp_id = tri_to_id(opp_tri)
        tdf    = feats[feats["teamId"] == tid]
        if tdf.empty:
            continue

        opp_ctx = _opp_team_context(feats, opp_id)

        out_set  = set()
        ques_set = set()
        if not live_inj.empty:
            ti       = live_inj[live_inj["team_tri"] == tri]
            out_set  = set(ti[ti["status_rank"] >= 4]["player_name"].str.lower())
            ques_set = set(ti[ti["status_rank"].isin([2, 3])]["player_name"].str.lower())

        recent_dates = tdf["game_date"].dropna().drop_duplicates().sort_values().iloc[-30:]
        recent_tdf   = tdf[tdf["game_date"].isin(recent_dates)]

        roster = (recent_tdf[recent_tdf["min_ewma"].fillna(0) > min_thr]
                  .sort_values("game_date")
                  .groupby("personId").last()
                  .reset_index()
                  .nlargest(n_rot, "min_ewma"))

        slots = _team_minute_slots(recent_tdf, n=n_rot)

        if not positions.empty:
            roster = roster.merge(positions, on="personId", how="left")
        if "position" not in roster.columns:
            roster["position"] = None
        roster["position"] = roster.apply(_infer_pos, axis=1)

        roster["_is_out"] = roster["playerName"].apply(lambda n: _name_match(n, out_set))
        _roster_ids = set(roster["personId"].astype(int))
        active_players = (roster[~roster["_is_out"]]
                          .sort_values("min_ewma", ascending=False)
                          .reset_index(drop=True))
        if active_players.empty:
            continue

        # Fill rotation gaps from bench when injuries thin the roster
        if len(active_players) < n_rot:
            bench_pool = (
                recent_tdf[recent_tdf["min_ewma"].fillna(0) > 0]
                .sort_values("game_date")
                .groupby("personId").last()
                .reset_index()
            )
            if not positions.empty:
                bench_pool = bench_pool.merge(positions, on="personId", how="left")
            if "position" not in bench_pool.columns:
                bench_pool["position"] = None
            bench_pool["position"] = bench_pool.apply(_infer_pos, axis=1)
            bench_pool = (bench_pool[
                (~bench_pool["personId"].astype(int).isin(_roster_ids)) &
                (~bench_pool["playerName"].apply(lambda n: _name_match(n, out_set)))
            ].sort_values("min_ewma", ascending=False).reset_index(drop=True))

            need = n_rot - len(active_players)
            if not bench_pool.empty and need > 0:
                pos_counts: dict = {}
                for p in active_players["position"]:
                    main = str(p).split("-")[0] if pd.notna(p) and str(p) != "nan" else "F"
                    pos_counts[main] = pos_counts.get(main, 0) + 1
                tgt_g = max(2, round(n_rot * 0.35))
                tgt_c = max(1, round(n_rot * 0.20))
                pos_targets = {"G": tgt_g, "F": n_rot - tgt_g - tgt_c, "C": tgt_c}

                used_fill: set = set()
                fillers = []
                for _ in range(need):
                    best_pos = max(pos_targets, key=lambda p: pos_targets[p] - pos_counts.get(p, 0))
                    cands = bench_pool[
                        (~bench_pool["personId"].astype(int).isin(used_fill)) &
                        (bench_pool["position"].fillna("").str.startswith(best_pos))
                    ]
                    if cands.empty:
                        cands = bench_pool[~bench_pool["personId"].astype(int).isin(used_fill)]
                    if cands.empty:
                        break
                    pick = cands.iloc[0]
                    fillers.append(pick)
                    used_fill.add(int(pick["personId"]))
                    mp_str = str(pick.get("position") or "F").split("-")[0]
                    pos_counts[mp_str] = pos_counts.get(mp_str, 0) + 1

                if fillers:
                    fill_df = pd.DataFrame(fillers)
                    active_players = (pd.concat([active_players, fill_df], ignore_index=True)
                                      .sort_values("min_ewma", ascending=False)
                                      .reset_index(drop=True))

        _LEAGUE_DEF = 113.0
        opp_drtg    = float(opp_ctx.get("opp_def_rating", _LEAGUE_DEF) or _LEAGUE_DEF)
        def_factor  = max(opp_drtg / _LEAGUE_DEF, 0.80)
        home_factor = 1.025 if is_home_flag else 1.0

        last_game_date = recent_tdf["game_date"].max()
        rest_days_v = 2
        if pd.notna(last_game_date):
            try:
                rest_days_v = max((date.today() - pd.to_datetime(last_game_date).date()).days, 1)
            except Exception:
                pass

        opp_tdf     = feats[feats["teamId"] == opp_id]
        opp_elo_v   = 1500.0
        opp_drtg_ml = opp_drtg
        if not opp_tdf.empty:
            opp_latest  = opp_tdf.sort_values("game_date").iloc[-1]
            opp_elo_v   = float(opp_latest.get("opp_elo", 1500) or 1500)
            opp_drtg_ml = float(opp_latest.get("drtg_roll5", opp_drtg) or opp_drtg)

        team_rows = []
        for idx, (_, pr) in enumerate(active_players.iterrows()):
            name         = pr["playerName"]
            pos          = pr["position"]
            pid          = int(pr.get("personId", 0))
            is_q         = _name_match(name, ques_set)
            assigned_min = slots[idx] if idx < len(slots) else max(4.0, slots[-1] * 0.6)
            base_min     = max(float(pr.get("min_season_avg", pr.get("min_ewma", 20)) or 20), 1.0)
            min_ratio    = assigned_min / base_min

            e_pts = _blend_stat(pr, "pts") * min_ratio * def_factor * home_factor
            e_reb = (_blend_stat(pr, "reb") + 0.18) * min_ratio
            e_ast = _blend_stat(pr, "ast") * min_ratio * home_factor
            e_stl = _blend_stat(pr, "stl") * min_ratio
            e_blk = _blend_stat(pr, "blk") * min_ratio
            e_tov = _blend_stat(pr, "tov") * min_ratio

            ml_pts = e_pts
            ml_reb = e_reb
            ml_ast = e_ast
            if player_models and player_feat_cols:
                try:
                    row_dict = {c: float(pr.get(c, 0) or 0) for c in player_feat_cols}
                    row_dict["is_home"]       = float(is_home_flag)
                    row_dict["rest_days"]      = float(rest_days_v)
                    row_dict["opp_def_rating"] = opp_drtg_ml
                    row_dict["opp_elo"]        = opp_elo_v
                    row_dict["elo_diff"]       = row_dict.get("team_elo", 1500) - opp_elo_v
                    X          = pd.DataFrame([row_dict])[player_feat_cols].fillna(0)
                    ml_min_raw = max(float(player_models["min"]["model"].predict(X)[0]), 1.0)
                    scale_ml   = assigned_min / ml_min_raw
                    ml_pts     = float(player_models["pts"]["model"].predict(X)[0]) * scale_ml \
                                 * def_factor * home_factor
                    ml_reb     = float(player_models["reb"]["model"].predict(X)[0]) * scale_ml
                    ml_ast     = float(player_models["ast"]["model"].predict(X)[0]) * scale_ml \
                                 * home_factor
                except Exception:
                    pass

            pts_raw = 0.60 * e_pts + 0.40 * ml_pts
            reb     = round(max(0, 0.60 * e_reb + 0.40 * ml_reb), 1)
            ast     = round(max(0, 0.60 * e_ast + 0.40 * ml_ast), 1)

            p_adv      = adv.get(pid, {})
            usg        = float(p_adv.get("usg_pct", _LEAGUE_AVG_USG) or _LEAGUE_AVG_USG)
            ts         = float(p_adv.get("ts_pct",  _LEAGUE_AVG_TS)  or _LEAGUE_AVG_TS)
            pts_raw    = pts_raw * (usg / _LEAGUE_AVG_USG) ** 0.35 * (ts / _LEAGUE_AVG_TS) ** 0.25

            team_rows.append({
                "Player":   name,
                "Pos":      pos,
                "Team":     tri,
                "Opp":      opp_tri,
                "H/A":      "Home" if is_home_flag else "Away",
                "Status":   "⚠️ Q/D" if is_q else "",
                "MIN":      round(assigned_min, 1),
                "_pts_raw": max(pts_raw, 0),
                "REB":      reb,
                "AST":      ast,
                "STL":      round(max(e_stl, 0), 1),
                "BLK":      round(max(e_blk, 0), 1),
                "TOV":      round(max(e_tov, 0), 1),
            })

        raw_pts_sum = sum(r["_pts_raw"] for r in team_rows)
        target      = team_pts_target.get(tri)
        scale       = (target / raw_pts_sum) if (target and raw_pts_sum > 0) else 1.0
        for r in team_rows:
            r["PTS"] = round(r.pop("_pts_raw") * scale, 1)

        rows.extend(team_rows)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# -- tab: player props --

with tab_props:
    st.subheader("Player Stat Predictions")
    st.caption(
        "EWMA form + tonight's opponent defensive context. "
        "Out/IR players excluded. ⚠️ = Questionable/Doubtful."
    )

    feats_p = load_features()
    today_s, inj_p, upcoming_p, live_pos_p = get_live_data()
    pos_p     = _build_positions(live_pos_p, feats_p)
    all_sched = pd.concat([today_s, upcoming_p], ignore_index=True) if not today_s.empty else upcoming_p

    game_opts = []
    if not all_sched.empty:
        all_sched["game_date"] = pd.to_datetime(
            all_sched.get("game_date", pd.NaT), errors="coerce").dt.date
        all_sched = all_sched[all_sched["game_date"].notna()]
        for _, g in all_sched.drop_duplicates(subset=["home_tri","away_tri","game_date"]).iterrows():
            d = g["game_date"]
            try:
                label = ("Today " if d == date.today() else
                         "Tomorrow " if d == date.today()+timedelta(1) else
                         d.strftime("%b %d "))
            except Exception:
                continue
            game_opts.append(f"{label}| {g['away_tri']} @ {g['home_tri']}")
    game_opts.append("Search player manually")

    sel = st.selectbox("Select game", game_opts)

    if sel == "Search player manually":
        search = st.text_input("Player name", placeholder="e.g. Jayson Tatum")
        if search.strip():
            hits = feats_p[feats_p["playerName"].str.contains(search.strip(), case=False, na=False)]
            if hits.empty:
                st.info("Player not found.")
            else:
                pid = hits.sort_values("game_date").iloc[-1]["personId"]
                pr  = hits[hits["personId"] == pid].sort_values("game_date").iloc[-1]
                st.markdown(f"**{pr['playerName']}** — {pr.get('teamTricode','')}")
                st.dataframe(pd.DataFrame([{
                    "Stat": s, "EWMA (projection)": round(float(pr.get(f"{s}_ewma", 0) or 0), 1)
                } for s in ["pts","reb","ast","stl","blk","tov","min"]]),
                use_container_width=True, hide_index=True)
    else:
        parts = sel.split("|")[-1].strip().split(" @ ")
        if len(parts) == 2:
            away_tri, home_tri = parts[0].strip(), parts[1].strip()
            with st.spinner("Predicting…"):
                wm_p, wfc_p, _ = load_win_model()
                game_result = None
                if wm_p is not None:
                    try:
                        game_result = predict_game(home_tri, away_tri, feats_p, wm_p, wfc_p,
                                                   mode_params=_mp)
                    except Exception:
                        pass
                home_pts_target = game_result[1] if game_result else None
                away_pts_target = game_result[2] if game_result else None

                _pm, _pfc = load_player_models()
                _adv      = get_advanced_stats()
                props_df  = predict_props_for_game(
                    home_tri, away_tri, feats_p, inj_p, pos_p,
                    home_team_pts=home_pts_target,
                    away_team_pts=away_pts_target,
                    mode_params=_mp,
                    adv_stats=_adv,
                    player_models=_pm,
                    player_feat_cols=_pfc,
                )

            if props_df.empty:
                st.warning("Could not generate predictions for this game.")
            else:
                stat_cols = [c for c in ["PTS","REB","AST","STL","BLK","TOV","MIN"]
                             if c in props_df.columns]
                disp_cols = ["Player","Pos","H/A","Status"] + stat_cols
                disp      = props_df[[c for c in disp_cols if c in props_df.columns]]

                home_df = disp[props_df["Team"] == home_tri].sort_values("PTS", ascending=False)
                away_df = disp[props_df["Team"] == away_tri].sort_values("PTS", ascending=False)

                if game_result:
                    _, hp, ap = game_result
                    st.markdown(
                        f"Predicted totals: **{away_tri} {ap}** — **{home_tri} {hp}** "
                        f"| Total {hp+ap:.0f}"
                    )

                col_h, col_a = st.columns(2)

                def _add_totals(df):
                    num_cols = [c for c in ["PTS","REB","AST","STL","BLK","TOV","MIN"]
                                if c in df.columns]
                    tot = {c: round(df[c].sum(), 1) for c in num_cols}
                    tot["Player"] = "TOTAL"
                    for c in df.columns:
                        if c not in tot:
                            tot[c] = ""
                    return pd.concat([df, pd.DataFrame([tot])], ignore_index=True)

                with col_h:
                    st.markdown(f"**🏠 {TEAM_NAMES.get(home_tri, home_tri)}**")
                    st.dataframe(_add_totals(home_df.reset_index(drop=True)),
                                 use_container_width=True, hide_index=True)
                with col_a:
                    st.markdown(f"**✈️ {TEAM_NAMES.get(away_tri, away_tri)}**")
                    st.dataframe(_add_totals(away_df.reset_index(drop=True)),
                                 use_container_width=True, hide_index=True)


# -- tab: team stats --

with tab_team:
    feats_t = load_features()
    st.subheader(f"{selected_name} — Current-Form EWMA Stats")

    DISPLAY = {
        "pts_ewma": "PTS", "reb_ewma": "REB", "ast_ewma": "AST",
        "stl_ewma": "STL", "blk_ewma": "BLK", "tov_ewma": "TOV",
        "min_ewma": "MIN", "ts_pct_ewma": "TS%", "usg_pct_ewma": "USG%",
    }

    team_feats = feats_t[feats_t["teamId"] == selected_id]
    latest_season = int(team_feats["season"].max()) if not team_feats.empty else 2026
    tfs = team_feats[team_feats["season"] == latest_season]
    if tfs.empty:
        tfs = team_feats

    if tfs.empty:
        st.warning("No data for this team.")
    else:
        latest = (tfs[tfs["min_ewma"].fillna(0) > 3]
                  .sort_values("game_date").groupby("personId").last().reset_index())
        show  = ["playerName"] + [c for c in DISPLAY if c in latest.columns]
        table = latest[show].rename(columns={**DISPLAY, "playerName": "Player"})
        table = table.sort_values("MIN", ascending=False).reset_index(drop=True)
        for col in DISPLAY.values():
            if col in table.columns:
                table[col] = table[col].round(1)
        st.dataframe(table, use_container_width=True, hide_index=True)

    eff_cols = {
        "ortg_szn": "Off Rtg", "drtg_szn": "Def Rtg",
        "net_rtg_szn": "Net Rtg", "pace_proxy_szn": "Pace",
    }
    avail_eff = [c for c in eff_cols if c in feats_t.columns]
    if avail_eff and not team_feats.empty:
        st.markdown("---")
        st.subheader("Season efficiency")
        row = team_feats.dropna(subset=avail_eff[:1]).sort_values("game_date").tail(1)
        if not row.empty:
            cols = st.columns(len(avail_eff))
            for i, ec in enumerate(avail_eff):
                cols[i].metric(eff_cols[ec], f"{row[ec].iloc[0]:.1f}")

    st.markdown("---")
    st.subheader("Search any player")
    search = st.text_input("Player name", placeholder="e.g. Jayson Tatum", key="ps_t")
    if search.strip():
        hits = feats_t[feats_t["playerName"].str.contains(search.strip(), case=False, na=False)]
        if hits.empty:
            st.info("Player not found.")
        else:
            lh   = hits.sort_values("game_date").groupby("personId").last().reset_index()
            show = ["playerName","teamTricode","season"] + [c for c in DISPLAY if c in lh.columns]
            tbl  = lh[show].rename(columns={**DISPLAY, "playerName":"Player","teamTricode":"Team","season":"Season"})
            for col in DISPLAY.values():
                if col in tbl.columns:
                    tbl[col] = tbl[col].round(1)
            st.dataframe(tbl.sort_values("MIN", ascending=False),
                         use_container_width=True, hide_index=True)


# -- tab: lineup optimizer --

with tab_lineup:
    feats_l = load_features()
    _, _, _, _lu_live_pos = get_live_data()
    positions = _build_positions(_lu_live_pos, feats_l)
    _lu_inj   = _get_fresh_injuries()

    st.subheader("Lineup Optimizer")
    st.caption(
        "Score = PTS + 1.2·REB + 1.5·AST + 2·STL + 2·BLK − 1.5·TOV + 20·TS% + 0.3·MIN  "
        "(EWMA-weighted · ILP positional constraints · Out/IR excluded)"
    )

    from src.optimizer import composite_score, optimize_lineup, _infer_position, _active_roster, MIN_MINUTES_THRESHOLD

    _POS_ORDER = {"G": 0, "G-F": 1, "F-G": 1, "F": 2, "F-C": 3, "C-F": 3, "C": 4}

    def get_lineup(team_id):
        th = feats_l[feats_l["teamId"] == int(team_id)]
        if th.empty:
            return pd.DataFrame()
        roster = _active_roster(th, n_recent=20)
        if not positions.empty:
            roster = roster.merge(positions, on="personId", how="left")
        roster["position"] = roster.apply(_infer_position, axis=1)
        roster["score"]    = composite_score(roster)

        tri = _tri_id_map().get(int(team_id), "")
        out_names = set()
        if not _lu_inj.empty and tri:
            ti = _lu_inj[_lu_inj["team_tri"] == tri]
            out_names = set(ti[ti["status_rank"] >= 4]["player_name"].str.lower())
        if out_names:
            roster = roster[~roster["playerName"].apply(lambda n: _name_match(n, out_names))]

        active = roster[roster["min_ewma"].fillna(0) >= MIN_MINUTES_THRESHOLD]
        if len(active) < 5:
            active = roster.nlargest(5, "score")
        lu = optimize_lineup(active)
        if not lu.empty and "position" in lu.columns:
            lu["_pos_ord"] = lu["position"].map(_POS_ORDER).fillna(2)
            lu = lu.sort_values("_pos_ord").drop(columns=["_pos_ord"])
        return lu

    STAT_COLS = {
        "pts_ewma":"PTS","reb_ewma":"REB","ast_ewma":"AST",
        "stl_ewma":"STL","blk_ewma":"BLK","tov_ewma":"TOV",
        "min_ewma":"MIN","ts_pct_ewma":"TS%",
    }

    st.markdown(f"#### {selected_name}")
    lineup = get_lineup(selected_id)
    if lineup.empty:
        st.warning("Not enough data to generate lineup.")
    else:
        team_latest = (
            feats_l[(feats_l["teamId"] == selected_id) & (feats_l["min_ewma"].fillna(0) > 3)]
            .sort_values("game_date").groupby("personId").last().reset_index()
        )
        merged = lineup.merge(
            team_latest[["personId"] + [c for c in STAT_COLS if c in team_latest.columns]],
            on="personId", how="left"
        ).rename(columns={**STAT_COLS, "playerName":"Player","position":"Pos","score":"Score"})
        for col in STAT_COLS.values():
            if col in merged.columns:
                merged[col] = merged[col].round(1)
        merged["Score"] = merged["Score"].round(1)
        show_cols = ["Player","Pos","Score"] + [c for c in STAT_COLS.values() if c in merged.columns]
        merged = merged[[c for c in show_cols if c in merged.columns]].reset_index(drop=True)
        merged.index += 1
        st.dataframe(merged, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Avg PTS", f"{merged['PTS'].mean():.1f}" if "PTS" in merged.columns else "—")
        m2.metric("Avg REB", f"{merged['REB'].mean():.1f}" if "REB" in merged.columns else "—")
        m3.metric("Avg AST", f"{merged['AST'].mean():.1f}" if "AST" in merged.columns else "—")

    st.markdown("---")
    st.subheader("Compare two lineups")
    c1, c2 = st.columns(2)
    with c1:
        h_idx = st.selectbox("Home team", range(len(tricodes)),
                             format_func=lambda i: team_label[i],
                             index=tricodes.index(selected_tri), key="lu_h")
        h_tri = tricodes[h_idx]; h_id = tri_to_id(h_tri)
    with c2:
        a_idx = st.selectbox("Away team", range(len(tricodes)),
                             format_func=lambda i: team_label[i],
                             index=min(tricodes.index("LAL"), len(tricodes)-1) if "LAL" in tricodes else 1,
                             key="lu_a")
        a_tri = tricodes[a_idx]; a_id = tri_to_id(a_tri)

    if h_id != a_id:
        lc, rc = st.columns(2)
        for ctx, tid, tname in [(lc, h_id, TEAM_NAMES.get(h_tri,h_tri)),
                                 (rc, a_id, TEAM_NAMES.get(a_tri,a_tri))]:
            with ctx:
                st.markdown(f"**{tname}**")
                lu = get_lineup(tid)
                if lu.empty:
                    st.info("No data.")
                else:
                    lu_show = lu[["playerName","position","score"]].rename(
                        columns={"playerName":"Player","position":"Pos","score":"Score"})
                    lu_show["Score"] = lu_show["Score"].round(1)
                    lu_show.index = range(1, len(lu_show)+1)
                    st.dataframe(lu_show, use_container_width=True)

        wm, wfc, _ = load_win_model()
        if wm is not None:
            feats_lu = load_features()
            res = predict_game(h_tri, a_tri, feats_lu, wm, wfc, mode_params=_mp)
            if res:
                wp, hp, ap = res
                st.markdown(_prob_bar(h_tri, a_tri, wp), unsafe_allow_html=True)
                st.markdown(
                    f"**Predicted:** {a_tri} **{ap}** — {h_tri} **{hp}**  "
                    f"| Total {hp+ap:.0f} | {h_tri if hp>ap else a_tri} by {abs(hp-ap):.1f}"
                )


# -- tab: injuries & risk --

with tab_injuries:
    _, inj_data, _, _ = get_live_data()
    feats_i = load_features()

    st.subheader("Live Injury Report")
    st.caption(f"Source: ESPN · auto-refreshes every 30 min · {date.today()}")

    if inj_data.empty:
        st.warning("Could not fetch injury data. Check internet connection.")
    else:
        c1f, c2f, _ = st.columns(3)
        show_team_f = c1f.selectbox("Team", ["All teams"] + sorted(tricodes), key="inj_t")
        sev_map     = {"All statuses": 0, "Questionable+": 2, "Doubtful+": 3, "Out only": 4}
        min_sev     = c2f.selectbox("Min severity", list(sev_map.keys()), key="inj_s")
        min_r       = sev_map[min_sev]

        filt = inj_data.copy()
        if show_team_f != "All teams":
            filt = filt[filt["team_tri"] == show_team_f]
        filt = filt[filt["status_rank"] >= min_r]
        filt = filt.sort_values(["status_rank","team_tri"], ascending=[False,True])

        if filt.empty:
            st.info("No players match these filters.")
        else:
            DISP = {"team_tri":"Team","player_name":"Player","position":"Pos",
                    "status":"Status","injury_detail":"Detail"}
            disp = filt[[c for c in DISP if c in filt.columns]].rename(columns=DISP)

            def _color(row):
                color = STATUS_COLORS.get(row.get("Status",""), "#999")
                return [f"color:{color}" if c == "Status" else "" for c in row.index]

            st.dataframe(disp.style.apply(_color, axis=1),
                         use_container_width=True, hide_index=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Out / IR",        int((filt["status_rank"] >= 4).sum()))
            m2.metric("Doubtful",         int((filt["status_rank"] == 3).sum()))
            m3.metric("Questionable/GTD", int((filt["status_rank"] == 2).sum()))

    st.markdown("---")
    st.subheader("Workload Overuse Risk")
    st.caption("Players whose recent minutes load is significantly above their season average. Risk > 0.6 = flagged.")

    report_path = PROCESSED / "workload_report_2026.parquet"
    if not report_path.exists():
        st.info("No workload report found. Run `python train_advanced.py --no-tune`.")
    else:
        report = pd.read_parquet(report_path)

        team_only = st.checkbox(f"Show {selected_tri} only", value=True, key="wl_t")
        show_rep  = report[report["teamTricode"] == selected_tri] if team_only else report

        if show_rep.empty and team_only:
            st.info(f"No high-risk players flagged for {selected_tri}.")
            show_rep = report

        RISK_COLS = {
            "playerName":"Player","teamTricode":"Team",
            "min_ewma":"Avg MIN","min_load_7d":"7d Load",
            "min_vs_season_avg":"vs Season Avg","btb_5g_count":"B2B/5g",
            "workload_index":"Risk Score","high_workload_flag":"⚠️",
        }
        disp_r = show_rep[[c for c in RISK_COLS if c in show_rep.columns]].rename(columns=RISK_COLS)
        for col in ["Avg MIN","7d Load","vs Season Avg","Risk Score"]:
            if col in disp_r.columns:
                disp_r[col] = disp_r[col].round(1)

        def _risk_row(row):
            flag = row.get("⚠️", 0)
            return ["background-color:#ffe0e0" if flag == 1 else "" for _ in row]

        st.dataframe(disp_r.style.apply(_risk_row, axis=1),
                     use_container_width=True, hide_index=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("High-risk players", int(show_rep.get("high_workload_flag", pd.Series(0)).sum()))
        if "workload_index" in show_rep.columns:
            r2.metric("Avg risk score", f"{show_rep['workload_index'].mean():.2f}")
            r3.metric("Max risk score", f"{show_rep['workload_index'].max():.2f}")
