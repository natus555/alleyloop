"""
Advanced Ensemble Models — AlleyLoop

Targets
-------
1. Player stats   : pts, reb, ast, stl, blk, tov, min  (per next game)
2. Game scores    : home_pts, away_pts, total_pts
3. Win probability: binary home_win
4. Workload index : injury-risk composite score

Models
------
LightGBM + CatBoost + XGBoost  →  stacked with Ridge meta-learner
Hyperparameters tuned with Optuna (50 trials per model, time-series CV)

Split
-----
Train : seasons 2020-2024  (≤ 2024)
Test  : seasons 2025-2026  (≥ 2025)
CV    : 5-fold time-series expanding window (no future leakage)
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error,
                             accuracy_score, roc_auc_score, brier_score_loss)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import catboost as cb
from xgboost import XGBRegressor, XGBClassifier

FEATURES_DIR  = Path("data/features")
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR   = Path("figures")

PLAYER_TARGETS = ["pts", "reb", "ast", "stl", "blk", "tov", "min"]
GAME_TARGETS   = ["home_pts", "away_pts"]

N_OPTUNA_TRIALS = 50
N_CV_SPLITS     = 5
RANDOM_SEED     = 42


# ── Feature engineering additions ─────────────────────────────────────────────

def add_elo_ratings(feats: pd.DataFrame,
                    sched: pd.DataFrame,
                    box:   pd.DataFrame) -> pd.DataFrame:
    """
    Compute game-by-game Elo ratings per team and attach to feature rows.
    Returns feats with two new columns: home_elo, away_elo (at game time).
    """
    K = 20.0
    elo: dict[int, float] = {}

    team_pts = (
        box.groupby(["gameId", "teamId"])["pts"].sum().reset_index()
    )
    team_pts["teamId"] = pd.to_numeric(team_pts["teamId"], errors="coerce").astype("int64")

    games = (
        sched.sort_values("game_date")[
            ["gameId", "game_date", "home_team_id", "away_team_id"]
        ].drop_duplicates("gameId")
    )

    elo_records = []
    for _, row in games.iterrows():
        gid, htid, atid = row["gameId"], int(row["home_team_id"]), int(row["away_team_id"])
        h_elo = elo.get(htid, 1500.0)
        a_elo = elo.get(atid, 1500.0)
        elo_records.append({"gameId": gid, "home_elo": h_elo, "away_elo": a_elo})

        # Update after game
        h_pts_row = team_pts[(team_pts["gameId"] == gid) & (team_pts["teamId"] == htid)]
        a_pts_row = team_pts[(team_pts["gameId"] == gid) & (team_pts["teamId"] == atid)]
        if h_pts_row.empty or a_pts_row.empty:
            continue
        h_score = float(h_pts_row["pts"].iloc[0])
        a_score = float(a_pts_row["pts"].iloc[0])
        h_actual = 1.0 if h_score > a_score else (0.5 if h_score == a_score else 0.0)
        h_expected = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo) / 400.0))
        elo[htid] = h_elo + K * (h_actual - h_expected)
        elo[atid] = a_elo + K * ((1 - h_actual) - (1 - h_expected))

    elo_df = pd.DataFrame(elo_records)

    # Player home_elo and away_elo based on their team role
    feats2 = feats.merge(elo_df, on="gameId", how="left")
    feats2["teamId_i64"] = pd.to_numeric(feats2["teamId"], errors="coerce").astype("float64")
    ht = sched.set_index("gameId")["home_team_id"].astype("float64")
    feats2["_home_tid"] = feats2["gameId"].map(ht)
    is_home = (feats2["teamId_i64"] == feats2["_home_tid"]).fillna(False)
    feats2["team_elo"]     = np.where(is_home, feats2["home_elo"], feats2["away_elo"])
    feats2["opp_elo"]      = np.where(is_home, feats2["away_elo"], feats2["home_elo"])
    feats2["elo_diff"]     = feats2["team_elo"] - feats2["opp_elo"]
    feats2.drop(columns=["teamId_i64", "_home_tid"], inplace=True)
    return feats2


def add_workload_features(feats: pd.DataFrame) -> pd.DataFrame:
    """
    Add injury-risk and workload features:
      min_load_7d        : rolling 7-game minutes sum (physical load accumulation)
      min_vs_season_avg  : current rolling-5 min vs season avg (deviation)
      btb_5g_count       : back-to-backs in last 5 games
      workload_index     : composite 0-1 score (higher = more at risk)
    """
    df = feats.sort_values(["personId", "game_date"]).copy()
    grp = df.groupby("personId")

    # 7-game cumulative minutes (shifted — excludes current game)
    df["min_load_7d"] = grp["min"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).sum()
    ).round(1)

    # deviation of recent minutes from season average
    df["min_vs_season_avg"] = (
        df["min_roll5"].fillna(0) - df["min_season_avg"].fillna(0)
    ).round(2)

    # back-to-back count in last 5 games
    df["btb_5g_count"] = grp["is_back_to_back"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    ).fillna(0).astype("int8")

    # composite workload index (normalised, higher = more load / more risk)
    # components: load excess (min_vs_season_avg > 0), b2b frequency, raw load
    df["workload_index"] = (
        np.clip(df["min_vs_season_avg"] / 10.0, -1, 1) * 0.3     # recent overload
        + df["btb_5g_count"].fillna(0) / 5.0 * 0.3                # b2b density
        + np.clip(df["min_load_7d"].fillna(0) / 240.0, 0, 1) * 0.4  # 7d volume
    ).round(3)

    # flag: high risk if workload_index > 0.6 and games_on_current_team < 15
    df["high_workload_flag"] = (
        (df["workload_index"] > 0.6) |
        (df["games_on_current_team"].fillna(999) < 10)   # recently traded
    ).astype("int8")

    return df


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_player_dataset() -> tuple[pd.DataFrame, list[str]]:
    """
    Load features.parquet, add Elo + workload features.
    Returns (df, feature_cols).
    """
    feats = pd.read_parquet(FEATURES_DIR / "features.parquet")
    sched = pd.read_parquet(PROCESSED_DIR / "game_schedule.parquet")
    box   = pd.read_parquet(PROCESSED_DIR / "box_scores.parquet")

    feats["teamId"] = pd.to_numeric(feats["teamId"], errors="coerce").astype("int64")
    box["teamId"]   = pd.to_numeric(box["teamId"],   errors="coerce").astype("int64")

    print("  Adding Elo ratings…")
    feats = add_elo_ratings(feats, sched, box)

    print("  Adding workload features…")
    feats = add_workload_features(feats)

    feature_cols = sorted([
        c for c in feats.columns
        if c.endswith(("_roll3", "_roll5", "_roll10", "_ewma",
                       "_season_avg", "_stint_ewma"))
        or c in ["is_home", "rest_days", "is_back_to_back", "opp_def_rating",
                 "games_on_current_team", "team_elo", "opp_elo", "elo_diff",
                 "min_load_7d", "min_vs_season_avg", "btb_5g_count", "workload_index"]
    ])

    print(f"  Dataset: {len(feats):,} rows  |  {len(feature_cols)} features")
    return feats, feature_cols


# ── Optuna hyperparameter search ───────────────────────────────────────────────

def _tune_lgbm(X_tr, y_tr, n_trials=N_OPTUNA_TRIALS, task="regression"):
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 400, 1200),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 63, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 0.9),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq":      trial.suggest_int("bagging_freq", 1, 10),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),
            "n_jobs": -1, "random_state": RANDOM_SEED, "verbose": -1,
        }
        scores = []
        for tr_idx, va_idx in tscv.split(X_tr):
            Xtr, Xva = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
            ytr, yva = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
            if task == "regression":
                m = lgb.LGBMRegressor(**params)
                m.fit(Xtr, ytr,
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                  lgb.log_evaluation(-1)],
                      eval_set=[(Xva, yva)])
                scores.append(mean_absolute_error(yva, m.predict(Xva)))
            else:
                m = lgb.LGBMClassifier(**params)
                m.fit(Xtr, ytr,
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                  lgb.log_evaluation(-1)],
                      eval_set=[(Xva, yva)])
                scores.append(-roc_auc_score(yva, m.predict_proba(Xva)[:, 1]))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _tune_catboost(X_tr, y_tr, n_trials=N_OPTUNA_TRIALS, task="regression"):
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    def objective(trial):
        params = {
            "iterations":    trial.suggest_int("iterations", 400, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "depth":         trial.suggest_int("depth", 5, 10),
            "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_seed": RANDOM_SEED, "verbose": False,
        }
        scores = []
        for tr_idx, va_idx in tscv.split(X_tr):
            Xtr, Xva = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
            ytr, yva = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
            if task == "regression":
                m = cb.CatBoostRegressor(**params)
                m.fit(Xtr, ytr, eval_set=(Xva, yva),
                      early_stopping_rounds=30, verbose=False)
                scores.append(mean_absolute_error(yva, m.predict(Xva)))
            else:
                m = cb.CatBoostClassifier(**params)
                m.fit(Xtr, ytr, eval_set=(Xva, yva),
                      early_stopping_rounds=30, verbose=False)
                scores.append(-roc_auc_score(yva, m.predict_proba(Xva)[:, 1]))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ── Stacked ensemble ───────────────────────────────────────────────────────────

class StackedRegressor:
    """LightGBM + CatBoost + XGBoost → Ridge meta-learner."""

    def __init__(self, lgbm_params, cb_params, xgb_params, n_splits=N_CV_SPLITS):
        self.lgbm_params = lgbm_params
        self.cb_params   = cb_params
        self.xgb_params  = xgb_params
        self.n_splits    = n_splits
        self.meta        = Ridge(alpha=1.0)
        self._l0_models  = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        oof  = np.zeros((len(X), 3))

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

            lgbm_p = {**self.lgbm_params, "n_jobs": -1, "random_state": RANDOM_SEED, "verbose": -1}
            cb_p   = {**self.cb_params, "random_seed": RANDOM_SEED, "verbose": False}
            xgb_p  = {**self.xgb_params, "n_jobs": -1, "random_state": RANDOM_SEED, "verbosity": 0}

            m0 = lgb.LGBMRegressor(**lgbm_p)
            m0.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                   callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])

            m1 = cb.CatBoostRegressor(**cb_p)
            m1.fit(Xtr, ytr, eval_set=(Xva, yva),
                   early_stopping_rounds=30, verbose=False)

            m2 = XGBRegressor(**xgb_p)
            m2.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

            oof[va_idx, 0] = m0.predict(Xva)
            oof[va_idx, 1] = m1.predict(Xva)
            oof[va_idx, 2] = m2.predict(Xva)

        self.meta.fit(oof, y)

        # Retrain base models on full data
        lgbm_p = {**self.lgbm_params, "n_jobs": -1, "random_state": RANDOM_SEED, "verbose": -1}
        cb_p   = {**self.cb_params, "random_seed": RANDOM_SEED, "verbose": False}
        xgb_p  = {**self.xgb_params, "n_jobs": -1, "random_state": RANDOM_SEED, "verbosity": 0}

        self.m0 = lgb.LGBMRegressor(**lgbm_p).fit(X, y)
        self.m1 = cb.CatBoostRegressor(**cb_p).fit(X, y, verbose=False)
        self.m2 = XGBRegressor(**xgb_p).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = np.column_stack([
            self.m0.predict(X),
            self.m1.predict(X),
            self.m2.predict(X),
        ])
        return self.meta.predict(p)


class StackedClassifier:
    """LightGBM + CatBoost + XGBoost → Logistic meta-learner."""

    def __init__(self, lgbm_params, cb_params, xgb_params, n_splits=N_CV_SPLITS):
        self.lgbm_params = lgbm_params
        self.cb_params   = cb_params
        self.xgb_params  = xgb_params
        self.n_splits    = n_splits
        from sklearn.linear_model import LogisticRegression
        self.meta = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        oof  = np.zeros((len(X), 3))

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

            lgbm_p = {**self.lgbm_params, "n_jobs": -1, "random_state": RANDOM_SEED, "verbose": -1}
            cb_p   = {**self.cb_params, "random_seed": RANDOM_SEED, "verbose": False}
            xgb_p  = {**self.xgb_params, "n_jobs": -1, "random_state": RANDOM_SEED, "verbosity": 0}

            m0 = lgb.LGBMClassifier(**lgbm_p)
            m0.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                   callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])

            m1 = cb.CatBoostClassifier(**cb_p)
            m1.fit(Xtr, ytr, eval_set=(Xva, yva),
                   early_stopping_rounds=30, verbose=False)

            m2 = XGBClassifier(**xgb_p)
            m2.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

            oof[va_idx, 0] = m0.predict_proba(Xva)[:, 1]
            oof[va_idx, 1] = m1.predict_proba(Xva)[:, 1]
            oof[va_idx, 2] = m2.predict_proba(Xva)[:, 1]

        self.meta.fit(oof, y)

        lgbm_p = {**self.lgbm_params, "n_jobs": -1, "random_state": RANDOM_SEED, "verbose": -1}
        cb_p   = {**self.cb_params, "random_seed": RANDOM_SEED, "verbose": False}
        xgb_p  = {**self.xgb_params, "n_jobs": -1, "random_state": RANDOM_SEED, "verbosity": 0}

        self.m0 = lgb.LGBMClassifier(**lgbm_p).fit(X, y)
        self.m1 = cb.CatBoostClassifier(**cb_p).fit(X, y, verbose=False)
        self.m2 = XGBClassifier(**xgb_p).fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.column_stack([
            self.m0.predict_proba(X)[:, 1],
            self.m1.predict_proba(X)[:, 1],
            self.m2.predict_proba(X)[:, 1],
        ])
        meta_p = self.meta.predict_proba(p)[:, 1]
        return np.column_stack([1 - meta_p, meta_p])


# ── Player stat models ─────────────────────────────────────────────────────────

def train_player_models(feats: pd.DataFrame,
                        feature_cols: list[str],
                        tune: bool = True) -> dict:
    """
    Train stacked ensemble for each player stat target.
    Returns dict: {target: {'model': StackedRegressor, 'mae': float, 'rmse': float}}
    """
    FIGURES_DIR.mkdir(exist_ok=True)

    clean = feats[feature_cols + PLAYER_TARGETS + ["season"]].copy()
    clean[feature_cols] = clean[feature_cols].fillna(0)
    clean = clean.dropna(subset=PLAYER_TARGETS)

    train_df = clean[clean["season"] <= 2025].drop(columns="season")
    test_df  = clean[clean["season"] >= 2026].drop(columns="season")
    X_train, y_train = train_df[feature_cols], train_df[PLAYER_TARGETS]
    X_test,  y_test  = test_df[feature_cols],  test_df[PLAYER_TARGETS]

    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}  |  Features: {len(feature_cols)}")

    results = {}
    all_records = []

    for target in PLAYER_TARGETS:
        print(f"\n── {target.upper()} ────────────────────────────────")
        ytr = y_train[target]
        yte = y_test[target]
        baseline_mae = mean_absolute_error(yte, np.full(len(yte), ytr.mean()))

        if tune:
            print("  Tuning LightGBM…", flush=True)
            lgbm_p = _tune_lgbm(X_train, ytr, n_trials=N_OPTUNA_TRIALS)
            print("  Tuning CatBoost…", flush=True)
            cb_p   = _tune_catboost(X_train, ytr, n_trials=N_OPTUNA_TRIALS)
        else:
            lgbm_p = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
                      "min_child_samples": 20, "feature_fraction": 0.7,
                      "bagging_fraction": 0.8, "bagging_freq": 5,
                      "reg_alpha": 0.1, "reg_lambda": 0.5}
            cb_p   = {"iterations": 400, "learning_rate": 0.05, "depth": 7,
                      "l2_leaf_reg": 3.0, "random_strength": 1.0, "bagging_temperature": 0.5}

        xgb_p = {"n_estimators": 350, "max_depth": 6, "learning_rate": 0.05,
                  "subsample": 0.8, "colsample_bytree": 0.75,
                  "min_child_weight": 5, "reg_alpha": 0.05, "reg_lambda": 1.5}

        n_splits = 3 if not tune else N_CV_SPLITS
        print("  Training stacked ensemble…", flush=True)
        model = StackedRegressor(lgbm_p, cb_p, xgb_p, n_splits=n_splits)
        model.fit(X_train, ytr)

        preds = model.predict(X_test)
        mae   = mean_absolute_error(yte, preds)
        rmse  = root_mean_squared_error(yte, preds)
        impv  = (baseline_mae - mae) / baseline_mae * 100

        print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  "
              f"baseline={baseline_mae:.3f}  ↓{impv:.1f}%")

        results[target] = {"model": model, "mae": mae, "rmse": rmse,
                           "baseline_mae": baseline_mae, "improvement": impv}
        all_records.append({"target": target, "MAE": round(mae, 3),
                             "RMSE": round(rmse, 3),
                             "baseline_MAE": round(baseline_mae, 3),
                             "improvement_%": round(impv, 1)})

    pd.DataFrame(all_records).to_parquet(
        PROCESSED_DIR / "advanced_player_results.parquet", index=False)
    return results


# ── Game score models ──────────────────────────────────────────────────────────

def _build_game_score_dataset() -> tuple[pd.DataFrame, list[str]]:
    """
    Build game-level dataset for score prediction.
    Features: team-level EWMA + Elo + pace + rest
    Targets: home_pts, away_pts
    """
    feats = pd.read_parquet(FEATURES_DIR / "features.parquet")
    sched = pd.read_parquet(PROCESSED_DIR / "game_schedule.parquet")
    box   = pd.read_parquet(PROCESSED_DIR / "box_scores.parquet")

    feats["teamId"] = pd.to_numeric(feats["teamId"], errors="coerce").astype("int64")
    box["teamId"]   = pd.to_numeric(box["teamId"],   errors="coerce").astype("int64")
    sched["home_team_id"] = sched["home_team_id"].astype("int64")
    sched["away_team_id"] = sched["away_team_id"].astype("int64")

    # Elo ratings at game time
    feats_elo = add_elo_ratings(feats, sched, box)

    ewma_cols = sorted([c for c in feats_elo.columns if c.endswith("_ewma")])

    # Extra team-level features (efficiency metrics & shot quality)
    _EFF_PREFIXES = ("ortg_", "drtg_", "net_rtg_", "pace_proxy_", "team_efg_", "opp_efg_")
    _SHOT_PREFIXES = ("shot_efg_roll", "shot_3pt_rate_roll", "shot_rim_rate_roll",
                      "opp_def_efg_roll")
    extra_cols = [c for c in feats_elo.columns
                  if any(c.startswith(p) for p in _EFF_PREFIXES + _SHOT_PREFIXES)]

    active = feats_elo[feats_elo["min_ewma"].fillna(0) > 5]
    agg_cols = ewma_cols + extra_cols + ["rest_days", "team_elo"]

    team_agg = (
        active.groupby(["gameId", "teamId"])[agg_cols].mean().reset_index()
    )
    roster_cnt = active.groupby(["gameId", "teamId"])["personId"].count().reset_index(name="active_players")
    team_agg = team_agg.merge(roster_cnt, on=["gameId", "teamId"])

    team_agg = team_agg.merge(
        sched[["gameId", "game_date", "season", "home_team_id", "away_team_id"]],
        on="gameId", how="inner"
    )
    team_agg["is_home"] = (team_agg["teamId"] == team_agg["home_team_id"]).astype(int)
    team_agg = team_agg[
        (team_agg["teamId"] == team_agg["home_team_id"]) |
        (team_agg["teamId"] == team_agg["away_team_id"])
    ]

    home_agg = team_agg[team_agg["is_home"] == 1].rename(
        columns={c: f"home_{c}" for c in agg_cols + ["active_players"]}
    )[["gameId", "game_date", "season"] + [f"home_{c}" for c in agg_cols] + ["home_active_players"]]

    away_agg = team_agg[team_agg["is_home"] == 0].rename(
        columns={c: f"away_{c}" for c in agg_cols + ["active_players"]}
    )[["gameId"] + [f"away_{c}" for c in agg_cols] + ["away_active_players"]]

    game_df = home_agg.merge(away_agg, on="gameId", how="inner")

    # Differentials for all aggregated metrics
    all_diff_cols = ewma_cols + extra_cols
    for col in all_diff_cols:
        hc, ac = f"home_{col}", f"away_{col}"
        if hc in game_df.columns and ac in game_df.columns:
            game_df[f"diff_{col}"] = game_df[hc] - game_df[ac]
    game_df["rest_diff"]   = game_df["home_rest_days"]   - game_df["away_rest_days"]
    game_df["roster_diff"] = game_df["home_active_players"] - game_df["away_active_players"]
    game_df["elo_diff"]    = game_df["home_team_elo"]    - game_df["away_team_elo"]

    # H2H features (already game-level, home-team oriented)
    h2h_cols = [c for c in feats_elo.columns if c.startswith("h2h_")]
    if h2h_cols:
        h2h_src = (
            feats_elo[feats_elo.get("is_home", pd.Series(0, index=feats_elo.index)) == 1]
            [["gameId"] + h2h_cols]
            .drop_duplicates("gameId")
        )
        if h2h_src.empty or "gameId" not in h2h_src.columns:
            h2h_src = feats_elo[["gameId"] + h2h_cols].drop_duplicates("gameId")
        game_df = game_df.merge(h2h_src, on="gameId", how="left")

    # Actual scores
    team_pts = box.groupby(["gameId", "teamId"])["pts"].sum().reset_index()
    home_pts = team_pts.merge(sched[["gameId", "home_team_id"]], on="gameId").query("teamId == home_team_id")[["gameId", "pts"]].rename(columns={"pts": "home_pts"})
    away_pts = team_pts.merge(sched[["gameId", "away_team_id"]], on="gameId").query("teamId == away_team_id")[["gameId", "pts"]].rename(columns={"pts": "away_pts"})
    game_df = game_df.merge(home_pts, on="gameId").merge(away_pts, on="gameId")
    game_df["total_pts"] = game_df["home_pts"] + game_df["away_pts"]
    game_df["home_win"]  = (game_df["home_pts"] > game_df["away_pts"]).astype(int)
    game_df = game_df.dropna(subset=["home_pts", "away_pts"])

    h2h_feat_cols = [c for c in (h2h_cols if h2h_cols else []) if c in game_df.columns]
    feat_cols = (
        [f"diff_{c}" for c in all_diff_cols if f"diff_{c}" in game_df.columns]
        + [f"home_{c}" for c in ewma_cols + extra_cols if f"home_{c}" in game_df.columns]
        + [f"away_{c}" for c in ewma_cols + extra_cols if f"away_{c}" in game_df.columns]
        + h2h_feat_cols
        + ["rest_diff", "roster_diff", "elo_diff",
           "home_rest_days", "away_rest_days",
           "home_team_elo", "away_team_elo",
           "home_active_players", "away_active_players"]
    )
    feat_cols = [c for c in feat_cols if c in game_df.columns]
    feat_cols = list(dict.fromkeys(feat_cols))  # deduplicate preserving order
    return game_df, feat_cols


def train_game_models(tune: bool = True) -> dict:
    """
    Train stacked ensembles for home_pts, away_pts, and win probability.
    """
    FIGURES_DIR.mkdir(exist_ok=True)
    print("\nBuilding game-score dataset…")
    game_df, feat_cols = _build_game_score_dataset()

    train = game_df[game_df["season"] <= 2025]
    test  = game_df[game_df["season"] >= 2026]
    X_tr = train[feat_cols].fillna(0)
    X_te = test[feat_cols].fillna(0)

    print(f"  Train: {len(train):,} games  |  Test: {len(test):,} games  |  Features: {len(feat_cols)}")

    results = {}

    # ── Score prediction (home_pts, away_pts) ─────────────────────────────────
    for target in ["home_pts", "away_pts"]:
        print(f"\n── {target} ───────────────────────────────")
        ytr = train[target]; yte = test[target]
        baseline = mean_absolute_error(yte, np.full(len(yte), ytr.mean()))

        if tune:
            lgbm_p = _tune_lgbm(X_tr, ytr, n_trials=N_OPTUNA_TRIALS)
            cb_p   = _tune_catboost(X_tr, ytr, n_trials=N_OPTUNA_TRIALS)
        else:
            lgbm_p = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
                      "min_child_samples": 20, "feature_fraction": 0.7,
                      "bagging_fraction": 0.8, "bagging_freq": 5,
                      "reg_alpha": 0.1, "reg_lambda": 0.5}
            cb_p   = {"iterations": 400, "learning_rate": 0.05, "depth": 7,
                      "l2_leaf_reg": 3.0, "random_strength": 1.0, "bagging_temperature": 0.5}

        xgb_p = {"n_estimators": 350, "max_depth": 6, "learning_rate": 0.05,
                  "subsample": 0.8, "colsample_bytree": 0.75,
                  "min_child_weight": 5, "reg_alpha": 0.05, "reg_lambda": 1.5}

        _n_splits = 3 if not tune else N_CV_SPLITS
        model = StackedRegressor(lgbm_p, cb_p, xgb_p, n_splits=_n_splits)
        model.fit(X_tr, ytr)
        preds = model.predict(X_te)
        mae  = mean_absolute_error(yte, preds)
        rmse = root_mean_squared_error(yte, preds)
        impv = (baseline - mae) / baseline * 100
        print(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  baseline={baseline:.2f}  ↓{impv:.1f}%")
        results[target] = {"model": model, "mae": mae, "rmse": rmse}

    # ── Win probability ────────────────────────────────────────────────────────
    print("\n── Win Probability ───────────────────────────")
    ytr = train["home_win"]; yte = test["home_win"]

    if tune:
        lgbm_p_c = _tune_lgbm(X_tr, ytr, n_trials=N_OPTUNA_TRIALS, task="classifier")
        cb_p_c   = _tune_catboost(X_tr, ytr, n_trials=N_OPTUNA_TRIALS, task="classifier")
    else:
        lgbm_p_c = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
                    "min_child_samples": 20, "feature_fraction": 0.7,
                    "bagging_fraction": 0.8, "bagging_freq": 5,
                    "reg_alpha": 0.1, "reg_lambda": 0.5}
        cb_p_c   = {"iterations": 400, "learning_rate": 0.05, "depth": 7,
                    "l2_leaf_reg": 3.0, "random_strength": 1.0, "bagging_temperature": 0.5}

    xgb_p_c = {"n_estimators": 350, "max_depth": 5, "learning_rate": 0.05,
               "subsample": 0.8, "colsample_bytree": 0.75,
               "min_child_weight": 5, "reg_alpha": 0.05, "reg_lambda": 1.5}

    _n_splits_c = 3 if not tune else N_CV_SPLITS
    win_model = StackedClassifier(lgbm_p_c, cb_p_c, xgb_p_c, n_splits=_n_splits_c)
    win_model.fit(X_tr, ytr)
    proba = win_model.predict_proba(X_te)[:, 1]
    preds_b = (proba >= 0.5).astype(int)
    acc   = accuracy_score(yte, preds_b)
    auc   = roc_auc_score(yte, proba)
    brier = brier_score_loss(yte, proba)
    print(f"  Accuracy={acc*100:.2f}%  AUC={auc:.4f}  Brier={brier:.4f}")
    results["win"] = {"model": win_model, "feat_cols": feat_cols,
                      "acc": acc, "auc": auc, "brier": brier}

    # ── Calibration plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(yte, proba, n_bins=10, strategy="uniform")
    ax.plot(prob_pred, prob_true, "o-", color="steelblue",
            label=f"Stacked Ensemble  AUC={auc:.3f}  Acc={acc*100:.1f}%")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Predicted home-win probability")
    ax.set_ylabel("Actual fraction of home wins")
    ax.set_title("Win Probability Calibration (2025-2026 holdout)")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "win_prob_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save results
    records = [
        {"target": "home_pts", "MAE": round(results["home_pts"]["mae"], 2),
         "RMSE": round(results["home_pts"]["rmse"], 2)},
        {"target": "away_pts", "MAE": round(results["away_pts"]["mae"], 2),
         "RMSE": round(results["away_pts"]["rmse"], 2)},
        {"target": "home_win", "Accuracy": round(acc * 100, 2),
         "AUC": round(auc, 4), "Brier": round(brier, 4)},
    ]
    pd.DataFrame(records).to_parquet(PROCESSED_DIR / "advanced_game_results.parquet", index=False)
    results["feat_cols"] = feat_cols
    results["game_df"]   = game_df
    return results


# ── Workload report ────────────────────────────────────────────────────────────

def workload_report(feats: pd.DataFrame, season: int = 2026,
                    top_n: int = 25) -> pd.DataFrame:
    """
    Generate injury-risk / workload report for the most recent season.
    Returns top_n highest-risk players with actionable flags.
    """
    feats_w = add_workload_features(feats)
    recent  = feats_w[feats_w["season"] == season]
    if recent.empty:
        recent = feats_w[feats_w["season"] == feats_w["season"].max()]

    latest = (
        recent[recent["min_ewma"].fillna(0) > 5]
        .sort_values("game_date")
        .groupby("personId")
        .last()
        .reset_index()
    )

    cols = ["playerName", "teamTricode", "min_ewma", "min_load_7d",
            "min_vs_season_avg", "btb_5g_count", "workload_index",
            "high_workload_flag", "games_on_current_team"]
    cols = [c for c in cols if c in latest.columns]
    report = (
        latest[cols]
        .sort_values("workload_index", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    report.index += 1

    out = PROCESSED_DIR / f"workload_report_{season}.parquet"
    report.to_parquet(out, index=False)
    return report


# ── Full run ───────────────────────────────────────────────────────────────────

def run(tune: bool = True):
    print("=" * 62)
    print("AlleyLoop Advanced Models — LightGBM + CatBoost + XGBoost Stack")
    print("=" * 62)

    print("\n[1/3] Building player dataset…")
    feats, feat_cols = build_player_dataset()

    print("\n[2/3] Training player stat models…")
    player_results = train_player_models(feats, feat_cols, tune=tune)

    print("\n[3/3] Training game score + win probability models…")
    game_results = train_game_models(tune=tune)

    print("\n── WORKLOAD REPORT (latest season) ───────────────────")
    report = workload_report(feats)
    print(report[["playerName", "teamTricode", "min_ewma",
                  "workload_index", "high_workload_flag"]].to_string())

    # ── Save models ───────────────────────────────────────────────────────────
    models_dir = PROCESSED_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    win_bundle = {
        "model":     game_results["win"]["model"],
        "feat_cols": game_results["win"]["feat_cols"],
        "acc":       game_results["win"]["acc"],
        "auc":       game_results["win"]["auc"],
        "brier":     game_results["win"]["brier"],
    }
    score_bundle = {
        "home_pts":     game_results["home_pts"]["model"],
        "away_pts":     game_results["away_pts"]["model"],
        "feat_cols":    game_results["feat_cols"],
        "home_pts_mae": game_results["home_pts"]["mae"],
        "away_pts_mae": game_results["away_pts"]["mae"],
    }
    joblib.dump(win_bundle,    models_dir / "win_model.joblib",    compress=3)
    joblib.dump(score_bundle,  models_dir / "score_model.joblib",  compress=3)
    joblib.dump(player_results, models_dir / "player_models.joblib", compress=3)
    joblib.dump(feat_cols,     models_dir / "player_feat_cols.joblib")
    print(f"\nModels saved to {models_dir}/")

    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print("\nPlayer stat prediction (test: 2025-2026):")
    for t in PLAYER_TARGETS:
        r = player_results[t]
        print(f"  {t:<5} MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  ↓{r['improvement']:.1f}% vs baseline")
    print(f"\nGame scores (MAE):")
    print(f"  home_pts  {game_results['home_pts']['mae']:.2f}")
    print(f"  away_pts  {game_results['away_pts']['mae']:.2f}")
    w = game_results["win"]
    print(f"\nWin probability: Acc={w['acc']*100:.2f}%  AUC={w['auc']:.4f}  Brier={w['brier']:.4f}")

    return player_results, game_results


if __name__ == "__main__":
    import sys
    tune = "--no-tune" not in sys.argv
    if not tune:
        print("Running without hyperparameter tuning (fast mode)")
    run(tune=tune)
