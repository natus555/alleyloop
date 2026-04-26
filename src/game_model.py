"""
Phase 7 — Game Outcome Prediction

Features : Team EWMA aggregates (home & away) + rest differential + active-player count
           Pre-game safe — EWMA in features.parquet uses shift(1), no leakage
Models   : LogisticRegression (baseline) → XGBoostClassifier
Target   : home_win  (1 = home team won)
Split    : 2020-2023 train | 2024 test (temporal)
Metrics  : Accuracy, ROC-AUC, Brier score, calibration plot
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

FEATURES_DIR  = Path("data/features")
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR   = Path("figures")

# Min-minutes threshold — excludes DNP / deep bench / injured players
ACTIVE_MIN_EWMA = 5.0


# ── Feature construction ───────────────────────────────────────────────────────

def build_game_features() -> pd.DataFrame:
    """
    Build one row per game with home-team and away-team EWMA aggregates,
    differentials, rest context, and active-roster count (injury proxy).

    Aggregation:
        For each game G, take each player's row in features.parquet where
        gameId == G.  Because features.py applied shift(1) before all rolling/EWMA
        transforms, those values represent stats *prior to* game G — no leakage.
    """
    feats = pd.read_parquet(FEATURES_DIR / "features.parquet")
    sched = pd.read_parquet(PROCESSED_DIR / "game_schedule.parquet")
    box   = pd.read_parquet(PROCESSED_DIR / "box_scores.parquet")

    ewma_cols = sorted([c for c in feats.columns if c.endswith("_ewma")])

    feats["teamId"] = pd.to_numeric(feats["teamId"], errors="coerce").astype("int64")
    box["teamId"]   = pd.to_numeric(box["teamId"],   errors="coerce").astype("int64")
    sched["home_team_id"] = sched["home_team_id"].astype("int64")
    sched["away_team_id"] = sched["away_team_id"].astype("int64")

    # ── Target: did the home team win? ────────────────────────────────────────
    team_pts = box.groupby(["gameId", "teamId"])["pts"].sum().reset_index()
    game_pts = team_pts.merge(
        team_pts.rename(columns={"teamId": "opp_id", "pts": "opp_pts"}), on="gameId"
    )
    game_pts = game_pts[game_pts["teamId"] != game_pts["opp_id"]]
    game_pts["won"] = (game_pts["pts"] > game_pts["opp_pts"]).astype(int)

    home_wins = (
        game_pts.merge(sched[["gameId", "home_team_id"]], on="gameId")
                .query("teamId == home_team_id")[["gameId", "won"]]
                .rename(columns={"won": "home_win"})
    )

    # ── Extra team-level columns (efficiency + shot quality) ─────────────────
    # These are uniform per player within a team-game so mean() == actual value
    EFF_COLS = [c for c in feats.columns if any(
        c.startswith(p) for p in ("ortg_", "drtg_", "net_rtg_", "pace_proxy_", "team_efg_", "opp_efg_")
    )]
    SHOT_TEAM_COLS = [c for c in feats.columns if any(
        c.startswith(p) for p in ("shot_efg_roll", "shot_3pt_rate_roll", "shot_rim_rate_roll",
                                   "opp_def_efg_roll")
    )]

    # ── Team-level EWMA aggregates ────────────────────────────────────────────
    active = feats[feats["min_ewma"].fillna(0) > ACTIVE_MIN_EWMA].copy()
    agg_cols = ewma_cols + ["rest_days"] + EFF_COLS + SHOT_TEAM_COLS

    # Only keep columns that actually exist in active
    agg_cols = [c for c in agg_cols if c in active.columns]

    team_agg = (
        active.groupby(["gameId", "teamId"])[agg_cols]
              .mean()
              .reset_index()
    )
    # active-roster size as injury proxy (fewer active players → injury pressure)
    roster_size = (
        active.groupby(["gameId", "teamId"])["personId"]
              .count()
              .reset_index(name="active_players")
    )
    team_agg = team_agg.merge(roster_size, on=["gameId", "teamId"])

    # attach game metadata for home/away split
    team_agg = team_agg.merge(
        sched[["gameId", "game_date", "season", "home_team_id", "away_team_id"]],
        on="gameId", how="inner"
    )
    team_agg["is_home"] = (team_agg["teamId"] == team_agg["home_team_id"]).astype(int)

    # only keep rows that belong to one of the two scheduled teams
    team_agg = team_agg[
        (team_agg["teamId"] == team_agg["home_team_id"]) |
        (team_agg["teamId"] == team_agg["away_team_id"])
    ]

    extra_cols = EFF_COLS + SHOT_TEAM_COLS
    extra_cols = [c for c in extra_cols if c in team_agg.columns]

    home_agg = (
        team_agg[team_agg["is_home"] == 1]
        .rename(columns={c: f"home_{c}" for c in agg_cols + ["active_players"]})
        [["gameId", "game_date", "season"]
         + [f"home_{c}" for c in agg_cols]
         + ["home_active_players"]]
    )
    away_agg = (
        team_agg[team_agg["is_home"] == 0]
        .rename(columns={c: f"away_{c}" for c in agg_cols + ["active_players"]})
        [["gameId"]
         + [f"away_{c}" for c in agg_cols]
         + ["away_active_players"]]
    )

    game_df = home_agg.merge(away_agg, on="gameId", how="inner")

    # ── H2H features — game-level, already home-oriented ─────────────────────
    H2H_COLS = [c for c in feats.columns if c.startswith("h2h_")]
    if H2H_COLS:
        h2h_src = (
            feats[feats["is_home"] == 1][["gameId"] + H2H_COLS]
            .dropna(subset=H2H_COLS[:1])
            .drop_duplicates("gameId")
        )
        if h2h_src.empty:
            h2h_src = (
                feats[["gameId"] + H2H_COLS]
                .drop_duplicates("gameId")
            )
        game_df = game_df.merge(h2h_src, on="gameId", how="left")

    # ── Differential features (home − away) ───────────────────────────────────
    for col in ewma_cols + extra_cols:
        hc, ac = f"home_{col}", f"away_{col}"
        if hc in game_df.columns and ac in game_df.columns:
            game_df[f"diff_{col}"] = game_df[hc] - game_df[ac]
    game_df["rest_diff"]    = game_df["home_rest_days"]     - game_df["away_rest_days"]
    game_df["roster_diff"]  = game_df["home_active_players"] - game_df["away_active_players"]

    # ── Attach target ─────────────────────────────────────────────────────────
    game_df = game_df.merge(home_wins, on="gameId", how="left")
    game_df = game_df.dropna(subset=["home_win"])
    game_df["home_win"] = game_df["home_win"].astype(int)

    out = PROCESSED_DIR / "game_features.parquet"
    game_df.to_parquet(out, index=False)
    print(f"Saved: {out}  ({len(game_df):,} games, {game_df.shape[1]} cols)")
    print(f"  Home win rate: {game_df['home_win'].mean()*100:.1f}%")
    return game_df


def _feature_cols(game_df: pd.DataFrame) -> list[str]:
    diff_cols  = [c for c in game_df.columns if c.startswith("diff_")]
    home_ewma  = [c for c in game_df.columns if c.startswith("home_") and c.endswith("_ewma")]
    away_ewma  = [c for c in game_df.columns if c.startswith("away_") and c.endswith("_ewma")]
    h2h        = [c for c in game_df.columns if c.startswith("h2h_")]
    ctx        = ["rest_diff", "roster_diff",
                  "home_rest_days", "away_rest_days",
                  "home_active_players", "away_active_players"]
    ctx = [c for c in ctx if c in game_df.columns]
    return diff_cols + home_ewma + away_ewma + h2h + ctx


# ── Training & evaluation ──────────────────────────────────────────────────────

def train_and_evaluate(game_df: pd.DataFrame) -> dict:
    """
    Train LR and XGBoost on 2020-2023, evaluate on 2024.
    Saves calibration plot to figures/win_prob_calibration.png.
    Returns dict of {model_name: {model, proba, acc, auc, brier}}.
    """
    FIGURES_DIR.mkdir(exist_ok=True)

    feat_cols = _feature_cols(game_df)
    train = game_df[game_df["season"] <= 2024]
    test  = game_df[game_df["season"] >= 2025]

    X_train = train[feat_cols].fillna(0)
    X_test  = test[feat_cols].fillna(0)
    y_train = train["home_win"]
    y_test  = test["home_win"]

    print(f"Train: {len(X_train):,} games  |  Test: {len(X_test):,} games  |  Features: {len(feat_cols)}")
    print(f"Train home win rate: {y_train.mean()*100:.1f}%  |  Test: {y_test.mean()*100:.1f}%")

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    results = {}

    # ── Logistic Regression baseline ──────────────────────────────────────────
    print("\n── LOGISTIC REGRESSION ──────────────────────────")
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_proba = lr.predict_proba(X_test_sc)[:, 1]
    lr_preds = (lr_proba >= 0.5).astype(int)

    lr_acc   = accuracy_score(y_test, lr_preds)
    lr_auc   = roc_auc_score(y_test, lr_proba)
    lr_brier = brier_score_loss(y_test, lr_proba)
    print(f"  Accuracy : {lr_acc*100:.2f}%")
    print(f"  ROC-AUC  : {lr_auc:.4f}")
    print(f"  Brier    : {lr_brier:.4f}")
    results["lr"] = {
        "model": lr, "scaler": scaler,
        "proba": lr_proba,
        "acc": lr_acc, "auc": lr_auc, "brier": lr_brier,
    }

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\n── XGBOOST ───────────────────────────────────────")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, verbosity=0,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_preds = (xgb_proba >= 0.5).astype(int)

    xgb_acc   = accuracy_score(y_test, xgb_preds)
    xgb_auc   = roc_auc_score(y_test, xgb_proba)
    xgb_brier = brier_score_loss(y_test, xgb_proba)
    print(f"  Accuracy : {xgb_acc*100:.2f}%")
    print(f"  ROC-AUC  : {xgb_auc:.4f}")
    print(f"  Brier    : {xgb_brier:.4f}")
    results["xgb"] = {
        "model": xgb,
        "proba": xgb_proba,
        "acc": xgb_acc, "auc": xgb_auc, "brier": xgb_brier,
    }

    # ── Calibration plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {"lr": ("LogReg", "steelblue"), "xgb": ("XGBoost", "darkorange")}
    for key, (label, color) in palette.items():
        prob_true, prob_pred = calibration_curve(
            y_test, results[key]["proba"], n_bins=10, strategy="uniform"
        )
        ax.plot(
            prob_pred, prob_true, marker="o",
            label=f"{label}  AUC={results[key]['auc']:.3f}  Acc={results[key]['acc']*100:.1f}%",
            color=color,
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability (home win)")
    ax.set_ylabel("Actual fraction of home wins")
    ax.set_title("Win Probability Calibration — 2024 Season Test Set")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_fig = FIGURES_DIR / "win_prob_calibration.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nCalibration plot saved: {out_fig}")

    # ── Results table ─────────────────────────────────────────────────────────
    records = [
        {"model": "LogReg",  **{k: results["lr"][k]  for k in ["acc", "auc", "brier"]}},
        {"model": "XGBoost", **{k: results["xgb"][k] for k in ["acc", "auc", "brier"]}},
    ]
    res_df = pd.DataFrame(records).round(4)
    res_df["acc"] = (res_df["acc"] * 100).round(2)
    res_df.to_parquet(PROCESSED_DIR / "game_model_results.parquet", index=False)
    print("\nResults:")
    print(res_df.to_string(index=False))

    return results


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(results: dict):
    print("\n" + "=" * 55)
    print("GAME MODEL SUMMARY — 2024 Season Test Set")
    print("=" * 55)
    for name, label in [("lr", "LogReg "), ("xgb", "XGBoost")]:
        r = results[name]
        print(f"  {label}  Accuracy={r['acc']*100:.2f}%  "
              f"AUC={r['auc']:.4f}  Brier={r['brier']:.4f}")
    best = "xgb" if results["xgb"]["auc"] >= results["lr"]["auc"] else "lr"
    print(f"\n  Best model (AUC): {'XGBoost' if best=='xgb' else 'LogReg'}")


# ── Entry point ────────────────────────────────────────────────────────────────

def run():
    print("Building game-level features...")
    game_df = build_game_features()

    print("\nTraining models...")
    results = train_and_evaluate(game_df)

    print_summary(results)
    return results


if __name__ == "__main__":
    run()
