"""
Phase 5 — Player Performance Prediction

Targets  : pts, reb, ast, min  (next-game stats)
Features : rolling/EWMA/season-avg + trade-aware stint EWMA + context flags
Split    : 2020-2024 train  |  2025-2026 test  (temporal, no shuffle)
Models   : RandomForestRegressor, XGBRegressor (tuned)
Metrics  : MAE, RMSE per target per model
SHAP     : TreeExplainer summary plots saved to figures/
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

FEATURES_DIR  = Path("data/features")
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR   = Path("figures")

TARGETS = ["pts", "reb", "ast", "min"]

FEATURE_COLS = None   # populated in build_datasets()

RF_PARAMS = dict(
    n_estimators=400, max_depth=12,
    min_samples_leaf=3, max_features=0.6,
    n_jobs=-1, random_state=42,
)
XGB_PARAMS = dict(
    n_estimators=600, max_depth=7, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.75, colsample_bylevel=0.75,
    min_child_weight=5, reg_alpha=0.05, reg_lambda=1.5,
    n_jobs=-1, random_state=42, verbosity=0,
)


# ── Data ─────────────────────────────────────────────────────────────────────

def build_datasets():
    """Load features, drop NaN rows, return train/test splits."""
    global FEATURE_COLS
    df = pd.read_parquet(FEATURES_DIR / "features.parquet")

    FEATURE_COLS = [
        c for c in df.columns
        if c.endswith(("_roll5", "_roll10", "_ewma", "_season_avg", "_stint_ewma"))
        or c in ["is_home", "rest_days", "is_back_to_back", "opp_def_rating",
                 "games_on_current_team"]
    ]

    needed = FEATURE_COLS + TARGETS + ["season"]
    clean = df[needed].dropna(subset=TARGETS).reset_index(drop=True)
    # Fill NaN features with 0 (e.g. first-game stint values)
    clean[FEATURE_COLS] = clean[FEATURE_COLS].fillna(0)

    train = clean[clean["season"] <= 2024].drop(columns="season")
    test  = clean[clean["season"] >= 2025].drop(columns="season")

    X_train, X_test = train[FEATURE_COLS], test[FEATURE_COLS]
    y_train, y_test = train[TARGETS],      test[TARGETS]

    print(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows  |  Features: {len(FEATURE_COLS)}")
    print(f"  Training seasons ≤ 2024  |  Test seasons ≥ 2025")
    return X_train, X_test, y_train, y_test


# ── Training & Evaluation ─────────────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Train RF and XGBoost for each target; return MAE/RMSE table."""
    FIGURES_DIR.mkdir(exist_ok=True)

    models   = {"rf": RandomForestRegressor(**RF_PARAMS),
                "xgb": XGBRegressor(**XGB_PARAMS)}
    records  = []
    trained  = {}   # (model_name, target) → fitted model

    for model_name, base_model in models.items():
        print(f"\n── {model_name.upper()} ──────────────────────────────")
        for target in TARGETS:
            import copy
            m = copy.deepcopy(base_model)
            m.fit(X_train, y_train[target])
            preds = m.predict(X_test)

            mae  = mean_absolute_error(y_test[target], preds)
            rmse = root_mean_squared_error(y_test[target], preds)
            baseline_mae = mean_absolute_error(
                y_test[target],
                np.full(len(y_test), y_train[target].mean())
            )
            improvement = (baseline_mae - mae) / baseline_mae * 100

            print(f"  {target:<4}  MAE={mae:.3f}  RMSE={rmse:.3f}  "
                  f"  baseline_MAE={baseline_mae:.3f}  improvement={improvement:.1f}%")

            records.append({
                "model": model_name, "target": target,
                "MAE": round(mae, 3), "RMSE": round(rmse, 3),
                "baseline_MAE": round(baseline_mae, 3),
                "improvement_%": round(improvement, 1),
            })
            trained[(model_name, target)] = m

    results = pd.DataFrame(records)
    results.to_parquet(PROCESSED_DIR / "predictor_results.parquet", index=False)
    return results, trained


# ── SHAP ─────────────────────────────────────────────────────────────────────

def plot_shap(trained: dict, X_test: pd.DataFrame, results: pd.DataFrame):
    """
    For each target, pick the better model (lower MAE) and generate a
    SHAP beeswarm summary plot. Saves to figures/shap_{target}.png.
    """
    shap.initjs()
    sample = X_test.sample(min(2000, len(X_test)), random_state=42)

    for target in TARGETS:
        # Pick best model for this target
        target_rows = results[results["target"] == target]
        best_model_name = target_rows.loc[target_rows["MAE"].idxmin(), "model"]
        model = trained[(best_model_name, target)]

        print(f"  SHAP for {target} ({best_model_name.upper()})...", flush=True)
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        # For RF, shap_values may be a list (one per output) — take index 0
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values, sample,
            feature_names=FEATURE_COLS,
            show=False, max_display=20,
            plot_type="dot",
        )
        plt.title(f"SHAP Feature Importance — {target.upper()} ({best_model_name.upper()})",
                  fontsize=13, pad=12)
        plt.tight_layout()
        out = FIGURES_DIR / f"shap_{target}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {out}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results: pd.DataFrame):
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY — MAE / RMSE vs Baseline")
    print("=" * 65)
    pivot = results.pivot(index="target", columns="model",
                          values=["MAE", "RMSE", "improvement_%"])
    print(pivot.to_string())
    print()
    print("Best model per target (by MAE):")
    for target in TARGETS:
        sub = results[results["target"] == target]
        best = sub.loc[sub["MAE"].idxmin()]
        print(f"  {target:<4}  {best['model'].upper():<4}  "
              f"MAE={best['MAE']}  RMSE={best['RMSE']}  "
              f"↓{best['improvement_%']}% vs baseline")


# ── Entry point ───────────────────────────────────────────────────────────────

def run():
    print("Loading features...")
    X_train, X_test, y_train, y_test = build_datasets()

    print("\nTraining models...")
    results, trained = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\nGenerating SHAP plots...")
    plot_shap(trained, X_test, results)

    print_summary(results)
    return results, trained


if __name__ == "__main__":
    run()
