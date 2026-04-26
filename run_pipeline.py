"""
AlleyLoop — End-to-End Pipeline Runner
Raw CSV → processed parquets → features → models → win probability

Usage
-----
    python run_pipeline.py                  # full run
    python run_pipeline.py --skip-raw       # skip Phase 1-2 (parquets already built)
    python run_pipeline.py --skip-features  # also skip Phase 3 feature engineering
    python run_pipeline.py --skip-models    # skip predictor + game model training
"""

import sys
import time
from pathlib import Path


def _step(name: str) -> float:
    print(f"\n{'='*62}")
    print(f"  {name}")
    print(f"{'='*62}")
    return time.time()


def _done(t0: float):
    print(f"  ✓ Completed in {time.time() - t0:.1f}s")


def main():
    skip_raw      = "--skip-raw"      in sys.argv
    skip_features = "--skip-features" in sys.argv or skip_raw and "--skip-features" in sys.argv
    skip_models   = "--skip-models"   in sys.argv

    # ── Phase 1-2 : raw CSV → processed parquets ──────────────────────────────
    if not skip_raw:
        t = _step("Phase 1 — Unify PBP CSVs → season parquets")
        from src.pipeline import run as _pbp_run
        _pbp_run()
        _done(t)

        t = _step("Phase 2a — Aggregate box scores")
        from src.pipeline import build_box_scores, clean_box_scores, build_lineup_stats
        build_box_scores()
        clean_box_scores()
        build_lineup_stats()
        _done(t)

        t = _step("Phase 2b — Score-tracked lineup ratings + player embeddings")
        from src.pipeline import build_lineup_ratings
        build_lineup_ratings()
        _done(t)

    # ── Phase 3 : feature engineering ─────────────────────────────────────────
    if not skip_features:
        t = _step("Phase 3 — Feature engineering (rolling / EWMA / season-avg / opp context)")
        from src.features import build as _feat_build
        _feat_build()
        _done(t)

    # ── Phase 4 : position cache ───────────────────────────────────────────────
    t = _step("Phase 4 — Cache player positions (nba_api)")
    from src.optimizer import fetch_player_positions
    fetch_player_positions()
    _done(t)

    # ── Phase 5 : player performance prediction ────────────────────────────────
    if not skip_models:
        t = _step("Phase 5 — Player performance prediction (RF + XGBoost + SHAP)")
        from src.predictor import run as _pred_run
        _pred_run()
        _done(t)

        # ── Phase 7 : game outcome prediction ─────────────────────────────────
        t = _step("Phase 7 — Game outcome prediction (LogReg + XGBoost)")
        from src.game_model import run as _gm_run
        _gm_run()
        _done(t)

        # ── Advanced models : stacked ensemble + workload ──────────────────
        t = _step("Advanced — Stacked ensemble (LightGBM+CatBoost+XGBoost) + workload risk")
        from src.advanced_models import run as _adv_run
        _adv_run(tune=False)
        _done(t)

    print(f"\n{'='*62}")
    print("  AlleyLoop pipeline complete — all outputs in data/ and figures/")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
