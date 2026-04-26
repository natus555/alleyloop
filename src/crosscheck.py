"""
Cross-validate box_scores.parquet against official NBA stats (nba_api).
Samples 5% of unique game IDs per season, fetches BoxScoreTraditionalV2,
and reports per-column accuracy.

Usage:
    python src/crosscheck.py
"""
import time
import random
import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import BoxScoreTraditionalV2

PROCESSED_DIR = Path("data/processed")

SEASON_MAP = {
    2020: "2020-21",
    2021: "2021-22",
    2022: "2022-23",
    2023: "2023-24",
    2024: "2024-25",
}

# nba_api column → our column
COL_MAP = {
    "PTS":        "pts",
    "REB":        "reb",
    "OREB":       "oreb",
    "DREB":       "dreb",
    "AST":        "ast",
    "STL":        "stl",
    "BLK":        "blk",
    "TO":         "tov",
    "PF":         "pf",
    "FGM":        "fgm",
    "FGA":        "fga",
    "FG3M":       "fg3m",
    "FG3A":       "fg3a",
    "FTM":        "ftm",
    "FTA":        "fta",
}


def _fetch_official(game_id: int, season_str: str, retries: int = 3) -> pd.DataFrame | None:
    for attempt in range(retries):
        try:
            bs = BoxScoreTraditionalV2(
                game_id=str(game_id).zfill(10),
                timeout=30,
            )
            df = bs.player_stats.get_data_frame()
            time.sleep(0.65)
            return df
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}] game {game_id}: {e} — waiting {wait}s")
            time.sleep(wait)
    return None


def run_crosscheck(sample_frac: float = 0.05, seed: int = 42) -> pd.DataFrame:
    box = pd.read_parquet(PROCESSED_DIR / "box_scores.parquet")

    all_results = []
    random.seed(seed)

    for season, season_str in SEASON_MAP.items():
        season_box = box[box["season"] == season]
        game_ids = season_box["gameId"].unique().tolist()
        n_sample = max(1, int(len(game_ids) * sample_frac))
        sampled = random.sample(game_ids, n_sample)

        print(f"\nSeason {season_str}: {len(game_ids)} games → sampling {n_sample}")

        for i, gid in enumerate(sampled, 1):
            if i % 20 == 0:
                print(f"  {i}/{n_sample} games fetched...")

            official = _fetch_official(gid, season_str)
            if official is None:
                print(f"  SKIP game {gid} (fetch failed)")
                continue

            official = official.rename(columns={"PLAYER_ID": "personId"})
            official["personId"] = pd.to_numeric(official["personId"], errors="coerce")

            our = season_box[season_box["gameId"] == gid].copy()
            our["personId"] = pd.to_numeric(our["personId"], errors="coerce")

            merged = official.merge(our, on="personId", suffixes=("_off", "_our"))
            if merged.empty:
                continue

            for off_col, our_col in COL_MAP.items():
                if off_col not in official.columns or our_col not in our.columns:
                    continue
                merged[f"diff_{our_col}"] = pd.to_numeric(merged[off_col], errors="coerce") \
                                           - pd.to_numeric(merged[our_col], errors="coerce")
            all_results.append(merged)

    if not all_results:
        print("No results collected.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    print("\n" + "=" * 60)
    print("CROSS-CHECK RESULTS")
    print("=" * 60)
    print(f"Player-game pairs compared: {len(combined):,}\n")

    report_rows = []
    for off_col, our_col in COL_MAP.items():
        diff_col = f"diff_{our_col}"
        if diff_col not in combined.columns:
            continue
        diffs = combined[diff_col].dropna()
        if diffs.empty:
            continue
        exact = (diffs == 0).mean() * 100
        mae   = diffs.abs().mean()
        max_e = diffs.abs().max()
        report_rows.append({
            "stat":     our_col,
            "exact_%":  round(exact, 1),
            "MAE":      round(mae, 4),
            "max_err":  int(max_e) if our_col != "min" else round(max_e, 2),
            "n":        len(diffs),
        })
        print(f"  {our_col:<12}  exact={exact:5.1f}%  MAE={mae:.4f}  max_err={max_e:.1f}")

    report = pd.DataFrame(report_rows)
    out = PROCESSED_DIR / "crosscheck_5pct.parquet"
    combined.to_parquet(out, index=False)
    print(f"\nFull comparison saved to {out}")
    return report


if __name__ == "__main__":
    report = run_crosscheck(sample_frac=0.05)
    print("\nSummary table:")
    print(report.to_string(index=False))
