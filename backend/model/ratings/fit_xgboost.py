#!/usr/bin/env python
"""
fit_xgboost.py
────────────────────────────────────────────────────────
Fit an XGBoost‑based point‑spread scaling model and write
team ratings into Supabase `power_ratings`.

• Uses team‑level net EPA as the core strength metric.
• Trains a one‑feature XGBRegressor on historical game
  margins (home − away) versus (home_net_epa_pp − away_net_epa_pp).
• The average slope of predictions → margin gives scale k.
• Final rating = (net_epa_pp * k) centered to zero mean.
• Rounds rating & rating_delta to 2 decimals.
"""

# ─── path / env bootstrap ─────────────────────────────────────
from pathlib import Path
import sys
from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

from utils.supabase_client import supabase  # noqa: E402
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from datetime import datetime, timezone
from sklearn.metrics import mean_absolute_error


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────
def pull_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (team_stats, game_df) for model fit."""
    # team stats – latest net_epa_pp per team
    stats_rows = supabase.table("epa_stats").select(
        "team,net_epa_pp,conference"
    ).execute().data
    if not stats_rows:
        raise RuntimeError("epa_stats empty.")
    stats = pd.DataFrame(stats_rows).rename(columns={"team": "team_id", "net_epa_pp": "net"})

    # finished games with scores
    game_rows = supabase.table("game_spreads").select(
        "home_team,away_team,home_points,away_points"
    ).execute().data
    if not game_rows:
        raise RuntimeError("game_spreads empty.")
    games = pd.DataFrame(game_rows)
    games = games[games["home_points"].notna() & games["away_points"].notna()].copy()

    # merge net EPA to build diff feature
    games = (
        games.merge(stats[["team_id", "net"]], left_on="home_team", right_on="team_id")
        .rename(columns={"net": "h"})
        .merge(stats[["team_id", "net"]], left_on="away_team", right_on="team_id")
        .rename(columns={"net": "a"})
    )
    games["diff"] = games.h - games.a
    games["margin"] = games.home_points - games.away_points
    return stats, games


def compute_scale_k(games: pd.DataFrame) -> float:
    """Train XGB to map diff→margin and return average slope k."""
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        n_jobs=1,
        verbosity=0,
    )
    model.fit(games[["diff"]], games["margin"])
    preds = model.predict(games[["diff"]])
    mae = mean_absolute_error(games["margin"], preds)
    print(f"Train MAE: {mae:.2f}")

    nonzero = games["diff"].abs() > 1e-6
    k = np.mean(preds[nonzero] / games.loc[nonzero, "diff"])
    print(f"Derived scale k = {k:.3f}")
    return float(k)


# ─────────────────────────────────────────────────────────────
# main upsert
# ─────────────────────────────────────────────────────────────
def upsert_ratings(stats: pd.DataFrame, k: float) -> None:
    """Scale EPA to ratings, compute delta, and upsert."""
    stats["rating_raw"] = stats["net"] * k
    stats["rating"] = stats["rating_raw"] - stats["rating_raw"].mean()

    # fetch previous rating, regardless of model_name
    old_rows = supabase.table("power_ratings").select("team,rating").execute().data
    old_map = {r["team"]: r["rating"] for r in old_rows} if old_rows else {}

    ts = datetime.now(timezone.utc).isoformat()
    recs = []
    for _, row in stats.iterrows():
        team = row["team_id"]
        r_new = round(float(row["rating"]), 2)
        r_old = old_map.get(team)
        recs.append(
            {
                "team": team,
                "conference": row.get("conference"),
                "rating": r_new,
                "rating_delta": round(r_new - float(r_old), 2) if r_old is not None else None,
                "model_name": "xgb",
                "season": None,
                "week": None,
                "last_updated": ts,
            }
        )

    supabase.table("power_ratings").upsert(recs, on_conflict="team").execute()
    print(f"✅ Upserted {len(recs)} XGBoost ratings")


# ─────────────────────────────────────────────────────────────
# entry
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    team_stats, game_df = pull_training_data()
    k_val = compute_scale_k(game_df)
    upsert_ratings(team_stats, k_val)