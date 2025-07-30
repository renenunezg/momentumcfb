from __future__ import annotations
"""
Fit a Bayesian point-spread ratings model and upsert into `power_ratings`.

Uses `game_spreads` for final scores.

Schema refs
-----------
game_spreads(game_id, season, week, home_team, away_team,
             home_points, away_points, …)

power_ratings(team, rating, rating_delta, model_name,
              season, week, last_updated, …)

Run:
    cd backend
    poetry run python model/ratings/fit_bayesian.py
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────
# set up backend path & env vars
# ────────────────────────────────────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

load_dotenv(BACKEND_ROOT / ".env")

from utils.supabase_client import supabase  # noqa: E402


# ────────────────────────────────────────────────────────────
# data helpers
# ────────────────────────────────────────────────────────────
def fetch_finished_games() -> pd.DataFrame:
    """Return finished-game rows with score differential."""
    q = supabase.table("game_spreads").select(
        "game_id, home_team, away_team, home_points, away_points"
    )

    rows = q.execute().data
    if not rows:
        raise RuntimeError("No finished games in `game_spreads`.")

    df = pd.DataFrame(rows)
    mask = df["home_points"].notna() & df["away_points"].notna()
    df = df.loc[mask].copy()

    df["score_diff"] = df["home_points"] - df["away_points"]  # + = home wins
    return df


# ────────────────────────────────────────────────────────────
# Bayesian model
# ────────────────────────────────────────────────────────────
def fit_bayesian(df: pd.DataFrame):
    home_arr = df["home_team"].unique()
    away_arr = df["away_team"].unique()
    teams = pd.Index(sorted(set(home_arr).union(away_arr)))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    home_i = df["home_team"].map(idx).to_numpy()
    away_i = df["away_team"].map(idx).to_numpy()
    y = df["score_diff"].to_numpy()

    with pm.Model() as m:
        raw = pm.Normal("raw_ratings", mu=0, sigma=10, shape=n)
        ratings = raw - pm.math.mean(raw)             # sum-to-zero constraint
        hfa = pm.Normal("hfa", mu=2.0, sigma=3.0)
        sigma = pm.HalfNormal("sigma", sigma=5.0)

        mu = ratings[home_i] - ratings[away_i] + hfa
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=42)

    return teams, trace


# ────────────────────────────────────────────────────────────
# write to Supabase
# ────────────────────────────────────────────────────────────
def upsert_power_ratings(
    teams: pd.Index,
    trace,
    season: int | None,
    week: int | None,
) -> None:
    mean_r = trace.posterior["raw_ratings"].stack(s=("chain", "draw")).mean("s")
    centered = mean_r - mean_r.mean()

    # fetch conference info from existing EPA stats
    conf_rows = supabase.table("epa_stats").select("team,conference").execute().data
    conf_map = {
        row["team"].lower(): row.get("conference")
        for row in conf_rows
        if row.get("conference")
    }

    # fetch previous Bayesian ratings for delta
    old_rows = supabase.table("power_ratings")\
        .select("team,rating")\
        .execute().data
    old_map = {row["team"]: row["rating"] for row in old_rows} if old_rows else {}

    ts = datetime.utcnow().isoformat(timespec="seconds")
    recs = []
    for t, r in zip(teams, centered.values):
        old = old_map.get(t)
        recs.append({
            "team": t,
            "conference": conf_map.get(t.lower()),
            "rating": round(float(r), 2),
            "rating_delta": round(float(r) - float(old), 2) if old is not None else None,
            "model_name": "bayesian",
            "season": season,
            "week": week,
            "last_updated": ts,
        })

    supabase.table("power_ratings").upsert(
        recs,
        on_conflict="team"
    ).execute()
    print(f"✅  Upserted {len(recs)} rows.")


# ────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = fetch_finished_games()
    print(f"Loaded {len(df):,} games.")

    teams, trace = fit_bayesian(df)

    upsert_power_ratings(teams, trace, season=None, week=None)