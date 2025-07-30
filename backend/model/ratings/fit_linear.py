#!/usr/bin/env python
"""
Fit a linear least-squares point-spread ratings model and upsert into power_ratings.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────
# set up backend path & env vars
# ────────────────────────────────────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

from utils.supabase_client import supabase  # noqa: E402

import numpy as np
import pandas as pd
from datetime import datetime, timezone

# ────────────────────────────────────────────────────────────
# data fetching
# ────────────────────────────────────────────────────────────
def fetch_game_data() -> pd.DataFrame:
    """Fetch finished games and score differentials."""
    rows = supabase.table("game_spreads").select(
        "home_team,away_team,home_points,away_points"
    ).execute().data
    if not rows:
        raise RuntimeError("No games in game_spreads")
    df = pd.DataFrame(rows)
    df = df[df["home_points"].notna() & df["away_points"].notna()].copy()
    return df

# ────────────────────────────────────────────────────────────
# model fitting
# ────────────────────────────────────────────────────────────
def fit_ratings(df: pd.DataFrame):
    """Compute team ratings via least-squares."""
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    m = len(df)
    n = len(teams)

    X = np.zeros((m, n))
    y = (df["home_points"] - df["away_points"]).to_numpy()

    for i, row in df.iterrows():
        X[i, idx[row["home_team"]]] = 1
        X[i, idx[row["away_team"]]] = -1

    # solve least squares: X @ ratings ≈ y
    ratings, *_ = np.linalg.lstsq(X, y, rcond=None)
    # center ratings to zero mean
    ratings = ratings - ratings.mean()
    return teams, ratings

# ────────────────────────────────────────────────────────────
# write to Supabase
# ────────────────────────────────────────────────────────────
def upsert_power_ratings(teams, ratings):
    """Upsert ratings into power_ratings."""
    ts = datetime.now(timezone.utc).isoformat()
    # fetch previous ratings for computing delta
    old_rows = supabase.table("power_ratings")\
        .select("team,rating")\
        .execute().data
    old_map = {row["team"]: row["rating"] for row in old_rows} if old_rows else {}
    recs = []
    for team, r in zip(teams, ratings):
        old = old_map.get(team)
        recs.append({
            "team": team,
            "rating": round(float(r), 2),
            "rating_delta": round(float(r) - float(old), 2) if old is not None else None,
            "model_name": "linear",
            "season": None,
            "week": None,
            "last_updated": ts,
        })
    supabase.table("power_ratings").upsert(
        recs,
        on_conflict="team"
    ).execute()
    print(f"✅ Upserted {len(recs)} linear ratings")

# ────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = fetch_game_data()
    teams, ratings = fit_ratings(df)
    upsert_power_ratings(teams, ratings)