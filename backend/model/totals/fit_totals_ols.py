#!/usr/bin/env python
"""
fit_totals_ols.py
────────────────────────────────────────────────────────
• Trains an OLS model to predict game total points.
• Uses sums of home/away EPA-based features.
• Writes predictions to Supabase `weekly_projection`
  (game_id, model_total).
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ─── env / path bootstrap ────────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

from utils.supabase_client import supabase  # noqa: E402
# ─────────────────────────────────────────────────────────


def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return totals_features (team-level) & finished games (with scores)."""
    feat_rows = supabase.table("totals_features").select("*").execute().data
    if not feat_rows:
        raise RuntimeError("totals_features empty.")
    feats = pd.DataFrame(feat_rows)

    game_rows = supabase.table("game_spreads").select(
        "game_id,home_team,away_team,home_points,away_points"
    ).execute().data
    if not game_rows:
        raise RuntimeError("game_spreads empty.")
    games = pd.DataFrame(game_rows)
    games = games.dropna(subset=["home_points", "away_points"]).copy()
    games["total_points"] = games.home_points + games.away_points
    return feats, games


def build_design_matrix(feats: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Merge per-team features into per-game sums and return X, y."""
    cols = [
        "off_epa_pp",
        "def_epa_pp",
        "net_epa",
        "plays",
        "pct_explosive",
        "var_off_epa",
        "var_def_epa",
    ]

    # home features
    g = games.merge(
        feats[["game_id", "team"] + cols],
        left_on=["game_id", "home_team"],
        right_on=["game_id", "team"],
        how="left",
    ).rename(columns={c: f"h_{c}" for c in cols})

    # away features
    g = g.merge(
        feats[["game_id", "team"] + cols],
        left_on=["game_id", "away_team"],
        right_on=["game_id", "team"],
        how="left",
        suffixes=("", "_away"),
    ).rename(columns={c: f"a_{c}" for c in cols})

    # build summed features
    for c in cols:
        g[f"sum_{c}"] = g[f"h_{c}"] + g[f"a_{c}"]

    X = g[[f"sum_{c}" for c in cols]].fillna(0)
    y = g["total_points"]
    return X, y, g[["game_id"]]


def fit_ols(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """Fit OLS and report MAE."""
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    print(f"Train MAE: {mae:.2f}")
    return model


def upsert_predictions(game_ids: pd.Series, preds: np.ndarray) -> None:
    """Upsert rounded totals into weekly_projection."""
    ts = datetime.now(timezone.utc).isoformat()
    recs = [
        {
            "game_id": int(gid),
            "generated_at": ts,
            "model_total": round(float(p), 2),
        }
        for gid, p in zip(game_ids, preds)
    ]
    supabase.table("weekly_projection").upsert(
        recs, on_conflict="game_id"
    ).execute()
    print(f"✅ Upserted {len(recs)} totals to weekly_projection")


# ─── run ───────────────────────────────────────────────────
if __name__ == "__main__":
    feats_df, games_df = fetch_data()
    X_train, y_train, game_ids = build_design_matrix(feats_df, games_df)
    ols = fit_ols(X_train, y_train)
    preds = ols.predict(X_train)
    upsert_predictions(game_ids["game_id"], preds)