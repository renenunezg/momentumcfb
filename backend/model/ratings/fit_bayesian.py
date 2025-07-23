#!/usr/bin/env python
# ─── path / env bootstrap ─────────────────────────────────────────────
from pathlib import Path
import sys
from dotenv import load_dotenv
import os
import requests
import pandas as pd
import pymc as pm
from sklearn.linear_model import LinearRegression
from datetime import datetime, timezone

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")
from utils.supabase_client import supabase  # noqa: E402
# ──────────────────────────────────────────────────────────────────────

YEAR = 2024

def fetch_epa_stats() -> pd.DataFrame:
    """Fetch play-by-play EPA and compute net EPA/play per team."""
    headers = {"Authorization": f"Bearer {os.getenv('CFBD_API_KEY')}"}
    rows = []
    for wk in range(1, 15):
        resp = requests.get(
            "https://api.collegefootballdata.com/plays",
            params={"year": YEAR, "week": wk},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        rows.extend(resp.json())
    pbp = pd.DataFrame(rows)
    pbp["off_epa_pp"] = pbp["offense"].apply(lambda d: d["overall"])
    pbp["def_epa_pp"] = pbp["defense"].apply(lambda d: d["overall"])
    stats = (
        pbp.groupby("team", as_index=False)[["off_epa_pp", "def_epa_pp"]]
           .mean()
    )
    stats["net_epa_pp"] = stats["off_epa_pp"] - stats["def_epa_pp"]
    return stats[["team", "net_epa_pp"]]

def fetch_games() -> pd.DataFrame:
    """Fetch schedule and final scores."""
    headers = {"Authorization": f"Bearer {os.getenv('CFBD_API_KEY')}"}
    resp = requests.get(
        "https://api.collegefootballdata.com/games",
        params={"year": YEAR, "seasonType": "regular"},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    g = pd.DataFrame(resp.json())
    return g[["id", "home_team", "away_team", "home_points", "away_points"]].rename(
        columns={"id": "game_id"}
    )

def build_game_df(epa_stats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Join EPA stats to games, compute epa_diff and actual margin."""
    df = games.merge(
        epa_stats.rename(columns={"team": "home_team", "net_epa_pp": "home_net_epa_pp"}),
        on="home_team",
        how="left",
    ).merge(
        epa_stats.rename(columns={"team": "away_team", "net_epa_pp": "away_net_epa_pp"}),
        on="away_team",
        how="left",
    )
    df = df.dropna(subset=["home_net_epa_pp", "away_net_epa_pp"])
    df["epa_diff"] = df["home_net_epa_pp"] - df["away_net_epa_pp"]
    df["spread_result"] = df["home_points"] - df["away_points"]
    return df[["game_id", "home_team", "away_team", "epa_diff", "spread_result"]]

def fit_bayes(df: pd.DataFrame) -> pd.DataFrame:
    """Fit hierarchical Bayesian model on EPA diff."""
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel("K"))
    idx = {t: i for i, t in enumerate(teams)}
    h = df["home_team"].map(idx).to_numpy()
    a = df["away_team"].map(idx).to_numpy()
    y = df["epa_diff"].to_numpy()

    with pm.Model() as model:
        σ_team = pm.HalfNormal("σ_team", sigma=1.0)
        σ_obs = pm.HalfNormal("σ_obs", sigma=1.0)
        rating = pm.Normal("rating", mu=0.0, sigma=σ_team, shape=len(teams))
        mu = rating[h] - rating[a]
        pm.Normal("obs", mu=mu, sigma=σ_obs, observed=y)
        trace = pm.sample(draws=1000, tune=1000, target_accept=0.9, progressbar=False)

    # Posterior mean rating per team
    stacked = trace.posterior["rating"].stack(samples=("chain", "draw"))
    mean_rating = stacked.mean("samples").values
    return pd.DataFrame({"team": teams, "raw_rating": mean_rating})

def main():
    # Fetch data
    epa_stats = fetch_epa_stats()
    games = fetch_games()
    game_df = build_game_df(epa_stats, games)

    # Fit model
    bayes_df = fit_bayes(game_df)

    # Calibrate to point spreads
    coef = LinearRegression().fit(
        game_df[["epa_diff"]], game_df["spread_result"]
    ).coef_[0]
    bayes_df["rating"] = (bayes_df["raw_rating"] * coef).round(2)
    bayes_df["rating"] -= bayes_df["rating"].mean()

    # Expand to all FBS teams
    headers = {"Authorization": f"Bearer {os.getenv('CFBD_API_KEY')}"}
    teams_data = requests.get(
        "https://api.collegefootballdata.com/teams/fbs",
        headers=headers,
        timeout=30,
    ).json()
    all_teams = [t["school"] for t in teams_data]
    final = (
        pd.DataFrame({"team": all_teams})
        .merge(bayes_df[["team", "rating"]], on="team", how="left")
        .fillna({"rating": 0})
    )

    # Prepare payload
    final["rating_delta"] = 0
    final["last_updated"] = datetime.now(timezone.utc).isoformat()
    payload = final[["team", "rating", "rating_delta", "last_updated"]].to_dict(
        "records"
    )

    # Upsert to Supabase
    supabase.table("power_ratings").upsert(payload, on_conflict="team").execute()
    print("✅ Bayesian power ratings updated")

if __name__ == "__main__":
    main()