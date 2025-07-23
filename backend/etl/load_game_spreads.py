#!/usr/bin/env python
"""
load_game_spreads.py
────────────────────────────────────────────────────────
Fetch final scores + pre‑game spreads, join with EPA/play
( from epa_stats table ), compute epa_diff & spread_result,
and upsert into `game_spreads`.
"""

from pathlib import Path
import sys
from dotenv import load_dotenv
import os
import requests
import pandas as pd

# ── path / env bootstrap ──────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")
# ──────────────────────────────────────────────────────

from utils.supabase_client import supabase  # noqa: E402

CFBD_API_KEY = os.getenv("CFBD_API_KEY")
YEAR = 2024


def fetch_games(year: int = YEAR) -> pd.DataFrame:
    """Fetch game scores for the season."""
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    resp = requests.get(
        "https://api.collegefootballdata.com/games",
        params={"year": year, "seasonType": "regular"},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    g = pd.DataFrame(resp.json())

    # camelCase → snake_case if needed
    if "home_team" not in g.columns:
        g = g.rename(
            columns={
                "homeTeam": "home_team",
                "awayTeam": "away_team",
                "homePoints": "home_points",
                "awayPoints": "away_points",
            }
        )
    # Do NOT rename id here; keep as 'id'
    keep = ["id", "home_team", "away_team", "home_points", "away_points"]
    return g[keep]


def fetch_spreads(year: int = YEAR) -> pd.DataFrame:
    """Pull pre‑game spreads week‑by‑week and return unique rows."""
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    wp_url = "https://api.collegefootballdata.com/metrics/wp/pregame"
    rows = []
    for w in range(1, 15):
        r = requests.get(
            wp_url,
            params={"year": year, "week": w, "seasonType": "regular"},
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        rows.extend(r.json())
    wp = pd.DataFrame(rows).rename(columns={"gameId": "id"})
    wp["spread"] = pd.to_numeric(wp["spread"], errors="coerce")
    return wp[["id", "spread"]].drop_duplicates("id")


def attach_epa(games: pd.DataFrame) -> pd.DataFrame:
    """Join EPA/play for home & away teams."""
    epa_stats = pd.DataFrame(supabase.table("epa_stats").select("*").execute().data)

    epa_stats["season"] = YEAR
    epa_stats["conference"] = epa_stats["conference"].fillna("")

    # Extract EPA fields
    epa_stats["off_cum_total"] = epa_stats["offense"].apply(lambda d: d["cumulative"]["total"])
    epa_stats["def_cum_total"] = epa_stats["defense"].apply(lambda d: d["cumulative"]["total"])
    epa_stats["net_epa"] = epa_stats["off_cum_total"] - epa_stats["def_cum_total"]
    epa_stats["off_epa_pp"] = epa_stats["offense"].apply(lambda d: d["overall"])
    epa_stats["def_epa_pp"] = epa_stats["defense"].apply(lambda d: d["overall"])
    epa_stats["net_epa_pp"] = epa_stats["off_epa_pp"] - epa_stats["def_epa_pp"]

    cols = [
        "season", "conference", "team", "offense", "defense",
        "off_cum_total", "def_cum_total", "net_epa",
        "off_epa_pp", "def_epa_pp", "net_epa_pp"
    ]
    epa_stats = epa_stats[cols]

    games = (
        games.merge(
            epa_stats[["team", "net_epa_pp"]],
            left_on="home_team",
            right_on="team",
            how="left",
        )
        .rename(columns={"net_epa_pp": "home_net_epa_pp"})
        .drop(columns=["team"])
    )

    games = (
        games.merge(
            epa_stats[["team", "net_epa_pp"]],
            left_on="away_team",
            right_on="team",
            how="left",
        )
        .rename(columns={"net_epa_pp": "away_net_epa_pp"})
        .drop(columns=["team"])
    )

    games["epa_diff"] = games["home_net_epa_pp"] - games["away_net_epa_pp"]
    # Spread result = (margin) + spread; positive if home covers, negative if not
    games["spread_result"] = (
        games["home_points"] - games["away_points"] + games["spread"]
    )

    return games.dropna(subset=["home_net_epa_pp", "away_net_epa_pp"])


def upsert(df: pd.DataFrame) -> None:
    df["id"] = pd.to_numeric(df["id"], errors="coerce", downcast="integer")
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)
    df["game_id"] = df["id"]  # mirror id to satisfy NOT NULL constraint
    print(f"🧪 Sample ids: {df['id'].head().tolist()}")
    print(f"🧪 id dtype after cast: {df['id'].dtype}")
    print(f"🧪 id dtype before upsert: {df['id'].dtype}")
    # --- ensure true integers in all integer columns ---
    INT_COLS = ["id", "game_id", "home_points", "away_points"]

    for col in INT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")

    df = df.dropna(subset=INT_COLS)          # drop rows still missing ints
    df[INT_COLS] = df[INT_COLS].astype(int)  # final cast
    # ---------------------------------------------------
    supabase.table("game_spreads").upsert(
        df.to_dict("records")
    ).execute()
    print(f"✅ Upserted {len(df)} rows into `game_spreads`")


if __name__ == "__main__":
    print("⏬ Fetching games & spreads …")
    games = fetch_games()
    spreads = fetch_spreads()
    merged = games.merge(spreads, on="id")
    df = attach_epa(merged)
    print(f"📊 Final DF size: {len(df)} rows")
    upsert(df)