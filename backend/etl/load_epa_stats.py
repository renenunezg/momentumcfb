#!/usr/bin/env python
"""
load_epa_stats.py
────────────────────────────────────────────────────────
Pull team‑level EPA/PPA metrics from CollegeFootballData
and upsert them into the `epa_stats` table in Supabase.

Columns inserted:
  season, conference, team, offense, defense,
  off_cum_total, def_cum_total, net_epa,
  off_epa_pp, def_epa_pp, net_epa_pp
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
# ──────────────────────────────────────────────────────

CFBD_API_KEY = os.getenv("CFBD_API_KEY")
YEAR = 2024  # change when rolling season
EXCLUDE_GARBAGE = True # True → remove garbage‑time plays


def fetch_team_epa(year: int = YEAR, exclude_garbage: bool = EXCLUDE_GARBAGE) -> pd.DataFrame:
    """Fetch team‑level PPA stats from CFBD."""
    print(f"⏬  Fetching team EPA/PPA for {year} (excludeGarbageTime={exclude_garbage}) …")
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    params = {"year": year, "excludeGarbageTime": exclude_garbage}
    resp = requests.get("https://api.collegefootballdata.com/ppa/teams",
                        params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    print(f"   → {len(df)} teams fetched")
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative and per‑play EPA metrics."""
    df["off_cum_total"] = df["offense"].apply(lambda d: d["cumulative"]["total"])
    df["def_cum_total"] = df["defense"].apply(lambda d: d["cumulative"]["total"])
    df["net_epa"] = df["off_cum_total"] - df["def_cum_total"]

    df["off_epa_pp"] = df["offense"].apply(lambda d: d["overall"])
    df["def_epa_pp"] = df["defense"].apply(lambda d: d["overall"])
    df["net_epa_pp"] = df["off_epa_pp"] - df["def_epa_pp"]

    keep = [
        "season",
        "conference",
        "team",
        "offense",
        "defense",
        "off_cum_total",
        "def_cum_total",
        "net_epa",
        "off_epa_pp",
        "def_epa_pp",
        "net_epa_pp",
    ]
    return df[keep]


def upsert(df: pd.DataFrame) -> None:
    """Upsert the transformed DataFrame into Supabase."""
    print("⏫  Upserting to Supabase table `epa_stats` …")
    supabase.table("epa_stats").upsert(df.to_dict("records"), on_conflict="team").execute()
    print("✅  Upsert complete")


if __name__ == "__main__":
    raw = fetch_team_epa()
    tidy = transform(raw)
    upsert(tidy)