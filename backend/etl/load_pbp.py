#!/usr/bin/env python
import math
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────
# Season constant: update each year
# ────────────────────────────────────────────────────────────
SEASON = 2024  # <- update each year

# ────────────────────────────────────────────────────────────
# set up backend path & env vars
# ────────────────────────────────────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

load_dotenv(BACKEND_ROOT / ".env")

from utils.supabase_client import supabase  # noqa: E402
import os
import requests

# Set API key and headers
API_KEY = os.getenv("CFBD_API_KEY")
headers = {"Authorization": f"Bearer {API_KEY}"}

# Fetch PBP data for SEASON across all weeks
url = "https://api.collegefootballdata.com/plays"
pbp_list = []
for w in range(1, 15):
    resp = requests.get(url, params={"year": SEASON, "week": w}, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # tag each play with season & week so they persist in the DB
    for play in data:
        play["season"] = SEASON
        play["week"] = w
    print(f"Week {w}: fetched {len(data)} plays")
    pbp_list.extend(data)
print(f"Total plays fetched: {len(pbp_list)}")

pbp = pd.DataFrame(pbp_list)
# convert CamelCase keys to snake_case to match table schema
pbp.columns = (
    pbp.columns
       .str.replace(r"([a-z0-9])([A-Z])", r"\1_\2", regex=True)
       .str.lower()
)
# ensure both season and week columns exist
if "season" not in pbp.columns:
    pbp["season"] = SEASON
if "week" not in pbp.columns:
    raise RuntimeError("week column missing after rename step")

# convert NaN to None for JSON
pbp = pbp.where(pd.notnull(pbp), None)

# manually fix any remaining column mismatches
pbp = pbp.rename(columns={
    "id": "play_id",
    "yardline": "yard_line",
    "wallclock": "wall_clock"
})

print(f"DataFrame created with {len(pbp)} rows and columns: {list(pbp.columns)}")


# ensure no NaN or infinite values remain
pbp = pbp.astype(object)
pbp = pbp.where(pd.notnull(pbp), None)
pbp = pbp.replace({np.inf: None, -np.inf: None})


# convert integer-like columns to pandas nullable Int64 (drops .0)
import pandas as pd  # ensure available
int_cols = [
    "game_id", "drive_number", "play_number",
    "offense_score", "defense_score", "period",
    "offense_timeouts", "defense_timeouts",
    "yards_to_goal", "down", "distance", "yards_gained"
]
for col in int_cols:
    if col in pbp.columns:
        pbp[col] = pd.to_numeric(pbp[col], errors='coerce').astype("Int64")
print("Post-cast dtypes:", pbp[int_cols].dtypes.to_dict())

records = pbp.to_dict("records")
print(f"Upserting {len(records)} records to cfb_pbp")

# batch upsert to avoid huge payloads
batch_size = 1000
total = len(records)
print(f"Total records to upsert: {total}")
for start in range(0, total, batch_size):
    end = min(start + batch_size, total)
    batch = records[start:end]
    print(f"Upserting records {start}–{end-1}")
    supabase.table("cfb_pbp") \
        .upsert(batch, on_conflict="game_id,play_id") \
        .execute()
print("✅ Loaded and upserted PBP for 2024 W1–W14")
