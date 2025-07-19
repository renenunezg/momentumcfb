import os, itertools, pandas as pd
import numpy as np
import math
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()  # pulls SUPABASE_URL & SUPABASE_SERVICE_ROLE_KEY from backend/.env
import requests
import datetime
CFBD_API_KEY = os.getenv("CFBD_API_KEY")
from supabase import create_client

YEARS = [2024]  
BATCH = 1000  # Supabase insert limit

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")     # service-role key bypasses RLS
)

def chunks(it, n):
    it = iter(it)
    while (c := list(itertools.islice(it, n))):
        yield c

BASE_URL = "https://api.collegefootballdata.com/plays"

# --------------------------------------------------
# Build a dict {year: [weeks_to_fetch]} so we can pull
# full prior seasons and only the latest finished week
weeks_map = {}

today = datetime.date.today()

for yr in YEARS:
    if yr < today.year:
        # Finished season – fetch every regular‑season week (1‑15)
        weeks_map[yr] = list(range(1, 16))
    else:
        # Current season – only the most‑recent completed week
        cal_resp = requests.get(
            "https://api.collegefootballdata.com/calendar",
            params={"year": yr, "seasonType": "regular"},
            headers={"Authorization": f"Bearer {CFBD_API_KEY}"}
        )
        cal = cal_resp.json()
        completed_weeks = [
            w["week"]
            for w in cal
            if w.get("lastGameStart")
            and datetime.date.fromisoformat(w["lastGameStart"][:10]) < today
        ]
        if not completed_weeks:
            print(f"No finished weeks yet for {yr} – nothing to pull.")
            continue
        weeks_map[yr] = [max(completed_weeks)]

for yr, weeks_to_fetch in weeks_map.items():
    for wk in weeks_to_fetch:
        params = {
            "year": yr,
            "week": wk,
            "seasonType": "both",   # include bowls later if needed
        }
        try:
            resp = requests.get(
                BASE_URL,
                params=params,
                headers={"Authorization": f"Bearer {CFBD_API_KEY}"}
            )
            resp.raise_for_status()
            import time
            time.sleep(0.25)
        except Exception as e:
            print(f"⚠️  {yr}-W{wk} API error ({e}); skipping.")
            continue

        try:
            plays = resp.json()
        except ValueError:
            print(f"⚠️  {yr}-W{wk} non‑JSON response; skipping.")
            continue

        if not plays:
            continue

        df = pd.DataFrame(plays)

        # --- harmonize column names ---
        rename_map = {
            "id": "play_id",
            "gameId": "game_id",
            "playType": "play_type",
            "ppa": "epa",
            "PPA": "epa",
            "EPA": "epa",
        }
        df = df.rename(columns=rename_map)

        # Add season / week if API omits them
        if "season" not in df.columns:
            df["season"] = yr
        if "week" not in df.columns:
            df["week"] = wk

        wanted = [
            "season", "week", "game_id", "play_id",
            "offense", "defense",
            "down", "distance", "epa", "play_type"
        ]
        missing = [c for c in wanted if c not in df.columns]
        if missing:
            print(f"⚠️  Missing cols {missing} on {yr}-W{wk}; skipping chunk.")
            continue

        # Slice wanted columns then scrub impossible numbers
        slim = (
            df[wanted]
            .replace({np.nan: None, np.inf: None, -np.inf: None})
            .astype(object)
        )

        # Build batch records after cleaning to avoid np.nan sneaking through
        records_iter = chunks(slim.to_dict('records'), BATCH)

        for batch in tqdm(records_iter, desc=f"Uploading {yr}-W{wk}"):
            supabase.table("cfb_pbp").upsert(
                batch,
                on_conflict="game_id,play_id"
            ).execute()

print("✅ Finished back-fill")