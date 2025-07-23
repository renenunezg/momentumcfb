# backend/etl/build_schedule.py
import os, requests, pandas as pd
from dotenv import load_dotenv
from utils.supabase_client import supabase

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

rows = []
for wk in range(1, 15):            # adjust when CFBD posts 2024 data
    resp = requests.get(
        "https://api.collegefootballdata.com/games",
        params={"year": 2024, "week": wk},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    for g in resp.json():
        rows.append(
            dict(
                game_id=g["id"],
                week=g["week"],
                home_team=g["home_team"],
                away_team=g["away_team"],
                season=2024,
            )
        )

supabase.table("schedule").upsert(rows).execute()
print(f"Upserted {len(rows)} games.")