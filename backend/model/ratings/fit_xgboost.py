#!/usr/bin/env python
# ─── path / env bootstrap ─────────────────────────────────────────────
from pathlib import Path, sys
from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")
from utils.supabase_client import supabase  # noqa: E402
# ──────────────────────────────────────────────────────────────────────

import pandas as pd
from xgboost import XGBRegressor
from datetime import datetime, timezone
from sklearn.metrics import mean_absolute_error


def pull_data():
    epa = supabase.table("team_game_epa").select("team,net_epa").execute().data
    sched = supabase.table("schedule").select("*").execute().data
    if not epa or not sched:
        raise RuntimeError("Need EPA & schedule.")
    stats = (
        pd.DataFrame(epa)
        .groupby("team", as_index=False)["net_epa"]
        .mean()
        .rename(columns={"team": "team_id", "net_epa": "net"})
    )
    sched = pd.DataFrame(sched)
    g = (
        sched.merge(stats, left_on="home_team", right_on="team_id")
        .rename(columns={"net": "h"})
        .merge(stats, left_on="away_team", right_on="team_id")
        .rename(columns={"net": "a"})
    )
    g["diff"] = g.h - g.a
    g["margin"] = g.home_points - g.away_points
    return stats, g


def main():
    stats, games = pull_data()
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3)
    model.fit(games[["diff"]], games["margin"])
    preds = model.predict(games[["diff"]])
    print("Train MAE:", mean_absolute_error(games["margin"], preds))

    k = 1.0  # (could derive a scale from model)
    stats["rating"] = stats.net * k - stats.net.mean()
    out = stats.rename(columns={"team_id": "team_id"})[["team_id", "rating"]]
    out["model"] = "xgb"
    out["updated_at"] = datetime.now(timezone.utc).isoformat()
    supabase.table("team_ratings").upsert(
        out.to_dict("records"), on_conflict="team_id,model"
    ).execute()
    print("✅ XGBoost ratings upserted")


if __name__ == "__main__":
    main()