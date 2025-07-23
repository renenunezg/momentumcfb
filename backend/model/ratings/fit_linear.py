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
from sklearn.linear_model import LinearRegression
from datetime import datetime, timezone


def build_stats() -> pd.DataFrame:
    epa = supabase.table("team_game_epa").select("team,net_epa").execute().data
    if not epa:
        raise RuntimeError("team_game_epa empty.")
    return (
        pd.DataFrame(epa)
        .groupby("team", as_index=False)["net_epa"]
        .mean()
        .rename(columns={"team": "team_id", "net_epa": "net"})
    )


def sos_adjust(stats: pd.DataFrame) -> pd.DataFrame:
    sched = supabase.table("schedule").select("home_team,away_team").execute().data
    if not sched:
        stats["adj"] = stats.net
        return stats

    sched = pd.DataFrame(sched)
    opp = (
        sched.melt(value_name="team_id")[["team_id"]]
        .merge(stats, on="team_id")
        .groupby("team_id", as_index=False)["net"]
        .mean()
        .rename(columns={"net": "sos"})
    )
    stats = stats.merge(opp, on="team_id", how="left").fillna({"sos": 0})
    stats["adj"] = stats.net - stats.sos
    return stats


def calibrate_k(stats: pd.DataFrame) -> float:
    sched = supabase.table("schedule").select("*").execute().data
    if not sched or "home_points" not in sched[0]:
        return 1.0
    sched = pd.DataFrame(sched)
    g = (
        sched.merge(stats[["team_id", "adj"]], left_on="home_team", right_on="team_id")
        .rename(columns={"adj": "h"})
        .merge(stats[["team_id", "adj"]], left_on="away_team", right_on="team_id")
        .rename(columns={"adj": "a"})
    )
    g["diff"] = g.h - g.a
    g["margin"] = g.home_points - g.away_points
    return LinearRegression().fit(g[["diff"]], g["margin"]).coef_[0]


def main():
    stats = sos_adjust(build_stats())
    k = calibrate_k(stats)
    stats["rating"] = stats.adj * k - stats.adj.mean()
    out = stats[["team_id", "rating"]]
    out["model"] = "linear"
    out["updated_at"] = datetime.now(timezone.utc).isoformat()
    supabase.table("team_ratings").upsert(
        out.to_dict("records"), on_conflict="team_id,model"
    ).execute()
    print("✅ Linear ratings upserted")


if __name__ == "__main__":
    main()