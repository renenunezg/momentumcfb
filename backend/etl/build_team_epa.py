from __future__ import annotations
"""
Build team‑game EPA features and store them in Supabase.

Workflow
--------
1. Pull raw plays from the `cfb_pbp` table.
2. Compute per‑team, per‑game:
    • offensive EPA/play  (off_epa)
    • defensive EPA/play  (def_epa)
    • net EPA             (off_epa − def_epa)
    • pace (number of offensive plays)
    • total_off_epa / total_def_epa
    • var_off_epa / var_def_epa
    • pct_explosive
    • run_epa / pass_epa
3. Upsert the result into the `team_game_epa` table on conflict (game_id, team).

Run locally:
    cd backend
    poetry run python etl/build_team_epa.py
"""

import math
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────
# set up backend path & env vars
# ────────────────────────────────────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

load_dotenv(BACKEND_ROOT / ".env")

from utils.supabase_client import supabase  # noqa: E402


# ────────────────────────────────────────────────────────────
# feature builder
# ────────────────────────────────────────────────────────────
def make_team_game_epa() -> pd.DataFrame:
    """Return a DataFrame with one row per (game_id, team)."""
    # paginate through all PBP rows
    print("⏬  Paginating plays from Supabase …")
    all_records = []
    start = 0
    page_size = 1000  # match Supabase max rows per request
    while True:
        resp = (
            supabase.table("cfb_pbp")
            .select("*")
            .range(start, start + page_size - 1)
            .execute()
        )
        chunk = resp.data
        if not chunk:
            break
        all_records.extend(chunk)
        start += len(chunk)  # advance by actual rows fetched
        # print(f"  fetched {len(all_records):,} plays…")

    plays = pd.DataFrame(all_records)
    # drop only rows missing offense or defense, but keep plays without PPA/EPA
    plays = plays.dropna(subset=["offense", "defense"])
    # Do NOT fill missing ppa/epa—leave as NaN
    missing_ppa = plays["ppa"].isna().sum()
    print(f"Note: {missing_ppa:,} plays had missing PPA and were left as NaN.")
    # Rename 'ppa' to 'epa' for downstream code compatibility
    plays = plays.rename(columns={"ppa": "epa"})

    print(f"Total plays loaded: {len(plays):,}")

    if plays.empty:
        raise RuntimeError("cfb_pbp is empty — run load_pbp.py first.")

    # offensive EPA/play
    off = (
        plays.groupby(["game_id", "offense"], as_index=False)["epa"]
        .mean(numeric_only=True)
        .rename(columns={"offense": "team", "epa": "off_epa"})
    )

    # defensive EPA/play
    deff = (
        plays.groupby(["game_id", "defense"], as_index=False)["epa"]
        .mean(numeric_only=True)
        .rename(columns={"defense": "team", "epa": "def_epa"})
    )

    # pace (number of offensive plays)
    pace = (
        plays.groupby(["game_id", "offense"], as_index=False)
        .size()
        .rename(columns={"offense": "team", "size": "plays"})
    )

    # season & week (first non-null per game/team)
    season_week = (
        plays.groupby(["game_id", "offense"], as_index=False)[["season", "week"]]
        .first()
        .rename(columns={"offense": "team"})
    )

    # merge base including season & week
    df = (
        off.merge(deff, on=["game_id", "team"])
           .merge(pace, on=["game_id", "team"])
           .merge(season_week, on=["game_id", "team"])
    )
    df["net_epa"] = df.off_epa - df.def_epa

    # ── extended features ───────────────────────────────

    # total EPA volume
    df["total_off_epa"] = df.off_epa * df.plays
    df["total_def_epa"] = df.def_epa * df.plays

    # variance of EPA/play
    var_off_df = (
        plays.groupby(["game_id", "offense"])["epa"]
        .var()
        .rename("var_off_epa")
        .reset_index()
        .rename(columns={"offense": "team"})
    )
    var_def_df = (
        plays.groupby(["game_id", "defense"])["epa"]
        .var()
        .rename("var_def_epa")
        .reset_index()
        .rename(columns={"defense": "team"})
    )
    df = (
        df.merge(var_off_df, on=["game_id", "team"], how="left")
        .merge(var_def_df, on=["game_id", "team"], how="left")
    )

    # explosive play percentage
    pct_expl_df = (
        plays.assign(expl=plays.epa > 1.0)
        .groupby(["game_id", "offense"])["expl"]
        .mean()
        .rename("pct_explosive")
        .reset_index()
        .rename(columns={"offense": "team"})
    )
    df = df.merge(pct_expl_df, on=["game_id", "team"], how="left")

    # inspect play_type values for debugging
    print("play_type unique values:", plays["play_type"].unique())

    # run vs pass EPA using play_type field
    # categorize run plays (Rush, Rushing Touchdown, Sack counts as rush-defense but here skip sacks)
    run_types = {"Rush", "Rushing Touchdown"}
    run_df = (
        plays[plays.play_type.isin(run_types)]
        .groupby(["game_id", "offense"])["epa"]
        .mean()
        .rename("run_epa")
        .reset_index()
        .rename(columns={"offense": "team"})
    )

    # categorize pass plays (all pass-related events)
    pass_types = {
        "Pass Reception", "Pass Incompletion", "Passing Touchdown",
        "Pass Interception Return"
    }
    pass_df = (
        plays[plays.play_type.isin(pass_types)]
        .groupby(["game_id", "offense"])["epa"]
        .mean()
        .rename("pass_epa")
        .reset_index()
        .rename(columns={"offense": "team"})
    )
    df = (
        df.merge(run_df, on=["game_id", "team"], how="left")
        .merge(pass_df, on=["game_id", "team"], how="left")
    )

    # fill missing stats with 0 (no variance/explosive/run/pass)
    df[["var_off_epa", "var_def_epa", "pct_explosive", "run_epa", "pass_epa"]] = (
        df[["var_off_epa", "var_def_epa", "pct_explosive", "run_epa", "pass_epa"]]
        .fillna(0)
    )

    print(f"✅  Built features for {len(df):,} team‑games.")
    return df


# ────────────────────────────────────────────────────────────
# upsert helper
# ────────────────────────────────────────────────────────────
def upsert_team_game_epa(df: pd.DataFrame) -> None:
    """Upsert the DataFrame into Supabase."""
    print("⏫  Writing to Supabase table `team_game_epa` …")

    # convert NaN → None so JSON serializer is happy
    records = []
    for rec in df.to_dict("records"):
        clean = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in rec.items()
        }
        records.append(clean)

    supabase.table("team_game_epa").upsert(
        records, on_conflict="game_id,team"  # pass as single comma‑separated string
    ).execute()
    print("✅  Upsert complete.")


# ────────────────────────────────────────────────────────────
# main entry
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    team_game_df = make_team_game_epa()
    upsert_team_game_epa(team_game_df)