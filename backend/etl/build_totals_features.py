#!/usr/bin/env python
"""
build_totals_features.py

Derives pace + efficiency features for totals modelling.
Each output row is (game_id, team) with:

    - off_epa_pp, def_epa_pp        : from team_game_epa table
    - plays                         : offensive play count in that game
    - rolling‑3‑game means of the above

Writes / upserts into `totals_features` on (game_id, team).
"""

from pathlib import Path
import sys
from dotenv import load_dotenv
import pandas as pd

def fetch_all_supabase_rows(table_name, select_cols, batch_size=1000):
    offset = 0
    all_rows = []
    while True:
        resp = supabase.table(table_name).select(select_cols).range(offset, offset + batch_size - 1).execute()
        data = resp.data
        if not data:
            break
        all_rows.extend(data)
        if len(data) < batch_size:
            break
        offset += batch_size
    return pd.DataFrame(all_rows)

# ── path / env bootstrap ──────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")
# ──────────────────────────────────────────────────────

from utils.supabase_client import supabase  # noqa: E402
# ──────────────────────────────────────────────────────


TEAM_GAME_EPA_TABLE = "team_game_epa"      # rename if different
PBP_TABLE           = "cfb_pbp"       # play‑by‑play table
OUTPUT_TABLE        = "totals_features"    # staging target


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch required source tables from Supabase."""
    epa_df = fetch_all_supabase_rows(
        TEAM_GAME_EPA_TABLE,
        "game_id, team, season, week, off_epa_pp, def_epa_pp, net_epa, run_epa, pass_epa, total_off_epa, total_def_epa, var_off_epa, var_def_epa, pct_explosive"
    )
    pbp_df = fetch_all_supabase_rows(
        PBP_TABLE,
        "game_id, offense"
    )
    print("epa_df columns:", epa_df.columns)
    print(epa_df[["off_epa_pp", "def_epa_pp"]].head())
    return epa_df, pbp_df


def derive_pace(pbp: pd.DataFrame) -> pd.DataFrame:
    """Return DF [game_id, team, plays] counting offensive snaps."""
    pace = (
        pbp.groupby(["game_id", "offense"])
        .size()
        .reset_index(name="pace_plays")
        .rename(columns={"offense": "team"})
    )
    return pace


def build_features() -> pd.DataFrame:
    epa_df, pbp_df = load_inputs()

    if epa_df.empty:
        raise RuntimeError("team_game_epa table returned 0 rows")

    # add plays per team×game
    pace_df = derive_pace(pbp_df)
    # 🛠 Ensure consistent types for join
    epa_df["game_id"] = epa_df["game_id"].astype("int64")
    pace_df["game_id"] = pace_df["game_id"].astype("int64")
    try:
        df = (
            epa_df.merge(
                pace_df,
                how="left",
                on=["game_id", "team"],
            )
            .rename(columns={"pace_plays": "plays"}, errors="ignore")
        )
        # keep only the columns we actually need
        cols_keep = [
            "game_id", "team", "season", "week",
            "off_epa_pp", "def_epa_pp", "net_epa",
            "run_epa", "pass_epa",
            "total_off_epa", "total_def_epa",
            "var_off_epa", "var_def_epa",
            "pct_explosive",
            "plays",
        ]
        df = df[cols_keep].copy()
        # df = df.dropna(subset=["season", "week"])
    except Exception as e:
        raise RuntimeError(f"Error merging pace and EPA data: {e}")

    # sort by team and game_id to get correct chronology for rolling means
    df = df.sort_values(["team", "game_id"]).reset_index(drop=True)

    for col in ["off_epa_pp", "def_epa_pp", "plays"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate rolling averages for each team over last 3 games
    df["off_epa_pp_roll3"] = (
        df.groupby("team")["off_epa_pp"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["def_epa_pp_roll3"] = (
        df.groupby("team")["def_epa_pp"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["plays_roll3"] = (
        df.groupby("team")["plays"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Remove any other *_roll3 columns except the three above
    roll3_cols_keep = ["off_epa_pp_roll3", "def_epa_pp_roll3", "plays_roll3"]
    roll3_cols_to_drop = [col for col in df.columns if col.endswith("_roll3") and col not in roll3_cols_keep]
    if roll3_cols_to_drop:
        df = df.drop(columns=roll3_cols_to_drop)

    return df


def upsert(df: pd.DataFrame) -> None:
    df = df.copy()
    df["game_id"] = df["game_id"].astype("Int64")
    for col in ["plays", "season", "week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.astype(object).where(pd.notnull(df), None)

    # Clean table before inserting (keep this if you want truncate behavior)
    supabase.table(OUTPUT_TABLE).delete().neq("game_id", -1).execute()

    records = df.to_dict("records")
    if not records:
        print("No records to upsert.")
        return

    # Chunk and upsert
    chunk_size = 1000
    total = len(records)
    for i in range(0, total, chunk_size):
        batch = records[i:i+chunk_size]
        supabase.table(OUTPUT_TABLE).upsert(
            batch, on_conflict="game_id,team"
        ).execute()
        print(f"Upserted rows {i+1}-{min(i+chunk_size, total)} of {total} into {OUTPUT_TABLE}")


if __name__ == "__main__":
    dataset = build_features()
    upsert(dataset)
