"""Atomically publish a validated draft workspace after the weekly college update."""

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import text

from backend.draft.board_source import parse_board
from backend.draft.positions import normalize_depth

PLAYER_COLUMNS = [
    "athlete_id",
    "athlete_name",
    "team",
    "position",
    "position_group",
    "games",
    "value_above_replacement",
    "position_rank",
]
HISTORY_COLUMNS = [
    "draft_year",
    "pick",
    "college_name",
    "collegeTeam",
    "draft_position",
    "identity_verified",
    "value_above_replacement",
    "nfl_first3_scrimmage_snaps",
    "outcome_status",
]


def records(frame):
    return [
        {k: v for k, v in row.items() if v is not None}
        for row in json.loads(frame.to_json(orient="records", date_format="iso"))
    ]


def fetch(url, receipts):
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    receipts.append(dict(url=url, sha256=hashlib.sha256(response.content).hexdigest()))
    return response.content


def current_depth(season, now, receipts):
    base = "https://github.com/nflverse/nflverse-data/releases/download/"
    depth = pd.read_parquet(
        io.BytesIO(fetch(f"{base}depth_charts/depth_charts_{season}.parquet", receipts))
    )
    depth["depth_as_of"] = pd.to_datetime(depth.dt, utc=True)
    depth = depth[depth.depth_as_of.le(now)].copy()
    latest = depth.groupby("team").depth_as_of.transform("max")
    depth = depth[depth.depth_as_of.eq(latest)].drop_duplicates(
        ["team", "pos_grp", "pos_slot", "pos_rank", "espn_id"]
    )
    if depth.team.nunique() != 32 or (now - depth.depth_as_of.min()).days > 14:
        raise ValueError("Incomplete or stale NFL depth charts")
    contracts = pd.read_parquet(
        io.BytesIO(fetch(f"{base}contracts/historical_contracts.parquet", receipts))
    )
    active = contracts[contracts.is_active.eq(True) & contracts.gsis_id.notna()].copy()
    active = active[~active.gsis_id.duplicated(keep=False)]
    depth = depth.merge(
        active[["gsis_id", "apy_cap_pct"]],
        on="gsis_id",
        how="left",
        validate="many_to_one",
    )
    columns = [
        "team",
        "pos_grp",
        "pos_abb",
        "pos_slot",
        "pos_rank",
        "player_name",
        "apy_cap_pct",
    ]
    return normalize_depth(records(depth[columns])), depth.depth_as_of.min().isoformat()


def publish(season, bootstrap=None, dry_run=False):
    from backend.db import engine

    now = datetime.now(timezone.utc)
    receipts = []
    with engine.connect() as connection:
        old = connection.execute(
            text("SELECT payload FROM cfb.draft_publications WHERE draft_year=:year"),
            {"year": season + 1},
        ).scalar_one_or_none()
        values = pd.read_sql(
            text(
                "SELECT * FROM cfb.player_values WHERE season=:season AND week=(SELECT max(week) FROM cfb.player_values WHERE season=:season)"
            ),
            connection,
            params={"season": season},
        )
    if values.empty or values.athlete_id.duplicated().any():
        raise ValueError("Latest college snapshot is missing or ambiguous")
    college_date = pd.to_datetime(values.as_of, utc=True).max()
    if not 0 <= (now - college_date).days <= 14:
        raise ValueError("College player snapshot is stale or in the future")
    seed = json.loads(Path(__file__).with_name("board_2027.json").read_text())
    if seed["season"] != season + 1:
        raise ValueError("A reviewed seed is required for a new draft class")
    pages = {
        key: fetch(url, receipts).decode("utf-8")
        for key, url in seed["sources"].items()
    }
    board = parse_board(pages, seed, now)
    depth, depth_date = current_depth(season, now, receipts)
    if {r["team"] for r in depth} != set(board["teams"]):
        raise ValueError("Draft board and NFL roster teams disagree")
    if bootstrap:
        roster = pd.read_parquet(bootstrap / "watchlist.parquet")
        history = records(
            pd.read_parquet(bootstrap / "historical_cohort.parquet")[HISTORY_COLUMNS]
        )
        history_date = json.loads((bootstrap / "audit.json").read_text())["as_of"]
        catalog = records(
            roster[
                ["athlete_id", "athlete_name", "team", "position", "height", "weight"]
            ]
        )
    elif old:
        catalog, history, history_date = (
            old["catalog"],
            old["history"],
            old["meta"]["history_as_of"],
        )
    else:
        raise ValueError("Initial publication requires an audited bootstrap directory")
    # The latest published college record owns team and position. Catalog-only
    # players remain searchable but never inherit a stale score after a transfer.
    measured = values[values.classification.eq("fbs")][PLAYER_COLUMNS].copy()
    measured.athlete_id = measured.athlete_id.astype(str)
    players = records(measured)
    for player in players:
        if player.get("position_group") == player.get("position"):
            player.pop("position_group", None)
    measured_ids = set(measured.athlete_id)
    players.extend(
        {
            k: v
            for k, v in p.items()
            if k in ("athlete_id", "athlete_name", "team", "position")
        }
        for p in catalog
        if p["athlete_id"] not in measured_ids
    )
    if len({p["athlete_id"] for p in players}) != len(players):
        raise ValueError("Duplicate college player identities")
    payload = dict(
        schema_version=1,
        board=board,
        roster=depth,
        players=players,
        history=history,
        catalog=catalog,
        meta=dict(
            published_at=now.isoformat(),
            college_as_of=college_date.isoformat(),
            college_week=int(values.week.max()),
            depth_as_of=depth_date,
            history_as_of=history_date,
            receipts=receipts,
        ),
    )
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    if not dry_run:
        with engine.begin() as connection:
            connection.execute(
                text("""INSERT INTO cfb.draft_publications (draft_year, payload, updated_at)
                VALUES (:year, CAST(:payload AS jsonb), :now)
                ON CONFLICT (draft_year) DO UPDATE SET payload=EXCLUDED.payload, updated_at=EXCLUDED.updated_at"""),
                dict(year=season + 1, payload=encoded, now=now),
            )
    return dict(
        draft_year=season + 1,
        college_week=int(values.week.max()),
        prospects=len(board["prospects"]),
        picks=len(board["picks"]),
        roster_rows=len(depth),
        players=len(players),
        bytes=len(encoded),
        published=not dry_run,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--bootstrap", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(publish(args.season, args.bootstrap, args.dry_run), indent=2))
