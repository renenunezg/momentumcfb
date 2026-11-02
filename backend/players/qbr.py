"""ESPN's season Total QBR, the ``espn_qbr`` value of the quarterback stat lines.

ESPN publishes the season-to-date Total QBR (``schedAdjQBR``, adjusted for the
defenses faced) of every FBS passer in one unmetered response. ESPN athlete ids
are CFBD athlete ids. The snapshot is taken on the weekly run, after the week's
games, so it covers the same games as the stat lines.
"""

from datetime import datetime, timezone

import pandas as pd
import requests

from backend.etl.ingest import write_parquet
from backend.players.ingest import players_dir

ESPN_QBR_URL = (
    "https://site.web.api.espn.com/apis/fitt/v3/sports/football/college-football/qbr"
)
TOTAL_QBR = "schedAdjQBR"
QBR_COLUMNS = ["athlete_id", "total_qbr", "fetched_at"]


def qbr_path(season: int):
    return players_dir(season) / "espn_qbr.parquet"


def parse_season_qbr(payload: dict, season: int) -> pd.DataFrame:
    """``QBR_COLUMNS`` rows from an ESPN season QBR response.

    The response must be the requested regular season on one page, and every
    passer must carry a Total QBR; anything else raises rather than publish a
    partial table.
    """
    requested = (payload.get("requestedSeason") or {}).get("year")
    if requested != season:
        raise ValueError(f"ESPN QBR answered season {requested}, not {season}")
    pagination = payload.get("pagination") or {}
    if pagination.get("pages", 1) != 1:
        raise ValueError(f"ESPN QBR spans {pagination.get('pages')} pages")
    names = next(
        category["names"]
        for category in payload["categories"]
        if category["name"] == "general"
    )
    column = names.index(TOTAL_QBR)
    rows = []
    for entry in payload["athletes"]:
        general = next(c for c in entry["categories"] if c["name"] == "general")
        rows.append(
            {
                "athlete_id": str(entry["athlete"]["id"]),
                "total_qbr": float(general["totals"][column]),
            }
        )
    out = pd.DataFrame(rows, columns=QBR_COLUMNS[:2])
    if out["athlete_id"].duplicated().any():
        raise ValueError("ESPN QBR lists a passer twice")
    return out


def ingest_season_qbr(season: int) -> pd.DataFrame:
    """Fetch the season's Total QBR for every FBS passer and store it beside the box scores."""
    response = requests.get(
        ESPN_QBR_URL,
        params={
            "region": "us",
            "lang": "en",
            "qbrType": "seasons",
            "seasontype": 2,
            "isqualified": "false",
            "season": season,
            "limit": 1000,
        },
        timeout=30,
    )
    response.raise_for_status()
    out = parse_season_qbr(response.json(), season).assign(
        fetched_at=datetime.now(timezone.utc).isoformat()
    )
    write_parquet(out, qbr_path(season))
    return out


def read_season_qbr(season: int) -> pd.DataFrame:
    return pd.read_parquet(qbr_path(season))
