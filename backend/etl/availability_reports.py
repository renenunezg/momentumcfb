"""Power 4 availability reports as pregame quarterback rows.

The SEC, Big Ten, ACC and Big 12 publish their mandated availability reports
through one vendor whose public archive lists every reported player per
conference game with a status at each report slot. The archive has no
positions and no timestamps, so a report row becomes a ``cfb.qb_availability``
row only when it names the quarterback who started the team's most recent
game, the passer with the most dropbacks in that game's play-by-play. That is
an observed fact about a played game, never an injury inference from text.
Only that starter's status is recorded, because the flat adjustment in
``backend.model.availability`` was sized on the week a team's previous
starter took no snaps; a backup listed out changes nothing.

The receipt time stands in for the report time, so a row can only inform a
forecast made after it was collected. Conference games only, Power 4 only.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime, timezone

import pandas as pd
import requests

from backend.config import RAW_DIR
from backend.etl import store

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://app.hdintelligence.com/api/get-archive-public"
LEGACY_ARCHIVE_URL = "https://app.hdintelligence.com/api/get-archive-legacy-public"
CONFERENCES = ("SEC", "B10", "ACC", "B12")
SOURCE = "hdintelligence_public_archive"
# Vendor statuses that the table's check constraint can hold. "Available",
# "Exempt" and blanks produce no row.
STATUS_MAP = {
    "out": "out",
    "doubtful": "doubtful",
    "questionable": "questionable",
    "game time decision": "questionable",
    "probable": "probable",
}
GAME_DATE_TOLERANCE_DAYS = 3
_PASSER = re.compile(r"#(\d{1,2}) (\S+) (?:pass|sacked)")
REPORT_COLUMNS = [
    "conference",
    "team",
    "player",
    "jersey",
    "opponent",
    "game_date",
    "vendor_week",
    "final_status",
    "slots",
]
TABLE_COLUMNS = [
    "season",
    "week",
    "team",
    "player",
    "status",
    "source",
    "source_url",
    "reported_at",
    "notes",
]


def fetch_conference_report(
    conference: str, legacy: bool = False, session: requests.Session | None = None
) -> dict:
    """One conference's public archive: ``data`` rows and ``report_days`` slots."""
    url = LEGACY_ARCHIVE_URL if legacy else ARCHIVE_URL
    response = (session or requests).post(
        url,
        json={
            "sport": "Football",
            "organization": conference,
            "conference": conference,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError(f"{conference}: unexpected availability payload")
    return payload


def parse_report(payload: dict, conference: str) -> pd.DataFrame:
    """Flatten one archive payload to one row per listed player and game.

    ``final_status`` is the latest filled report slot in the vendor's order.
    """
    slots = list(payload.get("report_days") or [])
    rows = []
    for item in payload.get("data") or []:
        filled = [(slot, item.get(slot)) for slot in slots if item.get(slot)]
        number = str(item.get("Number") or "").lstrip("#").strip()
        rows.append(
            {
                "conference": conference,
                "team": item.get("TeamDisplay") or item.get("Team"),
                "player": item.get("Player"),
                "jersey": int(number) if number.isdigit() else None,
                "opponent": item.get("OpponentDisplay") or item.get("Opponent"),
                "game_date": pd.Timestamp(item["Week"]).date(),
                "vendor_week": int(item.get("WeekNum")),
                "final_status": filled[-1][1] if filled else None,
                "slots": "; ".join(f"{slot}={status}" for slot, status in filled),
            }
        )
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def _fold(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    return re.sub(r"[^a-z]", "", text.encode("ascii", "ignore").decode().lower())


def _last_name(name: str) -> str:
    tokens = [_fold(token) for token in str(name or "").split()]
    tokens = [token for token in tokens if token not in ("jr", "sr", "ii", "iii", "iv")]
    return tokens[-1] if tokens else ""


def load_roster(season: int) -> pd.DataFrame:
    path = RAW_DIR / "players" / str(season) / "roster.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{season}: roster.parquet is required to find quarterbacks"
        )
    roster = pd.read_parquet(path)
    out = pd.DataFrame(
        {
            "team": roster["team"],
            "jersey": pd.to_numeric(roster["jersey"], errors="coerce"),
            "position": roster["position"],
            "name": (
                roster["first_name"].fillna("").astype(str).str.strip()
                + " "
                + roster["last_name"].fillna("").astype(str).str.strip()
            ).str.strip(),
            "athlete_id": roster["id"].astype(str),
        }
    )
    return out[out["jersey"].notna()].assign(
        jersey=lambda frame: frame["jersey"].astype(int)
    )


def match_quarterbacks(reports: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Keep report rows that name a rostered quarterback by team, jersey and
    last name."""
    quarterbacks = roster[roster["position"].eq("QB")]
    merged = reports.dropna(subset=["jersey"]).merge(
        quarterbacks[["team", "jersey", "name", "athlete_id"]],
        on=["team", "jersey"],
        how="inner",
    )
    same_name = merged["player"].map(_last_name).eq(merged["name"].map(_last_name))
    return merged[same_name].reset_index(drop=True)


def pbp_starters(season: int) -> pd.DataFrame:
    """The leading passer by dropbacks for every team and completed game.

    Read from the cached play-by-play text, which names the passer by jersey
    on every pass and sack. Returns game_id, week, season_type, team,
    starter_jersey, starter_token and dropbacks, one row per team and game.
    """
    plays = store.read_season_pbp(season)
    plays = plays[plays["play_text"].notna()]
    matches = plays["play_text"].astype(str).str.extract(_PASSER)
    plays = plays.assign(
        starter_jersey=pd.to_numeric(matches[0], errors="coerce"),
        starter_token=matches[1],
    ).dropna(subset=["starter_jersey"])
    counts = (
        plays.groupby(
            [
                "game_id",
                "week",
                "season_type",
                "offense",
                "starter_jersey",
                "starter_token",
            ]
        )
        .size()
        .rename("dropbacks")
        .reset_index()
    )
    counts["starter_jersey"] = counts["starter_jersey"].astype(int)
    starters = counts.sort_values("dropbacks", ascending=False).drop_duplicates(
        ["game_id", "offense"]
    )
    return starters.rename(columns={"offense": "team"}).sort_values(
        ["team", "season_type", "week", "game_id"], ignore_index=True
    )


def previous_starter(
    starters: pd.DataFrame, team: str, week: int
) -> tuple[int, str, int] | None:
    """Jersey, name token and week of the team's starter in its latest regular
    season game before ``week``."""
    rows = starters[
        starters["team"].eq(team)
        & starters["season_type"].eq("regular")
        & starters["week"].lt(week)
    ]
    if rows.empty:
        return None
    last = rows.sort_values("week").iloc[-1]
    return int(last["starter_jersey"]), str(last["starter_token"]), int(last["week"])


def map_games(reports: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach the CFBD game id and week by team and kickoff date."""
    schedule = games[
        ["id", "week", "season_type", "home_team", "away_team", "start_date"]
    ]
    schedule = schedule.assign(
        kickoff=pd.to_datetime(schedule["start_date"], utc=True)
        .dt.tz_convert("US/Eastern")
        .dt.date
    )
    long = pd.concat(
        [
            schedule.assign(team=schedule["home_team"]),
            schedule.assign(team=schedule["away_team"]),
        ]
    )[["id", "week", "season_type", "team", "kickoff"]]
    out = reports.merge(long, on="team", how="left")
    distance = (
        pd.to_datetime(out["kickoff"]) - pd.to_datetime(out["game_date"])
    ).dt.days.abs()
    out = out[distance.le(GAME_DATE_TOLERANCE_DAYS)]
    duplicates = out.duplicated(
        ["conference", "team", "player", "game_date"], keep=False
    )
    if duplicates.any():
        raise ValueError("a report row matched more than one scheduled game")
    return out.rename(columns={"id": "game_id"}).reset_index(drop=True)


def build_rows(
    season: int,
    reports: pd.DataFrame,
    starters: pd.DataFrame,
    fetched_at: datetime,
) -> pd.DataFrame:
    """Table rows for listed quarterbacks who started their team's last game."""
    rows = []
    for item in reports.itertuples():
        status = STATUS_MAP.get(str(item.final_status or "").strip().lower())
        if status is None:
            continue
        starter = previous_starter(starters, item.team, int(item.week))
        if starter is None or starter[0] != int(item.jersey):
            continue
        rows.append(
            {
                "season": season,
                "week": int(item.week),
                "team": item.team,
                "player": item.name,
                "status": status,
                "source": SOURCE,
                "source_url": ARCHIVE_URL,
                "reported_at": fetched_at,
                "notes": (
                    f"{item.conference} report, {item.opponent}, vendor week "
                    f"{item.vendor_week}, game {int(item.game_id)}; {item.slots}; "
                    f"started week {starter[2]} as #{starter[0]} {starter[1]}"
                ),
            }
        )
    out = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    if out.duplicated(["season", "week", "team"]).any():
        raise ValueError("two starters resolved for one team and week")
    return out.sort_values(["week", "team"], ignore_index=True)


def collect_quarterback_reports(
    season: int,
    legacy: bool,
    fetched_at: datetime | None = None,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch every conference, resolve starters, and return (table rows,
    matched quarterback report rows) for review."""
    fetched_at = fetched_at or datetime.now(timezone.utc)
    frames = []
    for conference in CONFERENCES:
        # One conference's outage or malformed reply must not drop the others;
        # its rows simply stay as the table last saw them.
        try:
            payload = fetch_conference_report(
                conference, legacy=legacy, session=session
            )
            frame = parse_report(payload, conference)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            log.warning("%s: availability archive unavailable: %s", conference, exc)
            continue
        log.info(
            "%s: %d report rows, weeks %s",
            conference,
            len(frame),
            sorted(frame["vendor_week"].unique().tolist()),
        )
        frames.append(frame)
    if not frames:
        raise RuntimeError("no conference availability archive could be read")
    reports = pd.concat(frames, ignore_index=True)
    quarterbacks = match_quarterbacks(reports, load_roster(season))
    quarterbacks = map_games(quarterbacks, store.read_games(season))
    rows = build_rows(season, quarterbacks, pbp_starters(season), fetched_at)
    return rows, quarterbacks


def changed_rows(rows: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Rows that are new or whose player or status differs from the table.

    An unchanged row keeps its original reported_at, so a report first seen
    before a forecast stays eligible for it.
    """
    if existing.empty:
        return rows
    current = existing[["season", "week", "team", "player", "status"]].rename(
        columns={"player": "existing_player", "status": "existing_status"}
    )
    merged = rows.merge(current, on=["season", "week", "team"], how="left")
    changed = merged["existing_player"].isna() | (
        merged["existing_player"].ne(merged["player"])
        | merged["existing_status"].ne(merged["status"])
    )
    return merged.loc[changed, TABLE_COLUMNS].reset_index(drop=True)


def write_rows(rows: pd.DataFrame) -> int:
    """Insert or update the table on its (season, week, team) key."""
    from sqlalchemy import text

    from backend.db import CFB_SCHEMA, engine, writes_allowed

    if rows.empty:
        return 0
    if not writes_allowed():
        raise PermissionError("set MOMENTUMCFB_DB_WRITES=1 to write availability rows")
    statement = text(
        f"INSERT INTO {CFB_SCHEMA}.qb_availability "
        "(season, week, team, player, status, source, source_url, reported_at, notes) "
        "VALUES (:season, :week, :team, :player, :status, :source, :source_url, "
        ":reported_at, :notes) "
        "ON CONFLICT (season, week, team) DO UPDATE SET player = EXCLUDED.player, "
        "status = EXCLUDED.status, source = EXCLUDED.source, "
        "source_url = EXCLUDED.source_url, reported_at = EXCLUDED.reported_at, "
        "notes = EXCLUDED.notes, updated_at = clock_timestamp()"
    )
    records = rows[TABLE_COLUMNS].to_dict("records")
    with engine.begin() as conn:
        conn.execute(statement, records)
    return len(records)
