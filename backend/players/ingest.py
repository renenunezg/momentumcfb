"""Snapshot the CFBD player sources behind player value and the Heisman board.

Per season: the team catalog, the roster (positions), and weekly poll
rankings. Per week: the box score for every game and, from 2019 on, the
per-play player stat rows that link players to plays. The play-stats endpoint
silently caps a response at 2000 rows, so it is pulled one FBS conference at
a time; that is the cheapest chunking against the monthly call quota.
"""

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from backend.cfbd.client import CFBDClient
from backend.config import MAX_REGULAR_WEEK, RAW_DIR
from backend.etl.ingest import SEASON_TYPES, ingest_games, to_snake, write_parquet

log = logging.getLogger(__name__)

PLAY_STATS_FIRST_SEASON = 2019
PLAY_STATS_ROW_CAP = 2000
PLAY_STATS_TIMEOUT = 240

BOX_COLUMNS = [
    "game_id",
    "season",
    "week",
    "season_type",
    "team",
    "conference",
    "home_away",
    "points",
    "category",
    "stat_name",
    "athlete_id",
    "athlete_name",
    "stat",
]

RANKING_COLUMNS = [
    "season",
    "season_type",
    "week",
    "poll",
    "rank",
    "team_id",
    "school",
    "conference",
    "points",
]


def players_dir(season: int):
    return RAW_DIR / "players" / str(season)


def weekly_path(season: int, kind: str, season_type: str, week: int):
    return players_dir(season) / f"{kind}_{season_type}_{week:02d}.parquet"


def _flatten_box(rows: list, season: int, week: int, season_type: str) -> pd.DataFrame:
    records = []
    for game in rows:
        for team in game.get("teams") or []:
            for category in team.get("categories") or []:
                for stat_type in category.get("types") or []:
                    for athlete in stat_type.get("athletes") or []:
                        records.append(
                            (
                                game["id"],
                                season,
                                week,
                                season_type,
                                team.get("team"),
                                team.get("conference"),
                                team.get("homeAway"),
                                team.get("points"),
                                category.get("name"),
                                stat_type.get("name"),
                                athlete.get("id"),
                                athlete.get("name"),
                                athlete.get("stat"),
                            )
                        )
    return pd.DataFrame.from_records(records, columns=BOX_COLUMNS)


def _flatten_rankings(rows: list) -> pd.DataFrame:
    records = []
    for entry in rows:
        for poll in entry.get("polls") or []:
            for rank in poll.get("ranks") or []:
                records.append(
                    (
                        entry["season"],
                        entry["seasonType"],
                        entry["week"],
                        poll["poll"],
                        rank.get("rank"),
                        rank.get("teamId"),
                        rank.get("school"),
                        rank.get("conference"),
                        rank.get("points"),
                    )
                )
    return pd.DataFrame.from_records(records, columns=RANKING_COLUMNS)


def _fetch_play_stats(
    client: CFBDClient,
    season: int,
    week: int,
    season_type: str,
    teams: pd.DataFrame,
    abbreviations: dict[str, str],
) -> pd.DataFrame:
    """Per-play stat rows for the week, one request per FBS conference.

    The endpoint filters on the conference abbreviation (B1G, MAC), not the
    name the team catalog carries. A conference call is slow but it is the
    cheapest chunking against the monthly call quota; a chunk that hits the
    row cap is split by team.
    """
    base = {"year": season, "week": week, "seasonType": season_type}
    frames = []
    fbs = teams[teams["classification"].eq("fbs")]
    for conference, members in fbs.groupby("conference"):
        abbreviation = abbreviations.get(str(conference))
        if abbreviation is None:
            raise RuntimeError(f"no CFBD abbreviation for conference {conference}")
        # A conference chunk can take over a minute to assemble server-side.
        rows = client.get(
            "/plays/stats",
            {**base, "conference": abbreviation},
            timeout=PLAY_STATS_TIMEOUT,
        )
        if len(rows) < PLAY_STATS_ROW_CAP:
            frames.append(pd.DataFrame(rows))
            continue
        for team in members["school"]:
            team_rows = client.get("/plays/stats", {**base, "team": team})
            if len(team_rows) >= PLAY_STATS_ROW_CAP:
                raise RuntimeError(
                    f"play stats for {team} {season} week {week} hit the row cap"
                )
            frames.append(pd.DataFrame(team_rows))
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    stats = to_snake(pd.concat(frames, ignore_index=True))
    # Cross-conference games arrive under both conferences.
    return stats.drop_duplicates(
        ["play_id", "athlete_id", "stat_type", "stat"], ignore_index=True
    )


def ingest_player_sources(
    client: CFBDClient,
    season: int,
    only_week: int | None = None,
) -> pd.DataFrame:
    """Snapshot every player source for one season and record a manifest."""
    destination = players_dir(season)
    fetched_at = datetime.now(timezone.utc).isoformat()
    manifest = []

    def record(name: str, endpoint: str, params: dict, frame: pd.DataFrame) -> None:
        manifest.append(
            {
                "source": name,
                "endpoint": endpoint,
                "params": json.dumps(params, sort_keys=True),
                "source_fetched_at": fetched_at,
                "row_count": len(frame),
                "is_empty": frame.empty,
            }
        )
        log.info(f"players {season} {name}: {len(frame)} rows")

    if not (RAW_DIR / "games" / f"{season}.parquet").exists():
        ingest_games(client, season)

    teams = to_snake(pd.DataFrame(client.get("/teams", {"year": season})))
    write_parquet(teams, destination / "teams.parquet")
    record("teams", "/teams", {"year": season}, teams)

    roster = to_snake(pd.DataFrame(client.get("/roster", {"year": season})))
    write_parquet(roster, destination / "roster.parquet")
    record("roster", "/roster", {"year": season}, roster)

    ranking_frames = []
    for season_type in SEASON_TYPES:
        params = {"year": season, "seasonType": season_type}
        ranking_frames.append(_flatten_rankings(client.get("/rankings", params)))
    rankings = pd.concat(ranking_frames, ignore_index=True)
    write_parquet(rankings, destination / "rankings.parquet")
    record("rankings", "/rankings", {"year": season}, rankings)

    include_play_stats = season >= PLAY_STATS_FIRST_SEASON
    abbreviations = (
        {
            entry["name"]: entry["abbreviation"]
            for entry in client.get("/conferences", {})
            if entry.get("abbreviation")
        }
        if include_play_stats
        else {}
    )
    for season_type in SEASON_TYPES:
        weeks = [only_week] if only_week is not None else range(1, MAX_REGULAR_WEEK + 1)
        for week in weeks:
            params = {"year": season, "week": week, "seasonType": season_type}
            box_path = weekly_path(season, "box", season_type, week)
            play_stats_path = weekly_path(season, "play_stats", season_type, week)
            # A season backfill resumes past weeks already on disk; only a
            # targeted week is refreshed, so restarts never re-spend calls.
            refresh = only_week is not None
            if refresh or not box_path.exists():
                box = _flatten_box(
                    client.get("/games/players", params), season, week, season_type
                )
                if box.empty:
                    continue
                write_parquet(box, box_path)
                record(f"box_{season_type}_{week:02d}", "/games/players", params, box)
            if include_play_stats and (refresh or not play_stats_path.exists()):
                stats = _fetch_play_stats(
                    client, season, week, season_type, teams, abbreviations
                )
                if stats.empty:
                    continue
                stats["season_type"] = season_type
                write_parquet(stats, play_stats_path)
                record(
                    f"play_stats_{season_type}_{week:02d}",
                    "/plays/stats",
                    params,
                    stats,
                )

    manifest_frame = pd.DataFrame(manifest)
    write_parquet(manifest_frame, destination / "manifest.parquet")
    return manifest_frame


def read_weekly(season: int, kind: str) -> pd.DataFrame:
    """Concatenate every stored weekly frame of one kind for a season."""
    files = sorted(players_dir(season).glob(f"{kind}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {kind} parquet for {season}, run ingest-players")
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def read_season_source(season: int, name: str) -> pd.DataFrame:
    return pd.read_parquet(players_dir(season) / f"{name}.parquet")
