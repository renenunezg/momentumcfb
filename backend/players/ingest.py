"""Snapshot the CFBD player sources behind player value and the Heisman board.

Per season: the team catalog, the roster (positions), and weekly poll
rankings. Per week: the box score for every game and, from 2019 on, the
per-play player stat rows that link players to plays. The play-stats endpoint
silently caps a response at 2000 rows, so it is pulled one FBS conference at
a time; that is the cheapest chunking against the monthly call quota.
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.cfbd.client import CFBDClient
from backend.config import RAW_DIR
from backend.etl.ingest import ingest_games, to_snake, write_parquet

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


def _completed_weeks(
    games: pd.DataFrame, only_week: int | None
) -> list[tuple[str, int, pd.DataFrame]]:
    required = {"id", "week", "season_type", "completed", "home_team", "away_team"}
    if not required.issubset(games.columns):
        raise ValueError(
            "Player ingestion requires a schedule with completion and team fields"
        )
    completed = games[games["completed"].fillna(False).eq(True)].copy()
    if {"home_classification", "away_classification"}.issubset(completed.columns):
        completed = completed[
            completed["home_classification"].eq("fbs")
            | completed["away_classification"].eq("fbs")
        ]
    if only_week is not None:
        completed = completed[completed["week"].eq(only_week)]
    return [
        (str(season_type), int(week), frame)
        for (season_type, week), frame in completed.groupby(
            ["season_type", "week"], sort=True
        )
    ]


def _game_ids(games: pd.DataFrame) -> list[int]:
    return sorted(int(game) for game in games["id"].unique())


def _conference_groups(teams: pd.DataFrame, games: pd.DataFrame):
    schools = set(games["home_team"]) | set(games["away_team"])
    active = teams[teams["classification"].eq("fbs") & teams["school"].isin(schools)]
    return list(active.groupby("conference", sort=True))


def _parts_directory(
    season: int, season_type: str, week: int, games: pd.DataFrame
) -> Path:
    digest = hashlib.sha256(json.dumps(_game_ids(games)).encode()).hexdigest()[:16]
    return players_dir(season) / "checkpoints" / f"{season_type}_{week:02d}" / digest


def _chunk_path(directory: Path, field: str, value: str) -> Path:
    digest = hashlib.sha256(value.encode()).hexdigest()[:16]
    return directory / f"{field}_{digest}.parquet"


def _play_stats_calls(teams: pd.DataFrame, games: pd.DataFrame, directory: Path) -> int:
    calls = 0
    for conference, members in _conference_groups(teams, games):
        path = _chunk_path(directory, "conference", str(conference))
        if not path.exists():
            calls += 1
        elif len(pd.read_parquet(path)) >= PLAY_STATS_ROW_CAP:
            calls += sum(
                not _chunk_path(directory, "team", str(team)).exists()
                for team in members["school"]
            )
    return calls


def _fetch_play_stats(
    client: CFBDClient,
    season: int,
    week: int,
    season_type: str,
    teams: pd.DataFrame,
    abbreviations: dict[str, str],
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch active conferences sequentially, persisting each quota-paid chunk."""
    base = {"year": season, "week": week, "seasonType": season_type}
    directory = _parts_directory(season, season_type, week, games)
    frames = []

    def fetch_chunk(
        field: str, value: str, cache_name: str, schools: list[str]
    ) -> tuple[pd.DataFrame, bool]:
        path = _chunk_path(directory, field, cache_name)
        if path.exists():
            return pd.read_parquet(path), True
        frame = to_snake(
            pd.DataFrame(
                client.get(
                    "/plays/stats", {**base, field: value}, timeout=PLAY_STATS_TIMEOUT
                )
            )
        )
        expected_games = games[
            games["home_team"].isin(schools) | games["away_team"].isin(schools)
        ]
        if len(frame) < PLAY_STATS_ROW_CAP:
            observed = set(frame["game_id"]) if "game_id" in frame else set()
            missing = set(_game_ids(expected_games)) - observed
            if missing:
                raise ValueError(
                    f"Incomplete play stats for {field} {value}: missing games {sorted(missing)}"
                )
        write_parquet(frame, path)
        return frame, False

    for conference, members in _conference_groups(teams, games):
        abbreviation = abbreviations.get(str(conference))
        if abbreviation is None:
            raise RuntimeError(f"no CFBD abbreviation for conference {conference}")
        frame, cached = fetch_chunk(
            "conference", abbreviation, str(conference), members["school"].tolist()
        )
        if len(frame) < PLAY_STATS_ROW_CAP:
            frames.append(frame)
            continue
        # Cached capped chunks are already included in the initial estimate.
        if not cached:
            client.extend_budget(
                sum(
                    not _chunk_path(directory, "team", str(team)).exists()
                    for team in members["school"]
                )
            )
        for team in members["school"]:
            team_frame, _ = fetch_chunk("team", str(team), str(team), [str(team)])
            if len(team_frame) >= PLAY_STATS_ROW_CAP:
                raise RuntimeError(
                    f"play stats for {team} {season} week {week} hit the row cap"
                )
            frames.append(team_frame)
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    stats = pd.concat(frames, ignore_index=True)
    # Cross-conference games arrive under both conferences. Exclude games
    # still live when the schedule used to plan this snapshot was captured.
    stats = stats[stats["game_id"].isin(_game_ids(games))]
    return stats.drop_duplicates(
        ["play_id", "athlete_id", "stat_type", "stat"], ignore_index=True
    )


def ingest_player_sources(
    client: CFBDClient,
    season: int,
    only_week: int | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Budget completed-game sources, resuming durable per-source checkpoints."""
    if refresh and only_week is None:
        raise ValueError("--refresh requires --week to bound paid corrections")
    destination = players_dir(season)
    games_path = RAW_DIR / "games" / f"{season}.parquet"
    if not games_path.exists():
        client.ensure_budget(1)
        ingest_games(client, season)
    weeks = _completed_weeks(pd.read_parquet(games_path), only_week)
    manifest_path = destination / "manifest.parquet"
    manifest = (
        pd.read_parquet(manifest_path).to_dict("records")
        if manifest_path.exists()
        else []
    )
    coverage_path = destination / "completed_games.json"
    coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
    include_play_stats = season >= PLAY_STATS_FIRST_SEASON

    def checkpoint_coverage(name: str, game_ids: list[int]) -> None:
        coverage[name] = game_ids
        temporary = coverage_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(coverage, sort_keys=True))
        temporary.replace(coverage_path)

    def pending(kind: str, season_type: str, week: int, games: pd.DataFrame) -> bool:
        name = f"{kind}_{season_type}_{week:02d}"
        path = weekly_path(season, kind, season_type, week)
        if refresh or not path.exists():
            return True
        if name not in coverage:
            # Adopt existing snapshots without paying for a migration ingest.
            existing = pd.read_parquet(path, columns=["game_id"])
            return not set(_game_ids(games)).issubset(existing["game_id"])
        return coverage[name] != _game_ids(games)

    work = [
        (kind, season_type, week, games)
        for season_type, week, games in weeks
        for kind in (["box", "play_stats"] if include_play_stats else ["box"])
        if pending(kind, season_type, week, games)
    ]
    if not work:
        log.info("players %s: all completed-game sources are checkpointed", season)
        return pd.DataFrame(manifest)

    fetched_at = datetime.now(timezone.utc).isoformat()

    def record(name: str, endpoint: str, params: dict, frame: pd.DataFrame) -> None:
        nonlocal manifest
        manifest = [entry for entry in manifest if entry["source"] != name]
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
        write_parquet(pd.DataFrame(manifest), manifest_path)
        log.info("players %s %s: %s rows", season, name, len(frame))

    teams_path = destination / "teams.parquet"
    if teams_path.exists():
        teams = pd.read_parquet(teams_path)
    else:
        client.ensure_budget(1)
        teams = to_snake(pd.DataFrame(client.get("/teams", {"year": season})))
        write_parquet(teams, teams_path)
        record("teams", "/teams", {"year": season}, teams)
    conferences_path = destination / "conferences.parquet"
    need_play_stats = any(kind == "play_stats" for kind, *_ in work)
    season_types = sorted({season_type for _, season_type, _, _ in work})
    snapshot_game_ids = sorted(
        {game_id for _, _, games in weeks for game_id in _game_ids(games)}
    )
    roster_path = destination / "roster.parquet"
    rankings_path = destination / "rankings.parquet"
    need_roster = (
        refresh
        or not roster_path.exists()
        or coverage.get("roster") != snapshot_game_ids
    )
    need_rankings = (
        refresh
        or not rankings_path.exists()
        or coverage.get("rankings") != snapshot_game_ids
    )
    estimated_calls = int(need_roster) + len(season_types) * int(need_rankings)
    estimated_calls += int(need_play_stats and not conferences_path.exists())
    estimated_calls += sum(
        1
        if kind == "box"
        else len(_conference_groups(teams, games))
        if refresh
        else _play_stats_calls(
            teams, games, _parts_directory(season, season_type, week, games)
        )
        for kind, season_type, week, games in work
    )
    client.ensure_budget(estimated_calls)
    if refresh:
        for season_type, week, _ in weeks:
            checkpoint = destination / "checkpoints" / f"{season_type}_{week:02d}"
            if checkpoint.exists():
                shutil.rmtree(checkpoint)
    if need_roster:
        roster = to_snake(pd.DataFrame(client.get("/roster", {"year": season})))
        write_parquet(roster, roster_path)
        record("roster", "/roster", {"year": season}, roster)
        checkpoint_coverage("roster", snapshot_game_ids)
    if need_rankings:
        ranking_frames = []
        if rankings_path.exists():
            previous = pd.read_parquet(rankings_path)
            ranking_frames.append(previous[~previous["season_type"].isin(season_types)])
        for season_type in season_types:
            params = {"year": season, "seasonType": season_type}
            ranking_frames.append(_flatten_rankings(client.get("/rankings", params)))
        rankings = pd.concat(ranking_frames, ignore_index=True)
        write_parquet(rankings, rankings_path)
        record("rankings", "/rankings", {"year": season}, rankings)
        checkpoint_coverage("rankings", snapshot_game_ids)
    abbreviations = {}
    if need_play_stats:
        if conferences_path.exists():
            conferences = pd.read_parquet(conferences_path)
        else:
            conferences = pd.DataFrame(client.get("/conferences", {}))
            write_parquet(conferences, conferences_path)
            record("conferences", "/conferences", {}, conferences)
        abbreviations = dict(zip(conferences["name"], conferences["abbreviation"]))

    for kind, season_type, week, games in work:
        params = {"year": season, "week": week, "seasonType": season_type}
        if kind == "box":
            frame = _flatten_box(
                client.get("/games/players", params), season, week, season_type
            )
            frame = frame[frame["game_id"].isin(_game_ids(games))]
            endpoint = "/games/players"
        else:
            frame = _fetch_play_stats(
                client, season, week, season_type, teams, abbreviations, games
            )
            if not frame.empty:
                frame["season_type"] = season_type
            endpoint = "/plays/stats"
        if frame.empty:
            raise ValueError(
                f"No {kind} rows for completed {season} {season_type} week {week}"
            )
        missing_games = set(_game_ids(games)) - set(frame["game_id"])
        if missing_games:
            raise ValueError(
                f"Incomplete {kind} snapshot: missing completed games {sorted(missing_games)}"
            )
        name = f"{kind}_{season_type}_{week:02d}"
        write_parquet(frame, weekly_path(season, kind, season_type, week))
        record(name, endpoint, params, frame)
        checkpoint_coverage(name, _game_ids(games))
    return pd.DataFrame(manifest)


def read_weekly(season: int, kind: str) -> pd.DataFrame:
    """Concatenate every stored weekly frame of one kind for a season."""
    files = sorted(players_dir(season).glob(f"{kind}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {kind} parquet for {season}, run ingest-players")
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def read_season_source(season: int, name: str) -> pd.DataFrame:
    return pd.read_parquet(players_dir(season) / f"{name}.parquet")
