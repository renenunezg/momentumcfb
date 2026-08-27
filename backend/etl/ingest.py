import json
import os
import re
from datetime import datetime, timezone

import pandas as pd

from backend.cfbd.client import CFBDClient
from backend.config import MAX_REGULAR_WEEK, RAW_DIR
from backend.odds.client import OddsAPIClient

SEASON_TYPES = ("regular", "postseason")

PRESEASON_SOURCES = {
    "teams": ("/teams", lambda season: {"year": season}),
    "games": (
        "/games",
        lambda season: {"year": season, "seasonType": "regular"},
    ),
    "talent": ("/talent", lambda season: {"year": season}),
    "returning": ("/player/returning", lambda season: {"year": season}),
    "portal": ("/player/portal", lambda season: {"year": season}),
    "coaches": ("/coaches", lambda season: {"year": season}),
    "recruiting": ("/recruiting/teams", lambda season: {"year": season}),
    "lines": ("/lines", lambda season: {"year": season}),
    "prior_coaches": ("/coaches", lambda season: {"year": season - 1}),
}


def to_snake(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(column)).lower()
        for column in df.columns
    ]
    return df


def write_parquet(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ingest_cfbd_plays(
    client: CFBDClient, season: int, only_week: int | None = None
) -> None:
    for season_type in SEASON_TYPES:
        weeks = [only_week] if only_week is not None else range(1, MAX_REGULAR_WEEK + 1)
        for week in weeks:
            rows = client.get(
                "/plays", {"year": season, "week": week, "seasonType": season_type}
            )
            if not rows:
                continue
            df = to_snake(pd.DataFrame(rows))
            df["season"] = season
            df["week"] = week
            df["season_type"] = season_type
            df["pbp_source"] = "cfbd"
            write_parquet(
                df,
                RAW_DIR / "pbp" / str(season) / f"{season_type}_{week:02d}.parquet",
            )
            print(f"plays {season} {season_type} week {week}: {len(df)} rows")


def ingest_games(client: CFBDClient, season: int) -> None:
    rows = client.get("/games", {"year": season, "seasonType": "both"})
    games = to_snake(pd.DataFrame(rows))
    write_parquet(games, RAW_DIR / "games" / f"{season}.parquet")
    print(f"games {season}: {len(rows)} rows")


def ingest_lines(client: CFBDClient, season: int) -> None:
    rows = client.get("/lines", {"year": season})
    df = pd.DataFrame(
        {
            "game_id": [r["id"] for r in rows],
            "lines": [r.get("lines") or [] for r in rows],
        }
    )
    write_parquet(df, RAW_DIR / "lines" / f"{season}.parquet")
    print(f"lines {season}: {len(df)} rows")


def ingest_talent(client: CFBDClient, season: int) -> None:
    rows = client.get("/talent", {"year": season})
    write_parquet(
        to_snake(pd.DataFrame(rows)), RAW_DIR / "talent" / f"{season}.parquet"
    )


def ingest_returning(client: CFBDClient, season: int) -> None:
    rows = client.get("/player/returning", {"year": season})
    write_parquet(
        to_snake(pd.DataFrame(rows)), RAW_DIR / "returning" / f"{season}.parquet"
    )


def ingest_preseason_sources(
    client: CFBDClient,
    season: int,
    odds_client: OddsAPIClient | None = None,
) -> pd.DataFrame:
    """Snapshot every source used by the preseason forecast with retrieval times."""
    destination = RAW_DIR / "preseason" / str(season)
    manifest_rows = []
    sources = dict(PRESEASON_SOURCES)
    sources["prior_talent"] = ("/talent", lambda year: {"year": year - 1})

    games_frame = None
    for name, (endpoint, build_params) in sources.items():
        params = build_params(season)
        rows = client.get(endpoint, params)
        fetched_at = datetime.now(timezone.utc).isoformat()
        frame = to_snake(pd.DataFrame(rows))
        frame["source_endpoint"] = endpoint
        frame["source_fetched_at"] = fetched_at
        write_parquet(frame, destination / f"{name}.parquet")
        if name == "games":
            games_frame = frame
        manifest_rows.append(
            {
                "source": name,
                "endpoint": endpoint,
                "params": json.dumps(params, sort_keys=True),
                "source_fetched_at": fetched_at,
                "row_count": len(frame),
                "is_empty": frame.empty,
            }
        )
        print(f"preseason {season} {name}: {len(frame)} rows")

    if odds_client is not None:
        if games_frame is None:
            raise ValueError("the schedule must be fetched before odds")
        week_one = games_frame[games_frame["week"].eq(1)].copy()
        starts = pd.to_datetime(week_one["start_date"], utc=True).dropna()
        if starts.empty:
            raise ValueError(f"no {season} Week 1 schedule is available")
        snapshot = odds_client.get_ncaaf_odds(
            starts.min().to_pydatetime(), starts.max().to_pydatetime()
        )
        fetched_at = snapshot.fetched_at.isoformat()
        odds = to_snake(pd.DataFrame(snapshot.events))
        odds["source_endpoint"] = "/v4/sports/americanfootball_ncaaf/odds"
        odds["source_fetched_at"] = fetched_at
        odds["execution_eligibility_verified"] = bool(snapshot.configured_bookmakers)
        write_parquet(odds, destination / "odds_api.parquet")
        manifest_rows.append(
            {
                "source": "odds_api",
                "endpoint": "/v4/sports/americanfootball_ncaaf/odds",
                "params": json.dumps(
                    {
                        "markets": ["spreads", "totals"],
                        "odds_format": "american",
                        "configured_bookmakers": snapshot.configured_bookmakers,
                    },
                    sort_keys=True,
                ),
                "source_fetched_at": fetched_at,
                "row_count": len(odds),
                "is_empty": odds.empty,
                "requests_remaining": snapshot.requests_remaining,
                "requests_used": snapshot.requests_used,
                "request_cost": snapshot.request_cost,
            }
        )
        print(
            f"preseason {season} odds_api: {len(odds)} events, "
            f"cost={snapshot.request_cost}, "
            f"remaining={snapshot.requests_remaining}"
        )

    manifest = pd.DataFrame(manifest_rows)
    write_parquet(manifest, destination / "manifest.parquet")
    return manifest


def ingest_season(
    client: CFBDClient,
    season: int,
    only_week: int | None = None,
) -> None:
    ingest_games(client, season)
    ingest_cfbd_plays(client, season, only_week)
    ingest_lines(client, season)
    ingest_talent(client, season)
    ingest_returning(client, season)
