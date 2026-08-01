import os

import pandas as pd
import pyarrow.parquet as parquet

from backend.cfbd.client import CFBDClient
from backend.config import MAX_REGULAR_WEEK, RAW_DIR
from backend.sportsdataverse.pbp import (
    CORE_SOURCE_COLUMNS,
    OPTIONAL_COLUMN_MAP,
    download_season_pbp,
    normalize_pbp,
)

SEASON_TYPES = ("regular", "postseason")


def to_snake(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.replace(r"([a-z0-9])([A-Z])", r"\1_\2", regex=True).str.lower()
    )
    return df


def write_parquet(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ingest_plays(client: CFBDClient, season: int, only_week: int | None = None) -> None:
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
            write_parquet(
                df,
                RAW_DIR
                / "pbp"
                / str(season)
                / f"{season_type}_{week:02d}.parquet",
            )
            print(f"plays {season} {season_type} week {week}: {len(df)} rows")


def _completed_fbs_games(games: pd.DataFrame) -> pd.DataFrame:
    completed = games["completed"].fillna(False).astype(bool)
    home_fbs = games["home_classification"].str.lower().eq("fbs")
    away_fbs = games["away_classification"].str.lower().eq("fbs")
    return games[completed & (home_fbs | away_fbs)].copy()


def _cfbd_missing_game_plays(
    client: CFBDClient,
    season: int,
    games: pd.DataFrame,
    present_game_ids: set[int],
) -> pd.DataFrame:
    targets = _completed_fbs_games(games)
    target_ids = set(pd.to_numeric(targets["id"], errors="coerce").dropna().astype(int))
    missing_ids = target_ids - present_game_ids
    if not missing_ids:
        return pd.DataFrame()

    missing_games = targets[targets["id"].isin(missing_ids)]
    fallback_frames = []
    for (season_type, week), group in missing_games.groupby(
        ["season_type", "week"], sort=True
    ):
        rows = client.get(
            "/plays",
            {"year": season, "week": int(week), "seasonType": season_type},
        )
        if not rows:
            continue
        week_plays = to_snake(pd.DataFrame(rows))
        game_ids = pd.to_numeric(week_plays["game_id"], errors="coerce")
        fallback = week_plays[game_ids.isin(group["id"])].copy()
        if fallback.empty:
            continue
        fallback["season"] = season
        fallback["week"] = int(week)
        fallback["season_type"] = season_type
        fallback["pbp_source"] = "cfbd_fallback"
        fallback["epa"] = fallback["ppa"]
        fallback["game_id"] = pd.to_numeric(
            fallback["game_id"], errors="raise"
        ).astype("Int64")
        fallback["id"] = pd.to_numeric(fallback["id"], errors="raise").astype(
            "Int64"
        )
        fallback["drive_id"] = fallback["drive_id"].astype("string")
        fallback["game_play_number"] = (
            fallback.groupby("game_id", sort=False).cumcount() + 1
        )
        write_parquet(
            fallback,
            RAW_DIR
            / "cfbd_fallback"
            / "pbp"
            / str(season)
            / f"{season_type}_{int(week):02d}.parquet",
        )
        fallback_frames.append(fallback)

    if not fallback_frames:
        return pd.DataFrame()
    return pd.concat(fallback_frames, ignore_index=True, sort=False)


def ingest_sportsdataverse_plays(
    client: CFBDClient, season: int, games: pd.DataFrame
) -> None:
    raw_path = RAW_DIR / "sportsdataverse" / "pbp" / f"{season}.parquet"
    normalized_path = RAW_DIR / "pbp" / str(season) / "canonical.parquet"
    download_season_pbp(season, raw_path)
    available = set(parquet.ParquetFile(raw_path).schema.names)
    selected = sorted(
        available & (set(CORE_SOURCE_COLUMNS) | set(OPTIONAL_COLUMN_MAP))
    )
    source = pd.read_parquet(raw_path, columns=selected)
    normalized = normalize_pbp(source)
    present_ids = set(
        pd.to_numeric(normalized["game_id"], errors="coerce").dropna().astype(int)
    )
    fallback = _cfbd_missing_game_plays(client, season, games, present_ids)
    combined = pd.concat([normalized, fallback], ignore_index=True, sort=False)
    write_parquet(combined, normalized_path)

    targets = _completed_fbs_games(games)
    target_ids = set(pd.to_numeric(targets["id"], errors="coerce").dropna().astype(int))
    covered_ids = set(
        pd.to_numeric(combined["game_id"], errors="coerce").dropna().astype(int)
    )
    uncovered = targets[targets["id"].isin(target_ids - covered_ids)]
    print(
        f"canonical plays {season}: {len(combined)} rows, "
        f"{combined['game_id'].nunique()} games, "
        f"{fallback['game_id'].nunique() if not fallback.empty else 0} CFBD fallback games"
    )
    if not uncovered.empty:
        descriptions = ", ".join(
            f"{row.away_team} at {row.home_team} ({int(row.id)})"
            for row in uncovered.itertuples()
        )
        print(f"WARNING: no play-by-play found for {descriptions}")


def ingest_games(client: CFBDClient, season: int) -> pd.DataFrame:
    rows = client.get("/games", {"year": season, "seasonType": "both"})
    games = to_snake(pd.DataFrame(rows))
    write_parquet(games, RAW_DIR / "games" / f"{season}.parquet")
    print(f"games {season}: {len(rows)} rows")
    return games


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
    write_parquet(to_snake(pd.DataFrame(rows)), RAW_DIR / "talent" / f"{season}.parquet")


def ingest_returning(client: CFBDClient, season: int) -> None:
    rows = client.get("/player/returning", {"year": season})
    write_parquet(to_snake(pd.DataFrame(rows)), RAW_DIR / "returning" / f"{season}.parquet")


def ingest_season(
    client: CFBDClient,
    season: int,
    only_week: int | None = None,
    pbp_source: str = "sportsdataverse",
) -> None:
    if pbp_source == "sportsdataverse":
        if only_week is not None:
            raise ValueError("--week is only supported with --pbp-source cfbd")
    elif pbp_source != "cfbd":
        raise ValueError(f"unsupported PBP source: {pbp_source}")

    games = ingest_games(client, season)
    if pbp_source == "sportsdataverse":
        ingest_sportsdataverse_plays(client, season, games)
    else:
        ingest_plays(client, season, only_week)
    ingest_lines(client, season)
    ingest_talent(client, season)
    ingest_returning(client, season)
