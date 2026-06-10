import pandas as pd

from backend.cfbd.client import CFBDClient
from backend.config import MAX_REGULAR_WEEK, RAW_DIR

SEASON_TYPES = ("regular", "postseason")


def to_snake(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.replace(r"([a-z0-9])([A-Z])", r"\1_\2", regex=True).str.lower()
    )
    return df


def write_parquet(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


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
            write_parquet(df, RAW_DIR / "pbp" / str(season) / f"{season_type}_{week:02d}.parquet")
            print(f"plays {season} {season_type} week {week}: {len(df)} rows")


def ingest_games(client: CFBDClient, season: int) -> None:
    rows = client.get("/games", {"year": season, "seasonType": "both"})
    write_parquet(to_snake(pd.DataFrame(rows)), RAW_DIR / "games" / f"{season}.parquet")
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
    write_parquet(to_snake(pd.DataFrame(rows)), RAW_DIR / "talent" / f"{season}.parquet")


def ingest_returning(client: CFBDClient, season: int) -> None:
    rows = client.get("/player/returning", {"year": season})
    write_parquet(to_snake(pd.DataFrame(rows)), RAW_DIR / "returning" / f"{season}.parquet")


def ingest_season(client: CFBDClient, season: int, only_week: int | None = None) -> None:
    ingest_plays(client, season, only_week)
    ingest_games(client, season)
    ingest_lines(client, season)
    ingest_talent(client, season)
    ingest_returning(client, season)
