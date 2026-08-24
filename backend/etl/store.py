import pandas as pd
import pyarrow.parquet

from backend.config import PROCESSED_DIR, RAW_DIR


def read_season_pbp(season: int) -> pd.DataFrame:
    season_dir = RAW_DIR / "pbp" / str(season)
    files = sorted(season_dir.glob("regular_*.parquet"))
    files.extend(sorted(season_dir.glob("postseason_*.parquet")))
    if not files:
        raise FileNotFoundError(f"no pbp parquet for {season}, run ingest first")

    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        if "pbp_source" not in frame:
            frame["pbp_source"] = "cfbd"
        elif not frame["pbp_source"].fillna("").eq("cfbd").all():
            raise ValueError(f"non-CFBD PBP rows found in {path}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def read_games(season: int) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "games" / f"{season}.parquet")


def read_lines(season: int) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "lines" / f"{season}.parquet")


def read_talent(season: int) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "talent" / f"{season}.parquet")


def read_returning(season: int) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "returning" / f"{season}.parquet")


def read_preseason_source(season: int, source: str) -> pd.DataFrame:
    return pd.read_parquet(
        RAW_DIR / "preseason" / str(season) / f"{source}.parquet"
    )


def write_processed(df: pd.DataFrame, *parts: str) -> None:
    path = PROCESSED_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_processed(
    *parts: str, columns: list[str] | None = None
) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR.joinpath(*parts), columns=columns)


def processed_names(*parts: str) -> list[str]:
    """Parquet file stems stored under a processed directory, if it exists."""
    directory = PROCESSED_DIR.joinpath(*parts)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.parquet"))


def processed_columns(*parts: str) -> list[str]:
    """Column names of a processed artifact, from parquet metadata only."""
    return list(
        pyarrow.parquet.read_schema(PROCESSED_DIR.joinpath(*parts)).names
    )
