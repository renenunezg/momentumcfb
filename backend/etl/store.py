import pandas as pd

from backend.config import PROCESSED_DIR, RAW_DIR


def read_season_pbp(season: int) -> pd.DataFrame:
    season_dir = RAW_DIR / "pbp" / str(season)
    canonical = season_dir / "canonical.parquet"
    if canonical.exists():
        return pd.read_parquet(canonical)

    files = sorted(season_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no pbp parquet for {season}, run ingest first")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def read_games(season: int) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "games" / f"{season}.parquet")


def read_lines(season: int) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "lines" / f"{season}.parquet")


def read_talent(season: int) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "talent" / f"{season}.parquet")


def read_returning(season: int) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "returning" / f"{season}.parquet")


def write_processed(df: pd.DataFrame, *parts: str) -> None:
    path = PROCESSED_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_processed(*parts: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR.joinpath(*parts))
