from datetime import datetime
from pathlib import Path

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


def raw_path(kind: str, season: int) -> Path:
    return RAW_DIR / kind / f"{season}.parquet"


def read_games(season: int) -> pd.DataFrame:
    return pd.read_parquet(raw_path("games", season))


def read_lines(season: int) -> pd.DataFrame:
    return pd.read_parquet(raw_path("lines", season))


def read_preseason_source(season: int, source: str) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "preseason" / str(season) / f"{source}.parquet")


def write_processed(df: pd.DataFrame, *parts: str) -> None:
    path = PROCESSED_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_processed(*parts: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR.joinpath(*parts), columns=columns)


def write_forecast_outputs(
    kind: str,
    season: int,
    week: int,
    created_at: datetime,
    outputs: dict[str, pd.DataFrame],
    canonical_prefix: tuple[str, ...] = (),
) -> Path:
    """Write each forecast frame to its canonical processed path and to an
    immutable timestamped run log under ``<kind>/forecast_log``."""
    filename = f"{season}_{week:02d}.parquet"
    timestamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    log_directory = (
        PROCESSED_DIR / kind / "forecast_log" / f"{season}_{week:02d}_{timestamp}"
    )
    log_directory.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        write_processed(frame, *canonical_prefix, name, filename)
        frame.to_parquet(log_directory / f"{name}.parquet", index=False)
    return log_directory


def read_preseason_forecast_artifact(
    season: int,
    week: int,
    name: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read a canonical forecast output or its immutable run-log copy."""
    filename = f"{season}_{week:02d}.parquet"
    canonical = PROCESSED_DIR / "preseason" / name / filename
    if canonical.exists():
        return pd.read_parquet(canonical, columns=columns)

    log_root = PROCESSED_DIR / "preseason" / "forecast_log"
    candidates = sorted(
        log_root.glob(f"{season}_{week:02d}_*/{name}.parquet"),
        key=lambda path: path.parent.name,
    )
    if not candidates:
        raise FileNotFoundError(canonical)
    return pd.read_parquet(candidates[-1], columns=columns)


def processed_names(*parts: str) -> list[str]:
    """Parquet file stems stored under a processed directory, if it exists."""
    directory = PROCESSED_DIR.joinpath(*parts)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.parquet"))


def processed_columns(*parts: str) -> list[str]:
    """Column names of a processed artifact, from parquet metadata only."""
    return list(pyarrow.parquet.read_schema(PROCESSED_DIR.joinpath(*parts)).names)
