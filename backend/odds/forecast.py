"""Load one immutable weekly run without mixing canonical forecast files."""

from pathlib import Path

import pandas as pd

from backend.serving.anchors import validate_serving_anchors


def load_weekly_capture_forecast(directory: str | Path, season: int):
    directory = Path(directory)
    frames = {
        name: pd.read_parquet(directory / f"{name}.parquet")
        for name in (
            "source_manifest",
            "ratings",
            "unit_ratings",
            "projections",
            "market_comparisons",
            "schedule_coverage",
        )
    }
    manifest = frames["source_manifest"]
    if len(manifest) != 1 or int(manifest.iloc[0]["season"]) != season:
        raise ValueError(
            "weekly capture requires one manifest for the requested season"
        )
    row = manifest.iloc[0]
    week = int(row["week"])
    created = pd.to_datetime(row["forecast_created_at"], utc=True, errors="coerce")
    if week < 1 or pd.isna(created):
        raise ValueError("weekly manifest has an invalid week or forecast timestamp")
    for name in ("ratings", "unit_ratings", "projections"):
        frame = frames[name]
        if frame.empty or not frame["season"].eq(season).all():
            raise ValueError(f"weekly {name} is empty or contains another season")
        if not frame["week"].eq(week).all():
            raise ValueError(f"weekly {name} contains another model week")
        if not pd.to_datetime(frame["as_of"], utc=True).eq(created).all():
            raise ValueError(f"weekly {name} comes from a different forecast run")
    projections = frames["projections"]
    ids = set(projections["game_id"])
    if len(projections) != int(row["projection_games"]):
        raise ValueError("weekly projection count disagrees with its manifest")
    for name in ("market_comparisons", "schedule_coverage"):
        frame = frames[name]
        if frame["game_id"].duplicated().any() or set(frame["game_id"]) != ids:
            raise ValueError(f"weekly {name} and projections cover different games")
    if set(frames["ratings"]["team_id"]) != set(frames["unit_ratings"]["team_id"]):
        raise ValueError("weekly ratings and unit ratings cover different teams")
    schedule = (
        frames["schedule_coverage"]
        .set_index("game_id")
        .loc[projections["game_id"]]
        .reset_index()
    )
    starts = pd.to_datetime(schedule["start_date"], utc=True, errors="coerce")
    projected_starts = pd.to_datetime(projections["start_date"], utc=True)
    if starts.isna().any() or not starts.eq(projected_starts).all():
        raise ValueError("weekly schedule and projections disagree on kickoff times")
    if starts.le(created).any():
        raise ValueError("weekly forecast was created after a target game started")
    if not schedule["model_week"].eq(week).all():
        raise ValueError("weekly schedule contains another model week")
    anchors = validate_serving_anchors(
        projections.rename(columns={"week": "model_week"}), str(directory)
    )
    # Derive outcome-free anchors from these exact frozen rows. Never load an
    # independently selected opening-week anchor file or refit predictions.
    frames["anchors"] = anchors
    frames["schedule_coverage"] = schedule
    return frames


def check_weekly_capture_readiness(directory, season, *, as_of, max_age_hours):
    try:
        frames = load_weekly_capture_forecast(directory, season)
        created = pd.to_datetime(
            frames["source_manifest"].iloc[0]["forecast_created_at"], utc=True
        )
        age = (pd.Timestamp(as_of) - created).total_seconds() / 3600
        if age < -5 / 60 or age > max_age_hours:
            raise ValueError(
                f"weekly forecast age {age:.1f}h is outside 0-{max_age_hours:.1f}h"
            )
    except (OSError, ValueError, KeyError) as exc:
        return [f"weekly forecast is not ready: {exc}"], [], []
    return (
        [],
        [],
        [
            f"weekly forecast: {len(frames['projections'])} games, created {created.isoformat()}",
            f"weekly anchors: {len(frames['anchors'])} matching frozen projections",
            "weekly source run uses frozen preseason priors; forecast freshness applies",
        ],
    )
