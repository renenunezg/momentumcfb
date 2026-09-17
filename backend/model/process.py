"""Frozen pregame margin correction from earlier games' process evidence.

Four prior-season games of support stabilize each team feature.
The symmetric ridge correction was fitted on 2020-2022 production-prior
replays, with its 14 features and penalty fixed before 2023-2025 validation.
No market data, target-game outcomes or 2026 outcomes enter the coefficients.
"""

from datetime import datetime, timezone
from hashlib import sha256

import numpy as np
import pandas as pd

from backend.etl import store

MODEL_VERSION = "process_margin_v1"
SCHEMA_VERSION = 1
PRIOR_GAMES = 4.0
RIDGE_LAMBDA = 1000.0
FEATURE_COLUMNS = (
    "offense_non_turnover_epa_per_play",
    "defense_non_turnover_epa_per_play_allowed",
    "offense_success_rate",
    "defense_success_rate_allowed",
    "offense_explosive_rate",
    "defense_explosive_rate_allowed",
    "offense_turnovers",
    "defense_turnovers_forced",
    "offense_touchdowns_per_opportunity",
    "defense_touchdowns_per_opportunity_allowed",
    "offense_average_start_yards_to_goal",
    "defense_average_start_yards_to_goal_allowed",
    "offense_seconds_per_play",
    "offense_plays_per_possession",
)
FEATURE_SCHEMA = "|".join(FEATURE_COLUMNS)
FALLBACK_COLUMNS = tuple(f"fallback_{column}" for column in FEATURE_COLUMNS)
# Standard deviations and coefficients from the frozen development replay.
# There is no centering or intercept: swapping teams negates the correction.
FEATURE_SCALE = np.array(
    [
        0.1402487029801564,
        0.147509159101797,
        0.07446374853422055,
        0.0690016438587127,
        0.04112197459637556,
        0.040956396049018726,
        0.6034664304256333,
        0.5387747814650563,
        0.15499554903058999,
        0.13874911224786501,
        3.618451169057648,
        4.315330831671717,
        6.011730902094625,
        0.6135463046139747,
    ]
)
COEFFICIENTS = np.array(
    [
        -0.5058855674261125,
        0.12308033012555662,
        0.009041714568504507,
        -0.1268085104185201,
        -0.2240290276502541,
        0.3292695981311714,
        0.32158742437257787,
        -0.37132836710499945,
        0.11769648565561472,
        0.5719825566176725,
        -0.5122218867478451,
        0.4699831520430449,
        -0.10467919637392321,
        0.38281516516360536,
    ]
)
PRIOR_METADATA = (
    "season",
    "source_season",
    "model_version",
    "schema_version",
    "prior_games",
    "feature_schema",
    "built_at",
    "source_last_game_at",
    "source_feature_sha256",
    "source_schedule_sha256",
)


def _features(team_games: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "team", *FEATURE_COLUMNS}
    missing = sorted(required - set(team_games.columns))
    if missing:
        raise ValueError("process features are missing columns: " + ", ".join(missing))
    if (
        team_games["team"].isna().any()
        or team_games.duplicated(["game_id", "team"]).any()
    ):
        raise ValueError("process features require one identified row per team-game")
    frame = team_games[["game_id", "team", *FEATURE_COLUMNS]].copy()
    frame[list(FEATURE_COLUMNS)] = frame[list(FEATURE_COLUMNS)].replace(
        [np.inf, -np.inf], np.nan
    )
    return frame


def validate_process_prior(frame: pd.DataFrame, season: int) -> None:
    """Reject incomplete, incompatible or ambiguous frozen process artifacts."""
    required = {"team", *PRIOR_METADATA, *FEATURE_COLUMNS, *FALLBACK_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("process prior is missing columns: " + ", ".join(missing))
    if frame.empty or frame["team"].isna().any() or frame["team"].duplicated().any():
        raise ValueError("process prior requires one identified row per team")
    expected = {
        "season": season,
        "source_season": season - 1,
        "model_version": MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "prior_games": PRIOR_GAMES,
        "feature_schema": FEATURE_SCHEMA,
    }
    if any(not frame[column].eq(value).all() for column, value in expected.items()):
        raise ValueError("process prior has incompatible season, schema or model")
    if frame[list(PRIOR_METADATA)].isna().any().any() or any(
        frame[column].nunique() != 1 for column in PRIOR_METADATA
    ):
        raise ValueError("process prior has inconsistent source metadata")
    if np.isinf(frame[list(FEATURE_COLUMNS)].to_numpy(float)).any():
        raise ValueError("process prior team means cannot be infinite")
    fallback = frame[list(FALLBACK_COLUMNS)].to_numpy(float)
    if not np.isfinite(fallback).all() or not (fallback == fallback[0]).all():
        raise ValueError("process prior requires one finite fallback per feature")
    for column in ("built_at", "source_last_game_at"):
        timestamp = pd.Timestamp(frame[column].iloc[0])
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError("process prior timestamps must be timezone-aware")
    for column in ("source_feature_sha256", "source_schedule_sha256"):
        digest = str(frame[column].iloc[0])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("process prior requires source SHA256 digests")


def build_process_prior(season: int) -> pd.DataFrame:
    """Summarize cached previous-season features without fetching or writing data.

    Previous-season means include all observed opponents and postseason games,
    matching the validated experiment; missing team means use pooled game means.
    Source hashes identify the retrospective data vintage used by the artifact.
    """
    year = season - 1
    feature_path = store.PROCESSED_DIR / "team_games" / f"{year}.parquet"
    schedule_path = store.raw_path("games", year)
    source = pd.read_parquet(feature_path)
    games = pd.read_parquet(schedule_path)
    if source.empty or not source["season"].eq(year).all():
        raise ValueError("process prior requires previous-season team-game features")
    if games["id"].duplicated().any() or not games["season"].eq(year).all():
        raise ValueError(
            "process prior requires an unambiguous previous-season schedule"
        )
    features = _features(source)
    context = games.set_index("id").reindex(features["game_id"])
    dates = pd.to_datetime(context["start_date"], utc=True, errors="coerce")
    if dates.isna().any() or not context["completed"].eq(True).all():
        raise ValueError("process prior contains unscheduled or incomplete games")
    means = features[list(FEATURE_COLUMNS)].mean()
    frame = features.groupby("team", sort=True)[list(FEATURE_COLUMNS)].mean()
    frame = frame.reset_index()
    for column in FEATURE_COLUMNS:
        frame[f"fallback_{column}"] = means[column]
    frame["season"] = season
    frame["source_season"] = year
    frame["model_version"] = MODEL_VERSION
    frame["schema_version"] = SCHEMA_VERSION
    frame["prior_games"] = PRIOR_GAMES
    frame["feature_schema"] = FEATURE_SCHEMA
    frame["built_at"] = datetime.now(timezone.utc)
    frame["source_last_game_at"] = dates.max()
    frame["source_feature_sha256"] = sha256(feature_path.read_bytes()).hexdigest()
    frame["source_schedule_sha256"] = sha256(schedule_path.read_bytes()).hexdigest()
    validate_process_prior(frame, season)
    return frame


def load_process_prior(season: int) -> pd.DataFrame:
    """Load required frozen state; never silently reconstruct a missing prior."""
    try:
        frame = store.read_preseason_forecast_artifact(season, 1, "process_prior")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{season}: frozen preseason process_prior is required; build it with "
            f"`python -m backend.model.process --season {season}` from cached "
            "previous-season features and install the reviewed runtime bundle"
        ) from exc
    validate_process_prior(frame, season)
    return frame


def process_margin_adjustments(
    projections: pd.DataFrame,
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    prior: pd.DataFrame,
    forecast_week: int,
    as_of,
) -> pd.Series:
    """Return raw symmetric point corrections keyed by target game ID.

    The caller applies its nonnegative-score boundary and recomputes all derived
    scores/probabilities from the applied correction, retaining the same total.
    Future results and games in the forecast's own model week never contribute.
    """
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("process forecast cutoff must be timezone-aware")
    seasons = games["season"].dropna().unique()
    if len(seasons) != 1:
        raise ValueError("process forecast requires one schedule season")
    validate_process_prior(prior, int(seasons[0]))
    safe_start = cutoff - pd.Timedelta(hours=8)
    if pd.Timestamp(prior["source_last_game_at"].iloc[0]) >= safe_start:
        raise ValueError("process prior contains games after the forecast cutoff")
    if games["game_id"].duplicated().any() or projections["game_id"].duplicated().any():
        raise ValueError("process forecast requires unique schedule and target games")
    if projections.empty:
        return pd.Series(dtype=float, name="process_margin_adjustment")
    features = _features(team_games)
    starts = pd.to_datetime(games["start_date"], utc=True, errors="raise")
    eligible = games.loc[
        games["completed"].eq(True)
        & games["model_week"].lt(forecast_week)
        & starts.lt(safe_start),
        "game_id",
    ]
    past = features[features["game_id"].isin(eligible)]
    columns = list(FEATURE_COLUMNS)
    sums = past.groupby("team", sort=True)[columns].sum()
    counts = past.groupby("team", sort=True)[columns].count()
    means = pd.Series(
        prior[list(FALLBACK_COLUMNS)].iloc[0].to_numpy(float), index=columns
    )
    previous = prior.set_index("team")[columns]
    previous = previous.reindex(previous.index.union(sums.index)).fillna(means)
    current = (sums.reindex(previous.index).fillna(0) + PRIOR_GAMES * previous) / (
        counts.reindex(previous.index).fillna(0) + PRIOR_GAMES
    )
    home = current.reindex(projections["home_team"]).fillna(means).to_numpy(float)
    away = current.reindex(projections["away_team"]).fillna(means).to_numpy(float)
    correction = ((home - away) / FEATURE_SCALE) @ COEFFICIENTS
    if not np.isfinite(correction).all():
        raise ValueError("process forecast produced nonfinite adjustments")
    return pd.Series(
        correction,
        index=projections["game_id"].to_numpy(),
        name="process_margin_adjustment",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a cached frozen process prior")
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()
    artifact = build_process_prior(args.season)
    store.write_processed(
        artifact, "preseason", "process_prior", f"{args.season}_01.parquet"
    )
