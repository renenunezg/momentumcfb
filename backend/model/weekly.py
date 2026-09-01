from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backend.etl import store
from backend.features.scoring import SCORING_COLUMNS, build_weekly_scoring_games
from backend.model.joint_scoring import fit_joint_scoring
from backend.model.market_blend import add_market_informed_margins
from backend.model.preseason import (
    MISSING_INPUT_COLUMNS,
    load_preseason_ratings,
    strength_prior_means_from_ratings,
)
from backend.model.unit_ratings import fit_unit_ratings
from backend.odds.markets import compare_priced_offers, flatten_odds_api_offers

MAX_MISSING_INPUT_COUNT = len(MISSING_INPUT_COLUMNS)


class WeeklyForecastNotReady(RuntimeError):
    """The scheduled weekly run is early or its source data is incomplete."""


@dataclass(frozen=True, slots=True)
class WeeklyForecastResult:
    week: int
    ratings: pd.DataFrame
    unit_ratings: pd.DataFrame
    projections: pd.DataFrame
    market_offers: pd.DataFrame
    market_comparisons: pd.DataFrame
    log_directory: Path


def load_weekly_games(season: int) -> pd.DataFrame:
    """Load the current schedule plus completed-game scoring features."""
    raw_games = store.read_games(season)
    try:
        team_games = store.read_processed("team_games", f"{season}.parquet")
    except FileNotFoundError:
        team_games = pd.DataFrame(
            columns=[
                "game_id",
                "team",
                "offense_possessions",
                "offense_epa_total",
                "game_possessions",
            ]
        )
    return build_weekly_scoring_games(raw_games, team_games)


def resolve_forecast_week(
    games: pd.DataFrame,
    requested_week: int | None,
    as_of: datetime,
) -> int:
    """Resolve the first entirely unstarted period after completed play."""
    if requested_week is not None:
        if requested_week < 1:
            raise ValueError("forecast week must be positive")
        return requested_week
    completed = games[games["completed"].fillna(False).astype(bool)]
    if completed.empty:
        raise WeeklyForecastNotReady(
            "no completed D1 games are available; use the preseason forecast "
            "until Week 0 is final"
        )

    periods = (
        games.groupby("model_week", as_index=False)
        .agg(first_kickoff=("start_date", "min"))
        .sort_values("first_kickoff", kind="stable")
    )
    completed_mask = games["completed"].fillna(False).astype(bool)
    for period in periods.itertuples():
        group = games[games["model_week"].eq(period.model_week)]
        group_completed = completed_mask.loc[group.index]
        if group_completed.all():
            continue
        started = group["start_date"].le(as_of) | group_completed
        if started.any():
            remaining = int((~group_completed).sum())
            raise WeeklyForecastNotReady(
                f"model Week {int(period.model_week)} has started but "
                f"{remaining} D1 games are not complete"
            )
        return int(period.model_week)
    raise WeeklyForecastNotReady("the schedule has no future D1 forecast period")


def resolve_ready_forecast_week(
    season: int,
    requested_week: int | None = None,
    as_of: datetime | None = None,
) -> int:
    """Resolve a scheduled run without fitting or consuming an odds request."""
    resolved_at = as_of or datetime.now(timezone.utc)
    if resolved_at.tzinfo is None or resolved_at.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    resolved_at = resolved_at.astimezone(timezone.utc)
    games = load_weekly_games(season)
    forecast_week = resolve_forecast_week(games, requested_week, resolved_at)
    _validate_weekly_inputs(games, forecast_week, resolved_at)
    return forecast_week


def _validate_weekly_inputs(
    games: pd.DataFrame,
    forecast_week: int,
    as_of: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = games["completed"].fillna(False).astype(bool)
    prior = games[games["model_week"].lt(forecast_week)]
    incomplete_prior = prior[~prior["completed"].fillna(False).astype(bool)]
    if not incomplete_prior.empty:
        descriptions = ", ".join(
            f"{row.away_team} at {row.home_team} ({int(row.game_id)})"
            for row in incomplete_prior.head(8).itertuples()
        )
        raise WeeklyForecastNotReady(
            f"{len(incomplete_prior)} earlier D1 games are not complete: {descriptions}"
        )

    prior_completed = games[games["model_week"].lt(forecast_week) & completed]
    missing_features = prior_completed[
        prior_completed[SCORING_COLUMNS].isna().any(axis=1)
    ]
    fbs_involved = missing_features[
        missing_features["home_classification"].str.lower().eq("fbs")
        | missing_features["away_classification"].str.lower().eq("fbs")
    ]
    if not fbs_involved.empty:
        descriptions = ", ".join(
            f"{row.away_team} at {row.home_team} ({int(row.game_id)})"
            for row in fbs_involved.head(8).itertuples()
        )
        raise WeeklyForecastNotReady(
            f"{len(fbs_involved)} completed FBS-involved games lack scoring "
            f"features: "
            f"{descriptions}"
        )
    missing_features = missing_features[
        [
            "game_id",
            "season",
            "week",
            "model_week",
            "start_date",
            "away_team",
            "away_classification",
            "home_team",
            "home_classification",
        ]
    ].copy()

    target = games[games["model_week"].eq(forecast_week)].copy()
    if target.empty:
        raise WeeklyForecastNotReady(
            f"no unplayed future D1 games are scheduled for model Week {forecast_week}"
        )
    started_target = target[
        target["completed"].fillna(False).astype(bool) | target["start_date"].le(as_of)
    ]
    if not started_target.empty:
        descriptions = ", ".join(
            f"{row.away_team} at {row.home_team} ({int(row.game_id)})"
            for row in started_target.head(8).itertuples()
        )
        raise WeeklyForecastNotReady(
            f"model Week {forecast_week} has already started: {descriptions}"
        )
    return target, missing_features


def _team_context(games: pd.DataFrame, preseason_ratings: pd.DataFrame) -> pd.DataFrame:
    home = games[
        [
            "home_team_id",
            "home_team",
            "home_conference",
            "home_classification",
        ]
    ].rename(
        columns={
            "home_team_id": "team_id",
            "home_team": "team",
            "home_conference": "schedule_conference",
            "home_classification": "schedule_classification",
        }
    )
    away = games[
        [
            "away_team_id",
            "away_team",
            "away_conference",
            "away_classification",
        ]
    ].rename(
        columns={
            "away_team_id": "team_id",
            "away_team": "team",
            "away_conference": "schedule_conference",
            "away_classification": "schedule_classification",
        }
    )
    schedule = pd.concat([home, away], ignore_index=True).drop_duplicates(
        "team_id", keep="last"
    )
    prior_columns = [
        "team_id",
        "conference",
        "classification",
        "missing_input_count",
    ]
    prior = preseason_ratings.reindex(columns=prior_columns).drop_duplicates(
        "team_id", keep="last"
    )
    context = schedule.merge(prior, on="team_id", how="left", validate="one_to_one")
    context["conference"] = context["schedule_conference"].fillna(context["conference"])
    context["classification"] = context["schedule_classification"].fillna(
        context["classification"]
    )
    context["classification"] = context["classification"].str.lower()
    context["missing_input_count"] = (
        pd.to_numeric(context["missing_input_count"], errors="coerce")
        .fillna(MAX_MISSING_INPUT_COUNT)
        .astype(int)
    )
    return context


def _decorate_outputs(
    ratings: pd.DataFrame,
    projections: pd.DataFrame,
    target: pd.DataFrame,
    context: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_team = context.set_index("team_id")
    ratings["conference"] = ratings["team_id"].map(by_team["conference"])
    ratings["classification"] = ratings["team_id"].map(by_team["classification"])
    ratings["missing_input_count"] = ratings["team_id"].map(
        by_team["missing_input_count"]
    )

    by_game = target.set_index("game_id")
    projections["start_date"] = projections["game_id"].map(by_game["start_date"])
    projections["home_classification"] = projections["game_id"].map(
        by_game["home_classification"]
    )
    projections["away_classification"] = projections["game_id"].map(
        by_game["away_classification"]
    )
    projections["home_missing_input_count"] = projections["home_team_id"].map(
        by_team["missing_input_count"]
    )
    projections["away_missing_input_count"] = projections["away_team_id"].map(
        by_team["missing_input_count"]
    )
    return ratings, projections


def _odds_frames(odds_client, target: pd.DataFrame):
    if odds_client is None:
        events = pd.DataFrame()
        offers, matches = flatten_odds_api_offers(events, target)
        return events, offers, matches, None

    starts = pd.to_datetime(target["start_date"], utc=True)
    snapshot = odds_client.get_ncaaf_odds(
        (starts.min() - timedelta(hours=2)).to_pydatetime(),
        (starts.max() + timedelta(hours=2)).to_pydatetime(),
    )
    events = pd.DataFrame(snapshot.events)
    events["source_fetched_at"] = snapshot.fetched_at.isoformat()
    events["execution_eligibility_verified"] = bool(snapshot.configured_bookmakers)
    offers, matches = flatten_odds_api_offers(events, target)
    return events, offers, matches, snapshot


def run_weekly_forecast(
    season: int,
    week: int | None = None,
    odds_client=None,
    require_market: bool = False,
    as_of: datetime | None = None,
) -> WeeklyForecastResult:
    """Fit and persist one leakage-safe in-season weekly forecast."""
    created_at = as_of or datetime.now(timezone.utc)
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    created_at = created_at.astimezone(timezone.utc)

    games = load_weekly_games(season)
    forecast_week = resolve_forecast_week(games, week, created_at)
    target, excluded_training_games = _validate_weekly_inputs(
        games, forecast_week, created_at
    )

    preseason_ratings, prior_source = load_preseason_ratings(season)
    strength_priors = strength_prior_means_from_ratings(preseason_ratings)
    fitted = fit_joint_scoring(
        games,
        forecast_week,
        created_at,
        strength_prior_means=strength_priors,
    )
    ratings = pd.DataFrame(rating.to_record() for rating in fitted.ratings())
    projections = pd.DataFrame(
        projection.to_record() for projection in fitted.project(target)
    )
    if len(projections) != len(target):
        raise ValueError(
            f"projected {len(projections)} of {len(target)} eligible games"
        )
    context = _team_context(games, preseason_ratings)
    ratings, projections = _decorate_outputs(ratings, projections, target, context)

    unit_games = store.read_processed("unit_games", f"{season}.parquet")
    units = fit_unit_ratings(
        unit_games,
        games,
        forecast_week,
        created_at,
    )
    unit_ratings = pd.DataFrame(units.to_records())
    unit_ratings["source_season"] = season
    unit_ratings["unit_history_missing"] = False

    odds_events, offers, matches, snapshot = _odds_frames(odds_client, target)
    priced_spreads = offers[offers["market"].eq("spreads") & offers["point"].notna()]
    if require_market and priced_spreads.empty:
        raise ValueError(
            "the Odds API returned no priced spread matched to the forecast week"
        )
    projections = add_market_informed_margins(projections, offers)
    comparisons = compare_priced_offers(projections, offers)

    coverage = target[
        [
            "game_id",
            "season",
            "week",
            "model_week",
            "start_date",
            "away_team",
            "home_team",
        ]
    ].copy()
    matched_ids = set(pd.to_numeric(offers["game_id"], errors="coerce").dropna())
    coverage["odds_market_matched"] = coverage["game_id"].isin(matched_ids)
    coverage["missing_market"] = ~coverage["odds_market_matched"]
    manifest = pd.DataFrame(
        [
            {
                "season": season,
                "week": forecast_week,
                "forecast_created_at": created_at.isoformat(),
                "preseason_prior_source": prior_source,
                "completed_prior_games": int(
                    (
                        games["model_week"].lt(forecast_week)
                        & games["completed"].fillna(False).astype(bool)
                    ).sum()
                ),
                "model_training_games": int(
                    (
                        games["model_week"].lt(forecast_week)
                        & games["completed"].fillna(False).astype(bool)
                        & ~games[SCORING_COLUMNS].isna().any(axis=1)
                    ).sum()
                ),
                "week_zero_training_games": int(
                    (
                        games["model_week"].eq(0)
                        & games["completed"].fillna(False).astype(bool)
                        & ~games[SCORING_COLUMNS].isna().any(axis=1)
                    ).sum()
                ),
                "excluded_fcs_training_games": len(excluded_training_games),
                "projection_games": len(projections),
                "odds_events": len(odds_events),
                "matched_odds_events": int(
                    matches["matched"].sum() if "matched" in matches else 0
                ),
                "priced_spread_games": int(priced_spreads["game_id"].nunique()),
                "requests_remaining": (
                    snapshot.requests_remaining if snapshot is not None else None
                ),
                "requests_used": (
                    snapshot.requests_used if snapshot is not None else None
                ),
                "request_cost": (
                    snapshot.request_cost if snapshot is not None else None
                ),
            }
        ]
    )
    log_directory = store.write_forecast_outputs(
        "weekly",
        season,
        forecast_week,
        created_at,
        {
            "ratings": ratings,
            "unit_ratings": unit_ratings,
            "projections": projections,
            "schedule_coverage": coverage,
            "excluded_training_games": excluded_training_games,
            "market_offers": offers,
            "market_comparisons": comparisons,
            "odds_match_coverage": matches,
            "odds_api_events": odds_events,
            "source_manifest": manifest,
        },
    )
    return WeeklyForecastResult(
        week=forecast_week,
        ratings=ratings,
        unit_ratings=unit_ratings,
        projections=projections,
        market_offers=offers,
        market_comparisons=comparisons,
        log_directory=log_directory,
    )
