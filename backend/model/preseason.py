from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backend.etl import store
from backend.features.scoring import build_scoring_games
from backend.model.joint_scoring import (
    DEFAULT_CONFIG,
    fit_joint_scoring,
    margin_total_distribution,
)
from backend.model.market_blend import add_market_informed_margins
from backend.model.outputs import GameProjection
from backend.model.unit_ratings import COLUMNS as UNIT_RATING_COLUMNS
from backend.model.unit_ratings import MODEL_VERSION as UNIT_RATING_MODEL_VERSION
from backend.model.unit_ratings import UnitRatings, fit_unit_ratings
from backend.odds.markets import (
    OFFER_COLUMNS,
    _stored_sequence,
    compare_priced_offers,
    flatten_odds_api_offers,
)

MODEL_VERSION = "preseason_v4"
# Power weights were fitted by least squares on 364 FBS games from the 2022
# through 2025 Week 1 and Week 2 slates: the closing margin regressed on the
# home-minus-away difference of each feature. Leave-one-season-out over 2023
# through 2025, the fitted weights cut Week 1 walk-forward margin MAE from
# 15.57 to 13.99 and the Week 1 gap to the closing line from 8.6 to 5.0
# points. QB continuity and recruiting class points carried no power weight
# once talent was fitted, so they only shape the environment and uncertainty.
PREVIOUS_POWER_WEIGHT = 0.87
PREVIOUS_ENVIRONMENT_WEIGHT = 0.40
PREVIOUS_PACE_WEIGHT = 0.35
LATEST_TALENT_POINTS = 3.70
CURRENT_TALENT_POINTS = 3.30
RECRUITING_POINTS = 0.00
RETURNING_POINTS = 1.70
TRANSFER_QUALITY_POINTS = 1.00
TRANSFER_COUNT_POINTS = 0.35
QB_CONTINUITY_POINTS = 0.00
# Calibration walk-forwards carry only the previous fit, without talent or
# returning production, and that regime evaluated best at full weight.
HISTORICAL_CARRYOVER_WEIGHT = 1.00
QB_TRANSFER_ENVIRONMENT_POINTS = 0.80
COACH_CONTINUITY_POINTS = 0.35
BASE_OFFSEASON_POWER_SD = 6.05
MISSING_INPUT_COLUMNS = [
    "previous_rating_missing",
    "current_talent_missing",
    "latest_talent_missing",
    "returning_production_missing",
    "qb_continuity_missing",
    "coach_continuity_missing",
    "recruiting_missing",
    "transfer_data_missing",
    "injury_availability_missing",
]


@dataclass(frozen=True, slots=True)
class PreseasonForecastResult:
    ratings: pd.DataFrame
    unit_ratings: pd.DataFrame
    projections: pd.DataFrame
    schedule_coverage: pd.DataFrame
    market_offers: pd.DataFrame
    market_comparisons: pd.DataFrame
    log_directory: Path


def strength_prior_means_from_ratings(
    ratings: pd.DataFrame,
) -> dict[int, tuple[float, float]]:
    """Convert published preseason point ratings into weekly-engine priors.

    The joint engine stores offense and defense strengths per possession,
    while the preseason artifact publishes their points-per-game equivalents.
    The average expected possession count recovers the common scale used when
    those ratings were built. This keeps market data out of the rating state:
    only the pure preseason team ratings enter the weekly fit.
    """
    required = {
        "team_id",
        "offense_points",
        "defense_points",
        "expected_possessions",
    }
    missing = sorted(required - set(ratings.columns))
    if missing:
        raise ValueError(
            "preseason ratings are missing prior columns: " + ", ".join(missing)
        )
    if ratings.empty:
        raise ValueError("preseason ratings must not be empty")
    base_possessions = float(
        pd.to_numeric(ratings["expected_possessions"], errors="coerce").mean()
    )
    if not np.isfinite(base_possessions) or base_possessions <= 0:
        raise ValueError("preseason expected possessions must have a positive mean")

    means = {}
    for row in ratings.itertuples():
        offense = float(row.offense_points) / base_possessions
        defense = float(row.defense_points) / base_possessions
        if np.isfinite(offense) and np.isfinite(defense):
            means[int(row.team_id)] = (offense, defense)
    if not means:
        raise ValueError("preseason ratings contain no finite strength priors")
    return means


def load_preseason_ratings(season: int, week: int = 1) -> tuple[pd.DataFrame, str]:
    """Load the pure preseason prior locally or from the serving database."""
    try:
        ratings = store.read_processed(
            "preseason", "ratings", f"{season}_{week:02d}.parquet"
        )
        return ratings, "local_preseason_artifact"
    except FileNotFoundError:
        from sqlalchemy import text

        from backend.db import CFB_SCHEMA, engine

        query = text(
            "SELECT team_id, team, conference, classification, "
            "offense_points, defense_points, expected_possessions, "
            f"missing_input_count FROM {CFB_SCHEMA}.team_ratings "
            "WHERE season = :season AND week = :week"
        )
        with engine.connect() as connection:
            ratings = pd.read_sql(
                query,
                connection,
                params={"season": season, "week": week},
            )
        if ratings.empty:
            raise FileNotFoundError(
                f"no published preseason ratings for {season} Week {week}"
            )
        return ratings, "published_team_ratings"


def build_historical_carryover_priors(
    previous_fit,
    current_games: pd.DataFrame,
    power_weight: float = HISTORICAL_CARRYOVER_WEIGHT,
    environment_weight: float = PREVIOUS_ENVIRONMENT_WEIGHT,
) -> dict[int, tuple[float, float]]:
    """Leakage-free prior means using only the previous season's final fit.

    This is the historical analogue of the previous-rating component in the
    richer current preseason model. Team names bridge any season-to-season ID
    changes; promoted or renamed teams deliberately receive no prior.
    """
    home = current_games[["home_team_id", "home_team"]].rename(
        columns={"home_team_id": "team_id", "home_team": "team"}
    )
    away = current_games[["away_team_id", "away_team"]].rename(
        columns={"away_team_id": "team_id", "away_team": "team"}
    )
    current_ids = {
        row.team: int(row.team_id)
        for row in pd.concat([home, away], ignore_index=True)
        .drop_duplicates(["team_id", "team"])
        .itertuples()
    }
    means = {}
    for team in previous_fit.teams.itertuples():
        team_id = current_ids.get(team.team)
        if team_id is None:
            continue
        index = previous_fit.team_index[int(team.team_id)]
        offense = float(previous_fit.offense_ppp[index])
        defense = float(previous_fit.defense_ppp[index])
        power = offense + defense
        environment = offense - defense
        means[team_id] = (
            0.5 * (power_weight * power + environment_weight * environment),
            0.5 * (power_weight * power - environment_weight * environment),
        )
    return means


def _manifest_time(manifest: pd.DataFrame, source: str) -> pd.Timestamp:
    row = manifest[manifest["source"].eq(source)]
    if len(row) != 1:
        raise ValueError(f"preseason manifest is missing {source}")
    return pd.to_datetime(row.iloc[0]["source_fetched_at"], utc=True)


def _neutral_zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    observed = numeric.dropna()
    if len(observed) < 2 or float(observed.std(ddof=0)) == 0.0:
        return pd.Series(0.0, index=values.index)
    return ((numeric - observed.mean()) / observed.std(ddof=0)).fillna(0.0)


def _latest_season_fit(season: int):
    games = build_scoring_games(
        store.read_games(season),
        store.read_processed("team_games", f"{season}.parquet"),
    )
    forecast_week = int(games["model_week"].max()) + 1
    as_of = pd.to_datetime(
        games["start_date"], utc=True
    ).max().to_pydatetime() + timedelta(seconds=1)
    return fit_joint_scoring(games, forecast_week, as_of)


def _latest_unit_ratings(
    season: int,
    target_season: int,
    target_week: int,
    as_of: datetime,
) -> pd.DataFrame:
    games = build_scoring_games(
        store.read_games(season),
        store.read_processed("team_games", f"{season}.parquet"),
    )
    forecast_week = int(games["model_week"].max()) + 1
    fitted = fit_unit_ratings(
        store.read_processed("unit_games", f"{season}.parquet"),
        games,
        forecast_week,
        as_of,
    )
    carried = UnitRatings(
        season=target_season,
        week=target_week,
        as_of=as_of,
        frame=fitted.frame,
    )
    return pd.DataFrame(carried.to_records())


def _align_preseason_unit_ratings(
    carried: pd.DataFrame,
    current_ratings: pd.DataFrame,
    source_season: int,
) -> pd.DataFrame:
    current = current_ratings[["team_id", "team", "classification"]].copy()
    history = carried[["team", *UNIT_RATING_COLUMNS]].copy()
    out = current.merge(history, on="team", how="left", validate="one_to_one")
    out["unit_history_missing"] = out[list(UNIT_RATING_COLUMNS)].isna().any(axis=1)
    out[list(UNIT_RATING_COLUMNS)] = out[list(UNIT_RATING_COLUMNS)].fillna(0.0)
    out["season"] = int(current_ratings["season"].iloc[0])
    out["week"] = int(current_ratings["week"].iloc[0])
    out["as_of"] = current_ratings["as_of"].iloc[0]
    out["model_version"] = UNIT_RATING_MODEL_VERSION
    out["source_season"] = source_season
    order = [
        "season",
        "week",
        "as_of",
        "model_version",
        "source_season",
        "team_id",
        "team",
        "classification",
        "unit_history_missing",
        *UNIT_RATING_COLUMNS,
    ]
    return out[order].sort_values("team", ignore_index=True)


def _previous_ratings(fitted) -> pd.DataFrame:
    frame = pd.DataFrame(rating.to_record() for rating in fitted.ratings())
    classifications = fitted.teams[["team", "classification"]].rename(
        columns={"classification": "previous_classification"}
    )
    frame = frame.merge(classifications, on="team", how="left", validate="one_to_one")
    return frame.rename(
        columns={
            "as_of": "previous_rating_as_of",
            "offense_points": "previous_offense_points",
            "defense_points": "previous_defense_points",
            "power_rating": "previous_power_rating",
            "scoring_environment": "previous_scoring_environment",
            "expected_possessions": "previous_expected_possessions",
            "power_rating_sd": "previous_power_rating_sd",
        }
    )[
        [
            "team",
            "previous_classification",
            "previous_rating_as_of",
            "previous_offense_points",
            "previous_defense_points",
            "previous_power_rating",
            "previous_scoring_environment",
            "previous_expected_possessions",
            "previous_power_rating_sd",
        ]
    ]


def _division_one_team_catalog(
    teams: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    source = teams.rename(columns={"school": "team", "id": "team_id"}).copy()
    source_columns = ["team_id", "team", "conference", "classification"]
    source = source[[column for column in source_columns if column in source.columns]]
    for column in source_columns:
        if column not in source:
            source[column] = pd.NA
    source["team_catalog_source"] = "teams_endpoint"

    home = games[
        ["home_id", "home_team", "home_conference", "home_classification"]
    ].rename(
        columns={
            "home_id": "team_id",
            "home_team": "team",
            "home_conference": "conference",
            "home_classification": "classification",
        }
    )
    away = games[
        ["away_id", "away_team", "away_conference", "away_classification"]
    ].rename(
        columns={
            "away_id": "team_id",
            "away_team": "team",
            "away_conference": "conference",
            "away_classification": "classification",
        }
    )
    home["team_catalog_source"] = "schedule_fallback"
    away["team_catalog_source"] = "schedule_fallback"
    catalog = pd.concat(
        [source[[*source_columns, "team_catalog_source"]], home, away],
        ignore_index=True,
    )
    catalog["classification"] = catalog["classification"].str.lower()
    catalog = catalog[catalog["classification"].isin({"fbs", "fcs"})]
    catalog["team_id"] = pd.to_numeric(catalog["team_id"], errors="coerce")
    catalog = catalog.dropna(subset=["team_id", "team"])
    catalog["team_id"] = catalog["team_id"].astype(int)
    catalog = catalog.drop_duplicates(["team_id", "team"], keep="first")
    duplicate_ids = catalog.groupby("team_id")["team"].nunique()
    duplicate_names = catalog.groupby("team")["team_id"].nunique()
    if duplicate_ids.gt(1).any() or duplicate_names.gt(1).any():
        raise ValueError("current Division I team IDs and names do not map one-to-one")
    return catalog.drop_duplicates("team_id", keep="first").reset_index(drop=True)


def _aggregate_portal(portal: pd.DataFrame, teams: pd.Series) -> pd.DataFrame:
    columns = [
        "team",
        "transfer_in_count",
        "transfer_out_count",
        "transfer_quality_balance",
        "transfer_count_balance",
        "qb_transfer_quality_balance",
    ]
    if portal.empty:
        return pd.DataFrame(columns=columns)

    frame = portal.copy()
    frame["rating"] = pd.to_numeric(frame.get("rating"), errors="coerce")
    frame["known_rating"] = (frame["rating"] - 0.85).clip(lower=0.0).fillna(0.0)
    frame["is_qb"] = frame.get(
        "position", pd.Series(index=frame.index, dtype="object")
    ).eq("QB")
    rows = []
    for team in teams:
        incoming = frame[frame["destination"].eq(team)]
        outgoing = frame[frame["origin"].eq(team)]
        rows.append(
            {
                "team": team,
                "transfer_in_count": len(incoming),
                "transfer_out_count": len(outgoing),
                "transfer_quality_balance": (
                    incoming["known_rating"].sum() - outgoing["known_rating"].sum()
                ),
                "transfer_count_balance": len(incoming) - len(outgoing),
                "qb_transfer_quality_balance": (
                    incoming.loc[incoming["is_qb"], "known_rating"].sum()
                    - outgoing.loc[outgoing["is_qb"], "known_rating"].sum()
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _coach_assignments(coaches: pd.DataFrame, season: int) -> pd.DataFrame:
    columns = ["team", "coach_id", "coach", "coach_games"]
    rows = []
    for coach in coaches.itertuples():
        seasons = coach.seasons if isinstance(coach.seasons, (list, np.ndarray)) else []
        current = [entry for entry in seasons if int(entry.get("year", 0)) == season]
        if not current:
            continue
        team = current[0].get("school")
        rows.append(
            {
                "team": team,
                "coach_id": int(coach.id),
                "coach": f"{coach.first_name} {coach.last_name}".strip(),
                "coach_games": int(current[0].get("games") or 0),
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["team", "coach_games"], ascending=[True, False])
        .drop_duplicates("team")
    )


def _coach_continuity(
    coaches: pd.DataFrame, prior_coaches: pd.DataFrame, season: int
) -> pd.DataFrame:
    current = _coach_assignments(coaches, season)
    previous = _coach_assignments(prior_coaches, season - 1).rename(
        columns={
            "coach_id": "previous_coach_id",
            "coach": "previous_coach",
            "coach_games": "previous_coach_games",
        }
    )
    continuity = current.merge(previous, on="team", how="outer")
    continuity["coach_continuity"] = (
        continuity["coach_id"].notna()
        & continuity["previous_coach_id"].notna()
        & continuity["coach_id"].eq(continuity["previous_coach_id"])
    )
    continuity["coach_continuity_missing"] = (
        continuity["coach_id"].isna() | continuity["previous_coach_id"].isna()
    )
    return continuity


def _merge_previous_ratings(ratings: pd.DataFrame, previous_fit) -> pd.DataFrame:
    previous = _previous_ratings(previous_fit)
    ratings = ratings.merge(previous, on="team", how="left")
    ratings["previous_rating_missing"] = ratings["previous_power_rating"].isna()
    previous_means = previous.groupby("previous_classification").mean(numeric_only=True)
    for column in (
        "previous_offense_points",
        "previous_defense_points",
        "previous_power_rating",
        "previous_scoring_environment",
    ):
        classification_fallback = ratings["classification"].map(
            previous_means[column] if column in previous_means else {}
        )
        ratings[column] = ratings[column].fillna(classification_fallback).fillna(0.0)
    ratings["previous_expected_possessions"] = (
        ratings["previous_expected_possessions"]
        .fillna(
            ratings["classification"].map(
                previous_means.get(
                    "previous_expected_possessions", pd.Series(dtype=float)
                )
            )
        )
        .fillna(previous_fit.base_possessions)
    )
    ratings["previous_power_rating_sd"] = ratings["previous_power_rating_sd"].fillna(
        BASE_OFFSEASON_POWER_SD
    )
    ratings["classification_prior_kind"] = np.where(
        ratings["previous_rating_missing"],
        "previous_season_classification_mean",
        "previous_season_team_rating",
    )
    return ratings


def _merge_talent(
    ratings: pd.DataFrame, talent: pd.DataFrame, prior_talent: pd.DataFrame
) -> pd.DataFrame:
    current = talent.rename(columns={"school": "team"})
    if "talent" in current:
        current = current[["team", "talent"]].rename(
            columns={"talent": "current_talent"}
        )
    else:
        current = pd.DataFrame(columns=["team", "current_talent"])
    ratings = ratings.merge(current, on="team", how="left")
    ratings["current_talent_missing"] = ratings["current_talent"].isna()
    ratings["current_talent_z"] = _neutral_zscore(ratings["current_talent"])

    prior = prior_talent.rename(columns={"school": "team"})
    if "talent" in prior:
        prior = prior[["team", "year", "talent"]].rename(
            columns={"year": "latest_talent_season", "talent": "latest_talent"}
        )
    else:
        prior = pd.DataFrame(columns=["team", "latest_talent_season", "latest_talent"])
    ratings = ratings.merge(prior, on="team", how="left")
    ratings["latest_talent_missing"] = ratings["latest_talent"].isna()
    ratings["latest_talent_z"] = _neutral_zscore(ratings["latest_talent"])
    return ratings


def _merge_returning_production(
    ratings: pd.DataFrame, returning: pd.DataFrame
) -> pd.DataFrame:
    source_empty = returning.empty
    renames = {
        "percent_ppa": "returning_percent_ppa",
        "usage": "returning_usage",
        "percent_passing_ppa": "qb_continuity",
        "percent_receiving_ppa": "returning_receiving_ppa",
    }
    if "team" in returning:
        available = ["team", *(column for column in renames if column in returning)]
        returning = returning[available].rename(columns=renames)
    else:
        returning = pd.DataFrame(columns=["team"])
    ratings = ratings.merge(returning, on="team", how="left")
    ratings["returning_source_kind"] = (
        "neutral_missing_cfbd_returning"
        if source_empty
        else "cfbd_returning_production"
    )
    for column in ("returning_percent_ppa", "qb_continuity", "returning_receiving_ppa"):
        if column not in ratings:
            ratings[column] = np.nan
    ratings["returning_production_missing"] = ratings["returning_percent_ppa"].isna()
    ratings["qb_continuity_missing"] = ratings["qb_continuity"].isna()
    ratings["returning_percent_ppa_z"] = _neutral_zscore(
        ratings["returning_percent_ppa"]
    )
    ratings["qb_continuity_z"] = _neutral_zscore(ratings["qb_continuity"])
    ratings["returning_receiving_ppa_z"] = _neutral_zscore(
        ratings["returning_receiving_ppa"]
    )
    return ratings


def _merge_recruiting(ratings: pd.DataFrame, recruiting: pd.DataFrame) -> pd.DataFrame:
    if {"team", "rank", "points"}.issubset(recruiting.columns):
        recruiting = recruiting[["team", "rank", "points"]].rename(
            columns={"rank": "recruiting_rank", "points": "recruiting_points"}
        )
    else:
        recruiting = pd.DataFrame(
            columns=["team", "recruiting_rank", "recruiting_points"]
        )
    ratings = ratings.merge(recruiting, on="team", how="left")
    ratings["recruiting_missing"] = ratings["recruiting_points"].isna()
    ratings["recruiting_points_z"] = _neutral_zscore(ratings["recruiting_points"])
    return ratings


def _merge_portal(
    ratings: pd.DataFrame, portal: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    ratings = ratings.merge(
        _aggregate_portal(portal, ratings["team"]), on="team", how="left"
    )
    ratings["transfer_data_missing"] = bool(
        manifest.loc[manifest["source"].eq("portal"), "is_empty"].iloc[0]
    )
    for column in (
        "transfer_in_count",
        "transfer_out_count",
        "transfer_quality_balance",
        "transfer_count_balance",
        "qb_transfer_quality_balance",
    ):
        ratings[column] = ratings[column].fillna(0.0)
        ratings[f"{column}_z"] = _neutral_zscore(ratings[column])
    return ratings


def _merge_coaches(
    ratings: pd.DataFrame,
    coaches: pd.DataFrame,
    prior_coaches: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    continuity = _coach_continuity(coaches, prior_coaches, season)
    ratings = ratings.merge(continuity, on="team", how="left")
    ratings["coach_continuity_missing"] = (
        ratings["coach_continuity_missing"].astype("boolean").fillna(True).astype(bool)
    )
    coach_value = ratings["coach_continuity"].map(
        {True: COACH_CONTINUITY_POINTS, False: -COACH_CONTINUITY_POINTS}
    )
    ratings["coach_continuity_contribution"] = coach_value.mask(
        ratings["coach_continuity_missing"], 0.0
    ).fillna(0.0)
    return ratings


def _combine_ratings(ratings: pd.DataFrame, previous_fit) -> pd.DataFrame:
    contributions = {
        "previous_power_contribution": (
            PREVIOUS_POWER_WEIGHT * ratings["previous_power_rating"]
        ),
        "latest_talent_contribution": LATEST_TALENT_POINTS * ratings["latest_talent_z"],
        "current_talent_contribution": (
            CURRENT_TALENT_POINTS * ratings["current_talent_z"]
        ),
        "recruiting_contribution": RECRUITING_POINTS * ratings["recruiting_points_z"],
        "returning_contribution": RETURNING_POINTS * ratings["returning_percent_ppa_z"],
        "transfer_contribution": (
            TRANSFER_QUALITY_POINTS * ratings["transfer_quality_balance_z"]
            + TRANSFER_COUNT_POINTS * ratings["transfer_count_balance_z"]
        ),
        "qb_continuity_contribution": QB_CONTINUITY_POINTS * ratings["qb_continuity_z"],
    }
    for column, values in contributions.items():
        ratings[column] = values
    contribution_columns = [*contributions, "coach_continuity_contribution"]
    ratings["power_rating"] = ratings[contribution_columns].sum(axis=1)

    fbs = ratings["classification"].eq("fbs")
    cohort = fbs if fbs.any() else pd.Series(True, index=ratings.index)
    ratings["power_rating"] -= ratings.loc[cohort, "power_rating"].mean()
    scoring_environment = (
        PREVIOUS_ENVIRONMENT_WEIGHT * ratings["previous_scoring_environment"]
        + QB_TRANSFER_ENVIRONMENT_POINTS * ratings["qb_transfer_quality_balance_z"]
        + 0.80 * ratings["qb_continuity_z"]
        + 0.30 * ratings["returning_receiving_ppa_z"]
    )
    ratings["scoring_environment"] = (
        scoring_environment - scoring_environment[cohort].mean()
    ).clip(-10.0, 10.0)
    ratings["offense_points"] = 0.5 * (
        ratings["power_rating"] + ratings["scoring_environment"]
    )
    ratings["defense_points"] = 0.5 * (
        ratings["power_rating"] - ratings["scoring_environment"]
    )
    ratings["expected_possessions"] = previous_fit.base_possessions + (
        PREVIOUS_PACE_WEIGHT
        * (ratings["previous_expected_possessions"] - previous_fit.base_possessions)
    )
    return ratings


def _add_uncertainty(ratings: pd.DataFrame) -> pd.DataFrame:
    ratings["injury_availability_missing"] = True
    missing_variance = (
        ratings["previous_rating_missing"].astype(float) * 3.0**2
        + ratings["current_talent_missing"].astype(float) * 1.0**2
        + ratings["returning_production_missing"].astype(float) * 1.5**2
        + ratings["qb_continuity_missing"].astype(float) * 1.25**2
        + ratings["coach_continuity_missing"].astype(float) * 0.75**2
        + ratings["recruiting_missing"].astype(float) * 0.75**2
        + ratings["transfer_data_missing"].astype(float) * 1.0**2
        + ratings["injury_availability_missing"].astype(float) * 1.0**2
        + ratings["classification"].eq("fcs").astype(float) * 3.0**2
    )
    ratings["power_rating_sd"] = np.sqrt(BASE_OFFSEASON_POWER_SD**2 + missing_variance)
    ratings["missing_input_count"] = ratings[MISSING_INPUT_COLUMNS].sum(axis=1)
    return ratings


def _build_ratings(
    season: int, previous_fit, sources: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    manifest = sources["manifest"]
    ratings = _division_one_team_catalog(sources["teams"], sources["games"])
    if ratings.empty:
        raise ValueError(f"no {season} Division I teams are available")
    ratings = _merge_previous_ratings(ratings, previous_fit)
    ratings = _merge_talent(ratings, sources["talent"], sources["prior_talent"])
    ratings = _merge_returning_production(ratings, sources["returning"])
    ratings = _merge_recruiting(ratings, sources["recruiting"])
    ratings = _merge_portal(ratings, sources["portal"], manifest)
    ratings = _merge_coaches(
        ratings, sources["coaches"], sources["prior_coaches"], season
    )
    ratings = _combine_ratings(ratings, previous_fit)
    ratings = _add_uncertainty(ratings)

    ratings["season"] = season
    ratings["week"] = 1
    ratings["as_of"] = max(
        pd.to_datetime(manifest["source_fetched_at"], utc=True)
    ).isoformat()
    ratings["model_version"] = MODEL_VERSION
    for source in (
        "teams",
        "talent",
        "prior_talent",
        "returning",
        "portal",
        "coaches",
        "prior_coaches",
        "recruiting",
    ):
        ratings[f"{source}_source_fetched_at"] = _manifest_time(
            manifest, source
        ).isoformat()
    order = [
        "season",
        "week",
        "as_of",
        "model_version",
        "team_id",
        "team",
        "conference",
        "classification",
        "offense_points",
        "defense_points",
        "power_rating",
        "scoring_environment",
        "expected_possessions",
        "power_rating_sd",
        "missing_input_count",
    ]
    return ratings.sort_values("power_rating", ascending=False, ignore_index=True)[
        [*order, *(column for column in ratings if column not in order)]
    ]


def _project_games(
    season: int,
    week: int,
    previous_fit,
    ratings: pd.DataFrame,
    games: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = games[games["week"].eq(week)].copy()
    schedule = schedule.rename(
        columns={
            "id": "game_id",
            "home_id": "home_team_id",
            "away_id": "away_team_id",
        }
    )
    schedule["start_date"] = pd.to_datetime(schedule["start_date"], utc=True)
    division_one = {"fbs", "fcs"}
    schedule["forecastable"] = (
        schedule["home_classification"].str.lower().isin(division_one)
        & schedule["away_classification"].str.lower().isin(division_one)
        & schedule["home_team_id"].isin(ratings["team_id"])
        & schedule["away_team_id"].isin(ratings["team_id"])
        & schedule["start_date"].gt(as_of)
    )
    schedule["forecast_exclusion_reason"] = np.select(
        [
            ~schedule["home_classification"].str.lower().isin(division_one)
            | ~schedule["away_classification"].str.lower().isin(division_one),
            ~schedule["home_team_id"].isin(ratings["team_id"])
            | ~schedule["away_team_id"].isin(ratings["team_id"]),
            schedule["start_date"].le(as_of),
        ],
        [
            "non_division_one_matchup",
            "missing_team_prior",
            "kickoff_not_after_snapshot",
        ],
        default="",
    )
    target = schedule[schedule["forecastable"]].copy()
    by_id = ratings.set_index("team_id")
    base_score_covariance = (
        previous_fit.score_residual_covariance
        * DEFAULT_CONFIG.score_covariance_scale**2
    )
    projections = []
    for game in target.itertuples():
        home = by_id.loc[int(game.home_team_id)]
        away = by_id.loc[int(game.away_team_id)]
        possessions = 0.5 * (
            float(home.expected_possessions) + float(away.expected_possessions)
        )
        base_points = previous_fit.base_ppp * possessions
        home_field = (
            0.0
            if bool(game.neutral_site)
            else previous_fit.base_possessions * previous_fit.hfa_ppp
        )
        expected_home = (
            base_points
            + float(home.offense_points)
            - float(away.defense_points)
            + 0.5 * home_field
        )
        expected_away = (
            base_points
            + float(away.offense_points)
            - float(home.defense_points)
            - 0.5 * home_field
        )
        prior_score_variance = 0.25 * (
            float(home.power_rating_sd) ** 2 + float(away.power_rating_sd) ** 2
        )
        score_covariance = base_score_covariance + np.eye(2) * prior_score_variance
        margin_sd, total_sd, correlation = margin_total_distribution(score_covariance)
        projection = GameProjection(
            season=season,
            week=week,
            as_of=as_of.to_pydatetime(),
            model_version=MODEL_VERSION,
            game_id=int(game.game_id),
            home_team_id=int(game.home_team_id),
            home_team=game.home_team,
            away_team_id=int(game.away_team_id),
            away_team=game.away_team,
            neutral_site=bool(game.neutral_site),
            home_field_points=float(home_field),
            expected_home_points=max(float(expected_home), 0.0),
            expected_away_points=max(float(expected_away), 0.0),
            margin_sd=margin_sd,
            total_sd=total_sd,
            margin_total_correlation=correlation,
            degrees_of_freedom=DEFAULT_CONFIG.student_t_degrees_of_freedom,
        ).to_record()
        projection["start_date"] = game.start_date
        projection["home_classification"] = game.home_classification
        projection["away_classification"] = game.away_classification
        projection["home_missing_input_count"] = int(home.missing_input_count)
        projection["away_missing_input_count"] = int(away.missing_input_count)
        projections.append(projection)
    return pd.DataFrame(projections), schedule


def _flatten_cfbd_offers(lines: pd.DataFrame) -> pd.DataFrame:
    """Reshape CFBD lines into the priced-offer schema, with no price, so the
    market blend and comparison code is shared with the Odds API feed."""
    rows = []
    for game in lines.itertuples():
        for offer in _stored_sequence(game.lines):
            spread = pd.to_numeric(offer.get("spread"), errors="coerce")
            total = pd.to_numeric(offer.get("overUnder"), errors="coerce")
            selections = []
            if pd.notna(spread):
                selections.append(("spreads", "home", float(spread)))
                selections.append(("spreads", "away", -float(spread)))
            if pd.notna(total):
                selections.append(("totals", "over", float(total)))
                selections.append(("totals", "under", float(total)))
            for market, selection, point in selections:
                rows.append(
                    {
                        "game_id": int(game.id),
                        "provider": offer.get("provider"),
                        "market": market,
                        "selection": selection,
                        "point": point,
                        "price": np.nan,
                        "execution_eligibility_verified": False,
                        "market_fetched_at": game.source_fetched_at,
                        "match_score": 1.0,
                    }
                )
    return pd.DataFrame(rows, columns=OFFER_COLUMNS)


def run_preseason_forecast(season: int, week: int = 1) -> PreseasonForecastResult:
    if week != 1:
        raise ValueError("the preseason prior currently supports Week 1 only")
    source_names = [
        "teams",
        "games",
        "talent",
        "prior_talent",
        "returning",
        "portal",
        "coaches",
        "prior_coaches",
        "recruiting",
        "lines",
        "manifest",
    ]
    sources = {name: store.read_preseason_source(season, name) for name in source_names}
    manifest = sources["manifest"]
    has_odds_api = manifest["source"].eq("odds_api").any()
    if has_odds_api:
        sources["odds_api"] = store.read_preseason_source(season, "odds_api")
    source_snapshot_as_of = max(pd.to_datetime(manifest["source_fetched_at"], utc=True))
    forecast_created_at = pd.Timestamp(datetime.now(timezone.utc))
    previous_fit = _latest_season_fit(season - 1)
    carried_unit_ratings = _latest_unit_ratings(
        season - 1,
        season,
        week,
        forecast_created_at.to_pydatetime(),
    )
    ratings = _build_ratings(season, previous_fit, sources)
    ratings["as_of"] = forecast_created_at.isoformat()
    ratings["source_snapshot_as_of"] = source_snapshot_as_of.isoformat()
    ratings["forecast_created_at"] = forecast_created_at.isoformat()
    unit_ratings = _align_preseason_unit_ratings(
        carried_unit_ratings,
        ratings,
        season - 1,
    )
    projections, coverage = _project_games(
        season,
        week,
        previous_fit,
        ratings,
        sources["games"],
        forecast_created_at,
    )
    projections["source_snapshot_as_of"] = source_snapshot_as_of.isoformat()
    projections["forecast_created_at"] = forecast_created_at.isoformat()
    coverage["source_snapshot_as_of"] = source_snapshot_as_of.isoformat()
    coverage["forecast_created_at"] = forecast_created_at.isoformat()
    odds_match_coverage = pd.DataFrame()
    if has_odds_api and not sources["odds_api"].empty:
        offers, odds_match_coverage = flatten_odds_api_offers(
            sources["odds_api"], coverage
        )
    else:
        offers = _flatten_cfbd_offers(sources["lines"])
    projections = add_market_informed_margins(projections, offers)
    comparisons = compare_priced_offers(projections, offers)
    comparisons["source_snapshot_as_of"] = source_snapshot_as_of.isoformat()
    comparisons["forecast_created_at"] = forecast_created_at.isoformat()
    outputs = {
        "ratings": ratings,
        "unit_ratings": unit_ratings,
        "projections": projections,
        "schedule_coverage": coverage,
        "market_offers": offers,
        "market_comparisons": comparisons,
        "odds_match_coverage": odds_match_coverage,
        "source_manifest": manifest,
    }
    log_directory = store.write_forecast_outputs(
        "preseason",
        season,
        week,
        forecast_created_at,
        outputs,
        canonical_prefix=("preseason",),
    )
    return PreseasonForecastResult(
        ratings=ratings,
        unit_ratings=unit_ratings,
        projections=projections,
        schedule_coverage=coverage,
        market_offers=offers,
        market_comparisons=comparisons,
        log_directory=log_directory,
    )
