from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from backend.config import PROCESSED_DIR
from backend.etl import store
from backend.features.scoring import build_scoring_games
from backend.model.joint_scoring import DEFAULT_CONFIG, fit_joint_scoring
from backend.model.outputs import GameProjection
from backend.odds.markets import compare_priced_offers, flatten_odds_api_offers

MODEL_VERSION = "preseason_v2"
PREVIOUS_POWER_WEIGHT = 1.00
PREVIOUS_ENVIRONMENT_WEIGHT = 0.40
PREVIOUS_PACE_WEIGHT = 0.35
LATEST_TALENT_POINTS = 1.50
CURRENT_TALENT_POINTS = 1.50
RECRUITING_POINTS = 0.80
RETURNING_POINTS = 1.20
TRANSFER_QUALITY_POINTS = 1.00
TRANSFER_COUNT_POINTS = 0.35
QB_CONTINUITY_POINTS = 1.00
QB_TRANSFER_ENVIRONMENT_POINTS = 0.80
COACH_CONTINUITY_POINTS = 0.35
BASE_OFFSEASON_POWER_SD = 6.05


@dataclass(frozen=True, slots=True)
class PreseasonForecastResult:
    ratings: pd.DataFrame
    projections: pd.DataFrame
    schedule_coverage: pd.DataFrame
    market_offers: pd.DataFrame
    market_comparisons: pd.DataFrame
    log_directory: Path


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
    as_of = (
        pd.to_datetime(games["start_date"], utc=True).max().to_pydatetime()
        + timedelta(seconds=1)
    )
    return fit_joint_scoring(games, forecast_week, as_of)


def _previous_ratings(fitted) -> pd.DataFrame:
    frame = pd.DataFrame(rating.to_record() for rating in fitted.ratings())
    classifications = fitted.teams[["team", "classification"]].rename(
        columns={"classification": "previous_classification"}
    )
    frame = frame.merge(
        classifications, on="team", how="left", validate="one_to_one"
    )
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
    source = source[
        [column for column in source_columns if column in source.columns]
    ]
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


def _build_ratings(season: int, previous_fit, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    manifest = sources["manifest"]
    teams = _division_one_team_catalog(sources["teams"], sources["games"])
    if teams.empty:
        raise ValueError(f"no {season} Division I teams are available")

    previous = _previous_ratings(previous_fit)
    ratings = teams.merge(previous, on="team", how="left")
    ratings["previous_rating_missing"] = ratings["previous_power_rating"].isna()
    previous_means = previous.groupby("previous_classification").mean(
        numeric_only=True
    )
    for column in (
        "previous_offense_points",
        "previous_defense_points",
        "previous_power_rating",
        "previous_scoring_environment",
    ):
        classification_fallback = ratings["classification"].map(
            previous_means[column] if column in previous_means else {}
        )
        ratings[column] = (
            ratings[column].fillna(classification_fallback).fillna(0.0)
        )
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

    current_talent = sources["talent"].rename(columns={"school": "team"})
    if "talent" in current_talent:
        current_talent = current_talent[["team", "talent"]].rename(
            columns={"talent": "current_talent"}
        )
    else:
        current_talent = pd.DataFrame(columns=["team", "current_talent"])
    ratings = ratings.merge(current_talent, on="team", how="left")
    ratings["current_talent_missing"] = ratings["current_talent"].isna()
    ratings["current_talent_z"] = _neutral_zscore(ratings["current_talent"])

    prior_talent = sources["prior_talent"].rename(columns={"school": "team"})
    if "talent" in prior_talent:
        prior_talent = prior_talent[["team", "year", "talent"]].rename(
            columns={"year": "latest_talent_season", "talent": "latest_talent"}
        )
    else:
        prior_talent = pd.DataFrame(
            columns=["team", "latest_talent_season", "latest_talent"]
        )
    ratings = ratings.merge(prior_talent, on="team", how="left")
    ratings["latest_talent_missing"] = ratings["latest_talent"].isna()
    ratings["latest_talent_z"] = _neutral_zscore(ratings["latest_talent"])

    returning = sources["returning"].copy()
    returning_source_empty = returning.empty
    returning_columns = {
        "percent_ppa": "returning_percent_ppa",
        "usage": "returning_usage",
        "percent_passing_ppa": "qb_continuity",
        "percent_receiving_ppa": "returning_receiving_ppa",
    }
    if "team" in returning:
        available = [
            "team", *(column for column in returning_columns if column in returning)
        ]
        returning = returning[available].rename(columns=returning_columns)
    else:
        returning = pd.DataFrame(columns=["team"])
    ratings = ratings.merge(returning, on="team", how="left")
    ratings["returning_source_kind"] = (
        "neutral_missing_cfbd_returning"
        if returning_source_empty
        else "cfbd_returning_production"
    )
    ratings["returning_production_missing"] = ratings.get(
        "returning_percent_ppa", pd.Series(index=ratings.index, dtype=float)
    ).isna()
    ratings["qb_continuity_missing"] = ratings.get(
        "qb_continuity", pd.Series(index=ratings.index, dtype=float)
    ).isna()
    ratings["returning_percent_ppa_z"] = _neutral_zscore(
        ratings.get("returning_percent_ppa", pd.Series(index=ratings.index, dtype=float))
    )
    ratings["qb_continuity_z"] = _neutral_zscore(
        ratings.get("qb_continuity", pd.Series(index=ratings.index, dtype=float))
    )
    ratings["returning_receiving_ppa_z"] = _neutral_zscore(
        ratings.get(
            "returning_receiving_ppa", pd.Series(index=ratings.index, dtype=float)
        )
    )

    recruiting = sources["recruiting"].copy()
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

    portal = _aggregate_portal(sources["portal"], ratings["team"])
    ratings = ratings.merge(portal, on="team", how="left")
    portal_source_missing = bool(
        manifest.loc[manifest["source"].eq("portal"), "is_empty"].iloc[0]
    )
    ratings["transfer_data_missing"] = portal_source_missing
    for column in (
        "transfer_in_count",
        "transfer_out_count",
        "transfer_quality_balance",
        "transfer_count_balance",
        "qb_transfer_quality_balance",
    ):
        ratings[column] = ratings[column].fillna(0.0)
        ratings[f"{column}_z"] = _neutral_zscore(ratings[column])

    coaches = _coach_continuity(
        sources["coaches"], sources["prior_coaches"], season
    )
    ratings = ratings.merge(coaches, on="team", how="left")
    ratings["coach_continuity_missing"] = (
        ratings["coach_continuity_missing"]
        .astype("boolean")
        .fillna(True)
        .astype(bool)
    )
    coach_value = ratings["coach_continuity"].map(
        {True: COACH_CONTINUITY_POINTS, False: -COACH_CONTINUITY_POINTS}
    )
    coach_value = coach_value.mask(ratings["coach_continuity_missing"], 0.0)

    ratings["injury_availability_missing"] = True
    ratings["previous_power_contribution"] = (
        PREVIOUS_POWER_WEIGHT * ratings["previous_power_rating"]
    )
    ratings["latest_talent_contribution"] = (
        LATEST_TALENT_POINTS * ratings["latest_talent_z"]
    )
    ratings["current_talent_contribution"] = (
        CURRENT_TALENT_POINTS * ratings["current_talent_z"]
    )
    ratings["recruiting_contribution"] = (
        RECRUITING_POINTS * ratings["recruiting_points_z"]
    )
    ratings["returning_contribution"] = (
        RETURNING_POINTS * ratings["returning_percent_ppa_z"]
    )
    ratings["transfer_contribution"] = (
        TRANSFER_QUALITY_POINTS * ratings["transfer_quality_balance_z"]
        + TRANSFER_COUNT_POINTS * ratings["transfer_count_balance_z"]
    )
    ratings["qb_continuity_contribution"] = (
        QB_CONTINUITY_POINTS * ratings["qb_continuity_z"]
    )
    ratings["coach_continuity_contribution"] = coach_value.fillna(0.0)
    contribution_columns = [
        "previous_power_contribution",
        "latest_talent_contribution",
        "current_talent_contribution",
        "recruiting_contribution",
        "returning_contribution",
        "transfer_contribution",
        "qb_continuity_contribution",
        "coach_continuity_contribution",
    ]
    ratings["power_rating"] = ratings[contribution_columns].sum(axis=1)
    fbs = ratings["classification"].eq("fbs")
    center = (
        ratings.loc[fbs, "power_rating"].mean()
        if fbs.any()
        else ratings["power_rating"].mean()
    )
    ratings["power_rating"] -= center

    scoring_environment = (
        PREVIOUS_ENVIRONMENT_WEIGHT * ratings["previous_scoring_environment"]
        + QB_TRANSFER_ENVIRONMENT_POINTS * ratings["qb_transfer_quality_balance_z"]
        + 0.80 * ratings["qb_continuity_z"]
        + 0.30 * ratings["returning_receiving_ppa_z"]
    )
    environment_center = (
        scoring_environment[fbs].mean() if fbs.any() else scoring_environment.mean()
    )
    ratings["scoring_environment"] = (scoring_environment - environment_center).clip(
        -10.0, 10.0
    )
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

    missing_variance = (
        ratings["previous_rating_missing"].astype(float) * 3.0**2
        + ratings["current_talent_missing"].astype(float) * 1.0**2
        + ratings["returning_production_missing"].astype(float) * 1.5**2
        + ratings["qb_continuity_missing"].astype(float) * 1.25**2
        + ratings["coach_continuity_missing"].fillna(True).astype(float) * 0.75**2
        + ratings["recruiting_missing"].astype(float) * 0.75**2
        + ratings["transfer_data_missing"].astype(float) * 1.0**2
        + ratings["injury_availability_missing"].astype(float) * 1.0**2
        + ratings["classification"].eq("fcs").astype(float) * 3.0**2
    )
    ratings["power_rating_sd"] = np.sqrt(
        BASE_OFFSEASON_POWER_SD**2 + missing_variance
    )
    missing_columns = [
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
    ratings["missing_input_count"] = ratings[missing_columns].sum(axis=1)
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
        transform = np.array([[1.0, -1.0], [1.0, 1.0]])
        margin_total_covariance = transform @ score_covariance @ transform.T
        margin_sd = float(np.sqrt(margin_total_covariance[0, 0]))
        total_sd = float(np.sqrt(margin_total_covariance[1, 1]))
        correlation = float(
            margin_total_covariance[0, 1] / (margin_sd * total_sd)
        )
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
            margin_total_correlation=float(np.clip(correlation, -0.999, 0.999)),
            degrees_of_freedom=DEFAULT_CONFIG.student_t_degrees_of_freedom,
        ).to_record()
        projection["start_date"] = game.start_date
        projection["home_classification"] = game.home_classification
        projection["away_classification"] = game.away_classification
        projection["home_missing_input_count"] = int(home.missing_input_count)
        projection["away_missing_input_count"] = int(away.missing_input_count)
        projections.append(projection)
    return pd.DataFrame(projections), schedule


def _flatten_market_offers(lines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game in lines.itertuples():
        offers = game.lines if isinstance(game.lines, (list, np.ndarray)) else []
        for offer in offers:
            raw_spread = offer.get("spread")
            formatted_spread = offer.get("formattedSpread") or ""
            rows.append(
                {
                    "game_id": int(game.id),
                    "away_team": game.away_team,
                    "home_team": game.home_team,
                    "provider": offer.get("provider"),
                    "raw_spread": raw_spread,
                    "formatted_spread": formatted_spread,
                    "home_spread": raw_spread,
                    "over_under": offer.get("overUnder"),
                    "home_moneyline": offer.get("homeMoneyline"),
                    "away_moneyline": offer.get("awayMoneyline"),
                    "spread_open": offer.get("spreadOpen"),
                    "over_under_open": offer.get("overUnderOpen"),
                    "market_fetched_at": game.source_fetched_at,
                    "spread_price_missing": True,
                    "total_price_missing": True,
                    "execution_eligibility_unverified": True,
                }
            )
    offers = pd.DataFrame(rows)
    for column in (
        "home_spread",
        "raw_spread",
        "over_under",
        "home_moneyline",
        "away_moneyline",
        "spread_open",
        "over_under_open",
    ):
        if column in offers:
            offers[column] = pd.to_numeric(offers[column], errors="coerce")
    return offers


def _market_comparisons(
    projections: pd.DataFrame, offers: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for projection in projections.itertuples():
        game_offers = offers[offers["game_id"].eq(projection.game_id)]
        spreads = game_offers.dropna(subset=["home_spread"])
        totals = game_offers.dropna(subset=["over_under"])
        row = {
            "game_id": projection.game_id,
            "start_date": projection.start_date,
            "away_team": projection.away_team,
            "home_team": projection.home_team,
            "model_home_spread": projection.home_spread,
            "model_total": projection.model_total,
            "margin_sd": projection.margin_sd,
            "total_sd": projection.total_sd,
            "model_as_of": projection.as_of,
            "market_available": not game_offers.empty,
            "executable_offer_available": False,
        }
        candidates = []
        if not spreads.empty:
            best_home = spreads.loc[spreads["home_spread"].idxmax()]
            best_away = spreads.loc[spreads["home_spread"].idxmin()]
            row.update(
                {
                    "best_home_spread": best_home.home_spread,
                    "best_home_spread_provider": best_home.provider,
                    "best_away_spread": -best_away.home_spread,
                    "best_away_spread_provider": best_away.provider,
                    "home_spread_edge_points": (
                        projection.home_margin + best_home.home_spread
                    ),
                    "away_spread_edge_points": (
                        -projection.home_margin - best_away.home_spread
                    ),
                }
            )
            candidates.extend(
                [
                    (
                        row["home_spread_edge_points"],
                        "spread",
                        projection.home_team,
                        best_home.home_spread,
                        best_home.provider,
                    ),
                    (
                        row["away_spread_edge_points"],
                        "spread",
                        projection.away_team,
                        -best_away.home_spread,
                        best_away.provider,
                    ),
                ]
            )
        if not totals.empty:
            best_over = totals.loc[totals["over_under"].idxmin()]
            best_under = totals.loc[totals["over_under"].idxmax()]
            row.update(
                {
                    "best_over": best_over.over_under,
                    "best_over_provider": best_over.provider,
                    "best_under": best_under.over_under,
                    "best_under_provider": best_under.provider,
                    "over_edge_points": projection.model_total - best_over.over_under,
                    "under_edge_points": best_under.over_under - projection.model_total,
                }
            )
            candidates.extend(
                [
                    (
                        row["over_edge_points"],
                        "total",
                        "Over",
                        best_over.over_under,
                        best_over.provider,
                    ),
                    (
                        row["under_edge_points"],
                        "total",
                        "Under",
                        best_under.over_under,
                        best_under.provider,
                    ),
                ]
            )
        if candidates:
            edge, market_type, selection, offer, provider = max(candidates)
            uncertainty = (
                projection.margin_sd if market_type == "spread" else projection.total_sd
            )
            scale = uncertainty * np.sqrt(
                (projection.degrees_of_freedom - 2.0)
                / projection.degrees_of_freedom
            )
            row.update(
                {
                    "largest_edge_points": edge,
                    "largest_edge_market": market_type,
                    "largest_edge_selection": selection,
                    "largest_edge_offer": offer,
                    "largest_edge_provider": provider,
                    "largest_edge_standardized": edge / uncertainty,
                    "estimated_cover_probability": student_t.cdf(
                        edge / scale, projection.degrees_of_freedom
                    ),
                    "review_status": (
                        "requires_current_source_review"
                        if edge >= 4.0
                        else "below_material_review_threshold"
                    ),
                    "recommendation_status": "not_recommended",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "largest_edge_points", ascending=False, na_position="last", ignore_index=True
    )


def _write_outputs(
    season: int,
    week: int,
    forecast_created_at: pd.Timestamp,
    outputs: dict[str, pd.DataFrame],
) -> Path:
    timestamp = forecast_created_at.strftime("%Y%m%dT%H%M%S%fZ")
    log_directory = (
        PROCESSED_DIR / "preseason" / "forecast_log" / f"{season}_{week:02d}_{timestamp}"
    )
    for name, frame in outputs.items():
        store.write_processed(frame, "preseason", name, f"{season}_{week:02d}.parquet")
        path = log_directory / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    return log_directory


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
    sources = {
        name: store.read_preseason_source(season, name) for name in source_names
    }
    manifest = sources["manifest"]
    has_odds_api = manifest["source"].eq("odds_api").any()
    if has_odds_api:
        sources["odds_api"] = store.read_preseason_source(season, "odds_api")
    source_snapshot_as_of = max(
        pd.to_datetime(manifest["source_fetched_at"], utc=True)
    )
    forecast_created_at = pd.Timestamp(datetime.now(timezone.utc))
    previous_fit = _latest_season_fit(season - 1)
    ratings = _build_ratings(season, previous_fit, sources)
    ratings["as_of"] = forecast_created_at.isoformat()
    ratings["source_snapshot_as_of"] = source_snapshot_as_of.isoformat()
    ratings["forecast_created_at"] = forecast_created_at.isoformat()
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
        comparisons = compare_priced_offers(projections, offers)
    else:
        offers = _flatten_market_offers(sources["lines"])
        comparisons = _market_comparisons(projections, offers)
    comparisons["source_snapshot_as_of"] = source_snapshot_as_of.isoformat()
    comparisons["forecast_created_at"] = forecast_created_at.isoformat()
    outputs = {
        "ratings": ratings,
        "projections": projections,
        "schedule_coverage": coverage,
        "market_offers": offers,
        "market_comparisons": comparisons,
        "odds_match_coverage": odds_match_coverage,
        "source_manifest": manifest,
    }
    log_directory = _write_outputs(
        season, week, forecast_created_at, outputs
    )
    return PreseasonForecastResult(
        ratings=ratings,
        projections=projections,
        schedule_coverage=coverage,
        market_offers=offers,
        market_comparisons=comparisons,
        log_directory=log_directory,
    )
