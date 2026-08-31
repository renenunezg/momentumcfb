from dataclasses import dataclass
from datetime import datetime
from math import isfinite, isnan

import numpy as np
import pandas as pd

from backend.model.outputs import GameProjection, TeamRating

MODEL_VERSION = "joint_scoring_v2"
PACE_PRIOR_SD = 2.0
HFA_PRIOR_POINTS = 2.5
HFA_PRIOR_SD_POINTS = 1.5


@dataclass(frozen=True, slots=True)
class JointScoringConfig:
    rating_half_life_weeks: float = 6.0
    strength_prior_sd_ppp: float = 0.55
    covariance_shrinkage: float = 0.1
    student_t_degrees_of_freedom: float = 7.0
    score_covariance_scale: float = 1.0

    def __post_init__(self) -> None:
        if isnan(self.rating_half_life_weeks) or self.rating_half_life_weeks <= 0:
            raise ValueError("rating_half_life_weeks must be positive")
        if not isfinite(self.strength_prior_sd_ppp) or self.strength_prior_sd_ppp <= 0:
            raise ValueError("strength_prior_sd_ppp must be positive")
        if not 0 <= self.covariance_shrinkage < 1:
            raise ValueError("covariance_shrinkage must be in [0, 1)")
        if (
            not isfinite(self.student_t_degrees_of_freedom)
            or self.student_t_degrees_of_freedom <= 2
        ):
            raise ValueError("student_t_degrees_of_freedom must exceed 2")
        if (
            not isfinite(self.score_covariance_scale)
            or self.score_covariance_scale <= 0
        ):
            raise ValueError("score_covariance_scale must be positive")


DEFAULT_CONFIG = JointScoringConfig(
    rating_half_life_weeks=float("inf"),
    strength_prior_sd_ppp=0.45,
    covariance_shrinkage=0.8,
    student_t_degrees_of_freedom=500.0,
    score_covariance_scale=1.125,
)


def _solve_ridge(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    prior_mean: np.ndarray,
    prior_sd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    precision = 1.0 / np.square(prior_sd)
    normal = design.T @ (weights[:, None] * design) + np.diag(precision)
    rhs = design.T @ (weights * target) + precision * prior_mean
    covariance = np.linalg.inv(normal)
    return covariance @ rhs, covariance


def _regularized_covariance(
    residuals: np.ndarray, floor: float, shrinkage: float
) -> np.ndarray:
    covariance = np.cov(residuals, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance).astype(float)
    diagonal = np.diag(np.diag(covariance))
    return (1 - shrinkage) * covariance + shrinkage * diagonal + np.eye(2) * floor


def _team_catalog(games: pd.DataFrame) -> pd.DataFrame:
    home = games[["home_team_id", "home_team", "home_classification"]].rename(
        columns={
            "home_team_id": "team_id",
            "home_team": "team",
            "home_classification": "classification",
        }
    )
    away = games[["away_team_id", "away_team", "away_classification"]].rename(
        columns={
            "away_team_id": "team_id",
            "away_team": "team",
            "away_classification": "classification",
        }
    )
    catalog = pd.concat([home, away], ignore_index=True).drop_duplicates(
        ["team_id", "team"]
    )
    duplicate_ids = catalog.groupby("team_id")["team"].nunique()
    duplicate_names = catalog.groupby("team")["team_id"].nunique()
    if duplicate_ids.gt(1).any() or duplicate_names.gt(1).any():
        raise ValueError("team IDs and names do not map one-to-one")
    return catalog.reset_index(drop=True)


@dataclass(slots=True)
class JointScoringFit:
    season: int
    week: int
    as_of: datetime
    teams: pd.DataFrame
    offense_ppp: np.ndarray
    defense_ppp: np.ndarray
    pace: np.ndarray
    base_ppp: float
    base_possessions: float
    hfa_ppp: float
    parameter_covariance: np.ndarray
    score_residual_covariance: np.ndarray
    config: JointScoringConfig

    @property
    def team_index(self) -> dict[int, int]:
        return {
            int(team_id): index for index, team_id in enumerate(self.teams["team_id"])
        }

    def ratings(self) -> list[TeamRating]:
        ratings = []
        for row in self.teams.itertuples():
            index = self.team_index[int(row.team_id)]
            vector = np.zeros(len(self.parameter_covariance))
            vector[index] = self.base_possessions
            vector[len(self.teams) + index] = self.base_possessions
            variance = float(vector @ self.parameter_covariance @ vector)
            ratings.append(
                TeamRating(
                    season=self.season,
                    week=self.week,
                    as_of=self.as_of,
                    model_version=MODEL_VERSION,
                    team_id=int(row.team_id),
                    team=row.team,
                    offense_points=float(
                        self.base_possessions * self.offense_ppp[index]
                    ),
                    defense_points=float(
                        self.base_possessions * self.defense_ppp[index]
                    ),
                    expected_possessions=float(
                        self.base_possessions + 0.5 * self.pace[index]
                    ),
                    power_rating_sd=float(np.sqrt(max(variance, 0.0))),
                )
            )
        return sorted(ratings, key=lambda rating: rating.power_rating, reverse=True)

    def project(self, schedule: pd.DataFrame) -> list[GameProjection]:
        projections = []
        index = self.team_index
        n_teams = len(self.teams)
        for game in schedule.itertuples():
            home = index[int(game.home_team_id)]
            away = index[int(game.away_team_id)]
            home_field = (
                0.0
                if bool(game.neutral_site)
                else (self.base_possessions * self.hfa_ppp)
            )
            possessions = self.base_possessions + 0.5 * (
                self.pace[home] + self.pace[away]
            )
            base_points = self.base_ppp * possessions
            home_strength = self.base_possessions * (
                self.offense_ppp[home] - self.defense_ppp[away]
            )
            away_strength = self.base_possessions * (
                self.offense_ppp[away] - self.defense_ppp[home]
            )
            expected_home = base_points + home_strength + 0.5 * home_field
            expected_away = base_points + away_strength - 0.5 * home_field

            score_design = np.zeros((2, 2 * n_teams + 1))
            score_design[0, home] = self.base_possessions
            score_design[0, n_teams + away] = -self.base_possessions
            score_design[1, away] = self.base_possessions
            score_design[1, n_teams + home] = -self.base_possessions
            if not bool(game.neutral_site):
                score_design[:, -1] = [
                    0.5 * self.base_possessions,
                    -0.5 * self.base_possessions,
                ]
            score_covariance = self.score_residual_covariance + (
                score_design @ self.parameter_covariance @ score_design.T
            )
            score_covariance *= self.config.score_covariance_scale**2
            transform = np.array([[1.0, -1.0], [1.0, 1.0]])
            margin_total_covariance = transform @ score_covariance @ transform.T
            margin_sd = float(np.sqrt(margin_total_covariance[0, 0]))
            total_sd = float(np.sqrt(margin_total_covariance[1, 1]))
            correlation = float(margin_total_covariance[0, 1] / (margin_sd * total_sd))
            projections.append(
                GameProjection(
                    season=int(game.season),
                    week=int(game.week),
                    as_of=self.as_of,
                    model_version=MODEL_VERSION,
                    game_id=int(game.game_id),
                    home_team_id=int(game.home_team_id),
                    home_team=game.home_team,
                    away_team_id=int(game.away_team_id),
                    away_team=game.away_team,
                    neutral_site=bool(game.neutral_site),
                    home_field_points=float(home_field),
                    # The linear strength model is unconstrained, but football
                    # scores are not. Extreme mismatches can otherwise produce
                    # a negative mean for the underdog and abort the forecast.
                    expected_home_points=max(float(expected_home), 0.0),
                    expected_away_points=max(float(expected_away), 0.0),
                    margin_sd=margin_sd,
                    total_sd=total_sd,
                    margin_total_correlation=float(np.clip(correlation, -0.999, 0.999)),
                    degrees_of_freedom=self.config.student_t_degrees_of_freedom,
                )
            )
        return projections


def fit_joint_scoring(
    games: pd.DataFrame,
    forecast_week: int,
    as_of: datetime,
    config: JointScoringConfig = DEFAULT_CONFIG,
    strength_prior_means: dict[int, tuple[float, float]] | None = None,
) -> JointScoringFit:
    """Fit ratings using only games strictly before the requested model week.

    strength_prior_means optionally maps team ID to offense and defense PPP
    prior means. Missing FBS teams default to zero, while missing FCS teams use
    the existing classification fallback learned from the initial fit.
    """
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    training = games[games["model_week"] < forecast_week].copy()
    if "completed" in training:
        training = training[training["completed"].fillna(False).astype(bool)]
    training = training.dropna(
        subset=[
            "home_points",
            "away_points",
            "game_possessions",
            "home_epa_per_possession",
            "away_epa_per_possession",
        ]
    )
    if training.empty:
        raise ValueError("at least one prior model week is required")
    if "start_date" in training:
        latest_training_start = pd.to_datetime(training["start_date"], utc=True).max()
        if latest_training_start.to_pydatetime() >= as_of:
            raise ValueError("training games must start before as_of")
    catalog = _team_catalog(games)
    team_index = {
        int(team_id): index for index, team_id in enumerate(catalog["team_id"])
    }
    n_teams = len(catalog)
    n_games = len(training)
    design = np.zeros((2 * n_games, 2 * n_teams + 1))
    points_per_possession = np.empty(2 * n_games)
    epa_per_possession = np.empty(2 * n_games)
    latest_week = int(training["model_week"].max())
    recency = np.empty(2 * n_games)

    for game_number, game in enumerate(training.itertuples()):
        home = team_index[int(game.home_team_id)]
        away = team_index[int(game.away_team_id)]
        home_row = 2 * game_number
        away_row = home_row + 1
        design[home_row, home] = 1.0
        design[home_row, n_teams + away] = -1.0
        design[away_row, away] = 1.0
        design[away_row, n_teams + home] = -1.0
        if not bool(game.neutral_site):
            design[home_row, -1] = 0.5
            design[away_row, -1] = -0.5
        points_per_possession[[home_row, away_row]] = [
            game.home_points / game.game_possessions,
            game.away_points / game.game_possessions,
        ]
        epa_per_possession[[home_row, away_row]] = [
            game.home_epa_per_possession,
            game.away_epa_per_possession,
        ]
        weight = 0.5 ** (
            (latest_week - game.model_week) / config.rating_half_life_weeks
        )
        recency[[home_row, away_row]] = weight

    base_ppp = float(np.average(points_per_possession, weights=recency))
    centered_points = points_per_possession - base_ppp
    centered_epa = epa_per_possession - np.average(epa_per_possession, weights=recency)
    epa_variance = np.average(np.square(centered_epa), weights=recency)
    epa_scale = float(
        np.clip(
            np.average(centered_epa * centered_points, weights=recency)
            / max(epa_variance, 1e-9),
            0.1,
            2.0,
        )
    )
    process_points = epa_scale * centered_epa
    target = 0.5 * (centered_points + process_points)
    base_possessions = float(np.average(training["game_possessions"]))
    prior_mean = np.zeros(2 * n_teams + 1)
    teams_with_priors = np.zeros(n_teams, dtype=bool)
    if strength_prior_means:
        for team_id, (offense_prior, defense_prior) in strength_prior_means.items():
            index = team_index.get(team_id)
            if index is not None:
                prior_mean[index] = offense_prior
                prior_mean[n_teams + index] = defense_prior
                teams_with_priors[index] = True
    prior_mean[-1] = HFA_PRIOR_POINTS / base_possessions
    prior_sd = np.full(2 * n_teams + 1, config.strength_prior_sd_ppp)
    prior_sd[-1] = HFA_PRIOR_SD_POINTS / base_possessions
    parameters, covariance = _solve_ridge(design, target, recency, prior_mean, prior_sd)
    classifications = catalog["classification"].fillna("").str.lower()
    fbs_mask = classifications.eq("fbs").to_numpy()
    fcs_mask = classifications.eq("fcs").to_numpy()
    if fbs_mask.any() and fcs_mask.any():
        initial_offense = parameters[:n_teams]
        initial_defense = parameters[n_teams : 2 * n_teams]
        missing_fcs_indices = np.flatnonzero(fcs_mask & ~teams_with_priors)
        prior_mean[missing_fcs_indices] = (
            initial_offense[fcs_mask].mean() - initial_offense[fbs_mask].mean()
        )
        prior_mean[n_teams + missing_fcs_indices] = (
            initial_defense[fcs_mask].mean() - initial_defense[fbs_mask].mean()
        )
    paired_residuals = np.column_stack(
        [centered_points - design @ parameters, process_points - design @ parameters]
    )
    process_covariance = _regularized_covariance(
        paired_residuals,
        floor=0.01,
        shrinkage=config.covariance_shrinkage,
    )
    inverse_process_covariance = np.linalg.inv(process_covariance)
    ones = np.ones(2)
    information = float(ones @ inverse_process_covariance @ ones)
    target = (
        np.column_stack([centered_points, process_points])
        @ inverse_process_covariance
        @ ones
        / information
    )
    parameters, covariance = _solve_ridge(
        design, target, recency * information, prior_mean, prior_sd
    )

    offense = parameters[:n_teams]
    defense = parameters[n_teams : 2 * n_teams]
    center_mask = fbs_mask if fbs_mask.any() else np.ones(n_teams, dtype=bool)
    offense_mean = float(offense[center_mask].mean())
    defense_mean = float(defense[center_mask].mean())
    offense = offense - offense_mean
    defense = defense - defense_mean
    base_ppp += offense_mean - defense_mean
    center_weights = np.zeros(n_teams)
    center_weights[center_mask] = 1.0 / center_mask.sum()
    centering = np.eye(n_teams) - np.tile(center_weights, (n_teams, 1))
    covariance_transform = np.zeros_like(covariance)
    covariance_transform[:n_teams, :n_teams] = centering
    covariance_transform[n_teams : 2 * n_teams, n_teams : 2 * n_teams] = centering
    covariance_transform[-1, -1] = 1.0
    covariance = covariance_transform @ covariance @ covariance_transform.T

    pace_design = np.zeros((n_games, n_teams))
    pace_target = training["game_possessions"].to_numpy(float)
    for row, game in enumerate(training.itertuples()):
        pace_design[row, team_index[int(game.home_team_id)]] = 0.5
        pace_design[row, team_index[int(game.away_team_id)]] = 0.5
    base_possessions = float(np.average(pace_target))
    pace, _ = _solve_ridge(
        pace_design,
        pace_target - base_possessions,
        recency[::2],
        np.zeros(n_teams),
        np.full(n_teams, PACE_PRIOR_SD),
    )
    base_possessions += float(pace.mean())
    pace -= pace.mean()

    home_index = training["home_team_id"].map(team_index).to_numpy(int)
    away_index = training["away_team_id"].map(team_index).to_numpy(int)
    expected_possessions = base_possessions + 0.5 * (
        pace[home_index] + pace[away_index]
    )
    home_field = (~training["neutral_site"].astype(bool)).to_numpy(float)
    hfa_points = base_possessions * parameters[-1]
    predicted_home = (
        base_ppp * expected_possessions
        + base_possessions * (offense[home_index] - defense[away_index])
        + 0.5 * hfa_points * home_field
    )
    predicted_away = (
        base_ppp * expected_possessions
        + base_possessions * (offense[away_index] - defense[home_index])
        - 0.5 * hfa_points * home_field
    )
    score_residuals = np.column_stack(
        [
            training["home_points"].to_numpy(float) - predicted_home,
            training["away_points"].to_numpy(float) - predicted_away,
        ]
    )
    score_covariance = _regularized_covariance(
        score_residuals,
        floor=4.0,
        shrinkage=config.covariance_shrinkage,
    )
    return JointScoringFit(
        season=int(training["season"].iloc[-1]),
        week=forecast_week,
        as_of=as_of,
        teams=catalog,
        offense_ppp=offense,
        defense_ppp=defense,
        pace=pace,
        base_ppp=base_ppp,
        base_possessions=base_possessions,
        hfa_ppp=float(parameters[-1]),
        parameter_covariance=covariance,
        score_residual_covariance=score_covariance,
        config=config,
    )
