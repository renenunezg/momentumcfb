from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import f, t

from backend.model.joint_scoring import JointScoringConfig, fit_joint_scoring

DEVELOPMENT_SEASONS = (2019, 2020, 2021, 2022)
HOLDOUT_SEASONS = (2023, 2024, 2025)
INTERVAL_LEVELS = (0.50, 0.80, 0.95)

_COARSE_CANDIDATES = tuple(
    JointScoringConfig(
        rating_half_life_weeks=half_life,
        strength_prior_sd_ppp=prior_sd,
        covariance_shrinkage=shrinkage,
        student_t_degrees_of_freedom=degrees_of_freedom,
        score_covariance_scale=covariance_scale,
    )
    for half_life, prior_sd, shrinkage, degrees_of_freedom, covariance_scale in product(
        (3.0, 6.0, 10.0),
        (0.35, 0.55, 0.65, 0.80),
        (0.0, 0.1, 0.3, 0.5),
        (7.0, 15.0, 30.0, 100.0),
        (1.0, 1.05, 1.10, 1.15),
    )
)

_REFINEMENT_CANDIDATES = tuple(
    JointScoringConfig(
        rating_half_life_weeks=half_life,
        strength_prior_sd_ppp=0.55,
        covariance_shrinkage=shrinkage,
        student_t_degrees_of_freedom=degrees_of_freedom,
        score_covariance_scale=covariance_scale,
    )
    for half_life, shrinkage, degrees_of_freedom, covariance_scale in product(
        (10.0, 14.0, 20.0, 30.0, 40.0, 60.0),
        (0.5, 0.6, 0.7),
        (100.0, 150.0, 250.0, 500.0),
        (1.075, 1.10, 1.125),
    )
)

_FINAL_REFINEMENT_CANDIDATES = tuple(
    JointScoringConfig(
        rating_half_life_weeks=half_life,
        strength_prior_sd_ppp=prior_sd,
        covariance_shrinkage=shrinkage,
        student_t_degrees_of_freedom=degrees_of_freedom,
        score_covariance_scale=covariance_scale,
    )
    for half_life, prior_sd, shrinkage, degrees_of_freedom, covariance_scale in product(
        (40.0, 60.0, 100.0, np.inf),
        (0.45, 0.50, 0.55),
        (0.4, 0.5, 0.6, 0.7),
        (250.0, 500.0, 1000.0),
        (1.10, 1.125, 1.15),
    )
)

_NO_DECAY_REFINEMENT_CANDIDATES = tuple(
    JointScoringConfig(
        rating_half_life_weeks=np.inf,
        strength_prior_sd_ppp=prior_sd,
        covariance_shrinkage=shrinkage,
        student_t_degrees_of_freedom=degrees_of_freedom,
        score_covariance_scale=covariance_scale,
    )
    for prior_sd, shrinkage, degrees_of_freedom, covariance_scale in product(
        (0.40, 0.45, 0.50),
        (0.7, 0.8, 0.9),
        (250.0, 500.0, 1000.0),
        (1.10, 1.125, 1.15),
    )
)

CANDIDATE_CONFIGS = tuple(
    dict.fromkeys(
        (
            *_COARSE_CANDIDATES,
            *_REFINEMENT_CANDIDATES,
            *_FINAL_REFINEMENT_CANDIDATES,
            *_NO_DECAY_REFINEMENT_CANDIDATES,
        )
    )
)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    selected_config: JointScoringConfig


def _config_id(config: JointScoringConfig) -> str:
    return (
        f"half_life={config.rating_half_life_weeks:g};"
        f"prior_sd={config.strength_prior_sd_ppp:g};"
        f"shrinkage={config.covariance_shrinkage:g};"
        f"df={config.student_t_degrees_of_freedom:g};"
        f"covariance_scale={config.score_covariance_scale:g}"
    )


def _student_t_nll(
    error: np.ndarray, standard_deviation: np.ndarray, degrees_of_freedom: np.ndarray
) -> np.ndarray:
    scale = standard_deviation * np.sqrt(
        (degrees_of_freedom - 2.0) / degrees_of_freedom
    )
    return (
        np.log(scale)
        + 0.5 * np.log(degrees_of_freedom * np.pi)
        + gammaln(degrees_of_freedom / 2.0)
        - gammaln((degrees_of_freedom + 1.0) / 2.0)
        + 0.5
        * (degrees_of_freedom + 1.0)
        * np.log1p(np.square(error / scale) / degrees_of_freedom)
    )


def _joint_student_t_nll(predictions: pd.DataFrame) -> np.ndarray:
    degrees_of_freedom = predictions["degrees_of_freedom"].to_numpy(float)
    margin_error = predictions["margin_error"].to_numpy(float)
    total_error = predictions["total_error"].to_numpy(float)
    margin_sd = predictions["margin_sd"].to_numpy(float)
    total_sd = predictions["total_sd"].to_numpy(float)
    correlation = predictions["margin_total_correlation"].to_numpy(float)
    one_minus_rho_squared = 1.0 - np.square(correlation)
    covariance_mahalanobis = (
        np.square(margin_error / margin_sd)
        - 2.0
        * correlation
        * margin_error
        * total_error
        / (margin_sd * total_sd)
        + np.square(total_error / total_sd)
    ) / one_minus_rho_squared
    covariance_to_scale = (degrees_of_freedom - 2.0) / degrees_of_freedom
    scale_mahalanobis = covariance_mahalanobis / covariance_to_scale
    scale_determinant = (
        np.square(covariance_to_scale)
        * np.square(margin_sd)
        * np.square(total_sd)
        * one_minus_rho_squared
    )
    dimensions = 2.0
    return (
        gammaln(degrees_of_freedom / 2.0)
        - gammaln((degrees_of_freedom + dimensions) / 2.0)
        + 0.5 * dimensions * np.log(degrees_of_freedom * np.pi)
        + 0.5 * np.log(scale_determinant)
        + 0.5
        * (degrees_of_freedom + dimensions)
        * np.log1p(scale_mahalanobis / degrees_of_freedom)
    )


def _add_proper_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    degrees_of_freedom = out["degrees_of_freedom"].to_numpy(float)
    for metric in ("home_score", "away_score", "margin", "total"):
        out[f"{metric}_nll"] = _student_t_nll(
            out[f"{metric}_error"].to_numpy(float),
            out[f"{metric}_sd"].to_numpy(float),
            degrees_of_freedom,
        )
    out["joint_margin_total_nll"] = _joint_student_t_nll(out)
    return out


def _validate_games(games: pd.DataFrame) -> None:
    required = {
        "away_points",
        "away_team_id",
        "game_id",
        "home_points",
        "home_team_id",
        "model_week",
        "season",
        "start_date",
    }
    missing = sorted(required - set(games.columns))
    if missing:
        raise ValueError(f"scoring games are missing columns: {', '.join(missing)}")
    if games.empty:
        raise ValueError("scoring games must not be empty")
    if games["game_id"].duplicated().any():
        raise ValueError("scoring games must contain one row per game_id")
    if games["season"].nunique() != 1:
        raise ValueError("walk-forward input must contain exactly one season")
    if games[list(required)].isna().any().any():
        raise ValueError("scoring games contain null chronological fields")


def fbs_calibration_cohort(games: pd.DataFrame) -> pd.DataFrame:
    """Return the stable FBS-vs-FBS population used by frozen calibration.

    Production scoring also covers mixed and FCS-only games. Those rows were
    added after the baseline was frozen and include postseason week labels
    that do not share the FBS regular-season chronology. This explicit cohort
    reproduces the original 4,958 IDs instead of silently changing the target.
    """
    required = {"home_classification", "away_classification"}
    missing = sorted(required - set(games.columns))
    if missing:
        raise ValueError(
            "scoring games are missing calibration columns: " + ", ".join(missing)
        )
    return games[
        games["home_classification"].str.lower().eq("fbs")
        & games["away_classification"].str.lower().eq("fbs")
    ].copy()


def walk_forward_season(
    games: pd.DataFrame,
    config: JointScoringConfig,
    strength_prior_means: dict[int, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Project each game using completed games from earlier model weeks only."""
    _validate_games(games)
    season = int(games["season"].iloc[0])
    frames = []
    for forecast_week in sorted(games["model_week"].unique()):
        training = games[games["model_week"] < forecast_week]
        if training.empty:
            continue
        target = games[games["model_week"].eq(forecast_week)]
        as_of = pd.to_datetime(target["start_date"], utc=True).min()
        latest_training_start = pd.to_datetime(training["start_date"], utc=True).max()
        if latest_training_start >= as_of:
            raise ValueError(
                f"{season} week {forecast_week} has an earlier-week game at or after "
                "the forecast cutoff"
            )

        fitted = fit_joint_scoring(
            games,
            forecast_week=int(forecast_week),
            as_of=as_of.to_pydatetime(),
            config=config,
            strength_prior_means=strength_prior_means,
        )
        projected = pd.DataFrame(
            projection.to_record() for projection in fitted.project(target)
        )
        if len(projected) != len(target):
            raise ValueError(
                f"{season} week {forecast_week} did not project every eligible game"
            )

        actual = target[
            [
                "game_id",
                "model_week",
                "season_type",
                "start_date",
                "home_points",
                "away_points",
            ]
        ].rename(
            columns={
                "home_points": "actual_home_points",
                "away_points": "actual_away_points",
            }
        )
        projected = projected.merge(
            actual, on="game_id", how="inner", validate="one_to_one"
        )

        prior_games = pd.concat(
            [training["home_team_id"], training["away_team_id"]],
            ignore_index=True,
        ).value_counts()
        projected["home_prior_games"] = (
            projected["home_team_id"].map(prior_games).fillna(0).astype(int)
        )
        projected["away_prior_games"] = (
            projected["away_team_id"].map(prior_games).fillna(0).astype(int)
        )
        projected["minimum_team_history"] = projected[
            ["home_prior_games", "away_prior_games"]
        ].min(axis=1)
        projected["history_bucket"] = pd.cut(
            projected["minimum_team_history"],
            bins=[-1, 1, 3, 6, np.inf],
            labels=["0-1", "2-3", "4-6", "7+"],
        ).astype(str)
        projected["training_games"] = len(training)
        projected["training_model_weeks"] = training["model_week"].nunique()
        projected["training_latest_start"] = latest_training_start
        projected["config_id"] = _config_id(config)
        projected["rating_half_life_weeks"] = config.rating_half_life_weeks
        projected["strength_prior_sd_ppp"] = config.strength_prior_sd_ppp
        projected["covariance_shrinkage"] = config.covariance_shrinkage
        projected["student_t_degrees_of_freedom"] = (
            config.student_t_degrees_of_freedom
        )
        projected["score_covariance_scale"] = config.score_covariance_scale

        projected["actual_margin"] = (
            projected["actual_home_points"] - projected["actual_away_points"]
        )
        projected["actual_total"] = (
            projected["actual_home_points"] + projected["actual_away_points"]
        )
        projected["home_score_error"] = (
            projected["expected_home_points"] - projected["actual_home_points"]
        )
        projected["away_score_error"] = (
            projected["expected_away_points"] - projected["actual_away_points"]
        )
        projected["margin_error"] = (
            projected["home_margin"] - projected["actual_margin"]
        )
        projected["total_error"] = (
            projected["model_total"] - projected["actual_total"]
        )
        frames.append(projected)

    if not frames:
        raise ValueError(f"{season} has no forecastable games")
    predictions = pd.concat(frames, ignore_index=True)
    predictions = predictions.sort_values(
        ["start_date", "game_id"], kind="stable", ignore_index=True
    )
    return _add_proper_scores(predictions)


def _interval_coverage(
    error: np.ndarray,
    standard_deviation: np.ndarray,
    degrees_of_freedom: np.ndarray,
    level: float,
) -> float:
    scale = standard_deviation * np.sqrt(
        (degrees_of_freedom - 2.0) / degrees_of_freedom
    )
    width = t.ppf((1.0 + level) / 2.0, degrees_of_freedom) * scale
    return float(np.mean(np.abs(error) <= width))


def _joint_calibration_values(predictions: pd.DataFrame) -> dict[str, float]:
    margin_error = predictions["margin_error"].to_numpy(float)
    total_error = predictions["total_error"].to_numpy(float)
    margin_sd = predictions["margin_sd"].to_numpy(float)
    total_sd = predictions["total_sd"].to_numpy(float)
    correlation = predictions["margin_total_correlation"].to_numpy(float)
    degrees_of_freedom = predictions["degrees_of_freedom"].to_numpy(float)
    one_minus_rho_squared = 1.0 - np.square(correlation)
    covariance_mahalanobis = (
        np.square(margin_error / margin_sd)
        - 2.0
        * correlation
        * margin_error
        * total_error
        / (margin_sd * total_sd)
        + np.square(total_error / total_sd)
    ) / one_minus_rho_squared

    standardized_margin = margin_error / margin_sd
    standardized_total = (
        total_error / total_sd - correlation * standardized_margin
    ) / np.sqrt(one_minus_rho_squared)
    standardized_correlation = (
        float(np.corrcoef(standardized_margin, standardized_total)[0, 1])
        if len(predictions) > 1
        else np.nan
    )
    values = {
        "standardized_variance_1": float(np.mean(np.square(standardized_margin))),
        "standardized_variance_2": float(np.mean(np.square(standardized_total))),
        "standardized_correlation": standardized_correlation,
    }
    for level in INTERVAL_LEVELS:
        threshold = (
            (degrees_of_freedom - 2.0)
            / degrees_of_freedom
            * 2.0
            * f.ppf(level, 2.0, degrees_of_freedom)
        )
        values[f"coverage_{int(level * 100)}"] = float(
            np.mean(covariance_mahalanobis <= threshold)
        )
    return values


def _coverage_status(row: dict[str, object], include_covariance: bool) -> None:
    failures = []
    n_games = int(row["n_games"])
    for level in INTERVAL_LEVELS:
        key = f"coverage_{int(level * 100)}"
        coverage = float(row[key])
        tolerance = max(
            0.03,
            1.96 * np.sqrt(level * (1.0 - level) / n_games),
        )
        if abs(coverage - level) > tolerance:
            failures.append(
                f"{int(level * 100)}% coverage is {coverage:.1%}"
            )
    if include_covariance:
        first_variance = float(row["standardized_variance_1"])
        second_variance = float(row["standardized_variance_2"])
        correlation = float(row["standardized_correlation"])
        if abs(first_variance - 1.0) > 0.20:
            failures.append(f"standardized margin variance is {first_variance:.2f}")
        if abs(second_variance - 1.0) > 0.20:
            failures.append(f"standardized total variance is {second_variance:.2f}")
        if abs(correlation) > 0.10:
            failures.append(f"standardized residual correlation is {correlation:.2f}")
    row["calibration_status"] = "not_calibrated" if failures else "calibrated"
    row["diagnostic"] = (
        "not calibrated: " + "; ".join(failures)
        if failures
        else "calibrated: 50%, 80%, and 95% coverage are within tolerance"
    )


def _evaluation_rows(
    predictions: pd.DataFrame,
    partition: str,
    scope: str,
    group_value: str,
    assess_status: bool,
) -> list[dict[str, object]]:
    rows = []
    degrees_of_freedom = predictions["degrees_of_freedom"].to_numpy(float)
    for metric in ("home_score", "away_score", "margin", "total"):
        error = predictions[f"{metric}_error"].to_numpy(float)
        standard_deviation = predictions[f"{metric}_sd"].to_numpy(float)
        row = {
            "summary_type": "evaluation",
            "partition": partition,
            "scope": scope,
            "group_value": group_value,
            "metric": metric,
            "n_games": len(predictions),
            "mean_error": float(np.mean(error)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "mean_nll": float(predictions[f"{metric}_nll"].mean()),
            "standardized_variance_1": float(
                np.mean(np.square(error / standard_deviation))
            ),
            "standardized_variance_2": np.nan,
            "standardized_correlation": np.nan,
        }
        for level in INTERVAL_LEVELS:
            row[f"coverage_{int(level * 100)}"] = _interval_coverage(
                error, standard_deviation, degrees_of_freedom, level
            )
        if assess_status:
            _coverage_status(row, include_covariance=False)
        else:
            row["calibration_status"] = "not_assessed"
            row["diagnostic"] = "segment reported for diagnosis only"
        rows.append(row)

    joint_values = _joint_calibration_values(predictions)
    joint_row = {
        "summary_type": "evaluation",
        "partition": partition,
        "scope": scope,
        "group_value": group_value,
        "metric": "joint_margin_total",
        "n_games": len(predictions),
        "mean_error": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "mean_nll": float(predictions["joint_margin_total_nll"].mean()),
        **joint_values,
    }
    if assess_status:
        _coverage_status(joint_row, include_covariance=True)
    else:
        joint_row["calibration_status"] = "not_assessed"
        joint_row["diagnostic"] = "segment reported for diagnosis only"
    rows.append(joint_row)
    return rows


def _build_evaluation_summary(
    predictions: pd.DataFrame,
    development_seasons: tuple[int, ...],
    holdout_seasons: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    partitions = {
        "all": predictions,
        "development": predictions[predictions["season"].isin(development_seasons)],
        "holdout": predictions[predictions["season"].isin(holdout_seasons)],
    }
    for partition, frame in partitions.items():
        rows.extend(
            _evaluation_rows(
                frame,
                partition=partition,
                scope="overall",
                group_value="all",
                assess_status=True,
            )
        )

    segment_specs = (
        ("season", ["season"]),
        ("model_week", ["model_week"]),
        ("team_history", ["history_bucket"]),
        ("season_week", ["season", "model_week"]),
    )
    for scope, columns in segment_specs:
        grouper = columns[0] if len(columns) == 1 else columns
        for keys, frame in predictions.groupby(grouper, sort=True, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            group_value = ":".join(str(value) for value in keys)
            seasons = set(frame["season"].astype(int))
            if seasons.issubset(development_seasons):
                partition = "development"
            elif seasons.issubset(holdout_seasons):
                partition = "holdout"
            else:
                partition = "all"
            rows.extend(
                _evaluation_rows(
                    frame,
                    partition=partition,
                    scope=scope,
                    group_value=group_value,
                    assess_status=scope == "team_history",
                )
            )
    return pd.DataFrame(rows)


def _candidate_row(
    config: JointScoringConfig, predictions: pd.DataFrame
) -> dict[str, object]:
    return {
        "summary_type": "hyperparameter",
        "partition": "development",
        "scope": "overall",
        "group_value": "all",
        "metric": "joint_margin_total",
        "config_id": _config_id(config),
        "rating_half_life_weeks": config.rating_half_life_weeks,
        "strength_prior_sd_ppp": config.strength_prior_sd_ppp,
        "covariance_shrinkage": config.covariance_shrinkage,
        "student_t_degrees_of_freedom": config.student_t_degrees_of_freedom,
        "score_covariance_scale": config.score_covariance_scale,
        "n_games": len(predictions),
        "mean_nll": float(predictions["joint_margin_total_nll"].mean()),
        "mean_score_nll": float(
            predictions[
                [
                    "home_score_nll",
                    "away_score_nll",
                    "margin_nll",
                    "total_nll",
                ]
            ].to_numpy(float).mean()
        ),
        "candidate_status": "valid",
        "diagnostic": "valid chronological predictions",
    }


def _invalid_candidate_row(
    config: JointScoringConfig, error: ValueError
) -> dict[str, object]:
    return {
        "summary_type": "hyperparameter",
        "partition": "development",
        "scope": "overall",
        "group_value": "all",
        "metric": "joint_margin_total",
        "config_id": _config_id(config),
        "rating_half_life_weeks": config.rating_half_life_weeks,
        "strength_prior_sd_ppp": config.strength_prior_sd_ppp,
        "covariance_shrinkage": config.covariance_shrinkage,
        "student_t_degrees_of_freedom": config.student_t_degrees_of_freedom,
        "score_covariance_scale": config.score_covariance_scale,
        "n_games": 0,
        "mean_nll": np.inf,
        "mean_score_nll": np.inf,
        "candidate_status": "invalid",
        "diagnostic": str(error),
    }


def _rescore_predictions(
    predictions: pd.DataFrame, config: JointScoringConfig
) -> pd.DataFrame:
    out = predictions.copy()
    previous_scale = float(out["score_covariance_scale"].iloc[0])
    scale_ratio = config.score_covariance_scale / previous_scale
    for column in ("home_score_sd", "away_score_sd", "margin_sd", "total_sd"):
        out[column] *= scale_ratio
    out["degrees_of_freedom"] = config.student_t_degrees_of_freedom
    out["config_id"] = _config_id(config)
    out["rating_half_life_weeks"] = config.rating_half_life_weeks
    out["strength_prior_sd_ppp"] = config.strength_prior_sd_ppp
    out["covariance_shrinkage"] = config.covariance_shrinkage
    out["student_t_degrees_of_freedom"] = config.student_t_degrees_of_freedom
    out["score_covariance_scale"] = config.score_covariance_scale
    return _add_proper_scores(out)


def run_calibration(
    games_by_season: Mapping[int, pd.DataFrame],
    candidate_configs: Iterable[JointScoringConfig] = CANDIDATE_CONFIGS,
    development_seasons: tuple[int, ...] = DEVELOPMENT_SEASONS,
    holdout_seasons: tuple[int, ...] = HOLDOUT_SEASONS,
    strength_priors_by_season: Mapping[
        int, dict[int, tuple[float, float]]
    ] | None = None,
    progress: Callable[[str], None] | None = None,
) -> CalibrationResult:
    required_seasons = set(development_seasons) | set(holdout_seasons)
    missing_seasons = sorted(required_seasons - set(games_by_season))
    if missing_seasons:
        raise ValueError(
            f"calibration is missing seasons: {', '.join(map(str, missing_seasons))}"
        )
    if set(development_seasons) & set(holdout_seasons):
        raise ValueError("development and holdout seasons must be disjoint")
    games_by_season = {
        season: fbs_calibration_cohort(games)
        for season, games in games_by_season.items()
    }
    strength_priors_by_season = strength_priors_by_season or {}

    configs = tuple(candidate_configs)
    if not configs:
        raise ValueError("at least one candidate config is required")
    candidate_rows = []
    prediction_cache: dict[tuple[float, float, float], pd.DataFrame] = {}
    invalid_cache: dict[tuple[float, float, float], ValueError] = {}
    reported_cores: set[tuple[float, float, float]] = set()
    core_count = len(
        {
            (
                config.rating_half_life_weeks,
                config.strength_prior_sd_ppp,
                config.covariance_shrinkage,
            )
            for config in configs
        }
    )
    for config in configs:
        core_config = (
            config.rating_half_life_weeks,
            config.strength_prior_sd_ppp,
            config.covariance_shrinkage,
        )
        try:
            if core_config in invalid_cache:
                raise invalid_cache[core_config]
            if core_config not in prediction_cache:
                prediction_cache[core_config] = pd.concat(
                    [
                        walk_forward_season(
                            games_by_season[season],
                            config,
                            strength_priors_by_season.get(season),
                        )
                        for season in development_seasons
                    ],
                    ignore_index=True,
                )
            development_predictions = _rescore_predictions(
                prediction_cache[core_config], config
            )
            row = _candidate_row(config, development_predictions)
        except ValueError as error:
            if str(error) != "expected points must not be negative":
                raise
            invalid_cache[core_config] = error
            row = _invalid_candidate_row(config, error)
        candidate_rows.append(row)
        if progress is not None and core_config not in reported_cores:
            reported_cores.add(core_config)
            if row["candidate_status"] == "valid":
                progress(
                    f"fit {len(reported_cores)}/{core_count} "
                    f"half_life={core_config[0]:g};prior_sd={core_config[1]:g};"
                    f"shrinkage={core_config[2]:g}"
                )
            else:
                progress(
                    f"fit {len(reported_cores)}/{core_count} "
                    f"half_life={core_config[0]:g};prior_sd={core_config[1]:g};"
                    f"shrinkage={core_config[2]:g} "
                    f"invalid={row['diagnostic']}"
                )

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["mean_nll", "config_id"], kind="stable", ignore_index=True
    )
    candidates["rank"] = np.arange(1, len(candidates) + 1)
    candidates["selected"] = candidates["rank"].eq(1)
    selected_row = candidates.iloc[0]
    if not np.isfinite(selected_row["mean_nll"]):
        raise ValueError("all candidate configs were invalid")
    selected_config = JointScoringConfig(
        rating_half_life_weeks=float(selected_row["rating_half_life_weeks"]),
        strength_prior_sd_ppp=float(selected_row["strength_prior_sd_ppp"]),
        covariance_shrinkage=float(selected_row["covariance_shrinkage"]),
        student_t_degrees_of_freedom=float(
            selected_row["student_t_degrees_of_freedom"]
        ),
        score_covariance_scale=float(selected_row["score_covariance_scale"]),
    )
    if progress is not None:
        progress(f"selected {_config_id(selected_config)} on development loss")

    predictions = pd.concat(
        [
            walk_forward_season(
                games_by_season[season],
                selected_config,
                strength_priors_by_season.get(season),
            )
            for season in (*development_seasons, *holdout_seasons)
        ],
        ignore_index=True,
    ).sort_values(["season", "start_date", "game_id"], ignore_index=True)
    evaluation = _build_evaluation_summary(
        predictions,
        development_seasons=development_seasons,
        holdout_seasons=holdout_seasons,
    )
    evaluation["config_id"] = _config_id(selected_config)
    evaluation["rating_half_life_weeks"] = selected_config.rating_half_life_weeks
    evaluation["strength_prior_sd_ppp"] = selected_config.strength_prior_sd_ppp
    evaluation["covariance_shrinkage"] = selected_config.covariance_shrinkage
    evaluation["student_t_degrees_of_freedom"] = (
        selected_config.student_t_degrees_of_freedom
    )
    evaluation["score_covariance_scale"] = selected_config.score_covariance_scale
    summary = pd.concat([candidates, evaluation], ignore_index=True, sort=False)
    return CalibrationResult(
        predictions=predictions,
        summary=summary,
        selected_config=selected_config,
    )


def format_diagnostic(summary: pd.DataFrame) -> str:
    selected = summary[
        summary["summary_type"].eq("hyperparameter")
        & summary["selected"].eq(True)
    ].iloc[0]
    holdout = summary[
        summary["summary_type"].eq("evaluation")
        & summary["partition"].eq("holdout")
        & summary["scope"].eq("overall")
    ]
    lines = [
        "Selected on 2019-2022 joint Student-t log loss: "
        f"{selected['config_id']}.",
        "Untouched 2023-2025 calibration:",
    ]
    for row in holdout.itertuples():
        if row.metric == "joint_margin_total":
            detail = (
                f"standardized variances={row.standardized_variance_1:.2f}/"
                f"{row.standardized_variance_2:.2f}, "
                f"residual correlation={row.standardized_correlation:.2f}"
            )
        else:
            detail = (
                f"mean error={row.mean_error:.2f}, MAE={row.mae:.2f}, "
                f"RMSE={row.rmse:.2f}"
            )
        lines.append(
            f"- {row.metric}: {row.calibration_status}; "
            f"50%={row.coverage_50:.1%}, 80%={row.coverage_80:.1%}, "
            f"95%={row.coverage_95:.1%}; {detail}"
        )
    early_history = summary[
        summary["summary_type"].eq("evaluation")
        & summary["scope"].eq("team_history")
        & summary["group_value"].eq("0-1")
        & summary["metric"].eq("joint_margin_total")
    ].iloc[0]
    lines.append(
        "Known weak segment - games with 0-1 prior games for either team: "
        f"{early_history.calibration_status}; "
        f"joint 50%={early_history.coverage_50:.1%}, "
        f"80%={early_history.coverage_80:.1%}, "
        f"95%={early_history.coverage_95:.1%}."
    )
    return "\n".join(lines)
