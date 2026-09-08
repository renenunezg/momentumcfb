"""Offline probability calibration with fixed chronological promotion checks.

Historical carryover is a research proxy, not archived production predictions.
Closing lines score probabilities only; missing executable prices prohibit
threshold optimization or a historical recommendation/ROI ledger.
"""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backend.model.distributions import marginal_cdf

FIT_SEASONS = (2020, 2021)
SELECTION_SEASONS = (2022,)
EVALUATION_SEASONS = (2023, 2024, 2025)
CONTRACT = "historical_carryover_with_prior_noise_v1"
# Candidate strength is selected on 2022, never on evaluation-year outcomes.
# Stronger shrinkage is simpler. This remains retrospective research.
CANDIDATES = (
    "raw",
    "market_reference",
    "market_relative",
    "keys_200",
    "keys_50",
    "keys_10",
)


@dataclass(frozen=True)
class ProbabilityCalibration:
    offset: float = 0.0
    sd_multiplier: float = 1.0
    key_weights: tuple[float, ...] = ()
    model_weight: float = 1.0

    def __post_init__(self):
        if (
            not np.isfinite(self.offset)
            or not np.isfinite(self.sd_multiplier)
            or self.sd_multiplier <= 0
            or not 0 <= self.model_weight <= 1
        ):
            raise ValueError("invalid probability calibration")
        if any(not np.isfinite(w) or w <= 0 for w in self.key_weights):
            raise ValueError("score weights must be finite and positive")


def _grid(market):
    if market not in ("spreads", "totals"):
        raise ValueError("unknown market")
    return np.arange(-70, 71) if market == "spreads" else np.arange(0, 121)


def _inputs(frame, market, calibration):
    mean = frame["home_margin" if market == "spreads" else "model_total"].to_numpy(
        float
    )
    sd = frame["margin_sd" if market == "spreads" else "total_sd"].to_numpy(float)
    if calibration.model_weight != 1:
        # One frozen reference line per game, held fixed across alternate offers.
        # Do not silently treat the queried offer itself as a new model anchor.
        reference = frame["market_reference"].to_numpy(float)
        if not np.isfinite(reference).all():
            raise ValueError("market-relative calibration requires a reference line")
        anchor = -reference if market == "spreads" else reference
        mean = anchor + calibration.model_weight * (mean - anchor)
    return (
        mean + calibration.offset,
        sd * calibration.sd_multiplier,
        frame.degrees_of_freedom.to_numpy(float),
    )


def _mass(mean, sd, df, values):
    return marginal_cdf(values + 0.5 - mean, sd, df) - marginal_cdf(
        values - 0.5 - mean, sd, df
    )


def _weighted_mass(frame, market, calibration):
    mean, sd, df = _inputs(frame, market, calibration)
    grid = _grid(market)
    groups = np.abs(grid) if market == "spreads" else grid
    masses = _mass(mean[:, None], sd[:, None], df[:, None], grid[None, :])
    weights = np.asarray(calibration.key_weights)
    if len(weights) != int(groups.max()) + 1:
        raise ValueError("calibration weights do not match the market")
    adjustment = masses * (weights[groups] - 1)
    # All scores outside the weighted grid retain their original tail mass.
    return grid, adjustment, 1 + adjustment.sum(axis=1)


def outcome_probabilities(frame, market, lines, calibration=ProbabilityCalibration()):
    """Home-cover / push / away-cover, or over / push / under probabilities.

    A common discrete distribution prices every line and both sides. Weights
    change integer score masses, including football key margins, and preserve
    monotonic probabilities and the unmodified distribution's outside tails.
    """
    mean, sd, df = _inputs(frame, market, calibration)
    threshold = (
        -np.asarray(lines, dtype=float)
        if market == "spreads"
        else np.asarray(lines, dtype=float)
    )
    win = marginal_cdf(mean - np.floor(threshold) - 0.5, sd, df)
    loss = marginal_cdf(np.ceil(threshold) - 0.5 - mean, sd, df)
    push = np.where(threshold == np.floor(threshold), np.maximum(0, 1 - win - loss), 0)
    if calibration.key_weights:
        grid, adjustment, normalizer = _weighted_mass(frame, market, calibration)
        win = (
            win + (adjustment * (grid[None, :] > threshold[:, None])).sum(axis=1)
        ) / normalizer
        loss = (
            loss + (adjustment * (grid[None, :] < threshold[:, None])).sum(axis=1)
        ) / normalizer
        push = (
            push + (adjustment * (grid[None, :] == threshold[:, None])).sum(axis=1)
        ) / normalizer
    probabilities = np.column_stack([win, push, loss])
    if not np.isfinite(probabilities).all() or np.min(probabilities) < -1e-10:
        raise ValueError("invalid calibrated probabilities")
    return np.clip(probabilities, 0, 1)


def _actual(frame, market):
    return frame["actual_margin" if market == "spreads" else "actual_total"].to_numpy(
        float
    )


def fit_calibration(frame, market, candidate):
    if candidate not in CANDIDATES or frame.empty:
        raise ValueError("invalid calibration candidate or empty fitting sample")
    if candidate == "raw":
        return ProbabilityCalibration()
    line_column = "closing_spread" if market == "spreads" else "closing_total"
    frame = frame[frame[line_column].notna()]
    if frame.empty:
        raise ValueError("no market lines in calibration sample")
    lines = frame[line_column].to_numpy(float)
    frame = frame.assign(market_reference=lines)
    threshold = -lines if market == "spreads" else lines
    balance = _actual(frame, market) - threshold
    labels = np.where(balance > 0, 0, np.where(balance == 0, 1, 2))

    def objective(parameters):
        calibration = ProbabilityCalibration(
            float(parameters[0]),
            float(np.exp(parameters[1])),
            model_weight=float(parameters[2]),
        )
        probabilities = outcome_probabilities(frame, market, lines, calibration)
        return -np.log(
            np.maximum(probabilities[np.arange(len(frame)), labels], 1e-14)
        ).sum()

    fitted = minimize(
        objective,
        [0.0, 0.0, 0.0 if candidate == "market_reference" else 0.5],
        method="L-BFGS-B",
        bounds=[
            (-20, 20),
            (np.log(0.5), np.log(3)),
            (0, 0 if candidate == "market_reference" else 1),
        ],
    )
    if not fitted.success:
        raise ValueError(f"location/scale fit failed: {fitted.message}")
    calibration = ProbabilityCalibration(
        float(fitted.x[0]), float(np.exp(fitted.x[1])), model_weight=float(fitted.x[2])
    )
    if candidate in ("market_reference", "market_relative"):
        return calibration
    penalty = float(candidate.split("_")[1])
    mean, sd, df = _inputs(frame, market, calibration)
    grid = _grid(market)
    groups = np.abs(grid) if market == "spreads" else grid
    masses = _mass(mean[:, None], sd[:, None], df[:, None], grid[None, :])
    # Symmetric margin weights avoid adding a second home-field correction.
    group_mass = np.column_stack(
        [masses[:, groups == g].sum(axis=1) for g in range(int(groups.max()) + 1)]
    )
    grid_outcome = np.where(
        grid[None, :] > threshold[:, None],
        0,
        np.where(grid[None, :] == threshold[:, None], 1, 2),
    )
    correct_mass = masses * (grid_outcome == labels[:, None])
    group_correct_mass = np.column_stack(
        [correct_mass[:, groups == g].sum(axis=1) for g in range(group_mass.shape[1])]
    )
    base_correct = outcome_probabilities(frame, market, lines, calibration)[
        np.arange(len(frame)), labels
    ]

    def weighted_objective(log_weights):
        weights = np.exp(log_weights)
        normalizer = 1 + group_mass @ (weights - 1)
        correct = base_correct + group_correct_mass @ (weights - 1)
        value = (
            np.log(normalizer).sum()
            - np.log(correct).sum()
            + penalty / 2 * (log_weights @ log_weights)
        )
        gradient = (
            weights
            * (group_mass.T @ (1 / normalizer) - group_correct_mass.T @ (1 / correct))
            + penalty * log_weights
        )
        return value, gradient

    fitted_weights = minimize(
        weighted_objective,
        np.zeros(group_mass.shape[1]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-4, 2)] * group_mass.shape[1],
    )
    if not fitted_weights.success:
        raise ValueError(f"key-score fit failed: {fitted_weights.message}")
    return ProbabilityCalibration(
        calibration.offset,
        calibration.sd_multiplier,
        tuple(np.exp(fitted_weights.x)),
        calibration.model_weight,
    )


def score_probabilities(frame, market, calibration):
    line_column = "closing_spread" if market == "spreads" else "closing_total"
    sample = frame[frame[line_column].notna()].copy()
    lines = sample[line_column].to_numpy(float)
    sample["market_reference"] = lines
    if (lines * 2 != np.round(lines * 2)).any():
        raise ValueError("evaluation requires integer or half-point lines")
    probabilities = outcome_probabilities(sample, market, lines, calibration)
    balance = (
        _actual(sample, market) + lines
        if market == "spreads"
        else _actual(sample, market) - lines
    )
    labels = np.where(balance > 0, 0, np.where(balance == 0, 1, 2))
    sample["log_loss"] = -np.log(
        np.maximum(probabilities[np.arange(len(sample)), labels], 1e-14)
    )
    sample["brier"] = ((probabilities - np.eye(3)[labels]) ** 2).sum(axis=1)
    sample["predicted_win"] = probabilities[:, 0]
    sample["predicted_push"] = probabilities[:, 1]
    sample["actual_win"] = labels == 0
    sample["actual_push"] = labels == 1
    sample["market"] = market
    return sample


def _paired_interval(frame, values):
    # Games in the same forecast week share fitted ratings and residual shocks.
    clusters = pd.DataFrame(
        {
            "cluster": frame.season.astype(str) + ":" + frame.model_week.astype(str),
            "value": values,
        }
    )
    grouped = clusters.groupby("cluster").value.agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(20260908)
    sampled = grouped[rng.integers(0, len(grouped), size=(2000, len(grouped)))]
    draws = sampled[:, :, 0].sum(axis=1) / sampled[:, :, 1].sum(axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def calibrate_probabilities(predictions):
    required = {
        "season",
        "game_id",
        "as_of",
        "start_date",
        "training_latest_start",
        "evaluation_contract",
    }
    if (
        not required <= set(predictions)
        or predictions.duplicated(["season", "game_id"]).any()
    ):
        raise ValueError("calibration requires unique chronological forecast records")
    if not predictions.evaluation_contract.eq(CONTRACT).all():
        raise ValueError("incompatible calibration forecast contract")
    dates = {
        c: pd.to_datetime(predictions[c], utc=True, errors="coerce")
        for c in ("as_of", "start_date", "training_latest_start")
    }
    if (
        any(d.isna().any() for d in dates.values())
        or not (
            (dates["training_latest_start"] < dates["as_of"])
            & (dates["as_of"] < dates["start_date"])
        ).all()
    ):
        raise ValueError("training and forecast timestamps must precede kickoff")
    permitted = set(FIT_SEASONS + SELECTION_SEASONS + EVALUATION_SEASONS)
    if set(predictions.season) != permitted:
        raise ValueError(
            "fixed calibration split requires 2020 through 2025, excluding 2026"
        )
    fit = predictions[predictions.season.isin(FIT_SEASONS)]
    selection = predictions[predictions.season.isin(SELECTION_SEASONS)]
    development = predictions[predictions.season.isin(FIT_SEASONS + SELECTION_SEASONS)]
    evaluation = predictions[predictions.season.isin(EVALUATION_SEASONS)]
    report, parameters, scored = [], {}, []
    for market in ("spreads", "totals"):
        scores = {
            name: score_probabilities(
                selection, market, fit_calibration(fit, market, name)
            )
            for name in CANDIDATES
        }
        best = min(CANDIDATES, key=lambda name: scores[name].log_loss.mean())
        # Prefer the earliest, simpler candidate within one paired standard error.
        selected = best
        for name in CANDIDATES:
            delta = scores[name].log_loss.to_numpy() - scores[best].log_loss.to_numpy()
            if delta.mean() <= delta.std(ddof=1) / np.sqrt(len(delta)) + 1e-12:
                selected = name
                break
        calibrated = fit_calibration(development, market, selected)
        parameters[market] = {"candidate": selected, **asdict(calibrated)}
        baseline = score_probabilities(evaluation, market, ProbabilityCalibration())
        adjusted = score_probabilities(evaluation, market, calibrated)
        market_reference = score_probabilities(
            evaluation, market, fit_calibration(development, market, "market_reference")
        )
        for label, frame in (
            ("raw", baseline),
            ("market_reference", market_reference),
            ("calibrated", adjusted),
        ):
            scored.append(frame.assign(method=label))
            for season, group in [("all", frame), *list(frame.groupby("season"))]:
                report.append(
                    dict(
                        market=market,
                        method=label,
                        season=season,
                        games=len(group),
                        log_loss=group.log_loss.mean(),
                        brier=group.brier.mean(),
                        predicted_push=group.predicted_push.mean(),
                        actual_push=group.actual_push.mean(),
                        win_bias=group.predicted_win.mean() - group.actual_win.mean(),
                    )
                )
        deltas = adjusted.log_loss.to_numpy() - baseline.log_loss.to_numpy()
        interval = _paired_interval(adjusted, deltas)
        reference_delta = (
            adjusted.log_loss.to_numpy() - market_reference.log_loss.to_numpy()
        )
        reference_interval = _paired_interval(adjusted, reference_delta)
        push_bias_interval = _paired_interval(
            adjusted, adjusted.predicted_push - adjusted.actual_push
        )
        parameters[market].update(
            selection_log_loss={
                name: float(scores[name].log_loss.mean()) for name in CANDIDATES
            },
            evaluation_log_loss_change=float(deltas.mean()),
            evaluation_log_loss_change_95ci=interval,
            probability_score_gate_passed=bool(
                interval[1] < 0 and adjusted.brier.mean() <= baseline.brier.mean()
            ),
            market_reference_log_loss_change=float(reference_delta.mean()),
            market_reference_log_loss_change_95ci=reference_interval,
            model_edge_supported=bool(
                calibrated.model_weight > 0 and reference_interval[1] < 0
            ),
            push_bias_95ci=push_bias_interval,
            push_calibration_status="no_detected_aggregate_bias"
            if push_bias_interval[0] <= 0 <= push_bias_interval[1]
            else "residual_bias",
        )
    return parameters, pd.DataFrame(report), pd.concat(scored, ignore_index=True)


def build_research_forecasts():
    from backend.etl import store
    from backend.features.scoring import build_scoring_games, load_scoring_team_games
    from backend.model.calibration import fbs_calibration_cohort, walk_forward_season
    from backend.model.joint_scoring import (
        DEFAULT_CONFIG,
        JointScoringPriors,
        fit_joint_scoring,
    )
    from backend.model.preseason import build_historical_carryover_priors
    from backend.serving.market import flatten_closing_lines, flatten_closing_totals

    previous, frames = None, []
    for season in range(2019, 2026):
        games = fbs_calibration_cohort(
            build_scoring_games(
                store.read_games(season), load_scoring_team_games(season)
            )
        )
        if previous is not None:
            means = build_historical_carryover_priors(previous, games)
            priors = JointScoringPriors(
                strength_means=means,
                strength_sds={
                    k: (DEFAULT_CONFIG.strength_prior_sd_ppp,) * 2 for k in means
                },
                expected_possessions={k: previous.base_possessions for k in means},
                base_possessions=previous.base_possessions,
                score_noise_covariance=previous.score_residual_covariance,
                score_noise_games=previous.training_games,
                score_noise_season=previous.season,
                score_noise_as_of=previous.as_of,
            )
            forecasts = walk_forward_season(games, DEFAULT_CONFIG, priors=priors)
            lines = store.read_lines(season)
            forecasts = forecasts.merge(
                flatten_closing_lines(lines)[["game_id", "closing_spread"]],
                on="game_id",
                how="left",
                validate="one_to_one",
            )
            forecasts = forecasts.merge(
                flatten_closing_totals(lines)[["game_id", "closing_total"]],
                on="game_id",
                how="left",
                validate="one_to_one",
            )
            forecasts["evaluation_contract"] = CONTRACT
            forecasts["prior_noise_season"] = previous.season
            forecasts["prior_noise_as_of"] = previous.as_of
            frames.append(forecasts)
        previous = fit_joint_scoring(
            games,
            int(games.model_week.max()) + 1,
            (
                pd.to_datetime(games.start_date, utc=True).max() + pd.Timedelta(days=1)
            ).to_pydatetime(),
        )
    return pd.concat(frames, ignore_index=True)


def run_recommendation_calibration(destination: Path):
    from backend.model.joint_scoring import DEFAULT_CONFIG, MODEL_VERSION

    predictions = build_research_forecasts()
    # Provider medians can create quarter-points. Round neither lines nor results:
    # these are unsuitable probability benchmarks and must be excluded per market.
    for column in ("closing_spread", "closing_total"):
        predictions.loc[
            predictions[column] * 2 != np.round(predictions[column] * 2), column
        ] = np.nan
    parameters, report, scored = calibrate_probabilities(predictions)
    destination.mkdir(parents=True, exist_ok=True)
    forecasts_path = destination / "predictions.parquet"
    predictions.to_parquet(forecasts_path, index=False)
    scored.to_parquet(destination / "evaluation.parquet", index=False)
    report.assign(season=report.season.astype(str)).to_parquet(
        destination / "summary.parquet", index=False
    )
    manifest = dict(
        probability_target="home_cover_push_away_cover_or_over_push_under_at_closing_line",
        study_design="retrospective_research_with_chronological_parameter_fitting_and_selection",
        model_version=MODEL_VERSION,
        model_config={
            k: v if np.isfinite(v) else "Infinity"
            for k, v in asdict(DEFAULT_CONFIG).items()
        },
        source_contract=CONTRACT,
        source_sha256=sha256(forecasts_path.read_bytes()).hexdigest(),
        fit_seasons=FIT_SEASONS,
        selection_seasons=SELECTION_SEASONS,
        evaluation_seasons=EVALUATION_SEASONS,
        parameters=parameters,
        production_activated=False,
        recommendation_policy="prospective_pure_model",
        gates_forward_recommendations=False,
        threshold_status="not_calibrated_missing_timestamped_executable_prices",
        research_limitations="missing_archived_rich_preseason_priors",
    )
    (destination / "calibration.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return manifest, report
