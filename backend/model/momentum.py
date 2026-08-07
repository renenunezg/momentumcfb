"""Latent in-game momentum layer over the frozen baseline win projection.

Process evidence accumulated from plays already run updates the latent
rest-of-game margin: each evidence family enters as a home-minus-away total
divided by (evidence plays + shrinkage prior), so early states stay anchored
to the pregame expectation and sustained process dominance earns weight as
evidence volume grows. Baseline parameters are loaded frozen and never refit;
momentum parameters are fitted on the development seasons only.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from backend.model.calibration import DEVELOPMENT_SEASONS, HOLDOUT_SEASONS
from backend.model.ingame import (
    REGULATION_SECONDS,
    IngameBaselineParams,
    _margin_bucket,
    _mean_log_loss,
    _phase,
    margin_distribution,
)

MODEL_VERSION = "ingame_momentum_v1"
FEATURE_NAMES = (
    "drive_efficiency",
    "success_rate",
    "sustained_stops",
    "turnovers",
    "field_position",
    "fourth_downs",
    "missed_kicks",
    "tempo",
)
# Keep optimizer dimensions comparably scaled: field position totals are in
# tens of yards and the tempo interaction is margin times a play-count gap.
FEATURE_SCALES = {"field_position": 10.0, "tempo": 10.0}
SHRINKAGE_GRID = (40.0, 80.0, 160.0, 320.0)


@dataclass(frozen=True, slots=True)
class MomentumParams:
    shrinkage_prior_plays: float
    league_scrimmage_plays_per_second: float
    betas: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.betas) != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} betas")
        if not all(isfinite(beta) for beta in self.betas):
            raise ValueError("betas must be finite")
        if not isfinite(self.shrinkage_prior_plays) or self.shrinkage_prior_plays <= 0:
            raise ValueError("shrinkage_prior_plays must be positive")
        if (
            not isfinite(self.league_scrimmage_plays_per_second)
            or self.league_scrimmage_plays_per_second <= 0
        ):
            raise ValueError("league_scrimmage_plays_per_second must be positive")

    def to_record(self) -> dict[str, float | str]:
        record: dict[str, float | str] = {
            "model_version": MODEL_VERSION,
            "shrinkage_prior_plays": self.shrinkage_prior_plays,
            "league_scrimmage_plays_per_second": self.league_scrimmage_plays_per_second,
        }
        record.update(
            {
                f"beta_{name}": beta
                for name, beta in zip(FEATURE_NAMES, self.betas, strict=True)
            }
        )
        return record


def league_scrimmage_rate(inputs: pd.DataFrame) -> float:
    """Scrimmage plays per regulation second, the tempo reference pace."""
    games = int(inputs["game_id"].nunique())
    if games == 0:
        raise ValueError("tempo reference requires at least one game")
    scrimmage = int(inputs["play_category"].eq("scrimmage").sum())
    return scrimmage / (games * REGULATION_SECONDS)


def evidence_numerators(inputs: pd.DataFrame, league_rate: float) -> np.ndarray:
    """Home-minus-away evidence totals, one column per feature family."""
    elapsed = REGULATION_SECONDS - inputs["seconds_remaining"].to_numpy(float)
    pace_gap = (
        inputs["scrimmage_plays_before"].to_numpy(float) - league_rate * elapsed
    )
    diffs = {
        "drive_efficiency": inputs["home_epa_total"] - inputs["away_epa_total"],
        "success_rate": inputs["home_successes"] - inputs["away_successes"],
        "sustained_stops": inputs["home_stops_forced"] - inputs["away_stops_forced"],
        "turnovers": (
            inputs["home_turnovers_forced"] - inputs["away_turnovers_forced"]
        ),
        "field_position": (
            inputs["home_field_position_total"] - inputs["away_field_position_total"]
        ),
        "fourth_downs": (
            inputs["home_fourth_down_conversions"]
            - inputs["away_fourth_down_conversions"]
        ),
        "missed_kicks": inputs["home_missed_kicks"] - inputs["away_missed_kicks"],
        "tempo": inputs["pregame_margin"].to_numpy(float) * pace_gap,
    }
    return np.column_stack(
        [
            np.asarray(diffs[name], dtype=float) / FEATURE_SCALES.get(name, 1.0)
            for name in FEATURE_NAMES
        ]
    )


def momentum_adjustment(inputs: pd.DataFrame, params: MomentumParams) -> np.ndarray:
    """Shrinkage-weighted update to the expected rest-of-game margin, in
    points per full game from the home perspective."""
    numerators = evidence_numerators(
        inputs, params.league_scrimmage_plays_per_second
    )
    plays = inputs["evidence_plays"].to_numpy(float)
    features = numerators / (plays + params.shrinkage_prior_plays)[:, None]
    return features @ np.asarray(params.betas, dtype=float)


def momentum_win_probability(
    inputs: pd.DataFrame,
    baseline_params: IngameBaselineParams,
    params: MomentumParams,
) -> np.ndarray:
    mean, sd = margin_distribution(inputs, baseline_params)
    fraction = inputs["fraction_remaining"].to_numpy(float)
    return norm.cdf((mean + fraction * momentum_adjustment(inputs, params)) / sd)


def fit_momentum(
    inputs: pd.DataFrame,
    baseline_params: IngameBaselineParams,
    shrinkage_grid: tuple[float, ...] = SHRINKAGE_GRID,
) -> MomentumParams:
    """Fit betas and the shrinkage prior by play-level log loss."""
    if inputs.empty:
        raise ValueError("momentum fitting requires at least one play state")
    league_rate = league_scrimmage_rate(inputs)
    outcome = inputs["home_win"].to_numpy(float)
    fraction = inputs["fraction_remaining"].to_numpy(float)
    mean, sd = margin_distribution(inputs, baseline_params)
    numerators = evidence_numerators(inputs, league_rate)
    plays = inputs["evidence_plays"].to_numpy(float)

    best: tuple[float, float, np.ndarray] | None = None
    for prior_plays in shrinkage_grid:
        features = numerators / (plays + prior_plays)[:, None]

        def loss(betas: np.ndarray) -> float:
            adjusted = mean + fraction * (features @ betas)
            return _mean_log_loss(outcome, norm.cdf(adjusted / sd))

        result = minimize(
            loss,
            x0=np.zeros(len(FEATURE_NAMES)),
            method="Powell",
            options={"xtol": 1e-4, "ftol": 1e-8, "maxiter": 20000},
        )
        if not result.success:
            raise ValueError(
                f"momentum fit did not converge at prior {prior_plays}: "
                f"{result.message}"
            )
        if best is None or result.fun < best[0]:
            best = (float(result.fun), float(prior_plays), result.x)

    _, prior_plays, betas = best
    return MomentumParams(
        shrinkage_prior_plays=prior_plays,
        league_scrimmage_plays_per_second=float(league_rate),
        betas=tuple(float(beta) for beta in betas),
    )


def _delta_row(
    frame: pd.DataFrame, partition: str, scope: str, group_value: str
) -> dict[str, object]:
    outcome = frame["home_win"].to_numpy(float)
    baseline = frame["baseline_win_probability"].to_numpy(float)
    momentum = frame["momentum_win_probability"].to_numpy(float)
    row = {
        "summary_type": "evaluation",
        "partition": partition,
        "scope": scope,
        "group_value": group_value,
        "n_states": len(frame),
        "n_games": int(frame["game_id"].nunique()),
        "log_loss": _mean_log_loss(outcome, momentum),
        "brier": float(np.mean(np.square(momentum - outcome))),
        "baseline_log_loss": _mean_log_loss(outcome, baseline),
        "baseline_brier": float(np.mean(np.square(baseline - outcome))),
    }
    row["log_loss_delta"] = row["log_loss"] - row["baseline_log_loss"]
    row["brier_delta"] = row["brier"] - row["baseline_brier"]
    row["diagnostic"] = (
        f"log loss {row['log_loss']:.5f} vs baseline "
        f"{row['baseline_log_loss']:.5f} ({row['log_loss_delta']:+.5f}); "
        f"Brier delta {row['brier_delta']:+.5f}"
    )
    return row


def evaluate_momentum(
    inputs: pd.DataFrame,
    params: MomentumParams,
    development_seasons: tuple[int, ...] = DEVELOPMENT_SEASONS,
    holdout_seasons: tuple[int, ...] = HOLDOUT_SEASONS,
) -> pd.DataFrame:
    """Score momentum against the frozen baseline, with the adoption verdict
    keyed on holdout log loss."""
    frame = inputs.copy()
    frame["phase"] = _phase(frame)
    frame["margin_bucket"] = _margin_bucket(frame)
    partitions = {
        "all": frame,
        "development": frame[frame["season"].isin(development_seasons)],
        "holdout": frame[frame["season"].isin(holdout_seasons)],
    }
    rows = []
    for partition, part in partitions.items():
        if part.empty:
            continue
        rows.append(_delta_row(part, partition, "overall", "all"))
    holdout = partitions["holdout"]
    for scope, column in (("phase", "phase"), ("margin", "margin_bucket")):
        for key, group in holdout.groupby(column, sort=True, observed=True):
            rows.append(_delta_row(group, "holdout", scope, str(key)))

    overall = next(
        (
            row
            for row in rows
            if row["partition"] == "holdout" and row["scope"] == "overall"
        ),
        None,
    )
    if overall is None:
        verdict = "not_assessed"
        diagnostic = "no holdout states; verdict requires the holdout seasons"
    elif overall["log_loss_delta"] < 0:
        verdict = "adopted"
        diagnostic = (
            "adopted: holdout log loss improved "
            f"{overall['baseline_log_loss']:.5f} -> {overall['log_loss']:.5f} "
            f"({overall['log_loss_delta']:+.5f}; Brier "
            f"{overall['brier_delta']:+.5f})"
        )
    else:
        verdict = "rejected"
        diagnostic = (
            "rejected: holdout log loss did not improve "
            f"{overall['baseline_log_loss']:.5f} -> {overall['log_loss']:.5f} "
            f"({overall['log_loss_delta']:+.5f}; Brier "
            f"{overall['brier_delta']:+.5f}); baseline retained"
        )
    verdict_row = {
        "summary_type": "verdict",
        "partition": "holdout",
        "scope": "overall",
        "group_value": "all",
        "verdict": verdict,
        "diagnostic": diagnostic,
    }
    if overall is not None:
        verdict_row["log_loss_delta"] = overall["log_loss_delta"]
        verdict_row["brier_delta"] = overall["brier_delta"]

    parameter_row = {
        "summary_type": "parameter",
        "partition": "development",
        "scope": "overall",
        "group_value": "all",
        "diagnostic": (
            "fitted on development seasons "
            + ",".join(str(season) for season in development_seasons)
        ),
        **params.to_record(),
    }
    return pd.concat(
        [pd.DataFrame(rows), pd.DataFrame([verdict_row, parameter_row])],
        ignore_index=True,
        sort=False,
    )


def format_momentum_diagnostic(summary: pd.DataFrame) -> str:
    parameters = summary[summary["summary_type"].eq("parameter")].iloc[0]
    betas = ", ".join(
        f"{name} {parameters[f'beta_{name}']:+.3f}" for name in FEATURE_NAMES
    )
    lines = [
        (
            f"{parameters['model_version']}: shrinkage prior "
            f"{parameters['shrinkage_prior_plays']:.0f} plays, tempo reference "
            f"{parameters['league_scrimmage_plays_per_second'] * REGULATION_SECONDS:.1f} "
            f"plays/game ({parameters['diagnostic']})"
        ),
        f"betas: {betas}",
    ]
    evaluation = summary[summary["summary_type"].eq("evaluation")]
    for row in evaluation[evaluation["scope"].eq("overall")].itertuples():
        lines.append(
            f"{row.partition}: {row.n_states} states / {row.n_games} games, "
            f"{row.diagnostic}"
        )
    lines.append("Holdout deltas by game phase and score margin:")
    for scope in ("phase", "margin"):
        for row in evaluation[evaluation["scope"].eq(scope)].itertuples():
            lines.append(f"- {scope} {row.group_value}: {row.diagnostic}")
    verdict = summary[summary["summary_type"].eq("verdict")].iloc[0]
    lines.append(f"VERDICT: {verdict['diagnostic']}")
    return "\n".join(lines)
