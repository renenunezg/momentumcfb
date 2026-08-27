"""Baseline in-game win projection anchored on pregame ratings.

P(home win) treats the remaining scoring margin as normal around the pregame
expectation scaled by the fraction of regulation left, plus the current score
and a possession value for whoever has the ball. There are deliberately no
momentum or streak features: this is the reference any latent-state momentum
model must beat.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from backend.model.calibration import DEVELOPMENT_SEASONS, HOLDOUT_SEASONS

MODEL_VERSION = "ingame_baseline_v1"
REGULATION_SECONDS = 3600.0
POSSESSION_REFERENCE_YARDS_TO_GOAL = 75.0
MARGIN_BUCKET_EDGES = (-np.inf, -16.5, -8.5, -3.5, 3.5, 8.5, 16.5, np.inf)
MARGIN_BUCKET_LABELS = ("<=-17", "-16:-9", "-8:-4", "-3:3", "4:8", "9:16", "17+")

SERVING_ANCHOR_COLUMNS = [
    "game_id",
    "model_week",
    "home_margin",
    "margin_sd",
]

OUTCOME_COLUMNS = ["actual_home_points", "actual_away_points"]

ANCHOR_COLUMNS = SERVING_ANCHOR_COLUMNS + OUTCOME_COLUMNS


@dataclass(frozen=True, slots=True)
class IngameBaselineParams:
    possession_points: float
    field_position_points_per_yard: float
    sd_floor_points: float

    def __post_init__(self) -> None:
        for name in ("possession_points", "field_position_points_per_yard"):
            if not isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not isfinite(self.sd_floor_points) or self.sd_floor_points <= 0:
            raise ValueError("sd_floor_points must be positive")

    def to_record(self) -> dict[str, float | str]:
        return {
            "model_version": MODEL_VERSION,
            "possession_points": self.possession_points,
            "field_position_points_per_yard": self.field_position_points_per_yard,
            "sd_floor_points": self.sd_floor_points,
        }


def build_serving_inputs(states: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """Join play-boundary states to chronological pregame projections.

    This is the live serving contract: only play states and the pregame
    anchor projection (game_id, model_week, home_margin, margin_sd) are
    read, never actual final scores, so it can score a game whose result
    is unknown.
    """
    missing = sorted(set(SERVING_ANCHOR_COLUMNS) - set(anchors.columns))
    if missing:
        raise ValueError(f"anchors are missing columns: {', '.join(missing)}")
    anchor = anchors[SERVING_ANCHOR_COLUMNS].rename(
        columns={"home_margin": "pregame_margin", "margin_sd": "pregame_margin_sd"}
    )
    if anchor["game_id"].duplicated().any():
        raise ValueError("anchors must contain one projection per game_id")

    merged = states.merge(anchor, on="game_id", how="inner", validate="many_to_one")
    merged = merged[merged["seconds_remaining"].notna()].copy()
    merged["fraction_remaining"] = np.where(
        merged["is_overtime"],
        0.0,
        merged["seconds_remaining"] / REGULATION_SECONDS,
    )
    scrimmage = merged["play_category"].eq("scrimmage")
    merged["possession_sign"] = np.where(
        scrimmage, np.where(merged["offense_is_home"], 1.0, -1.0), 0.0
    )
    merged["possession_yards_to_goal"] = np.where(
        scrimmage, merged["yards_to_goal"], np.nan
    )
    return merged.reset_index(drop=True)


def build_baseline_inputs(states: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """Serving inputs plus the outcome labels backtests fit and score on.

    Layers the home_win label and the tie filter from actual final scores on
    top of ``build_serving_inputs``; fitting and evaluation need this wrapper,
    live serving must never call it.
    """
    missing = sorted(set(ANCHOR_COLUMNS) - set(anchors.columns))
    if missing:
        raise ValueError(f"anchors are missing columns: {', '.join(missing)}")
    merged = build_serving_inputs(states, anchors).merge(
        anchors[["game_id", *OUTCOME_COLUMNS]],
        on="game_id",
        how="inner",
        validate="many_to_one",
    )
    merged = merged[
        merged["actual_home_points"].ne(merged["actual_away_points"])
    ].copy()
    merged["home_win"] = merged["actual_home_points"].gt(merged["actual_away_points"])
    # Stored baseline_predictions keep the pre-split column layout, with the
    # outcome labels between the anchor and derived serving columns.
    tail = [
        *OUTCOME_COLUMNS,
        "home_win",
        "fraction_remaining",
        "possession_sign",
        "possession_yards_to_goal",
    ]
    ordered = [column for column in merged.columns if column not in tail] + tail
    return merged[ordered].reset_index(drop=True)


def margin_distribution(
    inputs: pd.DataFrame, params: IngameBaselineParams
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the final home margin at each state."""
    yards_to_goal = (
        inputs["possession_yards_to_goal"]
        .fillna(POSSESSION_REFERENCE_YARDS_TO_GOAL)
        .to_numpy(float)
    )
    possession_value = inputs["possession_sign"].to_numpy(float) * (
        params.possession_points
        + params.field_position_points_per_yard
        * (POSSESSION_REFERENCE_YARDS_TO_GOAL - yards_to_goal)
    )
    fraction = inputs["fraction_remaining"].to_numpy(float)
    mean = (
        inputs["home_margin"].to_numpy(float)
        + fraction * inputs["pregame_margin"].to_numpy(float)
        + possession_value
    )
    variance = (
        fraction * np.square(inputs["pregame_margin_sd"].to_numpy(float))
        + params.sd_floor_points**2
    )
    return mean, np.sqrt(variance)


def win_probability(inputs: pd.DataFrame, params: IngameBaselineParams) -> np.ndarray:
    mean, sd = margin_distribution(inputs, params)
    return norm.cdf(mean / sd)


def _mean_log_loss(outcome: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return float(
        -np.mean(outcome * np.log(clipped) + (1.0 - outcome) * np.log1p(-clipped))
    )


def fit_baseline(inputs: pd.DataFrame) -> IngameBaselineParams:
    """Fit the three baseline parameters by play-level log loss."""
    if inputs.empty:
        raise ValueError("baseline fitting requires at least one play state")
    outcome = inputs["home_win"].to_numpy(float)

    def loss(theta: np.ndarray) -> float:
        params = IngameBaselineParams(
            possession_points=float(theta[0]),
            field_position_points_per_yard=float(theta[1]),
            sd_floor_points=float(np.exp(theta[2])),
        )
        return _mean_log_loss(outcome, win_probability(inputs, params))

    result = minimize(
        loss,
        x0=np.array([1.4, 0.03, np.log(3.0)]),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000},
    )
    if not result.success:
        raise ValueError(f"baseline fit did not converge: {result.message}")
    return IngameBaselineParams(
        possession_points=float(result.x[0]),
        field_position_points_per_yard=float(result.x[1]),
        sd_floor_points=float(np.exp(result.x[2])),
    )


def _phase(inputs: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(
            inputs["is_overtime"], "OT", "Q" + inputs["period"].astype(int).astype(str)
        ),
        index=inputs.index,
    )


def _margin_bucket(inputs: pd.DataFrame) -> pd.Series:
    return pd.cut(
        inputs["home_margin"],
        bins=MARGIN_BUCKET_EDGES,
        labels=MARGIN_BUCKET_LABELS,
    ).astype(str)


def _group_row(
    frame: pd.DataFrame,
    partition: str,
    scope: str,
    group_value: str,
    assess_status: bool,
) -> dict[str, object]:
    outcome = frame["home_win"].to_numpy(float)
    predicted = frame["win_probability"].to_numpy(float)
    n_games = int(frame["game_id"].nunique())
    row = {
        "summary_type": "evaluation",
        "partition": partition,
        "scope": scope,
        "group_value": group_value,
        "n_states": len(frame),
        "n_games": n_games,
        "mean_predicted": float(np.mean(predicted)),
        "empirical_rate": float(np.mean(outcome)),
        "gap": float(np.mean(predicted) - np.mean(outcome)),
        "brier": float(np.mean(np.square(predicted - outcome))),
        "log_loss": _mean_log_loss(outcome, predicted),
    }
    if not assess_status:
        row["calibration_status"] = "not_assessed"
        row["diagnostic"] = "segment reported for diagnosis only"
        return row
    # States within a game share one outcome, so the binomial tolerance uses
    # the game count, not the play count.
    rate = row["empirical_rate"]
    tolerance = max(
        0.02, 1.96 * np.sqrt(max(rate * (1.0 - rate), 1e-4) / max(n_games, 1))
    )
    if abs(row["gap"]) <= tolerance:
        row["calibration_status"] = "calibrated"
        row["diagnostic"] = f"calibrated: gap {row['gap']:+.1%} within ±{tolerance:.1%}"
    else:
        row["calibration_status"] = "not_calibrated"
        row["diagnostic"] = (
            f"not calibrated: predicted {row['mean_predicted']:.1%} vs "
            f"actual {rate:.1%} (gap {row['gap']:+.1%}, tolerance ±{tolerance:.1%})"
        )
    return row


def evaluate_baseline(
    inputs: pd.DataFrame,
    params: IngameBaselineParams,
    development_seasons: tuple[int, ...] = DEVELOPMENT_SEASONS,
    holdout_seasons: tuple[int, ...] = HOLDOUT_SEASONS,
) -> pd.DataFrame:
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
        rows.append(_group_row(part, partition, "overall", "all", assess_status=True))
    holdout = partitions["holdout"]
    segment_specs = (
        ("phase", ["phase"]),
        ("margin", ["margin_bucket"]),
        ("phase_margin", ["phase", "margin_bucket"]),
    )
    for scope, columns in segment_specs:
        grouper = columns[0] if len(columns) == 1 else columns
        for keys, group in holdout.groupby(grouper, sort=True, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rows.append(
                _group_row(
                    group,
                    "holdout",
                    scope,
                    ":".join(str(key) for key in keys),
                    assess_status=scope != "phase_margin",
                )
            )
    summary = pd.DataFrame(rows)
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
        [summary, pd.DataFrame([parameter_row])], ignore_index=True, sort=False
    )


def format_ingame_diagnostic(summary: pd.DataFrame) -> str:
    parameters = summary[summary["summary_type"].eq("parameter")].iloc[0]
    lines = [
        (
            f"{parameters['model_version']}: possession "
            f"{parameters['possession_points']:+.2f} pts, field position "
            f"{parameters['field_position_points_per_yard']:+.3f} pts/yd, "
            f"sd floor {parameters['sd_floor_points']:.2f} pts "
            f"({parameters['diagnostic']})"
        )
    ]
    evaluation = summary[summary["summary_type"].eq("evaluation")]
    for row in evaluation[evaluation["scope"].eq("overall")].itertuples():
        lines.append(
            f"{row.partition}: {row.n_states} states / {row.n_games} games, "
            f"log loss {row.log_loss:.4f}, Brier {row.brier:.4f}, "
            f"{row.calibration_status}"
        )
    lines.append("Holdout calibration by game phase and score margin:")
    for scope in ("phase", "margin"):
        for row in evaluation[evaluation["scope"].eq(scope)].itertuples():
            lines.append(
                f"- {scope} {row.group_value}: predicted "
                f"{row.mean_predicted:.1%} vs actual {row.empirical_rate:.1%} "
                f"({row.n_games} games); {row.calibration_status}"
            )
    return "\n".join(lines)
