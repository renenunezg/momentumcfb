"""Market-anchored rescoring of the frozen in-game baseline.

Answers one question with numbers: is the pregame anchor the binding
constraint on in-game holdout accuracy? The frozen baseline parameters and
play boundaries stay exactly as stored; the only variable is the anchor,
swapped from the chronological model projection to the market closing line.
The verdict keys on holdout log loss in the manner of the momentum
adopt-or-reject rows, and its diagnostic states what the result implies for
where effort goes next.
"""

import numpy as np
import pandas as pd

from backend.model.calibration import DEVELOPMENT_SEASONS, HOLDOUT_SEASONS
from backend.model.ingame import _margin_bucket, _mean_log_loss, _phase

MODEL_VERSION = "ingame_market_anchor_v1"


def _delta_row(
    frame: pd.DataFrame, partition: str, scope: str, group_value: str
) -> dict[str, object]:
    outcome = frame["home_win"].to_numpy(float)
    baseline = frame["baseline_win_probability"].to_numpy(float)
    market = frame["market_win_probability"].to_numpy(float)
    row = {
        "summary_type": "evaluation",
        "partition": partition,
        "scope": scope,
        "group_value": group_value,
        "n_states": len(frame),
        "n_games": int(frame["game_id"].nunique()),
        "log_loss": _mean_log_loss(outcome, market),
        "brier": float(np.mean(np.square(market - outcome))),
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


def evaluate_market_anchor(
    inputs: pd.DataFrame,
    anchor_note: str,
    development_seasons: tuple[int, ...] = DEVELOPMENT_SEASONS,
    holdout_seasons: tuple[int, ...] = HOLDOUT_SEASONS,
) -> pd.DataFrame:
    """Score the market anchor against the frozen baseline, with the verdict
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
            f"{overall['brier_delta']:+.5f}); the pregame anchor is the "
            "binding constraint on in-game accuracy; effort belongs on the "
            "anchor, not on further in-game adjustments"
        )
    else:
        verdict = "rejected"
        diagnostic = (
            "rejected: holdout log loss did not improve "
            f"{overall['baseline_log_loss']:.5f} -> {overall['log_loss']:.5f} "
            f"({overall['log_loss_delta']:+.5f}; Brier "
            f"{overall['brier_delta']:+.5f}); the in-game layer carries the "
            "work and the pregame anchor matters less than it appears"
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
        "partition": "holdout",
        "scope": "overall",
        "group_value": "all",
        "model_version": MODEL_VERSION,
        "diagnostic": anchor_note,
    }
    return pd.concat(
        [pd.DataFrame(rows), pd.DataFrame([verdict_row, parameter_row])],
        ignore_index=True,
        sort=False,
    )


def format_market_anchor_diagnostic(summary: pd.DataFrame) -> str:
    parameters = summary[summary["summary_type"].eq("parameter")].iloc[0]
    lines = [f"{parameters['model_version']}: {parameters['diagnostic']}"]
    evaluation = summary[summary["summary_type"].eq("evaluation")]
    for row in evaluation[evaluation["scope"].eq("overall")].itertuples():
        lines.append(
            f"{row.partition}: {row.n_states} states / {row.n_games} games, "
            f"{row.diagnostic}"
        )
    lines.append("Holdout deltas by game phase and score margin:")
    for scope in ("phase", "margin"):
        for row in evaluation[evaluation["scope"].eq(scope)].itertuples():
            lines.append(
                f"- {scope} {row.group_value}: {row.diagnostic} ({row.n_games} games)"
            )
    verdict = summary[summary["summary_type"].eq("verdict")].iloc[0]
    lines.append(f"verdict: {verdict['diagnostic']}")
    return "\n".join(lines)
