"""Market-anchored rescoring of the frozen in-game baseline.

Holds the baseline parameters and play boundaries fixed and swaps only the
pregame anchor for the market closing line, so the holdout log loss delta
isolates how much the anchor limits in-game accuracy.
"""

import pandas as pd

from backend.model.calibration import DEVELOPMENT_SEASONS, HOLDOUT_SEASONS
from backend.model.ingame import _margin_bucket, _phase, comparison_row

MODEL_VERSION = "ingame_market_anchor_v1"


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
        rows.append(
            comparison_row(part, "market_win_probability", partition, "overall", "all")
        )
    holdout = partitions["holdout"]
    for scope, column in (("phase", "phase"), ("margin", "margin_bucket")):
        for key, group in holdout.groupby(column, sort=True, observed=True):
            rows.append(
                comparison_row(
                    group, "market_win_probability", "holdout", scope, str(key)
                )
            )

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
