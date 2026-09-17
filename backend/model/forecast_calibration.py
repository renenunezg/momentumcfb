"""Frozen final-mean calibration shared by weekly forecasts and their replay.

Apply after the joint/SRS score fit and before availability or market layers.
The total offset was fitted on 2020-2021 and selected on 2022.
Process coefficients use only 2020-2022; both were checked on 2023-2025.
Distribution scales and underlying team ratings are unchanged.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from backend.model.joint_scoring import MODEL_VERSION as CORE_MODEL_VERSION
from backend.model.process import process_margin_adjustments

MODEL_VERSION = "joint_scoring_v14"
MEDIAN_TOTAL_OFFSET = -0.8326407989096936


def apply_forecast_calibration(
    projections: pd.DataFrame,
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    process_prior: pd.DataFrame,
    forecast_week: int,
    as_of: datetime,
) -> pd.DataFrame:
    """Return coherent score means, refusing repeated or out-of-order use."""
    if projections.empty:
        return projections.copy()
    if (
        not projections["model_version"].eq(CORE_MODEL_VERSION).all()
        or "process_margin_adjustment" in projections
        or "market_informed_home_margin" in projections
        or "qb_availability_points" in projections
    ):
        raise ValueError("forecast calibration requires unadjusted core projections")
    if projections["game_id"].duplicated().any():
        raise ValueError("forecast calibration requires unique game IDs")
    out = projections.copy()
    margin = out["home_margin"].to_numpy(dtype=float)
    total = out["model_total"].to_numpy(dtype=float)
    raw = (
        out["game_id"]
        .map(
            process_margin_adjustments(
                out, games, team_games, process_prior, forecast_week, as_of
            )
        )
        .to_numpy(dtype=float)
    )
    if not np.isfinite(np.column_stack([margin, total, raw])).all():
        raise ValueError("forecast calibration requires finite means and corrections")
    adjusted_total = np.maximum(np.abs(margin), total + MEDIAN_TOTAL_OFFSET)
    adjusted_margin = np.clip(margin + raw, -adjusted_total, adjusted_total)
    out["core_model_version"] = out["model_version"]
    out["model_version"] = MODEL_VERSION
    out["process_margin_raw_adjustment"] = raw
    out["process_margin_adjustment"] = adjusted_margin - margin
    out["median_total_adjustment"] = adjusted_total - total
    out["total_calibration_adjustment"] = (
        out["total_calibration_adjustment"] + out["median_total_adjustment"]
    )
    out["expected_home_points"] = (adjusted_total + adjusted_margin) / 2.0
    out["expected_away_points"] = (adjusted_total - adjusted_margin) / 2.0
    out["home_margin"] = adjusted_margin
    out["home_spread"] = -adjusted_margin
    out["model_total"] = adjusted_total
    return out
