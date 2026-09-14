"""Frozen calibration of weekly-model totals, separate from team strength.

Offsets are the negative mean forecast errors from 2020-2022 chronological
D1 forecasts, shrunk toward zero with 100 pseudo-games per matchup class.
2023-2025 validation: total MAE 13.15942 to 13.10901 on 4,439 games, with
unchanged margins; the season-week bootstrap MAE difference is -0.07415 to
-0.02513 points. Historical carryover priors differ from rich live preseason
inputs. No 2026 outcomes enter these coefficients.
"""

import numpy as np

TOTAL_OFFSETS = {
    "FBS vs FBS": -0.8305356532270571,
    "FBS vs FCS": 0.10800062688576359,
    "FCS vs FCS": -0.6189553229599813,
}


def bounded_model_total(total, home_margin, adjustment):
    """Preserve the margin without driving either team's expected score below zero."""
    return np.maximum(total + adjustment, np.abs(home_margin))


def calibrated_scores(
    home_points, away_points, home_classification, away_classification
):
    home = max(float(home_points), 0.0)
    away = max(float(away_points), 0.0)
    classes = (home_classification, away_classification)
    if classes == ("fbs", "fbs"):
        key = "FBS vs FBS"
    elif set(classes) == {"fbs", "fcs"}:
        key = "FBS vs FCS"
    elif classes == ("fcs", "fcs"):
        key = "FCS vs FCS"
    else:
        return home, away
    margin = home - away
    total = float(bounded_model_total(home + away, margin, TOTAL_OFFSETS[key]))
    return (total + margin) / 2, (total - margin) / 2
