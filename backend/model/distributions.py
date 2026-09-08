"""Shared marginal distribution contract: inputs are standard deviations."""

import numpy as np
from scipy.stats import t


def marginal_scale(standard_deviation, degrees_of_freedom):
    sd, df = np.broadcast_arrays(
        np.asarray(standard_deviation, dtype=float),
        np.asarray(degrees_of_freedom, dtype=float),
    )
    # Legacy normal forecasts have no finite degrees of freedom.
    df = np.where(np.isnan(df), np.inf, df)
    if np.any(df <= 2):
        raise ValueError("degrees_of_freedom must exceed 2")
    if np.any(~np.isfinite(sd) | (sd <= 0)):
        raise ValueError("standard deviations must be finite and positive")
    return sd * np.sqrt(1.0 - 2.0 / df)


def marginal_cdf(value, standard_deviation, degrees_of_freedom):
    scale = marginal_scale(standard_deviation, degrees_of_freedom)
    df = np.asarray(degrees_of_freedom, dtype=float)
    return t.cdf(np.asarray(value) / scale, np.where(np.isnan(df), np.inf, df))


def marginal_interval_half_width(level, standard_deviation, degrees_of_freedom):
    if not 0 < level < 1:
        raise ValueError("interval level must be between zero and one")
    scale = marginal_scale(standard_deviation, degrees_of_freedom)
    df = np.asarray(degrees_of_freedom, dtype=float)
    return t.ppf((1.0 + level) / 2.0, np.where(np.isnan(df), np.inf, df)) * scale
