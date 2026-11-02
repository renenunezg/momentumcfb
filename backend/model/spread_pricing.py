"""Frozen research spread distribution; never fits against serving outcomes."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from backend.model.distributions import marginal_cdf

# Fitted on 2020-2022 win/push/loss log loss at the posted line
# (examples/fit_spread_cover_weight.py); week-clustered se about 0.1. The
# model side carries no measurable cover information beyond the median
# market, so spread edges come from quotes away from that reference.
SPREAD_MODEL_WEIGHT = 0.0075
SPREAD_METHOD = "market_history_margin_blend_push_keys_v2"
SPREAD_CALIBRATION_VERSION = "spread_keys_cover_v2"


@lru_cache(maxsize=1)
def _key_weights():
    data = json.loads(Path(__file__).with_suffix(".json").read_text())
    weights = np.asarray(data["weights"], dtype=float)
    if (
        data["version"] != SPREAD_CALIBRATION_VERSION
        or data["model_weight"] != SPREAD_MODEL_WEIGHT
        or data["fit_seasons"] != [2020, 2021, 2022]
        or data["margin_sd"] != 15.35
        or data["degrees_of_freedom"] != 500
        or weights.shape != (81,)
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("invalid frozen spread calibration")
    weights.setflags(write=False)
    return weights


def spread_margin(projection):
    """Keep one frozen market reference across all queried spread offers."""
    pure = getattr(projection, "pure_home_margin", projection.home_margin)
    reference = getattr(projection, "market_home_spread", None)
    if pure is None or reference is None or not np.isfinite([pure, reference]).all():
        return None
    reference = -float(reference)
    history = getattr(projection, "market_history_home_margin", None)
    history_weight = getattr(projection, "market_history_weight", 0.0)
    if history is not None and np.isfinite(history):
        if (
            history_weight is None
            or not np.isfinite(history_weight)
            or not 0 <= history_weight < 1
        ):
            return None
        model_side = (1 - history_weight) * pure + history_weight * history
    else:
        model_side = pure
    mean = reference + SPREAD_MODEL_WEIGHT * (model_side - reference)
    return float(mean), float(model_side), reference


def spread_win_loss(mean, sd, df, threshold):
    """Key-number push mass with the unkeyed marginal's win/loss split.

    The symmetric key weights pile mass on small margins, which lie on the
    underdog's side of any line past 3. Letting them split win and loss moved
    the median off the market line: at the median close they priced 3.5 to
    10.5 point underdogs to cover 0.52 to 0.54 against 0.45 to 0.50 realized
    in 2020-2025. They now size only the push at integer lines.
    """
    grid = np.arange(-80, 81)
    masses = marginal_cdf(grid + 0.5 - mean, sd, df) - marginal_cdf(
        grid - 0.5 - mean, sd, df
    )
    adjustment = masses * (_key_weights()[np.abs(grid)] - 1)
    normalizer = 1 + adjustment.sum()
    win = marginal_cdf(mean - np.floor(threshold) - 0.5, sd, df)
    loss = marginal_cdf(np.ceil(threshold) - 0.5 - mean, sd, df)
    push = (1 - win - loss + adjustment[grid == threshold].sum()) / normalizer
    share = win / (win + loss)
    return float((1 - push) * share), float((1 - push) * (1 - share))
