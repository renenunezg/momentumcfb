"""Acceptance checks for the chronological recommendation calibration study."""

import numpy as np
import pandas as pd
import pytest

from backend.model.pick_calibration import (
    CONTRACT,
    ProbabilityCalibration,
    calibrate_probabilities,
    outcome_probabilities,
)
from backend.recommendations import _probabilities


def test_pick_calibration_preserves_price_math_and_withholds_evaluation_outcomes():
    rng = np.random.default_rng(51)
    frames = []
    for season in range(2020, 2026):
        n = 48
        margin = rng.normal(0, 12, n)
        total = rng.normal(52, 6, n)
        frames.append(
            pd.DataFrame(
                dict(
                    season=season,
                    game_id=np.arange(n),
                    model_week=np.arange(n) // 8 + 3,
                    home_margin=margin,
                    model_total=total,
                    margin_sd=14.0,
                    total_sd=12.0,
                    degrees_of_freedom=500.0,
                    actual_margin=np.round(margin + rng.normal(0, 16, n)),
                    actual_total=np.round(total + rng.normal(0, 14, n)),
                    closing_spread=np.round((-margin + rng.normal(0, 5, n)) * 2) / 2,
                    closing_total=np.round((total + rng.normal(0, 5, n)) * 2) / 2,
                    as_of=f"{season}-08-01T00:00:00Z",
                    start_date=f"{season}-09-01T00:00:00Z",
                    training_latest_start=f"{season}-07-01T00:00:00Z",
                    evaluation_contract=CONTRACT,
                )
            )
        )
    predictions = pd.concat(frames, ignore_index=True)
    parameters, report, evaluated = calibrate_probabilities(predictions)
    changed = predictions.copy()
    future = changed.season.ge(2023)
    changed.loc[future, ["actual_margin", "actual_total"]] = [99.0, 150.0]
    second, _, _ = calibrate_probabilities(changed)
    for market in ("spreads", "totals"):
        # The reported evaluation changes, but fitted parameters and candidate
        # choice must never respond to evaluation-year outcomes.
        for key in (
            "candidate",
            "offset",
            "sd_multiplier",
            "model_weight",
            "key_weights",
            "selection_log_loss",
        ):
            assert parameters[market][key] == second[market][key]
    assert set(evaluated.season) == {2023, 2024, 2025}
    assert set(report.method) == {"raw", "market_reference", "calibrated"}
    assert report.games.gt(0).all()

    # Production prices sides from the market-informed margin; research frames
    # carry only the pure margin, so pin them equal for the price-math parity.
    sample = predictions.iloc[[0]].assign(
        market_informed_home_margin=lambda f: f.home_margin
    )
    for market, side, line in [
        ("spreads", "home", -3.0),
        ("spreads", "away", 3.5),
        ("totals", "over", 43.0),
        ("totals", "under", 43.5),
    ]:
        canonical_line = -line if side == "away" else line
        result = outcome_probabilities(sample, market, [canonical_line])[0]
        if side in ("away", "under"):
            result = result[[2, 1, 0]]
        np.testing.assert_allclose(
            result,
            _probabilities(
                next(sample.itertuples()), dict(market=market, side=side, point=line)
            ),
            atol=1e-12,
        )

    # Key-score weighting must preserve coherent probabilities at every line,
    # including outside the weighted grid and at integer push boundaries.
    for market, size in [("spreads", 71), ("totals", 121)]:
        calibration = ProbabilityCalibration(
            1.0, 1.4, tuple(np.linspace(0.5, 2, size)), 0.3
        )
        thresholds = np.array([-200.0, -3.5, -3.0, 0.0, 3.0, 3.5, 43.0, 43.5, 300.0])
        lines = -thresholds if market == "spreads" else thresholds
        repeated = pd.concat([sample] * len(lines), ignore_index=True)
        repeated["market_reference"] = -3 if market == "spreads" else 43
        probabilities = outcome_probabilities(repeated, market, lines, calibration)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
        assert (np.diff(probabilities[:, 0]) <= 1e-12).all()
        assert (probabilities[thresholds % 1 != 0, 1] == 0).all()

    with pytest.raises(ValueError, match="precede kickoff"):
        calibrate_probabilities(predictions.assign(as_of=predictions.start_date))
    with pytest.raises(ValueError, match="excluding 2026"):
        calibrate_probabilities(
            pd.concat([predictions, predictions.iloc[[0]].assign(season=2026)])
        )
