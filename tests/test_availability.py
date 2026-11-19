import pandas as pd
import pytest

from backend.model import availability
from backend.model.availability import (
    UNPRICED_STARTER_POINTS,
    apply_qb_availability,
    pregame_qb_outs,
    starter_spread_values,
)

# Auburn's starter is unpriced below and plays a bowl: the average change plus
# the bowl intercept's excess, summed in the order the adjustment sums them.
QB_OUT_POSTSEASON_POINTS = UNPRICED_STARTER_POINTS + (
    availability.QB_OUT_POSTSEASON_POINTS - availability.QB_OUT_POINTS
)


def test_only_pregame_reports_adjust_and_a_re_decision_never_double_counts():
    reports = pd.DataFrame(
        [
            dict(
                season=2026,
                week=3,
                team="Utah",
                player="Cam Rising",
                status="out",
                reported_at="2026-09-09T12:00:00Z",
            ),
            dict(
                season=2026,
                week=3,
                team="Auburn",
                player="Payton Thorne",
                status="doubtful",
                reported_at="2026-09-12T12:00:00Z",
            ),
            dict(
                season=2026,
                week=3,
                team="Ohio",
                player="Parker Navarro",
                status="questionable",
                reported_at="2026-09-09T12:00:00Z",
            ),
            dict(
                season=2026,
                week=4,
                team="Georgia",
                player="Gunner Stockton",
                status="out",
                reported_at="2026-09-09T12:00:00Z",
            ),
        ]
    )
    forecast_cutoff = pd.Timestamp("2026-09-10T18:00:00Z")
    assert pregame_qb_outs(reports, 2026, 3, forecast_cutoff) == {"Utah"}
    # Utah's starter is priced by name from the PSR published before the cutoff,
    # and the line moves by exactly that PSR; a later publish is ignored.
    published_psr = pd.DataFrame(
        dict(
            team=["Utah", "Utah", "Auburn"],
            athlete_name=["Cam Rising", "Cam Rising", "Payton Thorne"],
            value=[3.0, 9.0, 2.0],
            as_of=[
                "2026-09-08T12:00:00Z",
                "2026-09-11T12:00:00Z",
                "2026-09-11T12:00:00Z",
            ],
        )
    )
    starter_psr = starter_spread_values(
        reports, published_psr, 2026, 3, forecast_cutoff
    )
    assert starter_psr == {"Utah": 3.0}
    QB_OUT_POINTS = 3.0
    projections = pd.DataFrame(
        [
            dict(
                game_id=1,
                home_team="Utah",
                away_team="Arkansas",
                expected_home_points=30.0,
                expected_away_points=20.0,
                home_margin=10.0,
                home_spread=-10.0,
                model_total=50.0,
            ),
            dict(
                game_id=2,
                home_team="Ohio",
                away_team="Auburn",
                expected_home_points=21.0,
                expected_away_points=28.0,
                home_margin=-7.0,
                home_spread=7.0,
                model_total=49.0,
            ),
        ]
    )
    forecast = apply_qb_availability(
        projections, {"Utah"}, postseason_game_ids={2}, starter_psr=starter_psr
    )
    utah = forecast.iloc[0]
    assert utah.home_qb_out and not utah.away_qb_out
    assert utah.expected_home_points == 30 - QB_OUT_POINTS
    assert (
        utah.home_margin == 10 - QB_OUT_POINTS and utah.home_spread == -utah.home_margin
    )
    assert utah.model_total == 50 - QB_OUT_POINTS
    assert utah.qb_availability_points == -QB_OUT_POINTS
    assert not forecast.iloc[1][["home_qb_out", "away_qb_out"]].any()
    assert forecast.iloc[1].qb_availability_points == 0
    # Re-deciding the published forecast three days later: Auburn's report is
    # now pregame and applies at the bowl size, Utah is already priced in, and
    # the market-informed line moves only by its model share.
    published = forecast.assign(
        pure_home_margin=forecast.home_margin,
        pure_home_spread=forecast.home_spread,
        market_weight=0.5,
        market_informed_home_margin=[0.0, -4.0],
        market_informed_home_spread=[0.0, 4.0],
        market_informed_total=[48.5, 49.0],
        market_total_weight=[0.0, 0.5],
        market_informed_home_points=[24.25, 22.5],
        market_informed_away_points=[24.25, 26.5],
    )
    decided = apply_qb_availability(
        published,
        pregame_qb_outs(reports, 2026, 3, pd.Timestamp("2026-09-12T15:00:00Z")),
        postseason_game_ids={2},
    )
    assert decided.iloc[0].expected_home_points == 30 - QB_OUT_POINTS
    assert decided.iloc[0].market_informed_home_margin == 0.0
    game = decided.iloc[1]
    assert game.away_qb_out and not game.home_qb_out
    assert game.expected_away_points == 28 - QB_OUT_POSTSEASON_POINTS
    assert game.pure_home_margin == -7 + QB_OUT_POSTSEASON_POINTS
    assert game.market_informed_home_margin == -4 + 0.5 * QB_OUT_POSTSEASON_POINTS
    assert game.qb_availability_points == QB_OUT_POSTSEASON_POINTS
    assert game.market_informed_total == 49.0 - 0.5 * QB_OUT_POSTSEASON_POINTS
    assert game.market_informed_home_points + game.market_informed_away_points == (
        pytest.approx(game.market_informed_total)
    )
    assert game.market_informed_home_points - game.market_informed_away_points == (
        pytest.approx(game.market_informed_home_margin)
    )
    assert published.iloc[1].market_informed_total == 49.0

    # Reproduced on the calibrated California-Wagner projection: a 0.55-point
    # away score was floored at zero but the margin and total still moved 1.5.
    near_zero = projections.iloc[[0]].assign(
        expected_home_points=30.0,
        expected_away_points=0.55,
        home_margin=29.45,
        home_spread=-29.45,
        model_total=30.55,
        pure_home_margin=29.45,
        pure_home_spread=-29.45,
        market_weight=0.5,
        market_history_weight=0.25,
        market_informed_home_margin=20.0,
        market_informed_home_spread=-20.0,
    )
    bounded = apply_qb_availability(near_zero, {"Arkansas"})
    row = bounded.iloc[0]
    assert row.expected_away_points == 0
    assert row.model_total == row.expected_home_points + row.expected_away_points
    assert row.home_margin == row.expected_home_points - row.expected_away_points
    assert row.pure_home_margin == row.home_margin
    assert row.market_informed_home_margin == 20.0 + 0.5 * 0.75 * 0.55
    pd.testing.assert_frame_equal(bounded, apply_qb_availability(bounded, {"Arkansas"}))
