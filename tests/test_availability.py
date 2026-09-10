import pandas as pd

from backend.model.availability import (
    QB_OUT_POINTS,
    QB_OUT_POSTSEASON_POINTS,
    apply_qb_availability,
    pregame_qb_outs,
)


def test_only_pregame_reports_adjust_and_a_re_decision_never_double_counts():
    reports = pd.DataFrame(
        [
            dict(season=2026, week=3, team="Utah", status="out", reported_at="2026-09-09T12:00:00Z"),
            dict(season=2026, week=3, team="Auburn", status="doubtful", reported_at="2026-09-12T12:00:00Z"),
            dict(season=2026, week=3, team="Ohio", status="questionable", reported_at="2026-09-09T12:00:00Z"),
            dict(season=2026, week=4, team="Georgia", status="out", reported_at="2026-09-09T12:00:00Z"),
        ]
    )
    forecast_cutoff = pd.Timestamp("2026-09-10T18:00:00Z")
    assert pregame_qb_outs(reports, 2026, 3, forecast_cutoff) == {"Utah"}
    projections = pd.DataFrame(
        [
            dict(game_id=1, home_team="Utah", away_team="Arkansas", expected_home_points=30.0, expected_away_points=20.0, home_margin=10.0, home_spread=-10.0, model_total=50.0),
            dict(game_id=2, home_team="Ohio", away_team="Auburn", expected_home_points=21.0, expected_away_points=28.0, home_margin=-7.0, home_spread=7.0, model_total=49.0),
        ]
    )
    forecast = apply_qb_availability(projections, {"Utah"}, postseason_game_ids={2})
    utah = forecast.iloc[0]
    assert utah.home_qb_out and not utah.away_qb_out
    assert utah.expected_home_points == 30 - QB_OUT_POINTS
    assert utah.home_margin == 10 - QB_OUT_POINTS and utah.home_spread == -utah.home_margin
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
