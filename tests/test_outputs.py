from datetime import datetime, timezone

import pandas as pd
import pytest

from backend.model import GameProjection, TeamRating
from backend.model.market_blend import (
    add_market_informed_margins,
    align_ratings_to_forecast,
)


def test_output_contract_preserves_rating_and_market_conventions():
    as_of = datetime(2026, 8, 25, 12, tzinfo=timezone.utc)
    rating = TeamRating(
        season=2026,
        week=1,
        as_of=as_of,
        model_version="v3",
        team_id=333,
        team="Alabama",
        offense_points=8.0,
        defense_points=5.0,
        expected_possessions=11.4,
        power_rating_sd=2.1,
    )
    projection = GameProjection(
        season=2026,
        week=1,
        as_of=as_of,
        model_version="v3",
        game_id=1,
        home_team_id=333,
        home_team="Alabama",
        away_team_id=61,
        away_team="Georgia",
        neutral_site=False,
        home_field_points=2.0,
        expected_home_points=30.0,
        expected_away_points=22.5,
        margin_sd=14.0,
        total_sd=13.0,
        margin_total_correlation=0.1,
        degrees_of_freedom=6.0,
    )

    assert rating.to_record()["power_rating"] == 13.0
    assert rating.to_record()["scoring_environment"] == 3.0
    assert projection.to_record()["home_margin"] == 7.5
    assert projection.to_record()["home_spread"] == -7.5
    assert projection.to_record()["model_total"] == 52.5

    blended = add_market_informed_margins(
        pd.DataFrame([projection.to_record()]),
        pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "market": "spreads",
                    "selection": "home",
                    "point": -3.5,
                },
                {"game_id": 1, "market": "totals", "selection": "over", "point": 48.5},
            ]
        ),
    ).iloc[0]
    assert blended.pure_home_margin == 7.5
    assert blended.market_informed_home_margin == 5.5
    assert blended.market_weight == 0.5
    # The published scores carry the blended line and total, not the pure ones.
    assert blended.market_informed_total == 50.5
    assert blended.market_informed_home_points == 28.0
    assert blended.market_informed_away_points == 22.5
    # Published ratings reproduce the published line; a team without a game
    # this week keeps its fitted rating.
    aligned = align_ratings_to_forecast(
        pd.DataFrame(
            {
                "team_id": [333, 61, 99],
                "power_rating": [13.0, 9.0, 1.0],
                "offense_points": [8.0, 4.0, 1.0],
                "defense_points": [5.0, 5.0, 0.0],
            }
        ),
        pd.DataFrame([blended]),
    ).set_index("team_id")
    assert aligned.power_rating[333] - aligned.power_rating[61] + 2.0 == pytest.approx(
        5.5
    )
    assert aligned.forecast_alignment_points.tolist() == pytest.approx(
        [-0.25, 0.25, 0.0]
    )
    assert (aligned.offense_points + aligned.defense_points).tolist() == pytest.approx(
        aligned.power_rating.tolist()
    )
    unpriced = add_market_informed_margins(
        pd.DataFrame([projection.to_record()]),
        pd.DataFrame(),
    ).iloc[0]
    assert unpriced.market_informed_home_margin == unpriced.pure_home_margin
    assert unpriced.market_informed_home_points == unpriced.expected_home_points
    assert unpriced.market_weight == 0.0
    with pytest.raises(ValueError, match="market weight"):
        add_market_informed_margins(
            pd.DataFrame([projection.to_record()]),
            pd.DataFrame(),
            weight=0.51,
        )

    with pytest.raises(ValueError, match="neutral-site"):
        GameProjection(
            season=2026,
            week=1,
            as_of=as_of,
            model_version="v3",
            game_id=2,
            home_team_id=333,
            home_team="Alabama",
            away_team_id=61,
            away_team="Georgia",
            neutral_site=True,
            home_field_points=2.0,
            expected_home_points=28.0,
            expected_away_points=24.0,
            margin_sd=14.0,
            total_sd=13.0,
            margin_total_correlation=0.1,
            degrees_of_freedom=6.0,
        )
