from datetime import datetime, timezone

import pytest

from backend.model import GameProjection, TeamRating


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
