import math

import pandas as pd
import pytest

from backend.model.availability import qb_out_points
from backend.players.traits.spread import shares, spread_values

TEAM_GAMES = pd.DataFrame(
    {
        "team": ["A", "A", "A", "A", "B"],
        "game_id": [1, 2, 3, 4, 2],
        "model_week": [1, 2, 3, 4, 2],
    }
)
PLAYED = pd.DataFrame(
    {
        "athlete_id": ["qb", "qb", "qb", "wr", "wr", "qb2"],
        "team": ["A", "A", "A", "A", "A", "A"],
        "game_id": [1, 2, 4, 3, 4, 3],
    }
)


def test_share_counts_team_games_through_the_week_weighted_by_age():
    flat = shares(PLAYED, TEAM_GAMES, 3, math.inf).set_index("athlete_id")["share"]
    assert flat["qb"] == pytest.approx(2 / 3)
    assert flat["wr"] == pytest.approx(1 / 3)

    # A one-week half-life weighs weeks 1-3 as 0.25, 0.5 and 1.
    decayed = shares(PLAYED, TEAM_GAMES, 3, 1.0).set_index("athlete_id")["share"]
    assert decayed["qb"] == pytest.approx(0.75 / 1.75)
    assert decayed["wr"] == pytest.approx(1 / 1.75)


def test_psr_is_the_forecast_move_for_each_teams_expected_starter():
    values = pd.DataFrame(
        {
            "week": [3, 4, 4, 4, 4],
            "athlete_id": ["qb", "qb", "qb2", "wr", "unplayed"],
            "team": ["A", "A", "A", "A", "A"],
            "position_group": ["QB", "QB", "QB", "WR", "QB"],
            "plays": [60.0, 90.0, 20.0, 15.0, 0.0],
            "shrunk_per_game": [4.0, 5.0, 6.0, 0.2, 3.0],
            "replacement_per_game": [1.0, 1.0, 1.0, 0.5, 1.0],
        }
    )
    out = spread_values(values, PLAYED, TEAM_GAMES, math.inf).set_index(
        ["week", "athlete_id"]
    )["value"]
    # Raw value is points above replacement per game times share.
    assert out[(3, "qb")] == pytest.approx(qb_out_points(3.0 * 2 / 3))
    assert out[(4, "qb")] == pytest.approx(qb_out_points(4.0 * 3 / 4))
    # Only the team's quarterback with the most plays carries PSR.
    assert list(out.index) == [(3, "qb"), (4, "qb")]
