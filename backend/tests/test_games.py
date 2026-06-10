import pandas as pd

from backend.features.games import build_games, flatten_lines


def test_flatten_lines_takes_median_spread():
    lines = pd.DataFrame(
        {
            "game_id": [10],
            "lines": [
                [
                    {"provider": "Bovada", "spread": -7.0},
                    {"provider": "DraftKings", "spread": -7.5},
                    {"provider": "ESPN Bet", "spread": -8.0},
                ]
            ],
        }
    )
    out = flatten_lines(lines)
    assert out.loc[0, "closing_spread"] == -7.5


def make_inputs():
    games = pd.DataFrame(
        {
            "id": [10, 11],
            "season": [2024, 2024],
            "week": [1, 1],
            "season_type": ["regular", "regular"],
            "completed": [True, True],
            "neutral_site": [False, True],
            "home_team": ["A", "C"],
            "away_team": ["B", "Tiny State"],
            "home_classification": ["fbs", "fbs"],
            "away_classification": ["fbs", "fcs"],
            "home_points": [28, 35],
            "away_points": [21, 3],
        }
    )
    lines = pd.DataFrame(
        {"game_id": [10], "lines": [[{"provider": "Bovada", "spread": -3.0}]]}
    )
    epa = pd.DataFrame(
        {
            "game_id": [10, 10, 11, 11],
            "team": ["A", "B", "C", "Tiny State"],
            "variant": ["filtered"] * 4,
            "total_off_epa": [9.0, 4.0, 20.0, -5.0],
        }
    )
    return games, lines, epa


def test_build_games_joins_and_pools_fcs():
    games, lines, epa = make_inputs()
    out = build_games(games, lines, epa, variant="filtered")

    g10 = out[out.game_id == 10].iloc[0]
    assert g10.margin == 7
    assert g10.closing_spread == -3.0
    assert g10.epa_margin == 5.0
    assert g10.week_index == 0

    g11 = out[out.game_id == 11].iloc[0]
    assert g11.away_team == "FCS"
    assert pd.isna(g11.closing_spread)
    assert g11.epa_margin == 25.0
