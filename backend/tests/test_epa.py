import pandas as pd

from backend.features.epa import flag_garbage_time, team_game_epa


def make_plays():
    return pd.DataFrame(
        {
            "game_id": [1] * 6,
            "season": [2024] * 6,
            "week": [1] * 6,
            "season_type": ["regular"] * 6,
            "offense": ["A", "A", "A", "B", "B", "A"],
            "defense": ["B", "B", "B", "A", "A", "B"],
            "offense_score": [0, 7, 14, 0, 0, 45],
            "defense_score": [0, 0, 0, 21, 21, 0],
            "period": [1, 1, 2, 2, 3, 4],
            "play_type": [
                "Rush",
                "Pass Reception",
                "Rush",
                "Pass Incompletion",
                "Rush",
                "Rush",
            ],
            "ppa": [0.5, 1.5, -0.2, -0.4, 0.3, 2.0],
        }
    )


def test_garbage_time_flag():
    plays = make_plays()
    flags = flag_garbage_time(plays)
    assert flags.tolist() == [False, False, False, False, False, True]


def test_team_game_epa_aggregates():
    out = team_game_epa(make_plays())
    a_all = out[(out.team == "A") & (out.variant == "all")].iloc[0]
    assert a_all.plays == 4
    assert abs(a_all.off_epa - (0.5 + 1.5 - 0.2 + 2.0) / 4) < 1e-9
    assert abs(a_all.total_off_epa - 3.8) < 1e-9
    assert abs(a_all.total_def_epa - (-0.4 + 0.3)) < 1e-9
    assert abs(a_all.run_epa - (0.5 - 0.2 + 2.0) / 3) < 1e-9
    assert abs(a_all.pass_epa - 1.5) < 1e-9
    assert abs(a_all.explosive_rate - 0.5) < 1e-9

    a_filt = out[(out.team == "A") & (out.variant == "filtered")].iloc[0]
    assert a_filt.plays == 3
    assert abs(a_filt.total_off_epa - 1.8) < 1e-9
