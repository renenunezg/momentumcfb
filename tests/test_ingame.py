import pandas as pd

from backend.features import ingame

HOME = "Home U"
AWAY = "Away St"


def _play(play_id, period, minutes, seconds, offense, play_type, **overrides):
    row = {
        "game_id": 9001,
        "id": play_id,
        "drive_id": overrides.pop("drive_id", "d1"),
        "drive_number": overrides.pop("drive_number", 1),
        "play_number": overrides.pop("play_number", play_id),
        "offense": offense,
        "defense": AWAY if offense == HOME else HOME,
        "home": HOME,
        "away": AWAY,
        "offense_score": 0,
        "defense_score": 0,
        "offense_timeouts": 3,
        "defense_timeouts": 3,
        "period": period,
        "clock": {"minutes": minutes, "seconds": seconds},
        "down": 1,
        "distance": 10,
        "yards_to_goal": 75,
        "yards_gained": 0,
        "scoring": False,
        "play_type": play_type,
        "play_text": play_type,
        "ppa": 0.1,
        "season": 2024,
        "week": 5,
        "season_type": "regular",
    }
    row.update(overrides)
    return row


def _game_plays():
    return pd.DataFrame(
        [
            _play(1, 1, 15, 0, HOME, "Kickoff"),
            _play(2, 1, 14, 30, AWAY, "Rush", drive_number=2),
            _play(
                3,
                1,
                13,
                50,
                AWAY,
                "Passing Touchdown",
                drive_number=2,
                offense_score=7,
                scoring=True,
            ),
            _play(
                4,
                2,
                8,
                0,
                AWAY,
                "Timeout",
                drive_number=3,
                offense_score=7,
                offense_timeouts=2,
            ),
            # Pick six thrown by the home offense: points land in defense_score.
            _play(
                5,
                2,
                7,
                40,
                HOME,
                "Interception Return Touchdown",
                drive_number=4,
                defense_score=14,
                scoring=True,
            ),
            _play(
                6,
                4,
                0,
                30,
                HOME,
                "Rush",
                drive_number=5,
                defense_score=14,
            ),
            _play(
                7,
                5,
                0,
                0,
                HOME,
                "Rush",
                drive_number=6,
                defense_score=14,
            ),
        ]
    )


def test_states_are_presnap_and_map_scores_to_home_away():
    states = ingame.build_game_states(_game_plays())

    assert list(states["play_index"]) == [1, 2, 3, 4, 5, 6, 7]
    # Scores are pre-play: 0-0 through the away touchdown play itself,
    # then 0-7 until the pick six, then 0-14.
    assert list(states["home_score"]) == [0, 0, 0, 0, 0, 0, 0]
    assert list(states["away_score"]) == [0, 0, 0, 7, 7, 14, 14]
    assert list(states["home_margin"]) == [0, 0, 0, -7, -7, -14, -14]

    # The Timeout row reports the decremented count; the state keeps the
    # pre-snap value from the prior play.
    timeout_state = states[states["play_index"].eq(4)].iloc[0]
    assert timeout_state["away_timeouts"] == 3
    assert timeout_state["home_timeouts"] == 3

    assert states["seconds_remaining"].iloc[0] == 3600.0
    assert states["seconds_remaining"].iloc[5] == 30.0
    overtime_state = states.iloc[6]
    assert bool(overtime_state["is_overtime"])
    assert overtime_state["seconds_remaining"] == 0.0


def test_leakage_check_passes_clean_states_and_catches_lookahead(monkeypatch):
    plays = _game_plays()
    assert ingame.leakage_problems(plays, [9001]) == []

    real_classify = ingame.classify_plays

    def contaminated_classify(frame):
        classified = real_classify(frame)
        classified["offense_score"] = classified.groupby("game_id")[
            "offense_score"
        ].transform("max")
        return classified

    monkeypatch.setattr(ingame, "classify_plays", contaminated_classify)
    problems = ingame.leakage_problems(plays, [9001])
    assert problems and "change when later plays are removed" in problems[0]
