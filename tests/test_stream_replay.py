import pandas as pd

from backend.features.ingame import build_game_states
from backend.model.ingame import (
    IngameBaselineParams,
    build_baseline_inputs,
    win_probability,
)
from backend.serving.replay import replay_game, stream_problems

HOME = "Home U"
AWAY = "Away St"

PARAMS = IngameBaselineParams(
    possession_points=0.3,
    field_position_points_per_yard=0.05,
    sd_floor_points=3.4,
)


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


def test_streamed_probabilities_equal_batch_and_skip_filtered_rows():
    plays = pd.DataFrame(
        [
            _play(1, 1, 15, 0, HOME, "Kickoff"),
            _play(2, 1, 14, 30, AWAY, "Rush", drive_number=2),
            _play(
                3, 1, 13, 50, AWAY, "Passing Touchdown",
                drive_number=2, offense_score=7, scoring=True,
            ),
            # Null clock inside regulation: the batch drops this boundary, so
            # the stream must produce an event without a probability.
            _play(4, 2, 0, 0, HOME, "Rush", drive_number=3, clock=None),
            _play(
                5, 3, 10, 0, HOME, "Timeout",
                drive_number=4, defense_score=7, offense_timeouts=2,
            ),
            _play(6, 4, 0, 30, HOME, "Rush", drive_number=5, defense_score=7),
            _play(7, 5, 0, 0, HOME, "Rush", drive_number=6, defense_score=7),
        ]
    )
    anchor = pd.DataFrame(
        [
            {
                "game_id": 9001,
                "model_week": 6,
                "home_margin": -3.5,
                "margin_sd": 11.0,
                "actual_home_points": 20.0,
                "actual_away_points": 27.0,
            }
        ]
    )
    stored = build_baseline_inputs(build_game_states(plays), anchor)
    stored["win_probability"] = win_probability(stored, PARAMS)

    events = replay_game(plays.sample(frac=1, random_state=7), anchor, PARAMS)

    assert stream_problems(events, stored) == []
    assert len(events) == len(plays)
    assert not events.loc[events["play_index"].eq(4), "emitted"].item()
    assert int(events["emitted"].sum()) == len(stored) == len(plays) - 1

    # The checker itself must catch a divergence, or the season-wide
    # equivalence proof proves nothing.
    perturbed = stored.copy()
    perturbed.loc[perturbed.index[2], "win_probability"] += 1e-12
    problems = stream_problems(events, perturbed)
    assert problems and "probabilities differ" in problems[0]
