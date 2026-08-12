import pandas as pd
import pytest

from backend.etl import store
from backend.features.ingame import build_game_states
from backend.model.ingame import (
    OUTCOME_COLUMNS,
    IngameBaselineParams,
    build_baseline_inputs,
    win_probability,
)
from backend.serving.replay import replay_game, stream_problems
from backend.serving.serve import MissingPlayFeed, serve_game

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

    # The serving path must score without knowing the result: stream from an
    # anchor that carries only the pregame projection columns.
    serving_anchor = anchor.drop(
        columns=["actual_home_points", "actual_away_points"]
    )
    events = replay_game(
        plays.sample(frac=1, random_state=7), serving_anchor, PARAMS
    )

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


def test_serve_game_reads_no_outcome_bearing_data(tmp_path, monkeypatch):
    """The serving entry point must be scoreable before a game has a result.

    Serving a 2026 game depends on this: the only artifacts it may open are
    the anchor contract columns, the frozen parameters, and the play feed.
    """
    monkeypatch.setattr(store, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(store, "RAW_DIR", tmp_path / "raw")

    plays = pd.DataFrame(
        [
            _play(1, 1, 15, 0, HOME, "Kickoff", season=2026),
            _play(2, 1, 14, 30, AWAY, "Rush", drive_number=2, season=2026),
            _play(3, 2, 9, 0, HOME, "Rush", drive_number=3, season=2026),
        ]
    )
    feed_dir = tmp_path / "raw" / "pbp" / "2026"
    feed_dir.mkdir(parents=True)
    plays.to_parquet(feed_dir / "regular_05.parquet", index=False)
    # Every source artifact carries outcome columns alongside the contract, so
    # a leak would be an ordinary column read rather than a missing file. Game
    # 9002 is anchored but unplayed.
    store.write_processed(
        pd.DataFrame(
            [
                {
                    "season": 2026,
                    "game_id": game_id,
                    "model_week": 6,
                    "home_margin": -3.5,
                    "margin_sd": 11.0,
                    "actual_home_points": 20.0,
                    "actual_away_points": 27.0,
                }
                for game_id in (9001, 9002)
            ]
        ),
        "calibration",
        "joint_scoring_predictions.parquet",
    )
    store.write_processed(
        pd.DataFrame([{"summary_type": "parameter", **PARAMS.to_record()}]),
        "ingame",
        "baseline_summary.parquet",
    )
    store.write_processed(
        pd.DataFrame([{"game_id": 9001, "win_probability": 0.5, "home_win": 1}]),
        "ingame",
        "baseline_predictions.parquet",
    )

    reads = []
    original = pd.read_parquet

    def record(path, *args, **kwargs):
        reads.append((str(path), kwargs.get("columns")))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", record)

    events = serve_game(2026, 9001, source="calibration")

    assert not [path for path, _ in reads if "baseline_predictions" in path]
    anchor_columns = [
        columns
        for path, columns in reads
        if "joint_scoring_predictions" in path and columns is not None
    ]
    assert anchor_columns and not [
        column
        for columns in anchor_columns
        for column in columns
        if column in OUTCOME_COLUMNS
    ]
    assert not [
        column for column in events.columns if column in OUTCOME_COLUMNS
    ]
    assert int(events["emitted"].sum()) == len(plays)

    # An anchored game with no feed is the ordinary pregame state, not an error
    # about the anchors.
    with pytest.raises(MissingPlayFeed, match="no play feed for game 9002"):
        serve_game(2026, 9002, source="calibration")
