import numpy as np
import pandas as pd
import pytest

from backend.etl import store
from backend.model.ingame import SERVING_ANCHOR_COLUMNS
from backend.serving.anchors import (
    load_serving_anchors,
    serving_anchor_artifact,
)
from backend.serving.market import MARKET_ANCHOR_WEEK, build_market_anchors


@pytest.fixture
def processed_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "PROCESSED_DIR", tmp_path)
    return tmp_path


def _projections(**overrides):
    frame = pd.DataFrame(
        {
            "game_id": [10, 11],
            "season": [2026, 2026],
            "week": [1, 1],
            "home_margin": [3.5, -7.0],
            "margin_sd": [12.0, 13.5],
            # Outcome-bearing columns must never leak into the contract.
            "actual_home_points": [70.0, 3.0],
        }
    )
    for column, values in overrides.items():
        frame[column] = values
    return frame


def test_projection_anchors_derive_model_week_and_round_trip(processed_dir):
    store.write_processed(
        _projections(), "preseason", "projections", "2026_01.parquet"
    )
    anchors = load_serving_anchors("preseason", season=2026, week=1)
    assert list(anchors.columns) == [
        "game_id",
        "model_week",
        "home_margin",
        "margin_sd",
    ]
    assert anchors["model_week"].tolist() == [1, 1]

    store.write_processed(anchors, *serving_anchor_artifact(2026, 1))
    reloaded = load_serving_anchors("serving", season=2026, week=1)
    assert reloaded.equals(anchors)


def test_loader_rejects_contract_violations(processed_dir):
    cases = [
        (_projections(game_id=[10, 10]), "one projection per game_id"),
        (_projections(margin_sd=[0.0, 13.5]), "margin_sd must be positive"),
        (_projections(home_margin=[np.nan, -7.0]), "home_margin must be finite"),
        (_projections(week=[1, 2]), "targets weeks other than 1"),
        (_projections(season=[2026, 2025]), "seasons other than 2026"),
        (
            _projections().drop(columns=["margin_sd"]),
            "missing anchor columns: margin_sd",
        ),
    ]
    for frame, message in cases:
        store.write_processed(
            frame, "preseason", "projections", "2026_01.parquet"
        )
        with pytest.raises(ValueError, match=message):
            load_serving_anchors("preseason", season=2026, week=1)


def test_market_anchors_flatten_sign_and_round_trip(processed_dir, monkeypatch):
    # A wrong median resolution or a flipped home_margin sign would serve
    # every live game from a corrupted anchor, so this boundary is pinned.
    lines = pd.DataFrame(
        {
            "game_id": [10, 11],
            "lines": [
                [
                    {"provider": "A", "spread": -2.5},
                    {"provider": "B", "spread": -3.5},
                    {"provider": "C", "spread": -10.0},
                ],
                [
                    {"provider": "A", "spread": 7.0},
                    {"provider": "B", "spread": None},
                ],
            ],
        }
    )
    games = pd.DataFrame(
        {
            "id": [10, 11],
            "week": [3, 1],
            "season_type": ["regular", "postseason"],
            "completed": [True, True],
            "home_points": [21.0, 10.0],
            "away_points": [17.0, 24.0],
        }
    )
    monkeypatch.setattr(store, "read_lines", lambda season: lines)
    monkeypatch.setattr(store, "read_games", lambda season: games)

    anchors = build_market_anchors(2026)
    # median across priced offers, unpriced offers skipped, home = -spread
    assert anchors["closing_spread"].tolist() == [-3.5, 7.0]
    assert anchors["home_margin"].tolist() == [3.5, -7.0]
    # postseason weeks offset past the schedule's last regular week
    assert anchors["model_week"].tolist() == [3, 4]
    assert (anchors["margin_sd"] > 0).all()
    assert anchors["margin_sd_method"].str.contains("development seasons").all()

    store.write_processed(
        anchors, *serving_anchor_artifact(2026, MARKET_ANCHOR_WEEK)
    )
    reloaded = load_serving_anchors(
        "serving", season=2026, week=MARKET_ANCHOR_WEEK
    )
    assert reloaded.equals(anchors[SERVING_ANCHOR_COLUMNS])
