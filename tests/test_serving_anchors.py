import numpy as np
import pandas as pd
import pytest

from backend.etl import store
from backend.serving.anchors import (
    load_serving_anchors,
    serving_anchor_artifact,
)


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
