"""The share model behaves like a ballot and the seed match fails loudly."""

import numpy as np
import pandas as pd
import pytest

from backend.players import heisman


def _synthetic_seasons(seed: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    truth = rng.normal(0.0, 1.0, len(heisman.FEATURES))
    frames = []
    for season in range(2010, 2018):
        design = rng.normal(0.0, 1.0, (40, len(heisman.FEATURES)))
        logits = 2.5 * (design @ truth)
        share = np.exp(logits - logits.max())
        share /= share.sum()
        frame = pd.DataFrame(design, columns=heisman.FEATURES)
        frame["season"] = season
        frame["athlete_id"] = [f"{season}-{index}" for index in range(40)]
        frame["athlete_name"] = frame["athlete_id"]
        frame["team"] = "T"
        frame["share"] = share
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), truth


def test_predicted_shares_form_a_ballot_and_recover_the_signal():
    rows, truth = _synthetic_seasons()
    model = heisman.fit_share_model(rows)
    rows["predicted_share"] = model.predict(rows)
    sums = rows.groupby("season")["predicted_share"].sum()
    assert np.allclose(sums, 1.0)
    assert np.corrcoef(model.beta, truth)[0, 1] > 0.9
    history = heisman.leave_one_season_out(rows)
    assert history["season"].tolist() == sorted(rows["season"].unique())
    assert history["top_three_hit"].mean() >= 0.75
    assert (history["actual_winner_predicted_share"] > 1 / 40).all()


def test_unmatched_ballot_row_raises(monkeypatch):
    players = pd.DataFrame(
        {
            "athlete_id": ["1"],
            "athlete_name": ["Real Player"],
            "team": ["State"],
            "games": [12.0],
        }
    )
    monkeypatch.setattr(
        heisman,
        "team_context",
        lambda season, through_week=None: pd.DataFrame(
            columns=["team", "win_pct", "ap_rank", "rank_points", "power_conference"]
        ),
    )
    monkeypatch.setattr(
        heisman,
        "read_season_source",
        lambda season, name: pd.DataFrame(
            columns=["id", "first_name", "last_name", "team", "position"]
        ),
    )
    seed = pd.DataFrame(
        [
            {"season": 2024, "player": "Real Player", "school": "State", "points": 100},
            {"season": 2024, "player": "Nobody Here", "school": "State", "points": 50},
        ]
    )
    with pytest.raises(ValueError, match="Nobody Here"):
        heisman._match_seed(seed, players, 2024)
    matched, _ = heisman._match_seed(seed.iloc[:1], players, 2024)
    assert matched["athlete_id"].tolist() == ["1"]
