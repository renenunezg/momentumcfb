"""Grading uses only pregame frozen records and keeps stored grades verbatim."""

from datetime import datetime, timezone

import pandas as pd

from backend import grading


def _projection(game_id, as_of, margin=7.0):
    return {
        "game_id": game_id,
        "season": 2026,
        "week": 1,
        "as_of": as_of,
        "model_version": "test",
        "start_date": "2026-08-29T16:00:00+00:00",
        "home_team_id": 1,
        "home_team": "Home",
        "away_team_id": 2,
        "away_team": "Away",
        "neutral_site": False,
        "conference_game": False,
        "home_classification": "fbs",
        "away_classification": "fbs",
        "home_missing_input_count": 0,
        "away_missing_input_count": 0,
        "home_margin": margin,
        "pure_home_margin": margin,
        "market_informed_home_margin": margin - 1.0,
        "market_weight": 0.5,
        "market_home_spread": -5.0,
        "model_total": 50.0,
        "margin_sd": 14.0,
        "total_sd": 12.0,
        "distribution": "bivariate_student_t",
        "degrees_of_freedom": 500.0,
    }


def test_grades_only_pregame_projections_and_keeps_stored_rows(monkeypatch, tmp_path):
    games = pd.DataFrame(
        [
            {
                "id": 1,
                "week": 1,
                "completed": True,
                "home_points": 24,
                "away_points": 20,
            },
            {
                "id": 2,
                "week": 1,
                "completed": True,
                "home_points": 10,
                "away_points": 30,
            },
            {
                "id": 3,
                "week": 1,
                "completed": False,
                "home_points": None,
                "away_points": None,
            },
            {"id": 4, "week": 1, "completed": True, "home_points": 3, "away_points": 0},
        ]
    ).assign(season_type="regular")
    lines = pd.DataFrame(
        {
            "game_id": [1, 2],
            "lines": [
                [
                    {"spread": -3.0, "overUnder": 44.0},
                    {"spread": -4.0, "overUnder": 46.0},
                ],
                [{"spread": 6.5, "overUnder": None}],
            ],
        }
    )
    raw = tmp_path / "games.parquet"
    raw.write_bytes(b"")
    monkeypatch.setattr(grading.store, "read_games", lambda season: games)
    monkeypatch.setattr(grading.store, "read_lines", lambda season: lines)
    monkeypatch.setattr(grading.store, "raw_path", lambda kind, season: raw)

    projections = pd.DataFrame(
        [
            _projection(1, "2026-08-27T18:00:00+00:00"),
            # Published after kickoff: not a forecast, never graded.
            _projection(2, "2026-08-29T17:00:00+00:00"),
            _projection(3, "2026-08-27T18:00:00+00:00"),
            _projection(4, "2026-08-27T18:00:00+00:00"),
        ]
    )
    stored = grading.build_graded_games(2026, projections.iloc[[3]]).assign(
        graded_at=datetime(2026, 8, 30, tzinfo=timezone.utc)
    )
    graded = grading.build_graded_games(2026, projections, existing=stored)

    assert graded["game_id"].tolist() == [1, 4]
    first = graded.set_index("game_id").loc[1]
    assert first["actual_margin"] == 4
    assert first["closing_spread"] == -3.5
    assert first["closing_total"] == 45.0
    assert first["n_spread_offers"] == 2
    assert 0.5 < first["home_win_probability"] < 1.0
    kept = graded.set_index("game_id").loc[4]
    assert kept["graded_at"] == pd.Timestamp("2026-08-30", tz="UTC")
    assert pd.isna(kept["closing_spread"])

    metrics = grading.compute_performance_metrics(graded)
    overall = metrics[metrics["segment_kind"].eq("overall")].set_index(
        "prediction_source"
    )
    assert overall.loc["pure_model", "games"] == 2
    assert overall.loc["pure_model", "games_with_market"] == 1
    assert overall.loc["pure_model", "margin_mae"] == 3.5
    assert overall.loc["closing_market", "games"] == 1
    assert overall.loc["closing_market", "margin_mae"] == 0.5
    assert overall.loc["pure_model", "model_minus_market_mae"] == 2.5
    assert bool(overall.loc["pure_model", "thin_sample"]) is True
