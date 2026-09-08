"""Grading uses only pregame frozen records and keeps stored grades verbatim."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from backend import grading
from backend.model.calibration import _interval_coverage
from backend.odds.markets import OFFER_COLUMNS, compare_priced_offers


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
        "home_spread": -margin,
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

    # Grading, priced probabilities and calibration share the same SD contract.
    student = projections.iloc[[0]].assign(degrees_of_freedom=7.0)
    student_grade = grading.build_graded_games(2026, student)
    expected = stats.t.cdf(7.0 / (14.0 * np.sqrt(5.0 / 7.0)), 7.0)
    assert student_grade.iloc[0].home_win_probability == pytest.approx(expected)
    offers = pd.DataFrame(
        [
            {
                "game_id": 1,
                "market": "spreads",
                "selection": "home",
                "point": 0.0,
                "price": -110.0,
                "execution_eligibility_verified": False,
            }
        ]
    ).reindex(columns=OFFER_COLUMNS)
    priced = compare_priced_offers(student, offers)
    assert priced.iloc[0].best_offer_model_cover_probability == pytest.approx(expected)
    boundary = student_grade.assign(actual_margin=7.0 + 18.0)
    assert grading._coverage(boundary, 0.8) == 0.0
    assert grading._coverage(boundary, 0.8) == _interval_coverage(
        np.array([18.0]), np.array([14.0]), np.array([7.0]), 0.8
    )

    # Repair the known derived-probability bug without changing frozen inputs.
    legacy = student_grade.assign(
        home_win_probability=stats.t.cdf(0.5, 7.0),
        probability_method=grading.LEGACY_PROBABILITY_METHOD,
    )
    migrated = grading.build_graded_games(2026, student, existing=legacy)
    assert migrated.iloc[0].home_win_probability == pytest.approx(expected)
    frozen_columns = [
        c for c in legacy if c not in {"home_win_probability", "probability_method"}
    ]
    pd.testing.assert_frame_equal(migrated[frozen_columns], legacy[frozen_columns])


def test_recommendation_flags_and_settlement_use_recorded_prices(monkeypatch):
    from backend.model import pick_calibration
    from backend.recommendations import build_recommendations, grade_recommendations

    # Forward picks and their settlement must work without research approval or
    # market-relative calibration, even when that research finds no model edge.
    def research_must_not_gate_picks(*args, **kwargs):
        raise AssertionError("historical calibration entered the recommendation path")

    monkeypatch.setattr(
        pick_calibration, "outcome_probabilities", research_must_not_gate_picks
    )
    monkeypatch.setattr(
        pick_calibration, "calibrate_probabilities", research_must_not_gate_picks
    )

    now = pd.Timestamp("2026-08-29T12:00:00Z")
    projections = pd.DataFrame(
        [_projection(i, now, margin=14 if i < 4 else -14) for i in range(1, 10)]
    )
    offers = pd.DataFrame(
        [
            dict(
                game_id=i,
                market=market,
                selection=side,
                point=point,
                price=price,
                provider="Book A",
                provider_key="book_a",
                provider_last_update=now,
                market_fetched_at=now,
                execution_eligibility_verified=True,
                match_score=1.0,
                odds_api_event_id=f"event-{i}",
                commence_time="2026-08-29T16:00:00Z",
            )
            for i in range(1, 10)
            for market, side, point, price in (
                ("h2h", "home", None, -180),
                ("h2h", "away", None, 160),
                ("spreads", "home", -3.0, -110),
                ("spreads", "away", 3.0, 120),
                ("totals", "over", 43.0, -110),
                ("totals", "under", 43.0, -110),
            )
        ]
    ).reindex(columns=OFFER_COLUMNS)
    projections.loc[projections.game_id.eq(4), "model_total"] = 36.0
    # Production includes the universal unavailable-injury flag on both teams.
    projections.loc[
        projections.game_id.eq(5),
        ["home_missing_input_count", "away_missing_input_count"],
    ] = 1
    # Gates apply per offer. Neither stale nor incomplete evidence earns a pick.
    projections.loc[projections.game_id.eq(6), "home_missing_input_count"] = 2
    offers.loc[offers.game_id.eq(7), "provider_last_update"] = now - pd.Timedelta(
        hours=2
    )
    offers.loc[offers.game_id.eq(8), "odds_api_event_id"] = None
    offers = offers[~(offers.game_id.eq(9) & offers.selection.isin(["home", "under"]))]
    decisions = build_recommendations(projections, offers, decision_at=now)
    assert decisions[decisions.game_id.le(5)].status.eq("recommended").all()
    assert decisions[decisions.game_id.ge(6)].status.eq("no_play").all()
    # Sides are priced from the market-informed margin, not the pure margin.
    blended = projections.set_index("game_id").market_informed_home_margin
    assert decisions.model_home_margin.eq(decisions.game_id.map(blended)).all()
    home = decisions[(decisions.game_id.eq(1)) & decisions.market.eq("spreads")].iloc[0]
    assert home.side == "home" and home.point == -3 and home.price == -110
    assert home.push_probability > 0
    assert home.expected_value_per_unit == pytest.approx(
        home.win_probability * 100 / 110
        - (1 - home.win_probability - home.push_probability)
    )
    assert decisions[decisions.game_id.eq(9)].reason.eq("unpaired_market").all()
    assert (
        build_recommendations(
            projections, offers, decision_at=now + pd.Timedelta(days=1)
        )
        .status.eq("no_play")
        .all()
    )
    # An eligible smaller edge wins over an unverified book with a higher EV.
    bad_book = offers[offers.game_id.eq(1)].assign(
        provider_key="bad", odds_api_event_id=None, price=300
    )
    selected = build_recommendations(
        projections.iloc[:1], pd.concat([offers, bad_book]), decision_at=now
    )
    assert selected.provider_key.eq("book_a").all()

    frozen = decisions.assign(outcome="pending")
    games = pd.DataFrame(
        [
            dict(
                id=i,
                start_date="2026-08-29T16:00:00Z",
                completed=True,
                home_team="Home",
                away_team="Away",
                home_points=home_score,
                away_points=away_score,
            )
            for i, home_score, away_score in (
                (1, 24, 20),
                (2, 23, 20),
                (3, 20, 23),
                (4, 20, 24),
                (5, 24, 20),
                (6, 24, 20),
                (7, 24, 20),
                (8, 24, 20),
                (9, 24, 20),
            )
        ]
    )
    # Missing kickoff data is not a postponement. Identity must be checked
    # even when the source also changes the kickoff.
    assert grade_recommendations(frozen, games.assign(start_date=None)).empty
    with pytest.raises(ValueError, match="identity changed"):
        grade_recommendations(frozen, games.assign(start_date=None, home_team="Wrong"))
    # Re-scheduling invalidates the recommendation at its original kickoff.
    games.loc[games.id.eq(5), "start_date"] = "2026-08-30T16:00:00Z"
    grades = grade_recommendations(frozen, games, graded_at=now + pd.Timedelta(days=2))
    spread = grades[grades.market.eq("spreads")].set_index("game_id")
    assert spread.loc[1, "outcome"] == "win"
    assert spread.loc[1, "profit_units"] == pytest.approx(100 / 110)
    assert spread.loc[2, "outcome"] == "push" and spread.loc[2, "profit_units"] == 0
    assert spread.loc[3, "outcome"] == "loss" and spread.loc[3, "profit_units"] == -1
    assert spread.loc[4, "outcome"] == "win" and spread.loc[4, "profit_units"] == 1.2
    assert spread.loc[5, "outcome"] == "void" and spread.loc[5, "profit_units"] == 0
    assert grades[grades.game_id.ge(6)].outcome.eq("no_play").all()
    moneyline = grades[grades.market.eq("h2h")].set_index("game_id")
    assert moneyline.loc[1, "outcome"] == "win"
    assert moneyline.loc[1, "profit_units"] == pytest.approx(100 / 180)
    assert moneyline.loc[3, "outcome"] == "loss"
    assert moneyline.loc[4, "profit_units"] == pytest.approx(1.6)
    assert moneyline.loc[5, "outcome"] == "void"
    assert decisions.loc[decisions.market.eq("h2h"), "point"].isna().all()
    assert decisions.loc[decisions.market.eq("h2h"), "push_probability"].eq(0).all()
    tied = grade_recommendations(
        frozen,
        games.assign(home_points=20, away_points=20),
        graded_at=now + pd.Timedelta(days=2),
    )
    assert (
        tied.loc[tied.market.eq("h2h") & tied.game_id.le(5), "outcome"].eq("void").all()
    )
    totals = grades[grades.market.eq("totals")].set_index("game_id")
    assert totals.loc[1, "outcome"] == "win"
    assert totals.loc[2, "outcome"] == "push"
    assert totals.loc[4, "outcome"] == "loss"
    mismatch = offers[offers.game_id.eq(1)].assign(
        commence_time=now - pd.Timedelta(hours=1)
    )
    assert (
        build_recommendations(projections.iloc[:1], mismatch, decision_at=now)
        .reason.eq("kickoff_mismatch")
        .all()
    )
    # Regional-feed quotes can qualify without a user-specified bookmaker list.
    regional = offers[offers.game_id.eq(1)].assign(execution_eligibility_verified=False)
    assert (
        build_recommendations(projections.iloc[:1], regional, decision_at=now)
        .status.eq("recommended")
        .all()
    )
    # A final-score correction never silently rewrites an already settled pick.
    settled = frozen.drop(columns="outcome").merge(
        grades[["game_id", "market", "outcome"]], on=["game_id", "market"]
    )
    assert grade_recommendations(
        settled, games.assign(home_points=99), graded_at=now + pd.Timedelta(days=3)
    ).empty
