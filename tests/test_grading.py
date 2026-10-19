"""Grading uses only pregame frozen records and keeps stored grades verbatim."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from backend import grading
from backend.model.calibration import _interval_coverage
from backend.odds.markets import OFFER_COLUMNS, compare_priced_offers
from backend.recommendations import PRICING_MARGIN_SD


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
        "market_home_spread": -(margin - 2.0),
        "model_total": 50.0,
        "market_informed_total": 46.5,
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

    from backend.diagnostics import scoring_diagnostics

    details = projections.assign(expected_game_possessions=12.0)
    scoring_games = games.rename(columns={"id": "game_id"}).assign(
        game_possessions=10.0
    )
    errors, summary = scoring_diagnostics(graded, details, scoring_games)
    assert np.allclose(
        errors.pace_error_points + errors.efficiency_error_points,
        errors.total_error,
    )
    assert summary.pace_decomposition_games.sum() == len(graded)
    # An updated forecast for the same game must not replace the frozen pace.
    mismatched = details.assign(as_of="2026-08-28T00:00:00+00:00")
    errors, _ = scoring_diagnostics(graded, mismatched, scoring_games)
    assert errors.expected_game_possessions.isna().all()

    import json

    from backend.diagnostics import grade_shadow_totals, shadow_totals

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "contract": "fall_d1_historical_carryover_v1",
                "development_seasons": [2020, 2021, 2022],
                "validation_seasons": [2023, 2024, 2025],
            }
        )
    )
    pd.DataFrame({"total_adjustment": [-0.8]}, index=["FBS vs FBS"]).to_parquet(
        tmp_path / "totals_offsets.parquet"
    )
    shadow_at = pd.Timestamp("2026-08-28T00:00:00Z")
    shadow = shadow_totals(projections, tmp_path, shadow_at)
    assert set(shadow.game_id) == {1, 3, 4}
    assert shadow.shadow_created_at.eq(shadow_at).all()
    assert shadow.model_total.eq(50).all()
    assert shadow.shadow_total.eq(49.2).all()
    extreme = projections.iloc[[0]].assign(pure_home_margin=50.0)
    clipped = shadow_totals(extreme, tmp_path, shadow_at)
    assert clipped.shadow_away_points.iloc[0] == 0
    assert clipped.shadow_total_adjustment.iloc[0] == 0
    with pytest.raises(ValueError, match="already include"):
        shadow_totals(
            projections.assign(model_version="joint_scoring_v10"), tmp_path, shadow_at
        )
    results = grade_shadow_totals(pd.concat([shadow, shadow]), graded)
    assert len(results) == len(shadow)
    assert results.result_status.eq("pending").sum() == 1
    evaluated = results[results.result_status.eq("graded")]
    assert np.allclose(
        evaluated.shadow_absolute_error,
        (evaluated.shadow_total - evaluated.actual_total).abs(),
    )
    with pytest.raises(ValueError, match="pregame creation"):
        grade_shadow_totals(
            shadow.assign(shadow_created_at="2026-09-01T00:00:00Z"), graded
        )
    assert shadow_totals(
        projections, tmp_path, pd.Timestamp("2026-09-01T00:00:00Z")
    ).empty
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
    # The comparison prices the published market-informed line with the
    # dispersion picks are priced with.
    assert priced.iloc[0].best_offer_model_cover_probability == pytest.approx(
        stats.t.cdf(6.0 / (PRICING_MARGIN_SD * np.sqrt(5.0 / 7.0)), 7.0)
    )
    boundary = student_grade.assign(actual_margin=7.0 + 18.0)
    pure_coverage = grading._coverage(
        boundary, boundary["pure_home_margin"], boundary["margin_sd"], 0.8
    )
    assert pure_coverage == 0.0
    assert pure_coverage == _interval_coverage(
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
    from backend import recommendations as recommendations_module
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
    # Two independent paired books anchor the reference; Book A offers the
    # better execution price. Repeat rows must never increase a book's vote.
    references = []
    for key in ("book_b", "book_c"):
        ref = offers[offers.market.eq("h2h")].copy()
        ref["provider_key"], ref["provider"] = key, key
        favoured = ref.selection.eq("home").eq(ref.game_id.lt(4))
        ref["price"] = np.where(favoured, -350, 280)
        references.append(ref)
    offers = pd.concat([offers, *references], ignore_index=True)
    projections.loc[projections.game_id.eq(4), "model_total"] = 36.0
    projections.loc[projections.game_id.eq(4), "market_informed_total"] = 39.5
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
    assert decisions[decisions.game_id.le(5)].reason.eq("qualifying_edge").all()
    h2h = decisions[decisions.market.eq("h2h") & decisions.game_id.eq(1)].iloc[0]
    market_home = (350 / 450) / (350 / 450 + 100 / 380)
    scale = PRICING_MARGIN_SD * np.sqrt(1 - 2 / 500)
    market_margin = scale * stats.t.ppf(market_home, 500)
    blended_margin = 0.8 * market_margin + 0.2 * 14
    expected_home = stats.t.cdf(blended_margin / scale, 500)
    assert h2h.win_probability == pytest.approx(expected_home)
    assert h2h.pricing_context["model_weight"] == 0.2
    assert h2h.pricing_context["blend_space"] == "margin"
    assert h2h.pricing_context["market_home_probability"] == pytest.approx(market_home)
    assert h2h.pricing_context["pricing_home_margin"] == pytest.approx(blended_margin)
    # Holding quotes fixed, changing model opinion must change the priced
    # probability and selection. This protects the research model's role.
    opposite_model = build_recommendations(
        projections.iloc[:1].assign(pure_home_margin=-40), offers, decision_at=now
    )
    contrary = opposite_model.query("market == 'h2h'").iloc[0]
    assert contrary.side == "away"
    assert contrary.pricing_context["home_probability"] == pytest.approx(
        stats.t.cdf((0.8 * market_margin - 0.2 * 40) / scale, 500)
    )
    assert len(h2h.pricing_context["reference_pairs"]) == 3
    assert h2h.pricing_context["decision_batch_game_stake_units"] == 3
    # No posted spread is needed to price a paired moneyline.
    without_spread = build_recommendations(
        projections.assign(market_home_spread=None), offers, decision_at=now
    )
    pd.testing.assert_series_equal(
        decisions.query("market == 'h2h'").win_probability,
        without_spread.query("market == 'h2h'").win_probability,
    )
    assert (
        without_spread.query("market == 'spreads' and game_id <= 5")
        .reason.eq("missing_spread_reference")
        .all()
    )
    # With the model matching the reference margin, a single book's own vig
    # cannot produce an edge. Model disagreement may legitimately create one.
    single_market_home = (180 / 280) / (180 / 280 + 100 / 260)
    single_market_margin = scale * stats.t.ppf(single_market_home, 500)
    single_book = build_recommendations(
        projections.iloc[:1].assign(pure_home_margin=single_market_margin),
        offers[offers.provider_key.eq("book_a")],
        decision_at=now,
    )
    assert single_book.query("market == 'h2h'").status.eq("no_play").all()
    duplicate = build_recommendations(
        projections.iloc[:1],
        pd.concat([offers, offers[offers.provider_key.eq("book_a")]]),
        decision_at=now,
    )
    pd.testing.assert_series_equal(
        decisions.iloc[:3].win_probability, duplicate.win_probability
    )
    # A newer capture missing the opposite side cannot resurrect an old pair.
    incomplete = offers[
        offers.game_id.eq(1) & offers.market.eq("h2h") & offers.selection.eq("home")
    ].assign(market_fetched_at=now + pd.Timedelta(seconds=1))
    newer = build_recommendations(
        projections.iloc[:1],
        pd.concat([offers, incomplete]),
        decision_at=now + pd.Timedelta(seconds=1),
    )
    assert newer.query("market == 'h2h'").reason.eq("missing_moneyline_reference").all()
    for invalid_price in (None, 50):
        invalid_capture = build_recommendations(
            projections.iloc[:1],
            pd.concat([offers, incomplete.assign(price=invalid_price)]),
            decision_at=now + pd.Timedelta(seconds=1),
        )
        assert (
            invalid_capture.query("market == 'h2h'")
            .reason.eq("missing_moneyline_reference")
            .all()
        )
    # Conflicting prices from one capture are ambiguous, not extra votes.
    conflict = offers[
        offers.game_id.eq(1) & offers.market.eq("h2h") & offers.selection.eq("home")
    ].assign(price=-400)
    ambiguous = build_recommendations(
        projections.iloc[:1], pd.concat([offers, conflict]), decision_at=now
    )
    assert (
        ambiguous.query("market == 'h2h'")
        .reason.eq("missing_moneyline_reference")
        .all()
    )
    # A gate no offer clears still yields exploratory picks, labelled as such.
    monkeypatch.setattr(recommendations_module, "MIN_EDGE_POINTS", 100.0)
    floored = build_recommendations(projections, offers, decision_at=now)
    monkeypatch.setattr(recommendations_module, "MIN_EDGE_POINTS", 2.0)
    assert floored.status.eq("recommended").sum() == min(
        recommendations_module.VOLUME_FLOOR, 10
    )
    assert floored[floored.status.eq("recommended")].reason.eq("volume_floor").all()
    assert floored[floored.game_id.ge(6)].status.eq("no_play").all()
    # Cap the original shortlist, never refill from the other market. Three
    # available slots and three same-market leaders yield two promotions,
    # even though eligible offers in the other market remain available.
    with monkeypatch.context() as patch:
        patch.setattr(recommendations_module, "MIN_EDGE_POINTS", 100.0)
        patch.setattr(recommendations_module, "VOLUME_FLOOR", 3)
        capped = build_recommendations(projections, offers, decision_at=now)
    eligible = floored[floored.status.eq("recommended")].sort_values(
        ["edge_points", "game_id", "market"], ascending=[False, True, True]
    )
    shortlist = eligible.head(3)
    assert shortlist.market.nunique() == 1
    assert eligible.market.nunique() == 2
    selected = capped[capped.status.eq("recommended")]
    assert len(selected) == 2
    assert set(selected.index) == set(shortlist.head(2).index)
    assert sum(c["floor_capped"] for c in capped.pricing_context) == 1
    assert all(c["floor_backfill"] is False for c in capped.pricing_context)
    assert decisions[decisions.game_id.ge(6)].status.eq("no_play").all()
    assert all(
        r["recommendation_tier"] == "exploratory"
        for r in floored.loc[floored.status.eq("recommended"), "pricing_context"]
    )
    assert all(
        r["recommendation_tier"] == "edge_qualified"
        for r in decisions.loc[decisions.status.eq("recommended"), "pricing_context"]
    )
    # Spread pricing preserves 20% of model opinion and a frozen market reference.
    indexed = projections.set_index("game_id")
    expected = -0.8 * indexed.market_home_spread + 0.2 * indexed.pure_home_margin
    spreads = decisions[decisions.market.eq("spreads")]
    assert np.allclose(spreads.model_home_margin, spreads.game_id.map(expected))
    # An archived history rating contributes within the 20% model component.
    history = (
        build_recommendations(
            projections.iloc[:1].assign(
                market_history_home_margin=4, market_history_weight=0.35
            ),
            offers,
            decision_at=now,
        )
        .query("market == 'spreads'")
        .iloc[0]
    )
    assert history.model_home_margin == pytest.approx(
        0.8 * 12 + 0.2 * (0.65 * 14 + 0.35 * 4)
    )
    assert history.pricing_context["published_home_margin"] == 13
    assert history.pricing_context["calibration_version"] == "spread_keys_020_v1"
    # Missing season type remains unknown; a schedule provides the flag
    # without changing selection or inferring the stage from a week number.
    assert history.pricing_context["is_postseason"] is None
    staged = build_recommendations(
        projections,
        offers,
        decision_at=now,
        games=pd.DataFrame({"id": range(1, 10), "season_type": ["postseason"] * 9}),
    )
    assert staged.status.equals(decisions.status)
    for row in staged.itertuples():
        assert row.pricing_context["is_postseason"] is True
        assert "postseason" in row.pricing_context["research_flags"]
        if row.market == "spreads":
            edge = row.pricing_context["spread_edge_breakdown"]
            assert edge["model_contribution_points"] == pytest.approx(
                0.2 * edge["model_disagreement_points"]
            )
            assert sum(
                edge[k]
                for k in (
                    "model_contribution_points",
                    "quote_difference_points",
                    "price_distribution_adjustment_points",
                )
            ) == pytest.approx(row.edge_points)
    # Audited key-number probability fixture and complementary side contracts.
    from backend.model.spread_pricing import spread_win_loss

    win, loss = spread_win_loss(7, 15.35, 500, 7)
    assert [win, loss] == pytest.approx([0.44512897515057426, 0.5073914035183481])
    assert 1 - win - loss > 0.04
    for threshold in (-7.5, -7, 0, 3, 3.5, 7, 80, 100):
        home = spread_win_loss(7, 15.35, 500, threshold)
        away = spread_win_loss(-7, 15.35, 500, -threshold)
        assert home == pytest.approx(away[::-1])
        assert 0 <= sum(home) <= 1 + 1e-12
    # Totals price from the published total; the posted total is recorded.
    totals = decisions[decisions.market.eq("totals")]
    assert totals.market_total.eq(43.0).all()
    published = projections.set_index("game_id").market_informed_total
    assert totals.model_total.eq(totals.game_id.map(published)).all()
    home = decisions[(decisions.game_id.eq(1)) & decisions.market.eq("spreads")].iloc[0]
    assert home.side == "home" and home.point == -3 and home.price == -110
    assert home.push_probability > 0
    assert home.expected_value_per_unit == pytest.approx(
        home.win_probability * 100 / 110
        - (1 - home.win_probability - home.push_probability)
    )
    assert (
        decisions[decisions.game_id.eq(9) & decisions.market.ne("h2h")]
        .reason.eq("unpaired_market")
        .all()
    )
    assert (
        decisions[decisions.game_id.eq(9) & decisions.market.eq("h2h")]
        .reason.eq("missing_moneyline_reference")
        .all()
    )
    assert floored.query("market == 'h2h'").status.eq("no_play").all()
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
    # CFBD medians: game 1 closed home -6 / 46.5 / -240 (moved toward the
    # home -3 and over 43 picks); game 3 closed home -1 / 40 (moved against).
    closing = pd.DataFrame(
        dict(
            game_id=[1, 3],
            closing_spread=[-6.0, -1.0],
            closing_total=[46.5, 40.0],
            closing_home_moneyline=[-240.0, -105.0],
            closing_away_moneyline=[200.0, -115.0],
        )
    ).set_index("game_id")
    grades = grade_recommendations(
        frozen, games, closing, graded_at=now + pd.Timedelta(days=2)
    )
    spread = grades[grades.market.eq("spreads")].set_index("game_id")
    assert spread.loc[1, "closing_point"] == -6 and spread.loc[1, "clv_points"] == 3
    assert spread.loc[3, "closing_point"] == -1 and spread.loc[3, "clv_points"] == -2
    assert spread.loc[1, "closing_source"] == "cfbd_lines_median"
    assert spread.loc[2, "closing_source"] is None and pd.isna(
        spread.loc[2, "clv_points"]
    )
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
    assert (
        decisions.loc[
            decisions.market.eq("h2h") & decisions.win_probability.notna(),
            "push_probability",
        ]
        .eq(0)
        .all()
    )
    tied = grade_recommendations(
        frozen,
        games.assign(home_points=20, away_points=20),
        graded_at=now + pd.Timedelta(days=2),
    )
    assert (
        tied.loc[tied.market.eq("h2h") & tied.game_id.le(5), "outcome"].eq("void").all()
    )
    totals = grades[grades.market.eq("totals")].set_index("game_id")
    assert totals.loc[1, "clv_points"] == 3.5 and totals.loc[3, "clv_points"] == -3
    assert moneyline.loc[1, "closing_price"] == -240 and pd.isna(
        moneyline.loc[1, "clv_points"]
    )
    assert totals.loc[1, "outcome"] == "win"
    assert totals.loc[2, "outcome"] == "push"
    assert totals.loc[4, "outcome"] == "loss"
    mismatch = offers[offers.game_id.eq(1)].assign(
        commence_time=now - pd.Timedelta(hours=1)
    )
    assert (
        build_recommendations(projections.iloc[:1], mismatch, decision_at=now)
        .reason.isin(["kickoff_mismatch", "missing_moneyline_reference"])
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

    from backend.diagnostics import recommendation_calibration

    audited = (
        frozen.drop(columns="outcome")
        .merge(
            grades[["game_id", "market", "outcome", "clv_points"]],
            on=["game_id", "market"],
            validate="one_to_one",
        )
        .assign(published_at=now)
    )
    calibration = recommendation_calibration(audited)
    overall = calibration[calibration.probability_bucket.eq("all")]
    eligible = audited[
        audited.status.eq("recommended") & audited.outcome.isin(["win", "loss", "push"])
    ]
    assert overall.selections.sum() == len(eligible)
    assert overall.pushes.sum() == eligible.outcome.eq("push").sum()
    for row in overall.itertuples():
        cohort = eligible[
            eligible.market.eq(row.market)
            & eligible.reason.eq(row.reason)
            & eligible.policy_version.eq(row.policy_version)
            & eligible.outcome.ne("push")
        ]
        assert row.predicted_decisive_win_rate == pytest.approx(
            (cohort.win_probability / (1 - cohort.push_probability)).mean()
        )
    with pytest.raises(ValueError, match="pregame"):
        recommendation_calibration(
            audited.assign(published_at=now + pd.Timedelta(days=30))
        )
