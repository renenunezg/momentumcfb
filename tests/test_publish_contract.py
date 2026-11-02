from pathlib import Path

import pandas as pd
import pytest

from backend.model.market_blend import add_market_informed_margins
from backend.model.preseason import _flatten_cfbd_offers
from backend.odds.markets import compare_priced_offers
from backend.publish import MARKET_COMPARISONS_COLUMNS, _serving_frame


def test_market_ratings_publish_only_with_the_matching_model_snapshot(
    tmp_path, monkeypatch
):
    from backend import publish
    from backend.etl import store

    monkeypatch.setattr(store, "PROCESSED_DIR", tmp_path)

    monkeypatch.setattr(publish, "PROCESSED_DIR", tmp_path)
    for kind in ("ratings", "market_ratings"):
        (tmp_path / kind).mkdir()
    snapshot = pd.DataFrame(
        {
            "team_id": [1, 2],
            "season": [2026, 2026],
            "week": [5, 5],
            "as_of": [pd.Timestamp("2026-09-28T12:00:00Z")] * 2,
            "power_rating": [20.0, 10.0],
        }
    )
    snapshot.to_parquet(tmp_path / "ratings/2026_05.parquet")
    # Older forecasts remain readable and never invent a market estimate.
    assert publish.load_team_ratings("weekly", 2026, 5).market_rating.isna().all()
    market = snapshot.drop(columns="power_rating").assign(
        market_rating=[21.5, 9.0], market_rating_sd=[3.2, 4.1], games_with_lines=[4, 0]
    )
    market.to_parquet(tmp_path / "market_ratings/2026_05.parquet")
    served = publish.load_team_ratings("weekly", 2026, 5)
    assert served.power_rating.tolist() == [20.0, 10.0]
    assert served.market_rating.tolist() == [21.5, 9.0]
    assert served.market_rating_sd.tolist() == [3.2, 4.1]
    assert served.market_rating_games.tolist() == [4, 0]
    # Publication consumes one completed run even if a later run stops halfway.
    created = pd.Timestamp("2026-09-28T13:00:00Z").to_pydatetime()
    run = store.write_forecast_outputs(
        "weekly",
        2026,
        5,
        created,
        {
            "ratings": snapshot,
            "market_ratings": market,
        },
    )
    assert store.completed_forecast_run("weekly", 2026, 5) == run
    original_write = pd.DataFrame.to_parquet

    def fail_second(frame, path, **kwargs):
        if Path(path).name == "market_ratings.parquet":
            raise OSError("interrupted run")
        return original_write(frame, path, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, "to_parquet", fail_second)
        with pytest.raises(OSError, match="interrupted run"):
            store.write_forecast_outputs(
                "weekly",
                2026,
                5,
                created + pd.Timedelta(seconds=1),
                {"ratings": snapshot, "market_ratings": market},
            )
    assert store.completed_forecast_run("weekly", 2026, 5) == run
    assert publish.load_team_ratings("weekly", 2026, 5, run).market_rating.tolist() == [
        21.5,
        9.0,
    ]
    market["as_of"] += pd.Timedelta(days=1)
    market.to_parquet(tmp_path / "market_ratings/2026_05.parquet")
    with pytest.raises(ValueError, match="must match the model rating snapshot"):
        publish.load_team_ratings("weekly", 2026, 5)


def test_cfbd_line_comparisons_fill_the_published_contract():
    lines = pd.DataFrame(
        [
            {
                "id": 401,
                "home_team": "Home",
                "away_team": "Away",
                "source_fetched_at": "2026-08-27T12:00:00+00:00",
                "lines": [
                    {"provider": "Book A", "spread": -3.5, "overUnder": 51.5},
                    {"provider": "Book B", "spread": -4.0, "overUnder": None},
                ],
            }
        ]
    )
    projections = pd.DataFrame(
        [
            {
                "game_id": 401,
                "start_date": "2026-08-29T16:00:00+00:00",
                "home_team": "Home",
                "away_team": "Away",
                "home_margin": 6.0,
                "home_spread": -6.0,
                "model_total": 55.0,
                "margin_sd": 13.0,
                "total_sd": 13.0,
                "as_of": "2026-08-27T13:00:00+00:00",
                "degrees_of_freedom": 8.0,
            }
        ]
    )

    offers = _flatten_cfbd_offers(lines)
    blended = add_market_informed_margins(projections, offers)
    published = _serving_frame(
        compare_priced_offers(blended, offers), MARKET_COMPARISONS_COLUMNS
    )
    row = published.iloc[0]

    assert blended["market_home_spread"].iloc[0] == -3.75
    assert row["market_available"] is True
    assert row["priced_offer_available"] is False
    assert row["review_status"] == "no_priced_offer"
    # The comparison carries the published market-informed line.
    assert row["model_home_spread"] == -4.875
    assert row["best_offer_point"] is None
    assert row["best_offer_expected_value_per_unit"] is None


def test_player_publish_checks_schema_before_removing_served_rows(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from backend import db, publish
    from backend.players import pipeline

    monkeypatch.delenv("DATABASE_URL", raising=False)
    statements = []

    def execute(statement, params):
        statements.append(str(statement))
        return SimpleNamespace(scalar=lambda: "NO")

    connection = SimpleNamespace(execute=execute)
    # Avoid invoking the lazy database constructor while installing the fake.
    monkeypatch.setitem(
        db.__dict__, "engine", SimpleNamespace(begin=lambda: nullcontext(connection))
    )
    monkeypatch.setattr(pipeline, "read_player_artifacts", lambda season: {})
    monkeypatch.setattr(publish, "_table_columns", lambda connection, table: [])
    with pytest.raises(ValueError, match="002_heisman_weekly_evaluation.sql"):
        publish.publish_players(2026)
    assert all(statement.startswith("SELECT") for statement in statements)


def test_missing_local_preseason_prior_recovers_pure_rating_contract(
    monkeypatch, tmp_path
):
    import os
    from contextlib import nullcontext
    from types import SimpleNamespace

    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    from backend import db
    from backend.etl import store
    from backend.model.preseason import load_preseason_ratings

    value = os.getenv("CFB_TEST_DATABASE_URL")
    if not value:
        pytest.skip("requires restored CFB schema in disposable localhost PostgreSQL")
    assert make_url(value).host in ("127.0.0.1", "localhost")
    monkeypatch.setattr(store, "PROCESSED_DIR", tmp_path)
    engine = create_engine(value)
    try:
        with engine.connect() as conn:
            transaction = conn.begin()
            conn.execute(
                text("""INSERT INTO cfb.team_ratings
                (season, week, as_of, model_version, team_id, team,
                 offense_points, defense_points, power_rating, forecast_alignment_points)
                VALUES (2099, 1, now(), 'preseason_acceptance', 99000001, 'Home', 5, 4, 9, 4)
            """)
            )
            monkeypatch.setitem(
                db.__dict__,
                "engine",
                SimpleNamespace(connect=lambda: nullcontext(conn)),
            )
            ratings, source = load_preseason_ratings(2099)
            assert source == "published_team_ratings"
            assert ratings[
                ["offense_points", "defense_points", "power_rating"]
            ].values.tolist() == [[3, 2, 5]]
            with pytest.raises(
                FileNotFoundError, match="no published preseason ratings"
            ):
                load_preseason_ratings(2099, week=2)
            transaction.rollback()
    finally:
        engine.dispose()


def test_adjusted_decision_forecast_survives_database_publication():
    import os

    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    from backend.model.availability import apply_qb_availability
    from backend.odds.markets import OFFER_COLUMNS
    from backend.publish import _publish_recommendations
    from backend.recommendations import build_recommendations

    value = os.getenv("CFB_TEST_DATABASE_URL")
    if not value:
        pytest.skip("requires restored CFB schema in disposable localhost PostgreSQL")
    assert make_url(value).host in ("127.0.0.1", "localhost")
    now = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=1)
    original = pd.DataFrame(
        [
            dict(
                game_id=99000001,
                season=2026,
                week=6,
                season_type="postseason",
                start_date=now + pd.Timedelta(days=1),
                as_of=now - pd.Timedelta(hours=1),
                model_version="decision-acceptance",
                home_team="Home",
                away_team="Away",
                home_margin=10.0,
                home_spread=-10.0,
                pure_home_margin=10.0,
                expected_home_points=30.0,
                expected_away_points=20.0,
                model_total=50.0,
                market_informed_total=49.0,
                market_total_weight=0.5,
                market_informed_home_margin=8.0,
                market_informed_home_spread=-8.0,
                market_weight=0.5,
                market_home_spread=-6.0,
                margin_sd=14.0,
                total_sd=14.0,
                degrees_of_freedom=8.0,
                home_missing_input_count=0,
                away_missing_input_count=0,
            )
        ]
    )
    adjusted = apply_qb_availability(original, {"Away"})
    evidence = {
        "version": 1,
        "published": {"model_total": 50.0},
        "adjusted": {"model_total": float(adjusted.model_total.iloc[0])},
    }
    adjusted["decision_forecast"] = [evidence]
    offers = pd.DataFrame(
        [
            dict(
                game_id=99000001,
                market="h2h",
                selection=side,
                point=None,
                price=price,
                provider=book,
                provider_key=book,
                provider_last_update=now,
                market_fetched_at=now,
                execution_eligibility_verified=True,
                match_score=1.0,
                odds_api_event_id="acceptance-event",
                commence_time=original.start_date.iloc[0],
            )
            for book, prices in [
                ("a", [-180, 160]),
                ("b", [-350, 280]),
                ("c", [-350, 280]),
            ]
            for side, price in zip(["home", "away"], prices)
        ]
    ).reindex(columns=OFFER_COLUMNS)
    offers["point"] = pd.to_numeric(offers.point).astype(float)
    spread_offers = offers[offers.provider_key.eq("a")].copy()
    spread_offers["market"] = "spreads"
    spread_offers["point"] = spread_offers.selection.map({"home": -3.0, "away": 3.0})
    spread_offers["price"] = -110
    offers = pd.concat([offers, spread_offers], ignore_index=True)
    decisions = build_recommendations(adjusted, offers, decision_at=now)
    assert decisions.query("market == 'h2h'").status.eq("recommended").all()
    # Parquet is the real handoff between decision generation and publication.
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "decisions.parquet"
        decisions.to_parquet(path)
        decisions = pd.read_parquet(path)
    engine = create_engine(value)
    try:
        with engine.connect() as conn:
            transaction = conn.begin()
            _publish_recommendations(conn, decisions)
            saved = conn.execute(
                text(
                    "SELECT market, decision_forecast, pricing_context, model_total FROM cfb.recommendations WHERE game_id=99000001"
                )
            ).all()
            assert len(saved) == 3
            assert all(row.decision_forecast == evidence for row in saved)
            assert all(
                row.model_total == adjusted.market_informed_total.iloc[0]
                for row in saved
            )
            assert original.model_total.iloc[0] == 50.0
            h2h = next(row for row in saved if row.market == "h2h")
            assert (
                h2h.pricing_context["probability_source"]
                == "paired_moneyline_margin_blend_v1"
            )
            assert h2h.pricing_context["model_weight"] == 0.2
            assert h2h.pricing_context["blend_space"] == "margin"
            assert h2h.pricing_context["pricing_home_margin"] == pytest.approx(
                0.8 * h2h.pricing_context["market_equivalent_home_margin"]
                + 0.2 * h2h.pricing_context["model_home_margin"]
            )
            assert len(h2h.pricing_context["reference_pairs"]) == 3
            assert h2h.pricing_context["decision_batch_game_stake_units"] == 2
            spread = next(row for row in saved if row.market == "spreads")
            assert spread.pricing_context["calibration_version"] == "spread_keys_020_v1"
            assert spread.pricing_context["recommendation_tier"] == "edge_qualified"
            assert spread.pricing_context["is_postseason"] is True
            assert spread.pricing_context["floor_backfill"] is False
            assert spread.pricing_context["spread_edge_breakdown"][
                "model_contribution_points"
            ] == pytest.approx(0.2 * (adjusted.pure_home_margin.iloc[0] - 6.0))
            assert (
                conn.execute(
                    text(
                        "SELECT picks FROM cfb.recommendation_summary(2026, 'all', null) WHERE segment_kind='qualification' AND segment='edge_qualified'"
                    )
                ).scalar_one()
                == 2
            )
            # Publication must preserve the original pick and its full pricing evidence.
            refreshed = build_recommendations(
                adjusted,
                offers.assign(price=-110),
                decision_at=now + pd.Timedelta(milliseconds=1),
            )
            _publish_recommendations(conn, refreshed)
            assert (
                conn.execute(
                    text(
                        "SELECT pricing_context FROM cfb.recommendations WHERE game_id=99000001 AND market='h2h'"
                    )
                ).scalar_one()
                == h2h.pricing_context
            )
            # A price-driven withdrawal updates the current list, never the ledger.
            ledger_before = (
                conn.execute(
                    text(
                        "SELECT to_jsonb(r) FROM cfb.recommendations r "
                        "WHERE game_id=99000001 AND status='recommended' ORDER BY market"
                    )
                )
                .scalars()
                .all()
            )
            withdrawn = build_recommendations(
                adjusted.assign(model_version="new-model"),
                offers.assign(price=-10000),
                decision_at=now + pd.Timedelta(milliseconds=2),
            )
            assert withdrawn.status.eq("no_play").all()
            _publish_recommendations(conn, withdrawn)
            assert (
                conn.execute(
                    text(
                        "SELECT count(*) FROM cfb.current_recommendations "
                        "WHERE game_id=99000001 AND status='recommended'"
                    )
                ).scalar_one()
                == 0
            )
            assert (
                conn.execute(
                    text(
                        "SELECT to_jsonb(r) FROM cfb.recommendations r "
                        "WHERE game_id=99000001 AND status='recommended' ORDER BY market"
                    )
                )
                .scalars()
                .all()
                == ledger_before
            )
            # An older run cannot resurrect a withdrawn recommendation.
            _publish_recommendations(conn, decisions)
            assert (
                conn.execute(
                    text(
                        "SELECT count(*) FROM cfb.current_recommendations "
                        "WHERE game_id=99000001 AND status='recommended'"
                    )
                ).scalar_one()
                == 0
            )
            # Anonymous app reads use the same current decisions under RLS.
            conn.execute(text("SET LOCAL ROLE anon"))
            assert (
                conn.execute(
                    text(
                        "SELECT count(*) FROM cfb.current_recommendations WHERE game_id=99000001"
                    )
                ).scalar_one()
                == 3
            )
            conn.execute(text("RESET ROLE"))
            # A late publisher cannot insert a decision for a started game.
            late = decisions.assign(
                game_id=99000009,
                start_date=now - pd.Timedelta(seconds=1),
            )
            # The ordinary publication path already filters started games;
            # verify the current-table SQL boundary independently as well.
            _publish_recommendations(conn, late)
            assert (
                conn.execute(
                    text(
                        "SELECT count(*) FROM cfb.recommendation_decisions WHERE game_id=99000009"
                    )
                ).scalar_one()
                == 0
            )

            from sqlalchemy.exc import IntegrityError

            # Database eligibility enforces the no-moneyline-floor rule too.
            invalid = decisions.copy()
            invalid["game_id"] = 99000002
            invalid.loc[invalid.market.eq("h2h"), "reason"] = "volume_floor"
            with pytest.raises(IntegrityError), conn.begin_nested():
                _publish_recommendations(conn, invalid)
            # Malformed spread evidence cannot silently bypass the DB contract.
            invalid = decisions.copy()
            invalid["game_id"] = 99000003
            mask = invalid.market.eq("spreads")
            invalid.loc[mask, "pricing_context"] = invalid.loc[
                mask, "pricing_context"
            ].map(lambda context: {**context, "model_weight": 0.0})
            with pytest.raises(IntegrityError), conn.begin_nested():
                _publish_recommendations(conn, invalid)
            invalid = decisions.copy()
            invalid["game_id"] = 99000005
            invalid["pricing_context"] = invalid.pricing_context.map(
                lambda context: {**context, "floor_backfill": True}
            )
            with pytest.raises(IntegrityError), conn.begin_nested():
                _publish_recommendations(conn, invalid)
            exploratory = build_recommendations(
                original.assign(
                    game_id=99000004,
                    home_margin=7.0,
                    pure_home_margin=7.0,
                    market_informed_home_margin=6.0,
                    market_home_spread=-5.0,
                ),
                spread_offers.assign(
                    game_id=99000004,
                    point=spread_offers.selection.map({"home": -6.0, "away": 6.0}),
                ),
                decision_at=now,
            )
            assert (
                exploratory.query("market == 'spreads'").reason.eq("volume_floor").all()
            )
            _publish_recommendations(conn, exploratory)
            # The total population stays intact and the filtered RPC/view agree.
            for source in (
                "cfb.recommendation_summary(2026, 'all', null)",
                "cfb.recommendation_performance",
            ):
                tiers = dict(
                    conn.execute(
                        text(
                            f"SELECT segment, picks FROM {source} WHERE segment_kind='qualification'"
                        )
                    ).all()
                )
                assert tiers == {"edge_qualified": 2, "exploratory": 1}
            assert (
                conn.execute(
                    text(
                        "SELECT picks FROM cfb.recommendation_summary(2026, 'spreads', null) WHERE segment_kind='qualification' AND segment='exploratory'"
                    )
                ).scalar_one()
                == 1
            )
            transaction.rollback()
    finally:
        engine.dispose()


def test_recommendation_schema_guard_follows_the_latest_eligibility_migration():
    import re

    from backend import publish

    version, migration = max(
        (int(match.group(1)), path.name)
        for path in (Path(__file__).parents[1] / "sql").glob("[0-9]*.sql")
        for match in re.finditer(
            r"add constraint recommendation_eligibility_v(\d+)", path.read_text()
        )
    )
    # A migration that renames the constraint must move the pre-run guard too.
    assert (
        publish.RECOMMENDATION_ELIGIBILITY_CONSTRAINT
        == f"recommendation_eligibility_v{version}"
    )
    assert publish.RECOMMENDATION_ELIGIBILITY_MIGRATION == f"sql/{migration}"
