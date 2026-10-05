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
    decisions = build_recommendations(
        adjusted, pd.DataFrame(columns=OFFER_COLUMNS), decision_at=now
    )
    engine = create_engine(value)
    try:
        with engine.connect() as conn:
            transaction = conn.begin()
            _publish_recommendations(conn, decisions)
            saved = conn.execute(
                text(
                    "SELECT decision_forecast, model_total FROM cfb.recommendations WHERE game_id=99000001"
                )
            ).all()
            assert len(saved) == 3
            assert all(row.decision_forecast == evidence for row in saved)
            assert all(
                row.model_total == adjusted.market_informed_total.iloc[0]
                for row in saved
            )
            assert original.model_total.iloc[0] == 50.0
            transaction.rollback()
    finally:
        engine.dispose()
