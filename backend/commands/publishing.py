"""Database publication command handlers."""

import logging
from argparse import Namespace

log = logging.getLogger(__name__)


def handle_publish(args: Namespace) -> None:
    from backend.publish import publish

    stored = publish(
        args.season,
        args.week,
        source=args.source,
        include_backtest=not args.skip_backtest,
    )
    for table, count in stored.items():
        log.info(f"cfb.{table}: {count} rows stored")


def handle_refresh_picks(args: Namespace) -> None:
    """Re-decide one published week's picks from the frozen projections and
    fresh prices. Published picks stay frozen unless a newer model version no
    longer makes them; those and No Play rows take the fresh decision."""
    from datetime import datetime, timezone

    import pandas as pd
    from sqlalchemy import text

    from backend.db import engine
    from backend.etl import store
    from backend.model.availability import apply_qb_availability, pregame_qb_outs
    from backend.model.weekly import odds_frames
    from backend.odds.client import OddsAPIClient
    from backend.publish import (
        CFB_SCHEMA,
        _publish_recommendations,
        ensure_recommendation_schema,
        fetch_qb_availability,
    )
    from backend.recommendations import build_recommendations

    ensure_recommendation_schema()
    with engine.connect() as conn:
        projections = pd.read_sql_query(
            text(
                f"SELECT * FROM {CFB_SCHEMA}.game_projections "
                "WHERE season = :season AND week = :week ORDER BY game_id"
            ),
            conn,
            params={"season": args.season, "week": args.week},
        )
    if projections.empty:
        raise ValueError(f"no published projections for {args.season} week {args.week}")
    _, offers, matches, snapshot = odds_frames(OddsAPIClient(), projections)
    # The decision is timed after the prices arrive: an offer fetched after
    # the decision timestamp would otherwise read as stale and block every
    # pick. The published forecast stays frozen; an absence reported since it
    # was made adjusts only the lines these fresh prices are decided against.
    decided_at = datetime.now(timezone.utc)
    games = store.read_games(args.season)
    published = projections.copy()
    availability = fetch_qb_availability(args.season)
    projections = apply_qb_availability(
        projections,
        pregame_qb_outs(availability, args.season, args.week, decided_at),
        set(games.loc[games["season_type"].eq("postseason"), "id"]),
    )
    # Persist both views with each fresh decision, including the exact reports
    # that were eligible at its cutoff. Existing recommended picks stay frozen.
    import json

    forecast_columns = [
        "game_id",
        "as_of",
        "model_total",
        "home_margin",
        "expected_home_points",
        "expected_away_points",
        "market_informed_home_margin",
        "market_informed_total",
        "market_informed_home_points",
        "market_informed_away_points",
        "market_total_weight",
        "home_qb_out",
        "away_qb_out",
        "qb_availability_points",
    ]
    reports = (
        availability[
            pd.to_datetime(availability["reported_at"], utc=True).lt(decided_at)
            & availability["season"].eq(args.season)
            & availability["week"].eq(args.week)
        ]
        if not availability.empty
        else availability
    )
    original_rows = json.loads(
        published.reindex(columns=forecast_columns).to_json(
            orient="records", date_format="iso"
        )
    )
    adjusted_rows = json.loads(
        projections.reindex(columns=forecast_columns).to_json(
            orient="records", date_format="iso"
        )
    )
    report_rows = json.loads(reports.to_json(orient="records", date_format="iso"))
    projections["decision_forecast"] = [
        {
            "version": 1,
            "decision_at": decided_at.isoformat(),
            "published": original,
            "adjusted": adjusted,
            "availability": [
                report
                for report in report_rows
                if report.get("team") in (game.home_team, game.away_team)
            ],
        }
        for original, adjusted, game in zip(
            original_rows, adjusted_rows, projections.itertuples()
        )
    ]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    store.write_processed(
        offers,
        "market_offers",
        f"{args.season}_{args.week:02d}_refresh_{stamp}.parquet",
    )
    decisions = build_recommendations(projections, offers, decision_at=decided_at)
    with engine.begin() as conn:
        replaced = _publish_recommendations(conn, decisions)
    picks = decisions[decisions["status"].eq("recommended")]
    log.info(
        f"refreshed {args.season} week {args.week}: {replaced} picks the current "
        f"model no longer makes replaced, {len(offers)} offers on "
        f"{int(matches['matched'].sum()) if not matches.empty else 0} matched events, "
        f"{len(picks)} qualifying picks "
        f"({picks['market'].value_counts().to_dict()}), "
        f"Odds API requests remaining {snapshot.requests_remaining}"
    )


def handle_publish_anchors(args: Namespace) -> None:
    from backend.publish import publish_serving_anchors

    count = publish_serving_anchors(args.season, args.week)
    log.info(
        f"cfb.serving_anchors: {count} rows stored for season "
        f"{args.season} anchor week {args.week:02d}"
    )


def handle_fetch_anchors(args: Namespace) -> None:
    from backend.publish import fetch_serving_anchors

    count = fetch_serving_anchors(args.season, args.week)
    log.info(
        f"hydrated {count} serving anchors for season {args.season} "
        f"anchor week {args.week:02d} from cfb.serving_anchors"
    )
