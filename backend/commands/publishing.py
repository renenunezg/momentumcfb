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
    fresh prices. Published picks stay frozen; only No Play rows can change."""
    from datetime import datetime, timezone

    import pandas as pd
    from sqlalchemy import text

    from backend.db import engine
    from backend.etl import store
    from backend.model.weekly import odds_frames
    from backend.odds.client import OddsAPIClient
    from backend.publish import (
        CFB_SCHEMA,
        _publish_recommendations,
        ensure_recommendation_schema,
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    store.write_processed(
        offers,
        "market_offers",
        f"{args.season}_{args.week:02d}_refresh_{stamp}.parquet",
    )
    decisions = build_recommendations(projections, offers)
    with engine.begin() as conn:
        _publish_recommendations(conn, decisions)
    picks = decisions[decisions["status"].eq("recommended")]
    log.info(
        f"refreshed {args.season} week {args.week}: {len(offers)} offers on "
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
