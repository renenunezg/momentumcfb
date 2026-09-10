"""Live-season grading command handlers."""

import logging
from argparse import Namespace

log = logging.getLogger(__name__)


def handle_grade(args: Namespace) -> None:
    from backend.etl import store
    from backend.grading import (
        build_graded_games,
        compute_performance_metrics,
        write_grading_artifacts,
    )
    from backend.publish import (
        fetch_graded_games,
        fetch_published_projections,
        fetch_recommendations,
    )
    from backend.recommendations import closing_line_backfill, grade_recommendations
    from backend.serving.market import (
        flatten_closing_lines,
        flatten_closing_moneylines,
        flatten_closing_totals,
    )

    projections = fetch_published_projections(args.season)
    # Player WPA needs the same immutable pregame inputs on an ephemeral runner.
    store.write_processed(
        projections, "players", "published_projections", f"{args.season}.parquet"
    )
    existing = None if args.regrade else fetch_graded_games(args.season)
    graded = build_graded_games(args.season, projections, existing)

    picks = fetch_recommendations(args.season)
    lines = store.read_lines(args.season)
    closing = (
        flatten_closing_lines(lines)
        .merge(flatten_closing_totals(lines), on="game_id", how="outer")
        .merge(flatten_closing_moneylines(lines), on="game_id", how="outer")
        .set_index("game_id")
    )
    settlements = grade_recommendations(picks, store.read_games(args.season), closing)
    store.write_processed(
        settlements, "grading", f"recommendations_{args.season}.parquet"
    )
    backfill = closing_line_backfill(picks, closing)
    store.write_processed(
        backfill, "grading", f"recommendation_closing_{args.season}.parquet"
    )
    log.info(
        f"settled {len(settlements)} picks, "
        f"{int(settlements['closing_source'].notna().sum())} with a closing line; "
        f"{len(backfill)} earlier settlements gain a closing line"
    )
    metrics = compute_performance_metrics(graded)
    write_grading_artifacts(args.season, graded, metrics)

    kept = 0 if existing is None else len(existing)
    log.info(
        f"graded {len(graded)} {args.season} games "
        f"({len(graded) - kept} new, {kept} kept); "
        f"{int(graded['closing_spread'].notna().sum())} with a closing spread"
    )
    overall = metrics[
        metrics["segment_kind"].eq("overall")
        & metrics["prediction_source"].isin(["pure_model", "closing_market"])
    ]
    if not overall.empty:
        log.info(
            overall[
                [
                    "prediction_source",
                    "games",
                    "margin_mae",
                    "margin_bias",
                    "coverage_80",
                ]
            ].to_string(index=False)
        )


def handle_publish_grading(args: Namespace) -> None:
    from backend.publish import publish_grading, publish_recommendation_grades

    stored = publish_grading(args.season)
    settled = publish_recommendation_grades(args.season)
    log.info(f"cfb.recommendations: {settled} newly settled for season {args.season}")
    for table, count in stored.items():
        log.info(f"cfb.{table}: {count} rows stored for season {args.season}")
