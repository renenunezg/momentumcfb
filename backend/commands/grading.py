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

    from backend.config import PROCESSED_DIR
    from backend.diagnostics import (
        recommendation_calibration,
        scoring_diagnostics,
        write_diagnostics,
    )

    # Settlement is computed before publication. Evaluate these new results,
    # retaining each original pregame price, probability and policy identity.
    settled_picks = picks.set_index(["game_id", "market"])
    settled_picks.update(settlements.set_index(["game_id", "market"]))
    errors, summary = scoring_diagnostics(graded)
    write_diagnostics(
        PROCESSED_DIR / "diagnostics" / str(args.season),
        {
            "scoring_errors": errors,
            "scoring_summary": summary,
            "recommendation_calibration": recommendation_calibration(
                settled_picks.reset_index()
            ),
        },
    )

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


def handle_diagnose(args: Namespace) -> None:
    from pathlib import Path

    import pandas as pd
    from sqlalchemy import text

    from backend.config import PROCESSED_DIR
    from backend.db import engine
    from backend.diagnostics import (
        disagreement_audit,
        grade_shadow_totals,
        recommendation_calibration,
        scoring_diagnostics,
        shadow_totals,
        write_diagnostics,
    )
    from backend.model.weekly import load_weekly_games
    from backend.publish import (
        fetch_graded_games,
        fetch_published_projections,
        fetch_qb_availability,
        fetch_recommendations,
    )

    all_graded = fetch_graded_games(args.season)
    graded = all_graded[all_graded.week.eq(args.week)]
    picks = fetch_recommendations(args.season)
    projections = fetch_published_projections(args.season)
    details = None
    if args.forecast_directory:
        details = pd.read_parquet(Path(args.forecast_directory) / "projections.parquet")
    errors, summary = scoring_diagnostics(
        graded,
        details,
        load_weekly_games(args.season),
    )
    with engine.connect() as conn:
        ratings = pd.read_sql_query(
            text("SELECT * FROM cfb.team_ratings WHERE season = :s"),
            conn,
            params={"s": args.season},
        )
    upcoming = projections[projections.week.gt(args.week)]
    destination = (
        Path(args.output_directory)
        if args.output_directory
        else (PROCESSED_DIR / "diagnostics" / f"{args.season}_{args.week:02d}")
    )
    write_diagnostics(
        destination,
        {
            "scoring_errors": errors,
            "scoring_summary": summary,
            "recommendation_calibration": recommendation_calibration(
                picks[picks.week.eq(args.week)]
            ),
            "upcoming_disagreements": disagreement_audit(
                upcoming,
                ratings,
                fetch_qb_availability(args.season),
            ),
        },
    )
    log.info("%s", summary.to_string(index=False))
    if args.shadow_calibration_directory:
        created_at = pd.Timestamp.now(tz="UTC")
        shadow = shadow_totals(
            upcoming, Path(args.shadow_calibration_directory), created_at
        )
        name = "shadow_totals_" + created_at.strftime("%Y%m%dT%H%M%S%fZ")
        write_diagnostics(destination, {name: shadow})
        log.info(
            "Froze %s prospective shadow totals; no selections repriced", len(shadow)
        )
    shadow_paths = sorted(
        set(
            (PROCESSED_DIR / "diagnostics").glob(
                f"{args.season}*/shadow_totals_*.parquet"
            )
        )
        | set(destination.glob("shadow_totals_*.parquet"))
    )
    if shadow_paths:
        shadows = pd.concat(
            [pd.read_parquet(path) for path in shadow_paths], ignore_index=True
        )
        shadows = shadows[shadows.season.eq(args.season)]
        results = grade_shadow_totals(shadows, all_graded)
        write_diagnostics(destination, {"shadow_results": results})
        log.info(
            "Shadow evaluation: %s", results.result_status.value_counts().to_dict()
        )
    log.info(
        "Wrote frozen diagnostics to %s; missing pace remains unknown", destination
    )
