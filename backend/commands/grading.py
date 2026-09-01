"""Live-season grading command handlers."""

from argparse import Namespace


def handle_grade(args: Namespace) -> None:
    from backend.grading import (
        build_graded_games,
        compute_performance_metrics,
        write_grading_artifacts,
    )
    from backend.publish import fetch_graded_games, fetch_published_projections

    projections = fetch_published_projections(args.season)
    existing = None if args.regrade else fetch_graded_games(args.season)
    graded = build_graded_games(args.season, projections, existing)
    metrics = compute_performance_metrics(graded)
    write_grading_artifacts(args.season, graded, metrics)

    kept = 0 if existing is None else len(existing)
    print(
        f"graded {len(graded)} {args.season} games "
        f"({len(graded) - kept} new, {kept} kept); "
        f"{int(graded['closing_spread'].notna().sum())} with a closing spread"
    )
    overall = metrics[
        metrics["segment_kind"].eq("overall")
        & metrics["prediction_source"].isin(["pure_model", "closing_market"])
    ]
    if not overall.empty:
        print(
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
    from backend.publish import publish_grading

    stored = publish_grading(args.season)
    for table, count in stored.items():
        print(f"cfb.{table}: {count} rows stored for season {args.season}")
