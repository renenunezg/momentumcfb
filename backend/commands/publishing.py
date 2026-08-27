"""Database publication command handlers."""

from argparse import Namespace


def handle_publish(args: Namespace) -> None:
    from backend.publish import publish

    stored = publish(
        args.season,
        args.week,
        source=args.source,
        include_backtest=not args.skip_backtest,
    )
    for table, count in stored.items():
        print(f"cfb.{table}: {count} rows stored")


def handle_publish_anchors(args: Namespace) -> None:
    from backend.publish import publish_serving_anchors

    count = publish_serving_anchors(args.season, args.week)
    print(
        f"cfb.serving_anchors: {count} rows stored for season "
        f"{args.season} anchor week {args.week:02d}"
    )


def handle_fetch_anchors(args: Namespace) -> None:
    from backend.publish import fetch_serving_anchors

    count = fetch_serving_anchors(args.season, args.week)
    print(
        f"hydrated {count} serving anchors for season {args.season} "
        f"anchor week {args.week:02d} from cfb.serving_anchors"
    )
