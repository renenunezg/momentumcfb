"""Serving anchor, scoring, and verification command handlers."""

from argparse import Namespace


def handle_serving_anchors(args: Namespace) -> None:
    from backend.etl import store
    from backend.serving.anchors import (
        load_serving_anchors,
        serving_anchor_artifact,
    )

    if args.source != "market" and args.market_feed is not None:
        raise SystemExit("--market-feed requires --source market")

    market_feed = args.market_feed or "cfbd"
    checked = None
    if args.source == "market":
        from backend.model.ingame import SERVING_ANCHOR_COLUMNS
        from backend.serving.market import (
            MARKET_ANCHOR_WEEK,
            build_live_market_anchors,
            build_market_anchors,
            cross_check_closing_spreads,
        )

        if args.week is not None:
            raise SystemExit("market anchors cover a whole season; drop --week")
        try:
            built = (
                build_live_market_anchors(args.season)
                if market_feed == "live-odds"
                else build_market_anchors(args.season)
            )
        except FileNotFoundError as exc:
            raise SystemExit(
                f"season {args.season} has no stored {market_feed} market "
                "data or schedule"
            ) from exc
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        checked, problems = (
            cross_check_closing_spreads(built, args.season)
            if market_feed == "cfbd"
            else (None, [])
        )
        for problem in problems:
            print(f"PROBLEM: {problem}")
        if problems:
            raise SystemExit(1)
        week = MARKET_ANCHOR_WEEK
        anchors = built[SERVING_ANCHOR_COLUMNS].reset_index(drop=True)
    else:
        week = args.week if args.week is not None else 1
        try:
            anchors = load_serving_anchors(args.source, season=args.season, week=week)
        except FileNotFoundError as exc:
            raise SystemExit(
                f"no stored {args.source} projections for season "
                f"{args.season} week {week}; run `python -m backend "
                f"{args.source} --season {args.season}` first"
            ) from exc
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        built = anchors
    artifact = serving_anchor_artifact(args.season, week)
    store.write_processed(built, *artifact)
    stored = load_serving_anchors("serving", season=args.season, week=week)
    if not stored.equals(anchors):
        raise SystemExit(
            f"stored {'/'.join(artifact)} does not round-trip through "
            "the serving anchor loader"
        )
    weeks = sorted(stored["model_week"].unique())
    source_note = (
        (
            "the latest stored pregame Odds API spreads"
            if market_feed == "live-odds"
            else "the CFBD market lines"
        )
        if args.source == "market"
        else f"the {args.source} projections"
    )
    print(
        f"wrote {'/'.join(artifact)}: {len(stored)} outcome-free anchors "
        f"for season {args.season} model week"
        f"{'s' if len(weeks) > 1 else ''} "
        + (
            f"{weeks[0]}-{weeks[-1]}"
            if len(weeks) > 2
            else ", ".join(str(week) for week in weeks)
        )
        + f" from {source_note}"
    )
    if args.source == "market":
        if market_feed == "live-odds":
            print(
                f"froze {built['closing_snapshot_id'].nunique()} latest "
                "pregame snapshots; live and post-kickoff offers excluded"
            )
        else:
            print(
                f"cross-checked {checked} closing spreads against "
                "backtest/predictions_filtered.parquet"
                + ("" if checked else " (season absent from the backtest)")
            )
        sd_note = built["margin_sd_method"].iloc[0]
        print(f"margin_sd {built['margin_sd'].iloc[0]:.3f} points ({sd_note})")


def handle_serve_game(args: Namespace) -> None:
    from backend.etl import store
    from backend.serving.replay import summarize_latency
    from backend.serving.serve import (
        MissingPlayFeed,
        serve_game,
        served_events_artifact,
    )

    if args.anchor_source == "serving" and args.week is None:
        raise SystemExit("--anchor-source serving requires --week")
    if args.anchor_source == "calibration" and args.week is not None:
        raise SystemExit("calibration anchors cover a whole season; drop --week")
    try:
        events = serve_game(
            args.season,
            args.game_id,
            source=args.anchor_source,
            week=args.week,
        )
    except (MissingPlayFeed, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    artifact = served_events_artifact(args.season, args.game_id)
    store.write_processed(events, *artifact)
    latency = summarize_latency(events["latency_seconds"])
    print(
        f"wrote {'/'.join(artifact)}: {int(events['emitted'].sum())} win "
        f"probabilities across {len(events)} play events for game "
        f"{args.game_id} (model week {int(events['model_week'].iloc[0])}, "
        f"{args.anchor_source} anchors)"
    )
    print(
        f"latency per event: median {latency['median_seconds'] * 1e3:.1f} ms, "
        f"p99 {latency['p99_seconds'] * 1e3:.1f} ms, "
        f"max {latency['max_seconds'] * 1e3:.1f} ms"
    )


def handle_serve_verify(args: Namespace) -> None:
    from backend.etl import store
    from backend.serving.verify import verify_served

    try:
        problems, frames = verify_served(args.season, args.game_id)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    for problem in problems:
        print(f"PROBLEM: {problem}")
    if args.game_id is None:
        store.write_processed(
            frames["games"],
            "serving",
            f"served_verification_games_{args.season}.parquet",
        )
        store.write_processed(
            frames["summary"],
            "serving",
            f"served_verification_{args.season}.parquet",
        )
    summary = frames["summary"].iloc[0]
    print(
        f"{summary['status']}: {summary['served_rows']} served "
        f"probabilities vs {summary['stored_rows']} stored baseline rows "
        f"across {summary['compared_games']} of {summary['served_games']} "
        f"served games"
    )
    if summary["unverifiable_games"]:
        print(
            f"{summary['unverifiable_games']} served games have no stored "
            "baseline predictions to compare against"
        )
    if problems:
        raise SystemExit(1)
