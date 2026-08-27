"""Live odds and kickoff command handlers."""

from argparse import Namespace


def handle_kickoff_check(args: Namespace) -> None:
    from backend.odds.kickoff import (
        check_kickoff_readiness,
        format_readiness,
    )

    result = check_kickoff_readiness(
        args.season,
        args.week,
        game_ids=args.game_id,
        cluster_minutes=args.cluster_minutes,
        lead_minutes=args.lead_minutes,
        post_minutes=args.post_minutes,
        interval_seconds=args.interval_seconds,
        min_quota=args.min_quota,
        max_forecast_age_hours=args.max_forecast_age_hours,
        max_source_age_hours=args.max_source_age_hours,
        max_poll_age_minutes=args.max_poll_age_minutes,
        max_offer_staleness_seconds=args.max_offer_staleness_seconds,
        min_providers=args.min_providers,
    )
    print(format_readiness(result))
    if not result.ready:
        raise SystemExit(1)


def handle_kickoff_run(args: Namespace) -> None:
    from backend.odds.kickoff import run_kickoff_window

    try:
        result = run_kickoff_window(
            args.season,
            args.week,
            game_ids=args.game_id,
            cluster_minutes=args.cluster_minutes,
            lead_minutes=args.lead_minutes,
            post_minutes=args.post_minutes,
            interval_seconds=args.interval_seconds,
            lookback_hours=args.lookback_hours,
            lookahead_hours=args.lookahead_hours,
            min_quota=args.min_quota,
            max_failures=args.max_failures,
            max_wait_hours=args.max_wait_hours,
            max_forecast_age_hours=args.max_forecast_age_hours,
            max_source_age_hours=args.max_source_age_hours,
            max_offer_staleness_seconds=args.max_offer_staleness_seconds,
            min_providers=args.min_providers,
        )
    except ValueError as exc:
        print(f"NOT READY\nBLOCKER: {exc}")
        raise SystemExit(1) from exc
    print("READY")
    for detail in result.target_details:
        print(f"OK: {detail}")
    print(
        f"OK: wrote {result.anchor_count} market anchors after "
        f"{result.plan.polls} verified polls"
    )


def handle_live_odds(args: Namespace) -> None:
    from backend.odds.live import load_division_one_schedule, run_live_polling

    schedule = load_division_one_schedule(args.season)
    completed = run_live_polling(
        args.season,
        schedule,
        polls=args.polls,
        interval_seconds=args.interval_seconds,
        lookback_hours=args.lookback_hours,
        lookahead_hours=args.lookahead_hours,
        days_from=args.days_from,
        min_quota=args.min_quota,
        max_failures=args.max_failures,
    )
    print(f"completed {completed} of {args.polls} polls")
    if completed != args.polls:
        raise SystemExit(1)


def handle_live_replay(args: Namespace) -> None:
    import pandas as pd

    from backend.odds.live import offers_available_at, verify_live_snapshots

    problems, frames = verify_live_snapshots(args.season)
    polls = frames["polls"]
    print(
        f"{len(polls)} polls stored "
        f"({int(polls['poll_status'].eq('ok').sum()) if not polls.empty else 0} ok, "
        f"{int(polls['poll_status'].eq('error').sum()) if not polls.empty else 0} failed), "
        f"{len(frames['offers'])} offers, {len(frames['events'])} event states"
    )
    if problems:
        for problem in problems:
            print(f"PROBLEM: {problem}")
    else:
        print("replay check passed: snapshots are complete and append-only")
    if args.as_of is not None:
        as_of = pd.to_datetime(args.as_of, utc=True)
        available = offers_available_at(polls, frames["offers"], as_of)
        print(f"\navailable at {as_of.isoformat()}: {len(available)} offers")
        if not available.empty:
            view = available[
                [
                    "game_id",
                    "phase",
                    "provider_key",
                    "market",
                    "selection",
                    "point",
                    "price",
                    "staleness_seconds",
                ]
            ].sort_values(["game_id", "market", "provider_key", "selection"])
            print(view.to_string(index=False))
    if problems:
        raise SystemExit(1)
