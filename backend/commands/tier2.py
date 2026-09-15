"""Paid CFBD data capture and offline diagnostics with explicit budgets."""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.cfbd.client import CFBDClient
from backend.cfbd.snapshots import capture, receipts
from backend.config import PROCESSED_DIR
from backend.model.paid_benchmarks import ADJUSTED_ENDPOINTS

log = logging.getLogger(__name__)


def handle_tier2_snapshot(args):
    client = CFBDClient(max_calls=args.max_calls, min_remaining=args.min_remaining)
    if args.weather and args.week is None and not args.historical_weather:
        from backend.model.weekly import resolve_ready_forecast_week

        week = resolve_ready_forecast_week(
            args.season, as_of=datetime.now(timezone.utc)
        )
    else:
        week = args.week
    requests = []
    if args.weather:
        params = {"year": args.season}
        if week is not None:
            params["week"] = week
        requests.append(("/games/weather", params))
    if args.adjusted:
        adjusted_year = args.adjusted_season or args.season - 1
        for endpoint in ADJUSTED_ENDPOINTS:
            cached = any(
                r["params"].get("year") == adjusted_year and r["payload"]
                for _, r in receipts(endpoint)
            )
            if not cached:
                requests.append((endpoint, {"year": adjusted_year}))
            else:
                log.info("Using cached %s %s", adjusted_year, endpoint)
    if not args.weather and not args.adjusted:
        raise ValueError("select --weather or --adjusted")
    client.ensure_budget(len(requests))
    for endpoint, params in requests:
        path = capture(client, endpoint, params)
        log.info("saved %s", path)


def handle_weather_evaluate(args):
    from backend.model.weather_evaluation import evaluate_weather

    result = evaluate_weather(
        pd.read_parquet(args.predictions), Path(args.output_directory)
    )
    log.info("%s", result.to_string(index=False))


def handle_tier2_benchmark(args):
    from backend.etl import store
    from backend.model.paid_benchmarks import (
        latest_adjusted,
        player_benchmark,
        team_benchmark,
    )
    from backend.model.preseason import _latest_season_fit

    now = datetime.now(timezone.utc)
    fit = _latest_season_fit(args.season)
    ratings = pd.DataFrame(x.to_record() for x in fit.ratings())
    directory = Path(args.output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    provider = latest_adjusted(ADJUSTED_ENDPOINTS[0], args.season, now)
    matched, summary = team_benchmark(ratings, provider)
    matched.to_parquet(directory / "team_benchmark.parquet", index=False)
    summary.to_csv(directory / "team_summary.csv", index=False)
    log.info("%s", summary.to_string(index=False))
    try:
        values = store.read_processed(
            "players", "player_values", f"{args.season}.parquet"
        )
    except FileNotFoundError:
        log.info("No cached player values; team benchmark completed")
        return
    for endpoint in ADJUSTED_ENDPOINTS[1:]:
        provider = latest_adjusted(endpoint, args.season, now)
        matched, summary = player_benchmark(values, provider)
        name = endpoint.rsplit("/", 1)[-1]
        matched.to_parquet(directory / f"{name}_benchmark.parquet", index=False)
        summary.to_csv(directory / f"{name}_summary.csv", index=False)
        log.info("%s: %s", name, summary.to_string(index=False))


def handle_live_pilot(args):
    from backend.serving.cfbd_live import run_pilot
    from backend.serving.serve import load_frozen_params

    projections = pd.read_parquet(args.forecast)
    if "model_week" not in projections:
        projections["model_week"] = projections.week
    client = CFBDClient(max_calls=args.max_calls, min_remaining=args.min_remaining)
    destination = (
        Path(args.output_directory)
        if args.output_directory
        else PROCESSED_DIR / "live_cfbd" / str(args.season)
    )
    result = run_pilot(
        client,
        projections,
        load_frozen_params(),
        destination,
        season=args.season,
        week=args.week,
        game_ids=args.game_ids or (),
        polls=args.polls,
        interval=args.interval,
        max_games=args.max_games,
    )
    log.info(
        "Pilot saved %s observations; %s calls used", len(result), client.calls_used
    )
