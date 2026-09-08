"""Data, feature, and forecast command handlers."""

import logging
import os
from argparse import Namespace

from backend.config import SEASONS

log = logging.getLogger(__name__)


def handle_ingest(args: Namespace) -> None:
    from backend.cfbd.client import CFBDClient
    from backend.etl.ingest import ingest_season

    client = CFBDClient(max_calls=args.max_calls, min_remaining=args.min_remaining)
    from backend.config import MAX_REGULAR_WEEK
    from backend.etl.ingest import SEASON_TYPES

    client.ensure_budget(
        len(args.seasons)
        * (4 + len(SEASON_TYPES) * (1 if args.week is not None else MAX_REGULAR_WEEK))
    )
    for season in args.seasons:
        ingest_season(client, season, only_week=args.week)


def handle_features(args: Namespace) -> None:
    from backend.etl import store
    from backend.features.possessions import build_possessions, build_team_games
    from backend.features.units import build_unit_games

    for season in args.seasons:
        try:
            plays = store.read_season_pbp(season)
        except FileNotFoundError:
            log.info(
                f"{season}: no CFBD play-by-play is available; no features to rebuild"
            )
            continue
        possessions = build_possessions(plays)
        team_games = build_team_games(possessions)
        unit_games = build_unit_games(plays)
        store.write_processed(possessions, "possessions", f"{season}.parquet")
        store.write_processed(team_games, "team_games", f"{season}.parquet")
        store.write_processed(unit_games, "unit_games", f"{season}.parquet")
        log.info(
            f"{season}: {len(possessions)} possessions, "
            f"{len(team_games)} team-game rows, "
            f"{len(unit_games)} unit-game rows"
        )


def handle_fit(args: Namespace) -> None:
    from backend.model.weekly import run_weekly_forecast

    result = run_weekly_forecast(args.season, args.week)
    log.info(result.ratings.head(30).to_string(index=False))
    log.info(
        f"wrote {len(result.ratings)} ratings, "
        f"{len(result.projections)} projections, and "
        f"{len(result.unit_ratings)} unit ratings for Week {result.week}"
    )
    log.info(f"forecast log: {result.log_directory}")


def handle_weekly_update(args: Namespace) -> None:
    from datetime import datetime, timezone

    from backend.model.joint_scoring import MODEL_VERSION
    from backend.model.weekly import (
        WeeklyForecastNotReady,
        resolve_ready_forecast_week,
        run_weekly_forecast,
    )
    from backend.odds.client import OddsAPIClient, OddsAPIError
    from backend.publish import (
        ensure_recommendation_schema,
        publish,
        weekly_forecast_is_published,
    )

    as_of = datetime.now(timezone.utc)
    try:
        forecast_week = resolve_ready_forecast_week(
            args.season,
            args.week,
            as_of,
        )
    except WeeklyForecastNotReady as exc:
        log.info(f"weekly update not ready: {exc}")
        return
    ensure_recommendation_schema()
    if args.week is None and weekly_forecast_is_published(
        args.season,
        forecast_week,
        MODEL_VERSION,
    ):
        log.info(
            f"weekly update already published for {args.season} "
            f"model Week {forecast_week}; no changes made"
        )
        return
    try:
        try:
            result = run_weekly_forecast(
                args.season,
                forecast_week,
                odds_client=OddsAPIClient(),
                require_market=True,
                as_of=as_of,
            )
        except OddsAPIError as exc:
            if "OUT_OF_USAGE_CREDITS" not in str(exc):
                raise
            log.info(
                "Odds API quota exhausted; publishing the pure-model forecast "
                "with market offers marked unavailable"
            )
            result = run_weekly_forecast(
                args.season,
                forecast_week,
                odds_client=None,
                require_market=False,
                as_of=as_of,
            )
    except WeeklyForecastNotReady as exc:
        log.info(f"weekly update not ready: {exc}")
        return
    totals = publish(
        args.season,
        result.week,
        source="fit",
        include_backtest=False,
    )
    log.info(
        f"published Week {result.week}: {len(result.ratings)} ratings, "
        f"{len(result.projections)} projections, "
        f"{len(result.market_comparisons)} market comparisons"
    )
    log.info(f"serving totals: {totals}")
    log.info(f"forecast log: {result.log_directory}")
    if (
        os.getenv("GITHUB_ACTIONS") == "true"
        and result.projections["market_home_spread"].notna().any()
    ):
        from backend.odds.scheduling import schedule_weekly_kickoff

        dispatch_at = schedule_weekly_kickoff(result.projections, args.season)
        log.info(f"kickoff capture rearmed for {dispatch_at}")


def handle_calibrate(args: Namespace) -> None:
    if getattr(args, "recommendations", False):
        if getattr(args, "production_replay", False) or getattr(args, "seasons", None):
            raise SystemExit(
                "--recommendations uses fixed splits and cannot combine with --production-replay or --seasons"
            )
        from pathlib import Path

        from backend.config import PROCESSED_DIR
        from backend.model.pick_calibration import run_recommendation_calibration

        destination = (
            Path(args.output_directory)
            if args.output_directory
            else PROCESSED_DIR / "recommendation_calibration"
        )
        manifest, report = run_recommendation_calibration(destination)
        log.info("\n%s", report[report.season.eq("all")].to_string(index=False))
        for market, result in manifest["parameters"].items():
            log.info(
                "%s: %s; paired log-loss change %.6f, 95%% CI %s; probability-score gate %s; model edge supported %s; push calibration %s",
                market,
                result["candidate"],
                result["evaluation_log_loss_change"],
                result["evaluation_log_loss_change_95ci"],
                result["probability_score_gate_passed"],
                result["model_edge_supported"],
                result["push_calibration_status"],
            )
        log.info(
            "Saved %s. Diagnostic research only. Forward recommendations continue using pure-model probabilities and the configured price/input flags; this report does not gate them.",
            destination,
        )
        return
    if getattr(args, "production_replay", False):
        _handle_production_replay(args)
        return
    if getattr(args, "seasons", None) or getattr(args, "output_directory", None):
        raise SystemExit("--seasons and --output-directory require --production-replay")
    from datetime import timedelta

    import pandas as pd

    from backend.etl import store
    from backend.features.scoring import build_scoring_games, load_scoring_team_games
    from backend.model.calibration import (
        fbs_calibration_cohort,
        format_diagnostic,
        run_calibration,
    )
    from backend.model.joint_scoring import fit_joint_scoring
    from backend.model.preseason import build_historical_carryover_priors

    games_by_season = {
        season: build_scoring_games(
            store.read_games(season),
            load_scoring_team_games(season),
        )
        for season in SEASONS
    }
    priors_by_season = {}
    for season in sorted(games_by_season)[1:]:
        previous = fbs_calibration_cohort(games_by_season[season - 1])
        forecast_week = int(previous["model_week"].max()) + 1
        as_of = (
            pd.to_datetime(previous["start_date"], utc=True).max()
            + timedelta(seconds=1)
        ).to_pydatetime()
        previous_fit = fit_joint_scoring(previous, forecast_week, as_of)
        priors_by_season[season] = build_historical_carryover_priors(
            previous_fit,
            fbs_calibration_cohort(games_by_season[season]),
        )
    result = run_calibration(
        games_by_season,
        strength_priors_by_season=priors_by_season,
        progress=log.info,
    )
    result.predictions["evaluation_contract"] = "historical_carryover"
    result.summary["evaluation_contract"] = "historical_carryover"
    store.write_processed(
        result.predictions,
        "calibration",
        "joint_scoring_predictions.parquet",
    )
    store.write_processed(
        result.summary,
        "calibration",
        "joint_scoring_summary.parquet",
    )
    log.info(format_diagnostic(result.summary))
    log.info(
        "Historical carryover calibration excludes rich preseason inputs and "
        "production week-zero grouping; use --production-replay for that contract."
    )
    log.info(
        f"wrote {len(result.predictions)} predictions and "
        f"{len(result.summary)} calibration rows"
    )


def _handle_production_replay(args: Namespace) -> None:
    from pathlib import Path

    import pandas as pd

    from backend.config import PROCESSED_DIR
    from backend.model.production_evaluation import (
        replay_production_season,
        summarize_production_replay,
    )

    if not args.seasons:
        raise SystemExit("--production-replay requires explicit --seasons")
    predictions = pd.concat(
        [replay_production_season(season) for season in args.seasons], ignore_index=True
    )
    summary = summarize_production_replay(predictions)
    destination = (
        Path(args.output_directory)
        if args.output_directory
        else PROCESSED_DIR / "calibration" / "production_replay"
    )
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(destination / "predictions.parquet", index=False)
    summary.to_parquet(destination / "summary.parquet", index=False)
    log.info(
        summary[
            [
                "group_value",
                "evaluation_stage",
                "metric",
                "n_games",
                "mae",
                "coverage_80",
            ]
        ].to_string(index=False)
    )
    log.info(
        f"wrote fixed-config retrospective production replay to {destination}; "
        "no tuning, live grades, or frozen baseline artifacts changed"
    )


def handle_preseason(args: Namespace) -> None:
    from backend.cfbd.client import CFBDClient
    from backend.etl.ingest import ingest_preseason_sources
    from backend.model.preseason import run_preseason_forecast
    from backend.odds.client import OddsAPIClient, OddsAPIError

    if args.with_odds_api and not args.refresh:
        raise SystemExit("--with-odds-api requires --refresh")
    if args.refresh:
        odds_client = None
        if args.with_odds_api:
            try:
                odds_client = OddsAPIClient()
            except OddsAPIError as exc:
                raise SystemExit(str(exc)) from exc
        ingest_preseason_sources(CFBDClient(), args.season, odds_client=odds_client)
    result = run_preseason_forecast(args.season, args.week)
    log.info(
        result.ratings[
            [
                "team",
                "classification",
                "power_rating",
                "offense_points",
                "defense_points",
                "power_rating_sd",
                "missing_input_count",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )
    log.info(
        f"wrote {len(result.ratings)} ratings, "
        f"{len(result.unit_ratings)} unit ratings, "
        f"{len(result.projections)} projections, and "
        f"{len(result.market_comparisons)} market comparisons"
    )
    log.info(f"forecast log: {result.log_directory}")
