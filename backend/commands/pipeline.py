"""Data, feature, and forecast command handlers."""

from argparse import Namespace

from backend.config import SEASONS


def handle_ingest(args: Namespace) -> None:
    from backend.cfbd.client import CFBDClient
    from backend.etl.ingest import ingest_season

    client = CFBDClient()
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
            print(
                f"{season}: no CFBD play-by-play is available; no features to rebuild"
            )
            continue
        possessions = build_possessions(plays)
        team_games = build_team_games(possessions)
        unit_games = build_unit_games(plays)
        store.write_processed(possessions, "possessions", f"{season}.parquet")
        store.write_processed(team_games, "team_games", f"{season}.parquet")
        store.write_processed(unit_games, "unit_games", f"{season}.parquet")
        print(
            f"{season}: {len(possessions)} possessions, "
            f"{len(team_games)} team-game rows, "
            f"{len(unit_games)} unit-game rows"
        )


def handle_fit(args: Namespace) -> None:
    from backend.model.weekly import run_weekly_forecast

    result = run_weekly_forecast(args.season, args.week)
    print(result.ratings.head(30).to_string(index=False))
    print(
        f"wrote {len(result.ratings)} ratings, "
        f"{len(result.projections)} projections, and "
        f"{len(result.unit_ratings)} unit ratings for Week {result.week}"
    )
    print(f"forecast log: {result.log_directory}")


def handle_weekly_update(args: Namespace) -> None:
    from datetime import datetime, timezone

    from backend.model.joint_scoring import MODEL_VERSION
    from backend.model.weekly import (
        WeeklyForecastNotReady,
        resolve_ready_forecast_week,
        run_weekly_forecast,
    )
    from backend.odds.client import OddsAPIClient, OddsAPIError
    from backend.publish import publish, weekly_forecast_is_published

    as_of = datetime.now(timezone.utc)
    try:
        forecast_week = resolve_ready_forecast_week(
            args.season,
            args.week,
            as_of,
        )
    except WeeklyForecastNotReady as exc:
        print(f"weekly update not ready: {exc}")
        return
    if args.week is None and weekly_forecast_is_published(
        args.season,
        forecast_week,
        MODEL_VERSION,
    ):
        print(
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
            print(
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
        print(f"weekly update not ready: {exc}")
        return
    totals = publish(
        args.season,
        result.week,
        source="fit",
        include_backtest=False,
    )
    print(
        f"published Week {result.week}: {len(result.ratings)} ratings, "
        f"{len(result.projections)} projections, "
        f"{len(result.market_comparisons)} market comparisons"
    )
    print(f"serving totals: {totals}")
    print(f"forecast log: {result.log_directory}")


def handle_calibrate(args: Namespace) -> None:
    from datetime import timedelta

    import pandas as pd

    from backend.etl import store
    from backend.features.scoring import build_scoring_games
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
            store.read_processed("team_games", f"{season}.parquet"),
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
        progress=print,
    )
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
    print(format_diagnostic(result.summary))
    print(
        f"wrote {len(result.predictions)} predictions and "
        f"{len(result.summary)} calibration rows"
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
    print(
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
    print(
        f"wrote {len(result.ratings)} ratings, "
        f"{len(result.unit_ratings)} unit ratings, "
        f"{len(result.projections)} projections, and "
        f"{len(result.market_comparisons)} market comparisons"
    )
    print(f"forecast log: {result.log_directory}")
