import argparse

from backend.config import SEASONS


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="backend")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="pull source data into raw parquet")
    ing.add_argument("--seasons", type=int, nargs="+", default=SEASONS)
    ing.add_argument("--week", type=int, default=None)
    ing.add_argument(
        "--pbp-source",
        choices=["sportsdataverse", "cfbd"],
        default="sportsdataverse",
    )

    feat = sub.add_parser("features", help="build possession and team-game features")
    feat.add_argument("--seasons", type=int, nargs="+", default=SEASONS)

    fit = sub.add_parser("fit", help="fit joint scoring ratings and projections")
    fit.add_argument("--season", type=int, required=True)
    fit.add_argument("--week", type=int, default=None)

    sub.add_parser(
        "calibrate",
        help="tune and diagnose chronological joint scoring projections",
    )

    preseason = sub.add_parser(
        "preseason",
        help="build timestamped preseason ratings and market comparisons",
    )
    preseason.add_argument("--season", type=int, required=True)
    preseason.add_argument("--week", type=int, default=1)
    preseason.add_argument(
        "--refresh",
        action="store_true",
        help="refresh the source snapshot from CFBD before forecasting",
    )

    ingame = sub.add_parser(
        "ingame-baseline",
        help="reconstruct play-boundary states and evaluate the baseline "
        "in-game win projection",
    )
    ingame.add_argument("--seasons", type=int, nargs="+", default=SEASONS)
    ingame.add_argument(
        "--leakage-games-per-season",
        type=int,
        default=2,
        help="games per season replayed from truncated play prefixes",
    )

    live = sub.add_parser(
        "live-odds",
        help="capture append-only live sportsbook line snapshots",
    )
    live.add_argument("--season", type=int, required=True)
    live.add_argument(
        "--polls",
        type=int,
        default=1,
        help="maximum number of poll cycles before stopping",
    )
    live.add_argument("--interval-seconds", type=float, default=60.0)
    live.add_argument(
        "--lookback-hours",
        type=float,
        default=8.0,
        help="include events that commenced up to this many hours ago",
    )
    live.add_argument(
        "--lookahead-hours",
        type=float,
        default=6.0,
        help="include events commencing up to this many hours ahead",
    )
    live.add_argument(
        "--days-from",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="also fetch completed games from up to N days back (extra quota)",
    )
    live.add_argument(
        "--min-quota",
        type=int,
        default=50,
        help="stop polling once remaining API requests fall below this",
    )
    live.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help="stop polling after this many consecutive failed polls",
    )

    replay = sub.add_parser(
        "live-replay",
        help="verify stored live odds snapshots and replay a point in time",
    )
    replay.add_argument("--season", type=int, required=True)
    replay.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="ISO timestamp; show the live market view available at that time",
    )

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.command == "ingest":
        from backend.cfbd.client import CFBDClient
        from backend.etl.ingest import ingest_season

        client = CFBDClient()
        for season in args.seasons:
            ingest_season(
                client,
                season,
                only_week=args.week,
                pbp_source=args.pbp_source,
            )

    elif args.command == "features":
        from backend.etl import store
        from backend.features.possessions import build_possessions, build_team_games

        for season in args.seasons:
            plays = store.read_season_pbp(season)
            possessions = build_possessions(plays)
            team_games = build_team_games(possessions)
            store.write_processed(possessions, "possessions", f"{season}.parquet")
            store.write_processed(team_games, "team_games", f"{season}.parquet")
            print(
                f"{season}: {len(possessions)} possessions, "
                f"{len(team_games)} team-game rows"
            )

    elif args.command == "fit":
        from datetime import timedelta

        import pandas as pd

        from backend.etl import store
        from backend.features.scoring import build_scoring_games
        from backend.model.joint_scoring import fit_joint_scoring

        games = build_scoring_games(
            store.read_games(args.season),
            store.read_processed("team_games", f"{args.season}.parquet"),
        )
        forecast_week = args.week or int(games["model_week"].max()) + 1
        target = games[games["model_week"].eq(forecast_week)]
        as_of = (
            target["start_date"].min()
            if not target.empty
            else games["start_date"].max() + timedelta(seconds=1)
        ).to_pydatetime()
        fitted = fit_joint_scoring(games, forecast_week, as_of)
        ratings = pd.DataFrame(rating.to_record() for rating in fitted.ratings())
        projections = pd.DataFrame(
            projection.to_record() for projection in fitted.project(target)
        )
        filename = f"{args.season}_{forecast_week:02d}.parquet"
        store.write_processed(ratings, "ratings", filename)
        store.write_processed(projections, "projections", filename)
        print(ratings.head(30).to_string(index=False))
        print(f"wrote {len(ratings)} ratings and {len(projections)} projections")

    elif args.command == "calibrate":
        from backend.etl import store
        from backend.features.scoring import build_scoring_games
        from backend.model.calibration import format_diagnostic, run_calibration

        games_by_season = {
            season: build_scoring_games(
                store.read_games(season),
                store.read_processed("team_games", f"{season}.parquet"),
            )
            for season in SEASONS
        }
        result = run_calibration(games_by_season, progress=print)
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

    elif args.command == "preseason":
        from backend.cfbd.client import CFBDClient
        from backend.etl.ingest import ingest_preseason_sources
        from backend.model.preseason import run_preseason_forecast
        from backend.odds.client import OddsAPIClient, OddsAPIError

        if args.refresh:
            try:
                odds_client = OddsAPIClient()
            except OddsAPIError as exc:
                odds_client = None
                print(f"WARNING: {exc}; retaining unpriced CFBD market fallback")
            ingest_preseason_sources(
                CFBDClient(), args.season, odds_client=odds_client
            )
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
            f"{len(result.projections)} projections, and "
            f"{len(result.market_comparisons)} market comparisons"
        )
        print(f"forecast log: {result.log_directory}")

    elif args.command == "ingame-baseline":
        import pandas as pd

        from backend.etl import store
        from backend.features.ingame import build_game_states, leakage_problems
        from backend.model.calibration import DEVELOPMENT_SEASONS
        from backend.model.ingame import (
            MODEL_VERSION,
            build_baseline_inputs,
            evaluate_baseline,
            fit_baseline,
            format_ingame_diagnostic,
            win_probability,
        )

        state_frames = []
        problems = []
        for season in args.seasons:
            plays = store.read_season_pbp(season)
            states = build_game_states(plays)
            store.write_processed(states, "ingame", "states", f"{season}.parquet")
            state_frames.append(states)
            game_ids = sorted(states["game_id"].unique())
            step = max(1, len(game_ids) // max(args.leakage_games_per_season, 1))
            sample = game_ids[::step][: args.leakage_games_per_season]
            problems.extend(leakage_problems(plays, sample))
            print(
                f"{season}: {len(states)} play states across "
                f"{states['game_id'].nunique()} games "
                f"(leakage-checked {len(sample)})"
            )
        for problem in problems:
            print(f"PROBLEM: {problem}")
        if problems:
            raise SystemExit(1)
        print("leakage check passed: states are prefix-stable")

        try:
            anchors = store.read_processed(
                "calibration", "joint_scoring_predictions.parquet"
            )
        except FileNotFoundError as exc:
            raise SystemExit(
                "missing chronological pregame projections; run "
                "`python -m backend calibrate` first"
            ) from exc
        states = pd.concat(state_frames, ignore_index=True)
        inputs = build_baseline_inputs(states, anchors)
        development = inputs[inputs["season"].isin(DEVELOPMENT_SEASONS)]
        params = fit_baseline(development)
        inputs["win_probability"] = win_probability(inputs, params)
        inputs["model_version"] = MODEL_VERSION
        summary = evaluate_baseline(inputs, params)
        store.write_processed(inputs, "ingame", "baseline_predictions.parquet")
        store.write_processed(summary, "ingame", "baseline_summary.parquet")
        print(
            f"anchored {inputs['game_id'].nunique()} of "
            f"{states['game_id'].nunique()} games with pregame projections; "
            f"wrote {len(inputs)} baseline predictions"
        )
        print(format_ingame_diagnostic(summary))

    elif args.command == "live-odds":
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

    elif args.command == "live-replay":
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
