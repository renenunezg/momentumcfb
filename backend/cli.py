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
