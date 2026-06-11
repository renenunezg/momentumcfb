import argparse

from backend.config import SEASONS


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="backend")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="pull CFBD data into raw parquet")
    ing.add_argument("--seasons", type=int, nargs="+", default=SEASONS)
    ing.add_argument("--week", type=int, default=None)

    feat = sub.add_parser("features", help="build team_game_epa parquet")
    feat.add_argument("--seasons", type=int, nargs="+", default=SEASONS)

    fit = sub.add_parser("fit", help="fit one season and print ratings")
    fit.add_argument("--season", type=int, required=True)
    fit.add_argument("--variant", choices=["all", "filtered"], default="filtered")
    fit.add_argument("--method", choices=["advi", "nuts"], default="nuts")

    bt = sub.add_parser("backtest", help="walk forward backtest vs closing lines")
    bt.add_argument("--seasons", type=int, nargs="+", default=SEASONS)
    bt.add_argument("--variant", choices=["all", "filtered"], default="filtered")
    bt.add_argument("--method", choices=["advi", "nuts"], default="nuts")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.command == "ingest":
        from backend.cfbd.client import CFBDClient
        from backend.etl.ingest import ingest_season

        client = CFBDClient()
        for season in args.seasons:
            ingest_season(client, season, only_week=args.week)

    elif args.command == "features":
        from backend.etl import store
        from backend.features.epa import team_game_epa

        for season in args.seasons:
            plays = store.read_season_pbp(season)
            out = team_game_epa(plays)
            store.write_processed(out, "team_game_epa", f"{season}.parquet")
            print(f"{season}: {len(out)} team game rows")

    elif args.command == "fit":
        from backend.backtest.harness import (
            build_season_prior,
            load_season_inputs,
            season_teams,
        )
        from backend.etl import store
        from backend.model.state_space import fit_ratings

        games, talent, returning = load_season_inputs(args.season, args.variant)
        prior = build_season_prior(args.season, season_teams(games), talent, returning, {})
        n_states = int(games["week_index"].max()) + 1
        ratings, _ = fit_ratings(games, prior, n_states, method=args.method)
        store.write_processed(ratings, "ratings", f"{args.season}.parquet")
        print(ratings.head(30).to_string(index=False))

    elif args.command == "backtest":
        from backend.backtest.harness import run_backtest
        from backend.backtest.metrics import ats_summary, calibration_table, mae
        from backend.etl import store

        preds = run_backtest(args.seasons, variant=args.variant, method=args.method)
        store.write_processed(preds, "backtest", f"predictions_{args.variant}.parquet")
        print(f"\nMAE vs actual margin: {mae(preds):.2f}")
        print("\nATS by edge threshold:")
        print(ats_summary(preds).to_string(index=False))
        print("\nCalibration:")
        print(calibration_table(preds).to_string(index=False))
        print("\nBy season:")
        for season, grp in preds.groupby("season"):
            row = ats_summary(grp, thresholds=(3.0,)).iloc[0]
            print(f"  {season}: {row.wins}-{row.losses} ({row.win_pct:.1%}) at 3+ edge")
