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

    for name, description in (
        (
            "ingame-momentum",
            "layer the cumulative-evidence momentum adjustment over the "
            "frozen baseline and record the adopt-or-reject verdict",
        ),
        (
            "ingame-momentum-recency",
            "layer the recency-weighted momentum adjustment over the frozen "
            "baseline and record the adopt-or-reject verdict",
        ),
    ):
        momentum = sub.add_parser(name, help=description)
        momentum.add_argument("--seasons", type=int, nargs="+", default=SEASONS)
        momentum.add_argument(
            "--leakage-games-per-season",
            type=int,
            default=2,
            help="games per season replayed from truncated play prefixes",
        )

    stream = sub.add_parser(
        "ingame-stream",
        help="replay stored plays as a simulated live feed and prove streamed "
        "baseline probabilities equal the stored batch outputs",
    )
    stream.add_argument("--season", type=int, required=True)
    stream.add_argument(
        "--game-id",
        type=int,
        default=None,
        help="replay a single game for diagnosis; skips writing artifacts",
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

    elif args.command in ("ingame-momentum", "ingame-momentum-recency"):
        import numpy as np
        import pandas as pd

        from backend.etl import store
        from backend.features.ingame import (
            build_momentum_states,
            build_process_evidence,
            leakage_problems,
        )
        from backend.model.calibration import DEVELOPMENT_SEASONS
        from backend.model.ingame import IngameBaselineParams, win_probability
        from backend.model.momentum import (
            MODEL_VERSION,
            RECENCY_MODEL_VERSION,
            evaluate_momentum,
            fit_momentum,
            fit_momentum_recency,
            format_momentum_diagnostic,
            momentum_recency_win_probability,
            momentum_win_probability,
        )

        recency = args.command == "ingame-momentum-recency"

        evidence_frames = []
        problems = []
        for season in args.seasons:
            plays = store.read_season_pbp(season)
            evidence = build_process_evidence(plays)
            evidence_frames.append(evidence)
            game_ids = sorted(evidence["game_id"].unique())
            step = max(1, len(game_ids) // max(args.leakage_games_per_season, 1))
            sample = game_ids[::step][: args.leakage_games_per_season]
            problems.extend(
                leakage_problems(plays, sample, builder=build_momentum_states)
            )
            print(
                f"{season}: evidence at {len(evidence)} play boundaries across "
                f"{evidence['game_id'].nunique()} games "
                f"(leakage-checked {len(sample)})"
            )
        for problem in problems:
            print(f"PROBLEM: {problem}")
        if problems:
            raise SystemExit(1)
        print(
            "extended leakage check passed: states and process evidence are "
            "prefix-stable"
        )

        try:
            baseline = store.read_processed("ingame", "baseline_predictions.parquet")
            baseline_summary = store.read_processed(
                "ingame", "baseline_summary.parquet"
            )
        except FileNotFoundError as exc:
            raise SystemExit(
                "missing baseline predictions; run "
                "`python -m backend ingame-baseline` first"
            ) from exc
        baseline = baseline[baseline["season"].isin(args.seasons)]
        parameter = baseline_summary[
            baseline_summary["summary_type"].eq("parameter")
        ].iloc[0]
        baseline_params = IngameBaselineParams(
            possession_points=float(parameter["possession_points"]),
            field_position_points_per_yard=float(
                parameter["field_position_points_per_yard"]
            ),
            sd_floor_points=float(parameter["sd_floor_points"]),
        )

        evidence = pd.concat(evidence_frames, ignore_index=True).drop(
            columns=["play_index"]
        )
        inputs = baseline.merge(
            evidence,
            on=["game_id", "source_play_id"],
            how="inner",
            validate="one_to_one",
        )
        if len(inputs) != len(baseline):
            raise SystemExit(
                f"process evidence covers {len(inputs)} of {len(baseline)} "
                "baseline play boundaries"
            )
        if not np.allclose(
            win_probability(inputs, baseline_params),
            inputs["win_probability"].to_numpy(float),
            atol=1e-9,
        ):
            raise SystemExit(
                "stored baseline probabilities do not match the frozen parameters"
            )
        inputs = inputs.rename(
            columns={
                "win_probability": "baseline_win_probability",
                "model_version": "baseline_model_version",
            }
        )

        development = inputs[inputs["season"].isin(DEVELOPMENT_SEASONS)]
        if recency:
            params = fit_momentum_recency(
                development, baseline_params, progress=print
            )
            inputs["momentum_win_probability"] = momentum_recency_win_probability(
                inputs, baseline_params, params
            )
            inputs["model_version"] = RECENCY_MODEL_VERSION
            summary = evaluate_momentum(
                inputs,
                params,
                rejection_note=(
                    "momentum iteration pauses until Rene agrees on a "
                    "structurally different approach"
                ),
            )
            prefix = "momentum_recency"
        else:
            params = fit_momentum(development, baseline_params)
            inputs["momentum_win_probability"] = momentum_win_probability(
                inputs, baseline_params, params
            )
            inputs["model_version"] = MODEL_VERSION
            summary = evaluate_momentum(inputs, params)
            prefix = "momentum"
        store.write_processed(inputs, "ingame", f"{prefix}_predictions.parquet")
        store.write_processed(summary, "ingame", f"{prefix}_summary.parquet")
        print(
            f"momentum probabilities at {len(inputs)} play boundaries across "
            f"{inputs['game_id'].nunique()} games (every baseline boundary)"
        )
        print(format_momentum_diagnostic(summary))

    elif args.command == "ingame-stream":
        from time import perf_counter

        import pandas as pd

        from backend.etl import store
        from backend.model.ingame import MODEL_VERSION, IngameBaselineParams
        from backend.serving.replay import (
            latency_conclusion,
            replay_game,
            stream_problems,
            summarize_latency,
        )

        try:
            stored = store.read_processed("ingame", "baseline_predictions.parquet")
            baseline_summary = store.read_processed(
                "ingame", "baseline_summary.parquet"
            )
        except FileNotFoundError as exc:
            raise SystemExit(
                "missing baseline predictions; run "
                "`python -m backend ingame-baseline` first"
            ) from exc
        stored = stored[stored["season"].eq(args.season)]
        if args.game_id is not None:
            stored = stored[stored["game_id"].eq(args.game_id)]
        if stored.empty:
            raise SystemExit(
                f"no stored baseline predictions for season {args.season}"
                + (f" game {args.game_id}" if args.game_id is not None else "")
            )
        versions = sorted(stored["model_version"].unique())
        if versions != [MODEL_VERSION]:
            raise SystemExit(
                f"stored predictions carry model versions {versions}; "
                f"the streaming harness serves {MODEL_VERSION}"
            )
        parameter = baseline_summary[
            baseline_summary["summary_type"].eq("parameter")
        ].iloc[0]
        params = IngameBaselineParams(
            possession_points=float(parameter["possession_points"]),
            field_position_points_per_yard=float(
                parameter["field_position_points_per_yard"]
            ),
            sd_floor_points=float(parameter["sd_floor_points"]),
        )
        anchors = store.read_processed(
            "calibration", "joint_scoring_predictions.parquet"
        )
        plays_by_game = dict(
            iter(store.read_season_pbp(args.season).groupby("game_id", sort=False))
        )

        game_ids = list(dict.fromkeys(stored["game_id"]))
        problems = []
        event_frames = []
        started = perf_counter()
        for number, game_id in enumerate(game_ids, start=1):
            game_plays = plays_by_game.get(game_id)
            anchor = anchors[anchors["game_id"].eq(game_id)]
            if game_plays is None or game_plays.empty:
                problems.append(f"game {game_id}: no raw plays to replay")
                continue
            if anchor.empty:
                problems.append(f"game {game_id}: no stored pregame anchor")
                continue
            events = replay_game(game_plays, anchor, params)
            problems.extend(
                stream_problems(
                    events,
                    stored[stored["game_id"].eq(game_id)].reset_index(drop=True),
                )
            )
            event_frames.append(events)
            if number % 50 == 0 or number == len(game_ids):
                print(
                    f"replayed {number}/{len(game_ids)} games "
                    f"({sum(len(f) for f in event_frames)} events, "
                    f"{perf_counter() - started:.0f}s elapsed)",
                    flush=True,
                )

        for problem in problems:
            print(f"PROBLEM: {problem}")
        if not event_frames:
            raise SystemExit(1)
        events = pd.concat(event_frames, ignore_index=True)
        latency = summarize_latency(events["latency_seconds"])
        conclusion = latency_conclusion(latency)
        equivalence = {
            "summary_type": "equivalence",
            "season": args.season,
            "model_version": MODEL_VERSION,
            "games": int(events["game_id"].nunique()),
            "events": len(events),
            "streamed_rows": int(events["emitted"].sum()),
            "stored_rows": len(stored),
            "problem_count": len(problems),
            "status": "exact_match" if not problems else "mismatch",
            "diagnostic": (
                "every streamed probability equals the stored baseline "
                "prediction row"
                if not problems
                else "streamed outputs diverge from stored baseline predictions"
            ),
        }
        summary = pd.DataFrame(
            [
                equivalence,
                {
                    "summary_type": "latency",
                    "season": args.season,
                    "model_version": MODEL_VERSION,
                    **latency,
                },
                {
                    "summary_type": "conclusion",
                    "season": args.season,
                    "model_version": MODEL_VERSION,
                    **conclusion,
                },
            ]
        )
        if args.game_id is None:
            store.write_processed(
                events, "serving", f"stream_replay_events_{args.season}.parquet"
            )
            store.write_processed(
                summary, "serving", f"stream_replay_summary_{args.season}.parquet"
            )
        print(
            f"{equivalence['status']}: {equivalence['streamed_rows']} streamed "
            f"probabilities vs {equivalence['stored_rows']} stored rows across "
            f"{equivalence['games']} games ({equivalence['events']} play events)"
        )
        print(
            f"latency per event: median {latency['median_seconds'] * 1e3:.1f} ms, "
            f"p99 {latency['p99_seconds'] * 1e3:.1f} ms, "
            f"mean {latency['mean_seconds'] * 1e3:.1f} ms, "
            f"max {latency['max_seconds'] * 1e3:.1f} ms"
        )
        print(f"conclusion: {conclusion['diagnostic']}")
        if problems:
            raise SystemExit(1)

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
