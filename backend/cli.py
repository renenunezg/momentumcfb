import argparse

from backend.config import EVAL_SEASONS, SEASONS


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

    weekly = sub.add_parser(
        "weekly-update",
        help="fit, price, validate, and publish the next in-season week",
    )
    weekly.add_argument("--season", type=int, required=True)
    weekly.add_argument("--week", type=int, default=None)

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
    stream.add_argument(
        "--outcome-free",
        action="store_true",
        help="serve from outcome-free anchors: the game list and every anchor "
        "column handed to the driver come from the serving anchor loader "
        "alone, with no actual_* columns present",
    )

    anchors = sub.add_parser(
        "serving-anchors",
        help="build outcome-free serving anchors from a stored projection "
        "artifact or the raw market lines and prove the stored artifact "
        "round-trips through the anchor loader",
    )
    anchors.add_argument("--season", type=int, required=True)
    anchors.add_argument(
        "--week",
        type=int,
        default=None,
        help="projection week naming a preseason artifact (default 1); "
        "market anchors cover the whole season and take no week",
    )
    anchors.add_argument(
        "--source",
        choices=["preseason", "market"],
        default="preseason",
        help="anchor on a stored projection artifact, or flatten the raw "
        "market lines into closing-spread anchors "
        "(weekly fit projections were considered and dropped: the live "
        "model always starts after kickoff, when the closing line is known)",
    )
    anchors.add_argument(
        "--market-feed",
        choices=["cfbd", "live-odds"],
        default=None,
        help="market input feed (default: cfbd); live-odds freezes the latest "
        "stored pregame Odds API snapshot for each game",
    )

    market_eval = sub.add_parser(
        "ingame-market-anchor",
        help="rescore the frozen in-game baseline on market closing-line "
        "anchors - identical play boundaries and parameters, anchor as the "
        "only variable - and record the adopt-or-reject verdict",
    )
    market_eval.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=EVAL_SEASONS,
        help="seasons with stored market anchor artifacts",
    )

    serve = sub.add_parser(
        "serve-game",
        help="score one game from its play feed and serving anchors alone, "
        "without reading any stored prediction or outcome",
    )
    serve.add_argument("--season", type=int, required=True)
    serve.add_argument("--game-id", type=int, required=True)
    serve.add_argument(
        "--anchor-source",
        choices=["calibration", "serving"],
        default="calibration",
        help="calibration anchors cover a completed season; serving anchors "
        "come from a stored serving anchor artifact and need --week",
    )
    serve.add_argument(
        "--week",
        type=int,
        default=None,
        help="projection week naming the serving anchor artifact",
    )

    verify = sub.add_parser(
        "serve-verify",
        help="compare already-served events against the stored baseline "
        "predictions; the only step that reads them",
    )
    verify.add_argument("--season", type=int, required=True)
    verify.add_argument(
        "--game-id",
        type=int,
        nargs="+",
        default=None,
        help="verify only these served games (default: every served game)",
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

    kickoff_commands = (
        (
            "kickoff-check",
            "run the read-only, fail-closed readiness gate for the next "
            "market-covered kickoff window",
        ),
        (
            "kickoff-run",
            "poll across one kickoff window, prove the closing snapshot is "
            "pregame, and rebuild market serving anchors",
        ),
    )
    for name, description in kickoff_commands:
        kickoff = sub.add_parser(name, help=description)
        kickoff.add_argument("--season", type=int, required=True)
        kickoff.add_argument("--week", type=int, default=1)
        kickoff.add_argument(
            "--game-id",
            type=int,
            nargs="+",
            default=None,
            help="target explicit games; otherwise use the next stored "
            "market-covered kickoff cluster",
        )
        kickoff.add_argument("--cluster-minutes", type=float, default=15.0)
        kickoff.add_argument("--lead-minutes", type=float, default=5.0)
        kickoff.add_argument("--post-minutes", type=float, default=5.0)
        kickoff.add_argument("--interval-seconds", type=float, default=60.0)
        kickoff.add_argument("--min-quota", type=int, default=50)
        kickoff.add_argument(
            "--max-forecast-age-hours", type=float, default=48.0
        )
        kickoff.add_argument("--max-source-age-hours", type=float, default=48.0)
        kickoff.add_argument(
            "--max-offer-staleness-seconds", type=float, default=300.0
        )
        kickoff.add_argument("--min-providers", type=int, default=2)
        if name == "kickoff-check":
            kickoff.add_argument(
                "--max-poll-age-minutes", type=float, default=15.0
            )
        else:
            kickoff.add_argument("--lookback-hours", type=float, default=1.0)
            kickoff.add_argument("--lookahead-hours", type=float, default=2.0)
            kickoff.add_argument("--max-failures", type=int, default=3)
            kickoff.add_argument("--max-wait-hours", type=float, default=2.0)

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

    pub = sub.add_parser(
        "publish",
        help="publish serving tables to the cfb schema of the momentum "
        "Supabase project (the website's only data interface)",
    )
    pub.add_argument("--season", type=int, required=True)
    pub.add_argument("--week", type=int, default=1)
    pub.add_argument(
        "--source",
        choices=["preseason", "fit"],
        default="preseason",
        help="publish a preseason forecast artifact or an in-season fit "
        "artifact",
    )
    pub.add_argument(
        "--skip-backtest",
        action="store_true",
        help="skip the full refresh of the backtest history table",
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
        from backend.features.units import build_unit_games

        for season in args.seasons:
            plays = store.read_season_pbp(season)
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

    elif args.command == "fit":
        from backend.model.weekly import run_weekly_forecast

        result = run_weekly_forecast(args.season, args.week)
        print(result.ratings.head(30).to_string(index=False))
        print(
            f"wrote {len(result.ratings)} ratings, "
            f"{len(result.projections)} projections, and "
            f"{len(result.unit_ratings)} unit ratings for Week {result.week}"
        )
        print(f"forecast log: {result.log_directory}")

    elif args.command == "weekly-update":
        from datetime import datetime, timezone

        from backend.model.joint_scoring import MODEL_VERSION
        from backend.model.weekly import (
            WeeklyForecastNotReady,
            resolve_ready_forecast_week,
            run_weekly_forecast,
        )
        from backend.odds.client import OddsAPIClient
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
            result = run_weekly_forecast(
                args.season,
                forecast_week,
                odds_client=OddsAPIClient(),
                require_market=True,
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

    elif args.command == "calibrate":
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
            f"{len(result.unit_ratings)} unit ratings, "
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
        from backend.serving.anchors import load_serving_anchors
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
        if args.outcome_free:
            # Serving must not learn which games to score, or anything else,
            # from outcome-bearing artifacts: the driver sees only contract
            # rows from the anchor loader, on its calibration source so
            # historical replays keep the anchors they were scored with.
            try:
                anchors = load_serving_anchors(season=args.season)
            except (FileNotFoundError, ValueError) as exc:
                raise SystemExit(str(exc)) from exc
            game_ids = list(dict.fromkeys(anchors["game_id"]))
            if args.game_id is not None:
                game_ids = [gid for gid in game_ids if gid == args.game_id]
            if not game_ids:
                raise SystemExit(
                    f"no pregame anchors for season {args.season}"
                    + (
                        f" game {args.game_id}"
                        if args.game_id is not None
                        else ""
                    )
                )
        else:
            anchors = store.read_processed(
                "calibration", "joint_scoring_predictions.parquet"
            )
            game_ids = list(dict.fromkeys(stored["game_id"]))
        plays_by_game = dict(
            iter(store.read_season_pbp(args.season).groupby("game_id", sort=False))
        )
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
            "outcome_free": bool(args.outcome_free),
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
        mode = " (outcome-free)" if args.outcome_free else ""
        print(
            f"{equivalence['status']}{mode}: {equivalence['streamed_rows']} "
            f"streamed probabilities vs {equivalence['stored_rows']} stored "
            f"rows across {equivalence['games']} games "
            f"({equivalence['events']} play events)"
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

    elif args.command == "serving-anchors":
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
                raise SystemExit(
                    "market anchors cover a whole season; drop --week"
                )
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
                anchors = load_serving_anchors(
                    args.source, season=args.season, week=week
                )
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
        stored = load_serving_anchors(
            "serving", season=args.season, week=week
        )
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
            print(
                f"margin_sd {built['margin_sd'].iloc[0]:.3f} points ({sd_note})"
            )

    elif args.command == "ingame-market-anchor":
        import numpy as np
        import pandas as pd

        from backend.etl import store
        from backend.model.ingame import IngameBaselineParams, win_probability
        from backend.model.market_anchor import (
            MODEL_VERSION,
            evaluate_market_anchor,
            format_market_anchor_diagnostic,
        )
        from backend.serving.anchors import load_serving_anchors
        from backend.serving.market import MARKET_ANCHOR_WEEK

        anchor_frames = []
        for season in args.seasons:
            try:
                season_anchors = load_serving_anchors(
                    "serving", season=season, week=MARKET_ANCHOR_WEEK
                )
            except (FileNotFoundError, ValueError) as exc:
                raise SystemExit(
                    f"no stored market anchors for season {season} "
                    f"(run `python -m backend serving-anchors --season "
                    f"{season} --source market` first): {exc}"
                ) from exc
            anchor_frames.append(season_anchors)
        anchors = pd.concat(anchor_frames, ignore_index=True)

        try:
            baseline = store.read_processed(
                "ingame", "baseline_predictions.parquet"
            )
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
        params = IngameBaselineParams(
            possession_points=float(parameter["possession_points"]),
            field_position_points_per_yard=float(
                parameter["field_position_points_per_yard"]
            ),
            sd_floor_points=float(parameter["sd_floor_points"]),
        )
        if not np.allclose(
            win_probability(baseline, params),
            baseline["win_probability"].to_numpy(float),
            atol=1e-9,
        ):
            raise SystemExit(
                "stored baseline probabilities do not match the frozen "
                "parameters"
            )

        # Identical play boundaries: every stored baseline state must carry a
        # market anchor, so the anchor is the only variable in the comparison.
        anchored_games = set(anchors["game_id"])
        missing = baseline[~baseline["game_id"].isin(anchored_games)]
        if not missing.empty:
            raise SystemExit(
                f"{missing['game_id'].nunique()} baseline games in seasons "
                f"{sorted(missing['season'].unique())} have no market anchor; "
                "the comparison requires identical play boundaries"
            )
        inputs = baseline.rename(
            columns={
                "win_probability": "baseline_win_probability",
                "pregame_margin": "baseline_pregame_margin",
                "pregame_margin_sd": "baseline_pregame_margin_sd",
                "model_version": "baseline_model_version",
            }
        ).merge(
            anchors.rename(
                columns={
                    "model_week": "market_model_week",
                    "home_margin": "pregame_margin",
                    "margin_sd": "pregame_margin_sd",
                }
            ),
            on="game_id",
            how="inner",
            validate="many_to_one",
        )
        inputs["market_win_probability"] = win_probability(inputs, params)
        inputs["model_version"] = MODEL_VERSION

        anchor_note = (
            "frozen baseline parameters and play boundaries, anchor swapped "
            "to the market closing line (home_margin = -closing_spread, "
            f"margin_sd {inputs['pregame_margin_sd'].iloc[0]:.3f} constant) "
            f"across seasons {min(args.seasons)}-{max(args.seasons)}"
        )
        summary = evaluate_market_anchor(inputs, anchor_note)

        # The frozen holdout numbers must reproduce exactly from the same
        # states, or the boundaries are not identical after all.
        frozen = baseline_summary[
            baseline_summary["summary_type"].eq("evaluation")
            & baseline_summary["partition"].eq("holdout")
            & baseline_summary["scope"].eq("overall")
        ].iloc[0]
        rescored = summary[
            summary["summary_type"].eq("evaluation")
            & summary["partition"].eq("holdout")
            & summary["scope"].eq("overall")
        ].iloc[0]
        if int(rescored["n_states"]) != int(frozen["n_states"]) or not np.isclose(
            rescored["baseline_log_loss"], frozen["log_loss"], atol=1e-9
        ):
            raise SystemExit(
                "holdout baseline does not reproduce the frozen summary "
                f"({rescored['n_states']} states, log loss "
                f"{rescored['baseline_log_loss']:.6f} vs frozen "
                f"{frozen['n_states']:.0f} states, {frozen['log_loss']:.6f})"
            )

        store.write_processed(
            inputs, "ingame", "market_anchor_predictions.parquet"
        )
        store.write_processed(summary, "ingame", "market_anchor_summary.parquet")
        print(
            f"market-anchored probabilities at {len(inputs)} play boundaries "
            f"across {inputs['game_id'].nunique()} games "
            "(every baseline boundary in "
            + ", ".join(str(season) for season in args.seasons)
            + ")"
        )
        print(format_market_anchor_diagnostic(summary))

    elif args.command == "serve-game":
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
            raise SystemExit(
                "calibration anchors cover a whole season; drop --week"
            )
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

    elif args.command == "serve-verify":
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

    elif args.command == "kickoff-check":
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

    elif args.command == "kickoff-run":
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

    elif args.command == "publish":
        from backend.publish import publish

        stored = publish(
            args.season,
            args.week,
            source=args.source,
            include_backtest=not args.skip_backtest,
        )
        for table, count in stored.items():
            print(f"cfb.{table}: {count} rows stored")
