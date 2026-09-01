"""In-game model and evaluation command handlers."""

import logging
from argparse import Namespace

log = logging.getLogger(__name__)


def handle_ingame_baseline(args: Namespace) -> None:
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
        log.info(
            f"{season}: {len(states)} play states across "
            f"{states['game_id'].nunique()} games "
            f"(leakage-checked {len(sample)})"
        )
    for problem in problems:
        log.info(f"PROBLEM: {problem}")
    if problems:
        raise SystemExit(1)
    log.info("leakage check passed: states are prefix-stable")

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
    log.info(
        f"anchored {inputs['game_id'].nunique()} of "
        f"{states['game_id'].nunique()} games with pregame projections; "
        f"wrote {len(inputs)} baseline predictions"
    )
    log.info(format_ingame_diagnostic(summary))


def handle_ingame_momentum(args: Namespace) -> None:
    import pandas as pd

    from backend.etl import store
    from backend.features.ingame import (
        build_momentum_states,
        build_process_evidence,
        leakage_problems,
    )
    from backend.model.calibration import DEVELOPMENT_SEASONS
    from backend.model.ingame import check_frozen_probabilities, load_baseline_params
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
        problems.extend(leakage_problems(plays, sample, builder=build_momentum_states))
        log.info(
            f"{season}: evidence at {len(evidence)} play boundaries across "
            f"{evidence['game_id'].nunique()} games "
            f"(leakage-checked {len(sample)})"
        )
    for problem in problems:
        log.info(f"PROBLEM: {problem}")
    if problems:
        raise SystemExit(1)
    log.info(
        "extended leakage check passed: states and process evidence are prefix-stable"
    )

    try:
        baseline = store.read_processed("ingame", "baseline_predictions.parquet")
        baseline_summary = store.read_processed("ingame", "baseline_summary.parquet")
    except FileNotFoundError as exc:
        raise SystemExit(
            "missing baseline predictions; run "
            "`python -m backend ingame-baseline` first"
        ) from exc
    baseline = baseline[baseline["season"].isin(args.seasons)]
    baseline_params = load_baseline_params(baseline_summary)

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
    try:
        check_frozen_probabilities(inputs, baseline_params)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    inputs = inputs.rename(
        columns={
            "win_probability": "baseline_win_probability",
            "model_version": "baseline_model_version",
        }
    )

    development = inputs[inputs["season"].isin(DEVELOPMENT_SEASONS)]
    if recency:
        params = fit_momentum_recency(development, baseline_params, progress=log.info)
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
    log.info(
        f"momentum probabilities at {len(inputs)} play boundaries across "
        f"{inputs['game_id'].nunique()} games (every baseline boundary)"
    )
    log.info(format_momentum_diagnostic(summary))


def handle_ingame_stream(args: Namespace) -> None:
    from time import perf_counter

    import pandas as pd

    from backend.etl import store
    from backend.model.ingame import MODEL_VERSION, load_baseline_params
    from backend.serving.anchors import load_serving_anchors
    from backend.serving.replay import (
        latency_conclusion,
        replay_game,
        stream_problems,
        summarize_latency,
    )

    try:
        stored = store.read_processed("ingame", "baseline_predictions.parquet")
        baseline_summary = store.read_processed("ingame", "baseline_summary.parquet")
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
    params = load_baseline_params(baseline_summary)
    if args.outcome_free:
        # The driver sees only anchor-loader rows, never outcome-bearing
        # artifacts, so serving cannot learn which games to score from them.
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
                + (f" game {args.game_id}" if args.game_id is not None else "")
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
            log.info(
                f"replayed {number}/{len(game_ids)} games "
                f"({sum(len(f) for f in event_frames)} events, "
                f"{perf_counter() - started:.0f}s elapsed)",
            )

    for problem in problems:
        log.info(f"PROBLEM: {problem}")
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
            "every streamed probability equals the stored baseline prediction row"
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
    log.info(
        f"{equivalence['status']}{mode}: {equivalence['streamed_rows']} "
        f"streamed probabilities vs {equivalence['stored_rows']} stored "
        f"rows across {equivalence['games']} games "
        f"({equivalence['events']} play events)"
    )
    log.info(
        f"latency per event: median {latency['median_seconds'] * 1e3:.1f} ms, "
        f"p99 {latency['p99_seconds'] * 1e3:.1f} ms, "
        f"mean {latency['mean_seconds'] * 1e3:.1f} ms, "
        f"max {latency['max_seconds'] * 1e3:.1f} ms"
    )
    log.info(f"conclusion: {conclusion['diagnostic']}")
    if problems:
        raise SystemExit(1)


def handle_ingame_market_anchor(args: Namespace) -> None:
    import numpy as np
    import pandas as pd

    from backend.etl import store
    from backend.model.ingame import (
        check_frozen_probabilities,
        load_baseline_params,
        win_probability,
    )
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
        baseline = store.read_processed("ingame", "baseline_predictions.parquet")
        baseline_summary = store.read_processed("ingame", "baseline_summary.parquet")
    except FileNotFoundError as exc:
        raise SystemExit(
            "missing baseline predictions; run "
            "`python -m backend ingame-baseline` first"
        ) from exc
    baseline = baseline[baseline["season"].isin(args.seasons)]
    params = load_baseline_params(baseline_summary)
    try:
        check_frozen_probabilities(baseline, params)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Every baseline state needs a market anchor so the anchor is the only
    # variable in the comparison.
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

    # The frozen holdout numbers must reproduce exactly from the same states.
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

    store.write_processed(inputs, "ingame", "market_anchor_predictions.parquet")
    store.write_processed(summary, "ingame", "market_anchor_summary.parquet")
    log.info(
        f"market-anchored probabilities at {len(inputs)} play boundaries "
        f"across {inputs['game_id'].nunique()} games "
        "(every baseline boundary in "
        + ", ".join(str(season) for season in args.seasons)
        + ")"
    )
    log.info(format_market_anchor_diagnostic(summary))
