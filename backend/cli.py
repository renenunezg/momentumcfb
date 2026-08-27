import argparse

from backend.config import EVAL_SEASONS, SEASONS


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="backend")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="pull CFBD data into raw parquet")
    ing.add_argument("--seasons", type=int, nargs="+", default=SEASONS)
    ing.add_argument("--week", type=int, default=None)

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
    preseason.add_argument(
        "--with-odds-api",
        action="store_true",
        help="also buy a two-market Odds API snapshot; routine refreshes use "
        "the CFBD lines feed",
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
    live.add_argument("--interval-seconds", type=float, default=120.0)
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
        kickoff.add_argument("--interval-seconds", type=float, default=120.0)
        kickoff.add_argument("--min-quota", type=int, default=50)
        kickoff.add_argument("--max-forecast-age-hours", type=float, default=48.0)
        kickoff.add_argument("--max-source-age-hours", type=float, default=48.0)
        kickoff.add_argument("--max-offer-staleness-seconds", type=float, default=300.0)
        kickoff.add_argument("--min-providers", type=int, default=2)
        if name == "kickoff-check":
            kickoff.add_argument("--max-poll-age-minutes", type=float, default=15.0)
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
        help="publish a preseason forecast artifact or an in-season fit artifact",
    )
    pub.add_argument(
        "--skip-backtest",
        action="store_true",
        help="skip the full refresh of the backtest history table",
    )

    anchor_sync_commands = (
        (
            "publish-anchors",
            "publish a stored serving anchor artifact to cfb.serving_anchors "
            "so an ephemeral capture runner's output survives the job",
        ),
        (
            "fetch-anchors",
            "hydrate the local serving anchor artifact from "
            "cfb.serving_anchors and prove it round-trips through the loader",
        ),
    )
    for name, description in anchor_sync_commands:
        anchor_sync = sub.add_parser(name, help=description)
        anchor_sync.add_argument("--season", type=int, required=True)
        anchor_sync.add_argument(
            "--week",
            type=int,
            required=True,
            help="artifact week: 0 is the frozen market capture, >= 1 a "
            "projection artifact",
        )

    return p.parse_args(argv)


def main(argv=None):
    from backend.commands import run

    run(parse_args(argv))
