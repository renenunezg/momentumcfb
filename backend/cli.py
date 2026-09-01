import argparse
import logging
import sys

from backend.config import EVAL_SEASONS, SEASONS


def _seasons_argument(parser: argparse.ArgumentParser, default: list[int]) -> None:
    parser.add_argument("--seasons", type=int, nargs="+", default=default)


def _leakage_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--leakage-games-per-season",
        type=int,
        default=2,
        help="games per season replayed from truncated play prefixes",
    )


def _add_pipeline_commands(sub) -> None:
    ingest = sub.add_parser("ingest", help="pull CFBD data into raw parquet")
    _seasons_argument(ingest, SEASONS)
    ingest.add_argument("--week", type=int, default=None)

    features = sub.add_parser(
        "features", help="build possession and team-game features"
    )
    _seasons_argument(features, SEASONS)

    fit = sub.add_parser(
        "fit", help="fit one in-season week locally without publishing"
    )
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
        help="also buy a two-market Odds API snapshot instead of CFBD lines",
    )


def _add_research_commands(sub) -> None:
    """Frozen in-game experiments; their verdicts are recorded in the README."""
    baseline = sub.add_parser(
        "ingame-baseline",
        help="rebuild play-boundary states and evaluate the in-game baseline",
    )
    _seasons_argument(baseline, SEASONS)
    _leakage_argument(baseline)

    for name, description in (
        ("ingame-momentum", "cumulative-evidence momentum over the frozen baseline"),
        ("ingame-momentum-recency", "recency-weighted momentum over the baseline"),
    ):
        momentum = sub.add_parser(name, help=description)
        _seasons_argument(momentum, SEASONS)
        _leakage_argument(momentum)

    market_eval = sub.add_parser(
        "ingame-market-anchor",
        help="rescore the frozen baseline with the closing line as the anchor",
    )
    _seasons_argument(market_eval, EVAL_SEASONS)


def _add_serving_commands(sub) -> None:
    stream = sub.add_parser(
        "ingame-stream",
        help="replay stored plays as a live feed and prove streamed "
        "probabilities equal the stored batch outputs",
    )
    stream.add_argument("--season", type=int, required=True)
    stream.add_argument(
        "--game-id",
        type=int,
        default=None,
        help="replay one game for diagnosis without writing artifacts",
    )
    stream.add_argument(
        "--outcome-free",
        action="store_true",
        help="drive the replay from the serving anchor loader alone",
    )

    anchors = sub.add_parser(
        "serving-anchors",
        help="build outcome-free serving anchors and prove they round-trip",
    )
    anchors.add_argument("--season", type=int, required=True)
    anchors.add_argument(
        "--week",
        type=int,
        default=None,
        help="projection week of a preseason artifact (default 1); "
        "market anchors cover the whole season",
    )
    anchors.add_argument(
        "--source",
        choices=["preseason", "market"],
        default="preseason",
        help="anchor on a stored projection artifact or on closing spreads",
    )
    anchors.add_argument(
        "--market-feed",
        choices=["cfbd", "live-odds"],
        default=None,
        help="closing-spread feed (default cfbd); live-odds freezes the latest "
        "stored pregame Odds API snapshot per game",
    )

    serve = sub.add_parser(
        "serve-game",
        help="score one game from its play feed and serving anchors alone",
    )
    serve.add_argument("--season", type=int, required=True)
    serve.add_argument("--game-id", type=int, required=True)
    serve.add_argument(
        "--anchor-source",
        choices=["calibration", "serving"],
        default="calibration",
        help="calibration anchors cover a completed season; serving anchors "
        "come from a stored artifact and need --week",
    )
    serve.add_argument("--week", type=int, default=None)

    verify = sub.add_parser(
        "serve-verify",
        help="compare served events against the stored baseline predictions",
    )
    verify.add_argument("--season", type=int, required=True)
    verify.add_argument(
        "--game-id",
        type=int,
        nargs="+",
        default=None,
        help="verify only these served games (default: every served game)",
    )


def _add_market_commands(sub) -> None:
    live = sub.add_parser(
        "live-odds", help="capture append-only live sportsbook line snapshots"
    )
    live.add_argument("--season", type=int, required=True)
    live.add_argument("--polls", type=int, default=1, help="maximum poll cycles")
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

    for name, description in (
        ("kickoff-check", "read-only readiness gate for the next kickoff window"),
        (
            "kickoff-run",
            "poll one kickoff window, prove the closing snapshot is pregame, "
            "and rebuild market serving anchors",
        ),
    ):
        kickoff = sub.add_parser(name, help=description)
        kickoff.add_argument("--season", type=int, required=True)
        kickoff.add_argument("--week", type=int, default=1)
        kickoff.add_argument(
            "--game-id",
            type=int,
            nargs="+",
            default=None,
            help="target explicit games instead of the next kickoff cluster",
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
        help="ISO timestamp; show the market view available at that time",
    )


def _add_publish_commands(sub) -> None:
    publish = sub.add_parser(
        "publish", help="publish serving tables to the cfb Supabase schema"
    )
    publish.add_argument("--season", type=int, required=True)
    publish.add_argument("--week", type=int, default=1)
    publish.add_argument(
        "--source",
        choices=["preseason", "fit"],
        default="preseason",
        help="publish a preseason forecast artifact or an in-season fit artifact",
    )
    publish.add_argument(
        "--skip-backtest",
        action="store_true",
        help="skip the full refresh of the backtest history table",
    )

    for name, description in (
        ("publish-anchors", "publish a serving anchor artifact to cfb.serving_anchors"),
        ("fetch-anchors", "hydrate a local serving anchor artifact from the database"),
    ):
        anchor_sync = sub.add_parser(name, help=description)
        anchor_sync.add_argument("--season", type=int, required=True)
        anchor_sync.add_argument(
            "--week",
            type=int,
            required=True,
            help="artifact week: 0 is the frozen market capture, >= 1 a "
            "projection artifact",
        )

    grade = sub.add_parser(
        "grade",
        help="grade completed games against the frozen published projection "
        "and the CFBD closing line",
    )
    grade.add_argument("--season", type=int, required=True)
    grade.add_argument(
        "--regrade",
        action="store_true",
        help="rebuild every graded row instead of keeping stored grades",
    )

    publish_grading = sub.add_parser(
        "publish-grading",
        help="publish graded games and performance metrics to the cfb schema",
    )
    publish_grading.add_argument("--season", type=int, required=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="backend")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_pipeline_commands(sub)
    _add_research_commands(sub)
    _add_serving_commands(sub)
    _add_market_commands(sub)
    _add_publish_commands(sub)
    return parser.parse_args(argv)


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True
    )
    from backend.commands import run

    run(parse_args(argv))
