"""Command dispatch for the backend CLI."""

from argparse import Namespace
from collections.abc import Callable

from backend.commands.grading import handle_grade, handle_publish_grading
from backend.commands.ingame import (
    handle_ingame_baseline,
    handle_ingame_market_anchor,
    handle_ingame_momentum,
    handle_ingame_stream,
)
from backend.commands.odds import (
    handle_kickoff_check,
    handle_kickoff_run,
    handle_live_odds,
    handle_live_replay,
)
from backend.commands.pipeline import (
    handle_calibrate,
    handle_features,
    handle_fit,
    handle_ingest,
    handle_preseason,
    handle_weekly_update,
)
from backend.commands.publishing import (
    handle_fetch_anchors,
    handle_publish,
    handle_publish_anchors,
)
from backend.commands.serving import (
    handle_serve_game,
    handle_serve_verify,
    handle_serving_anchors,
)

Handler = Callable[[Namespace], None]

HANDLERS: dict[str, Handler] = {
    "ingest": handle_ingest,
    "features": handle_features,
    "fit": handle_fit,
    "weekly-update": handle_weekly_update,
    "calibrate": handle_calibrate,
    "preseason": handle_preseason,
    "ingame-baseline": handle_ingame_baseline,
    "ingame-momentum": handle_ingame_momentum,
    "ingame-momentum-recency": handle_ingame_momentum,
    "ingame-stream": handle_ingame_stream,
    "ingame-market-anchor": handle_ingame_market_anchor,
    "serving-anchors": handle_serving_anchors,
    "serve-game": handle_serve_game,
    "serve-verify": handle_serve_verify,
    "kickoff-check": handle_kickoff_check,
    "kickoff-run": handle_kickoff_run,
    "live-odds": handle_live_odds,
    "live-replay": handle_live_replay,
    "publish": handle_publish,
    "publish-anchors": handle_publish_anchors,
    "fetch-anchors": handle_fetch_anchors,
    "grade": handle_grade,
    "publish-grading": handle_publish_grading,
}


def run(args: Namespace) -> None:
    HANDLERS[args.command](args)
