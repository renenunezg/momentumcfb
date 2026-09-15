"""Command dispatch for the backend CLI."""

from argparse import Namespace
from collections.abc import Callable

from backend.commands.grading import (
    handle_diagnose,
    handle_grade,
    handle_publish_grading,
)
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
    handle_preseason_bundle,
    handle_srs_prior,
    handle_weekly_update,
)
from backend.commands.players import (
    handle_heisman,
    handle_heisman_train,
    handle_ingest_players,
    handle_player_values,
    handle_publish_players,
    handle_qb_starters,
)
from backend.commands.publishing import (
    handle_fetch_anchors,
    handle_publish,
    handle_publish_anchors,
    handle_refresh_picks,
)
from backend.commands.serving import (
    handle_serve_game,
    handle_serve_verify,
    handle_serving_anchors,
)
from backend.commands.tier2 import (
    handle_live_pilot,
    handle_tier2_benchmark,
    handle_tier2_snapshot,
    handle_weather_evaluate,
)

Handler = Callable[[Namespace], None]

HANDLERS: dict[str, Handler] = {
    "tier2-snapshot": handle_tier2_snapshot,
    "weather-evaluate": handle_weather_evaluate,
    "tier2-benchmark": handle_tier2_benchmark,
    "cfbd-live-pilot": handle_live_pilot,
    "ingest": handle_ingest,
    "qb-starters": handle_qb_starters,
    "features": handle_features,
    "fit": handle_fit,
    "weekly-update": handle_weekly_update,
    "calibrate": handle_calibrate,
    "preseason": handle_preseason,
    "srs-prior": handle_srs_prior,
    "preseason-bundle": handle_preseason_bundle,
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
    "refresh-picks": handle_refresh_picks,
    "fetch-anchors": handle_fetch_anchors,
    "grade": handle_grade,
    "diagnose": handle_diagnose,
    "publish-grading": handle_publish_grading,
    "ingest-players": handle_ingest_players,
    "player-values": handle_player_values,
    "heisman": handle_heisman,
    "heisman-train": handle_heisman_train,
    "publish-players": handle_publish_players,
}


def run(args: Namespace) -> None:
    HANDLERS[args.command](args)
