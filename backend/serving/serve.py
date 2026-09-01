"""Score one game from a play feed and serving anchors alone.

Everything read here is available before a game ends: the pregame anchor via
``load_serving_anchors``, the frozen baseline parameters, and the plays so
far. Stored baseline predictions are never opened; proving served output
matches a batch run lives in ``backend.serving.verify``. A game with no stored
plays raises ``MissingPlayFeed``, the same condition a live feed reports
before kickoff.
"""

import pandas as pd

from backend.etl import store
from backend.model.ingame import MODEL_VERSION, IngameBaselineParams
from backend.serving.anchors import load_serving_anchors
from backend.serving.replay import EVENT_COLUMNS, replay_game

BASELINE_SUMMARY_ARTIFACT = ("ingame", "baseline_summary.parquet")

# The frozen baseline is defined by these fitted constants alone; reading the
# summary column-restricted keeps every evaluation column out of the serving
# path even though they sit in the same artifact.
PARAMETER_COLUMNS = [
    "summary_type",
    "model_version",
    "possession_points",
    "field_position_points_per_yard",
    "sd_floor_points",
]

SERVED_EVENT_COLUMNS = [
    "season",
    *EVENT_COLUMNS,
    "model_week",
    "anchor_source",
    "model_version",
]


class MissingPlayFeed(LookupError):
    """No plays exist yet for a game the serving anchors already cover."""


def served_events_artifact(season: int, game_id: int) -> tuple[str, ...]:
    """Processed-store location of one game's served events."""
    return ("serving", "served", str(season), f"{game_id}.parquet")


def load_frozen_params() -> IngameBaselineParams:
    """Read the frozen baseline parameters, and nothing else, from storage."""
    location = "/".join(BASELINE_SUMMARY_ARTIFACT)
    try:
        summary = store.read_processed(
            *BASELINE_SUMMARY_ARTIFACT, columns=PARAMETER_COLUMNS
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"no frozen baseline parameters at {location}; run "
            "`python -m backend ingame-baseline` first"
        ) from exc
    parameters = summary[summary["summary_type"].eq("parameter")]
    if parameters.empty:
        raise ValueError(f"{location} carries no fitted parameter row")
    parameter = parameters.iloc[0]
    if parameter["model_version"] != MODEL_VERSION:
        raise ValueError(
            f"{location} holds parameters for {parameter['model_version']!r}; "
            f"the serving path runs {MODEL_VERSION!r}"
        )
    return IngameBaselineParams(
        possession_points=float(parameter["possession_points"]),
        field_position_points_per_yard=float(
            parameter["field_position_points_per_yard"]
        ),
        sd_floor_points=float(parameter["sd_floor_points"]),
    )


def load_play_feed(season: int, game_id: int) -> pd.DataFrame:
    """The game's play feed, standing in for a live low-latency feed."""
    try:
        plays = store.read_season_pbp(season)
    except FileNotFoundError as exc:
        raise MissingPlayFeed(
            f"no play feed for game {game_id}: season {season} has no stored plays yet"
        ) from exc
    feed = plays[plays["game_id"].eq(game_id)]
    if feed.empty:
        raise MissingPlayFeed(
            f"no play feed for game {game_id}: season {season} has stored "
            "plays, but none for this game yet"
        )
    return feed.reset_index(drop=True)


def serve_game(
    season: int,
    game_id: int,
    *,
    source: str,
    week: int | None = None,
) -> pd.DataFrame:
    """Serve one game's win probability sequence from anchors and plays.

    Anchors resolve first so a game that is anchored but not yet played fails
    on the feed, which is the ordinary pregame state, rather than on a
    misleading anchor error.
    """
    anchors = load_serving_anchors(source, season=season, week=week)
    anchor = anchors[anchors["game_id"].eq(game_id)]
    if anchor.empty:
        where = f"season {season}" + (f" week {week}" if week is not None else "")
        raise ValueError(f"no {source} serving anchor for game {game_id} in {where}")
    params = load_frozen_params()
    feed = load_play_feed(season, game_id)
    events = replay_game(feed, anchor, params)
    events.insert(0, "season", season)
    events["model_week"] = int(anchor["model_week"].iloc[0])
    events["anchor_source"] = source
    events["model_version"] = MODEL_VERSION
    return events[SERVED_EVENT_COLUMNS]
