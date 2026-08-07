"""Streaming serving harness for the frozen in-game baseline.

Replays a stored game as a simulated live feed: plays arrive one at a time in
chronological order, and after each event the play-boundary state is rebuilt
from only the plays seen so far, then scored with the frozen baseline
parameters and the game's stored pregame anchor. This rebuild-per-play loop is
the reference serving path; ``stream_problems`` proves each streamed
probability sequence equals the stored batch predictions exactly.
"""

from time import perf_counter

import numpy as np
import pandas as pd

from backend.features.ingame import build_game_states
from backend.model.ingame import (
    IngameBaselineParams,
    build_baseline_inputs,
    win_probability,
)

# Live plays arrive tens of seconds apart, so the streamed probability must be
# out well before the next snap. A p99 within this budget means the
# rebuild-per-play reference can serve live traffic without an incremental
# state builder.
LIVE_LATENCY_BUDGET_SECONDS = 1.0

EVENT_COLUMNS = [
    "game_id",
    "play_index",
    "source_play_id",
    "emitted",
    "win_probability",
    "latency_seconds",
]


def chronological_plays(game_plays: pd.DataFrame) -> pd.DataFrame:
    """Order one game's raw plays into the canonical arrival sequence.

    The harness may read the whole stored game to learn the arrival order; the
    model input at event N is strictly the first N plays of that sequence.
    """
    raw = game_plays.reset_index(drop=True)
    order = build_game_states(raw)["source_play_id"]
    if not order.is_unique:
        raise ValueError("duplicate source play ids; cannot form a play feed")
    position = raw["id"].astype("string").map(
        {play_id: index for index, play_id in enumerate(order)}
    )
    return raw.iloc[
        np.argsort(position.to_numpy(), kind="stable")
    ].reset_index(drop=True)


def replay_game(
    game_plays: pd.DataFrame,
    anchor: pd.DataFrame,
    params: IngameBaselineParams,
) -> pd.DataFrame:
    """Stream one game and return a row per play event.

    Each event's latency covers exactly the serving work: rebuild the
    play-boundary state from the prefix, join the pregame anchor, and produce
    the baseline win probability. Events the batch pipeline filters out (no
    anchor row survives the join) are recorded with ``emitted`` False.
    """
    feed = chronological_plays(game_plays)
    game_id = feed["game_id"].iloc[0]
    rows = []
    for count in range(1, len(feed) + 1):
        prefix = feed.iloc[:count]
        started = perf_counter()
        boundary = build_game_states(prefix).iloc[[-1]]
        inputs = build_baseline_inputs(boundary, anchor)
        emitted = bool(len(inputs))
        probability = (
            float(win_probability(inputs, params)[0]) if emitted else np.nan
        )
        latency = perf_counter() - started
        rows.append(
            {
                "game_id": game_id,
                "play_index": count,
                "source_play_id": boundary["source_play_id"].iloc[0],
                "emitted": emitted,
                "win_probability": probability,
                "latency_seconds": latency,
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def stream_problems(events: pd.DataFrame, stored: pd.DataFrame) -> list[str]:
    """Prove one game's streamed probabilities equal the stored batch rows."""
    game_id = events["game_id"].iloc[0]
    streamed = events[events["emitted"]].reset_index(drop=True)
    if len(streamed) != len(stored):
        return [
            f"game {game_id}: streamed {len(streamed)} probabilities but the "
            f"batch stored {len(stored)}"
        ]
    if not np.array_equal(
        streamed["play_index"].to_numpy(int), stored["play_index"].to_numpy(int)
    ) or streamed["source_play_id"].astype(str).tolist() != stored[
        "source_play_id"
    ].astype(str).tolist():
        return [
            f"game {game_id}: streamed events do not line up with the stored "
            "play boundaries"
        ]
    streamed_values = streamed["win_probability"].to_numpy(float)
    stored_values = stored["win_probability"].to_numpy(float)
    # NaN != NaN, so a NaN on either side always counts as a mismatch: a NaN
    # probability must be flagged, never tolerated as "equal".
    mismatch = np.flatnonzero(streamed_values != stored_values)
    if mismatch.size:
        first = int(mismatch[0])
        return [
            f"game {game_id}: {mismatch.size} of {len(stored)} probabilities "
            f"differ; first at play_index {int(stored['play_index'].iloc[first])} "
            f"(streamed {streamed_values[first]!r}, "
            f"stored {stored_values[first]!r})"
        ]
    return []


def summarize_latency(latency_seconds) -> dict[str, float | int]:
    values = np.asarray(latency_seconds, dtype=float)
    return {
        "events": int(values.size),
        "median_seconds": float(np.median(values)),
        "p99_seconds": float(np.percentile(values, 99)),
        "mean_seconds": float(values.mean()),
        "max_seconds": float(values.max()),
    }


def latency_conclusion(latency: dict[str, float | int]) -> dict[str, object]:
    """The explicit fast-enough-or-not verdict for the rebuild-per-play path."""
    fast_enough = latency["p99_seconds"] <= LIVE_LATENCY_BUDGET_SECONDS
    scale = (
        f"median {latency['median_seconds'] * 1e3:.1f} ms, "
        f"p99 {latency['p99_seconds'] * 1e3:.1f} ms, "
        f"max {latency['max_seconds'] * 1e3:.1f} ms per event against the "
        f"{LIVE_LATENCY_BUDGET_SECONDS:.1f} s live budget"
    )
    if fast_enough:
        diagnostic = (
            f"rebuild-per-play is fast enough for live use ({scale}); "
            "no incremental state builder is required"
        )
    else:
        diagnostic = (
            f"rebuild-per-play misses the live budget ({scale}); "
            "an incremental state builder is required"
        )
    return {
        "fast_enough": fast_enough,
        "budget_seconds": LIVE_LATENCY_BUDGET_SECONDS,
        "diagnostic": diagnostic,
    }
