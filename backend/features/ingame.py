"""Chronological in-game state at every play boundary.

The state stored for play N is what was knowable pre-snap: the score and
timeout counts produced by plays 1..N-1, plus the situation of the play about
to run (period, clock, possession, field position, down, distance).
``classify_plays`` supplies the canonical chronological ordering so these
states line up with the possession pipeline.
"""

import numpy as np
import pandas as pd

from backend.features.possessions import classify_plays

REGULATION_PERIODS = 4
PERIOD_SECONDS = 900.0

STATE_COLUMNS = [
    "season",
    "week",
    "season_type",
    "game_id",
    "home_team",
    "away_team",
    "play_index",
    "source_play_id",
    "drive_number",
    "play_number",
    "play_category",
    "period",
    "clock_seconds",
    "seconds_remaining",
    "is_overtime",
    "wallclock",
    "offense",
    "defense",
    "offense_is_home",
    "down",
    "distance",
    "yards_to_goal",
    "home_timeouts",
    "away_timeouts",
    "home_score",
    "away_score",
    "home_margin",
]


def build_game_states(plays: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the pre-snap state at every play boundary."""
    classified = classify_plays(plays)
    offense_is_home = classified["offense"].eq(classified["home"])

    def _numeric(column: str) -> pd.Series:
        if column in classified:
            return pd.to_numeric(classified[column], errors="coerce").astype(float)
        return pd.Series(np.nan, index=classified.index, dtype=float)

    def _home_away(offense_column: str, defense_column: str):
        offense_values = _numeric(offense_column)
        defense_values = _numeric(defense_column)
        home = offense_values.where(offense_is_home, defense_values)
        away = defense_values.where(offense_is_home, offense_values)
        return home, away

    home_score_post, away_score_post = _home_away("offense_score", "defense_score")
    home_timeouts_row, away_timeouts_row = _home_away(
        "offense_timeouts", "defense_timeouts"
    )

    games = classified.groupby("game_id", sort=False)
    home_score = home_score_post.groupby(classified["game_id"]).shift().fillna(0.0)
    away_score = away_score_post.groupby(classified["game_id"]).shift().fillna(0.0)

    # The provider decrements the charged team's count on the Timeout row
    # itself, so only those rows need the prior row's value to stay pre-snap.
    is_timeout_row = classified["play_type"].eq("Timeout")
    home_timeouts = home_timeouts_row.where(
        ~is_timeout_row, home_timeouts_row.groupby(classified["game_id"]).shift()
    )
    away_timeouts = away_timeouts_row.where(
        ~is_timeout_row, away_timeouts_row.groupby(classified["game_id"]).shift()
    )

    in_regulation = classified["period"].between(1, REGULATION_PERIODS)
    seconds_remaining = pd.Series(
        np.where(
            in_regulation,
            (REGULATION_PERIODS - classified["period"]) * PERIOD_SECONDS
            + classified["clock_seconds_remaining"],
            0.0,
        ),
        index=classified.index,
    )

    states = pd.DataFrame(
        {
            "season": classified["season"],
            "week": classified["week"],
            "season_type": classified["season_type"],
            "game_id": classified["game_id"],
            "home_team": classified["home"],
            "away_team": classified["away"],
            "play_index": games.cumcount() + 1,
            "source_play_id": classified["id"].astype("string"),
            "drive_number": classified["drive_number"],
            "play_number": classified["play_number"],
            "play_category": classified["play_category"],
            "period": classified["period"],
            "clock_seconds": classified["clock_seconds_remaining"],
            "seconds_remaining": seconds_remaining,
            "is_overtime": classified["period"].gt(REGULATION_PERIODS),
            "wallclock": classified.get("wallclock"),
            "offense": classified["offense"],
            "defense": classified["defense"],
            "offense_is_home": offense_is_home,
            "down": classified["down"],
            "distance": classified["distance"],
            "yards_to_goal": classified["yards_to_goal"],
            "home_timeouts": home_timeouts,
            "away_timeouts": away_timeouts,
            "home_score": home_score,
            "away_score": away_score,
        }
    )
    states["home_margin"] = states["home_score"] - states["away_score"]
    return states[STATE_COLUMNS].reset_index(drop=True)


def leakage_problems(
    plays: pd.DataFrame, game_ids, checkpoints: int = 4
) -> list[str]:
    """Prove states are prefix-stable: rebuilding from only the first N plays
    must reproduce states 1..N exactly, so no state can depend on later plays.
    """
    problems = []
    for game_id in game_ids:
        game_plays = plays[plays["game_id"].eq(game_id)]
        full = build_game_states(game_plays)
        total = len(full)
        if total == 0:
            problems.append(f"game {game_id}: no plays to verify")
            continue
        cuts = sorted(
            {max(1, round(total * step / checkpoints)) for step in range(1, checkpoints)}
            | {1, total}
        )
        for cut in cuts:
            kept = full["source_play_id"].iloc[:cut]
            truncated = build_game_states(
                game_plays[game_plays["id"].astype("string").isin(set(kept))]
            )
            if len(truncated) != cut or not truncated.equals(full.iloc[:cut].reset_index(drop=True)):
                problems.append(
                    f"game {game_id}: states for plays 1..{cut} change when "
                    "later plays are removed"
                )
                break
    return problems
