"""Chronological in-game state at every play boundary.

The state stored for play N is what was knowable pre-snap: the score and
timeout counts produced by plays 1..N-1, plus the situation of the play about
to run (period, clock, possession, field position, down, distance).
``classify_plays`` supplies the canonical chronological ordering so these
states line up with the possession pipeline.
"""

import numpy as np
import pandas as pd

from backend.features.possessions import (
    GARBAGE_MARGIN,
    GARBAGE_MARGIN_LATE,
    MISSED_KICK_TYPES,
    PUNT_TYPES,
    classify_plays,
)

REGULATION_PERIODS = 4
PERIOD_SECONDS = 900.0
FIELD_POSITION_REFERENCE_YARDS_TO_GOAL = 75.0
# Recency-weighted evidence decays by half-life measured in scrimmage plays;
# the tick counter is scrimmage plays so dead time never erodes evidence.
DECAY_HALF_LIVES = (30, 60, 120)
DECAYED_FAMILIES = (
    "epa_total",
    "successes",
    "stops_forced",
    "turnovers_forced",
    "field_position_total",
    "fourth_down_conversions",
    "missed_kicks",
)

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


def _decayed_columns() -> list[str]:
    columns = []
    for half_life in DECAY_HALF_LIVES:
        columns.append(f"evidence_plays_decayed_hl{half_life}")
        columns.append(f"scrimmage_plays_decayed_hl{half_life}")
        columns.append(f"elapsed_seconds_decayed_hl{half_life}")
        columns.extend(
            f"{side}_{family}_hl{half_life}"
            for family in DECAYED_FAMILIES
            for side in ("home", "away")
        )
    return columns


EVIDENCE_COLUMNS = [
    "game_id",
    "source_play_id",
    "play_index",
    "evidence_plays",
    "scrimmage_plays_before",
    "home_epa_total",
    "away_epa_total",
    "home_successes",
    "away_successes",
    "home_stops_forced",
    "away_stops_forced",
    "home_turnovers_forced",
    "away_turnovers_forced",
    "home_field_position_total",
    "away_field_position_total",
    "home_fourth_down_attempts",
    "away_fourth_down_attempts",
    "home_fourth_down_conversions",
    "away_fourth_down_conversions",
    "home_missed_kicks",
    "away_missed_kicks",
    *_decayed_columns(),
]


def build_process_evidence(plays: pd.DataFrame) -> pd.DataFrame:
    """Cumulative process evidence knowable pre-snap at every play boundary.

    Evidence for play N sums contributions of plays 1..N-1 only, mirroring the
    score shift in ``build_game_states``. Stop and turnover credit goes to the
    row's defense: the kicking or punting team is always listed as offense,
    including on return touchdowns.
    """
    classified = classify_plays(plays)
    offense_is_home = classified["offense"].eq(classified["home"])

    # Same lopsided-margin rule as is_garbage_time, but computable for
    # special-teams rows too so punts and kicks share the filter.
    threshold = classified["period"].map(GARBAGE_MARGIN).fillna(GARBAGE_MARGIN_LATE)
    lopsided = (
        (classified["offense_score"] - classified["defense_score"]).abs().gt(threshold)
    )

    epa = classified["epa"].where(classified["is_competitive"], 0.0).fillna(0.0)
    fourth_attempt = classified["is_competitive"] & classified["down"].eq(4)
    fourth_converted = fourth_attempt & classified["is_success"]
    turnover = classified["is_turnover"] & ~lopsided
    punt = classified["play_type"].isin(PUNT_TYPES) & ~lopsided
    missed_kick = classified["play_type"].isin(MISSED_KICK_TYPES) & ~lopsided
    stop = punt | missed_kick | (fourth_attempt & ~fourth_converted & ~turnover)

    # Possession starts reuse the possession split rule on scrimmage rows.
    scrimmage = classified[classified["is_scrimmage"]]
    by_game = scrimmage.groupby("game_id", sort=False)
    previous_offense = by_game["offense"].shift()
    new_possession = (
        previous_offense.isna()
        | scrimmage["offense"].ne(previous_offense)
        | scrimmage["defense"].ne(by_game["defense"].shift())
        | scrimmage["boundary_before"].ne(by_game["boundary_before"].shift())
        | scrimmage["source_drive_id"].ne(by_game["source_drive_id"].shift())
    )
    start_advantage = (
        (FIELD_POSITION_REFERENCE_YARDS_TO_GOAL - scrimmage["yards_to_goal"])
        .where(new_possession & ~lopsided.loc[scrimmage.index], 0.0)
        .fillna(0.0)
        .reindex(classified.index, fill_value=0.0)
    )

    def _split(values: pd.Series, credit_home: pd.Series):
        numeric = values.astype(float)
        return numeric.where(credit_home, 0.0), numeric.where(~credit_home, 0.0)

    contributions = pd.DataFrame(index=classified.index)
    contributions["evidence_plays"] = classified["is_competitive"].astype(float)
    contributions["scrimmage_plays_before"] = classified["is_scrimmage"].astype(float)
    offense_families = {
        "epa_total": epa,
        "successes": classified["is_success"],
        "field_position_total": start_advantage,
        "fourth_down_attempts": fourth_attempt,
        "fourth_down_conversions": fourth_converted,
        "missed_kicks": missed_kick,
    }
    for name, values in offense_families.items():
        home, away = _split(values, offense_is_home)
        contributions[f"home_{name}"] = home
        contributions[f"away_{name}"] = away
    for name, values in {"stops_forced": stop, "turnovers_forced": turnover}.items():
        home, away = _split(values, ~offense_is_home)
        contributions[f"home_{name}"] = home
        contributions[f"away_{name}"] = away

    game_ids = classified["game_id"]
    evidence = contributions.groupby(game_ids, sort=False).cumsum()
    evidence = evidence.groupby(game_ids, sort=False).shift().fillna(0.0)

    # Recency-weighted counterparts: contribution c_i decays by
    # lambda ** (scrimmage plays run since play i), evaluated pre-snap, via an
    # exact scaled cumulative sum so truncated-prefix replays match bit-for-bit.
    decay_source = contributions[
        [f"{side}_{family}" for family in DECAYED_FAMILIES for side in ("home", "away")]
        + ["evidence_plays", "scrimmage_plays_before"]
    ].copy()
    elapsed = classified.groupby("game_id", sort=False)[
        "regulation_seconds_elapsed"
    ].ffill()
    decay_source["elapsed_seconds"] = (
        elapsed.groupby(game_ids, sort=False).diff().clip(lower=0.0).fillna(0.0)
    )
    ticks = (
        classified["is_scrimmage"]
        .groupby(game_ids, sort=False)
        .cumsum()
        .to_numpy(float)
    )
    decayed_frames = []
    for half_life in DECAY_HALF_LIVES:
        decay = 0.5 ** (1.0 / half_life)
        scaled = decay_source.mul(
            pd.Series(np.power(decay, -ticks), index=classified.index), axis=0
        )
        sums = scaled.groupby(game_ids, sort=False).cumsum()
        shifted = sums.groupby(game_ids, sort=False).shift()
        rescale = (
            pd.Series(np.power(decay, ticks), index=classified.index)
            .groupby(game_ids, sort=False)
            .shift()
        )
        decayed = shifted.mul(rescale, axis=0).fillna(0.0)
        renamed = {
            "evidence_plays": f"evidence_plays_decayed_hl{half_life}",
            "scrimmage_plays_before": f"scrimmage_plays_decayed_hl{half_life}",
            "elapsed_seconds": f"elapsed_seconds_decayed_hl{half_life}",
        }
        decayed.columns = [
            renamed.get(column, f"{column}_hl{half_life}") for column in decayed.columns
        ]
        decayed_frames.append(decayed)

    evidence = pd.concat([evidence, *decayed_frames], axis=1)
    evidence.insert(0, "game_id", classified["game_id"])
    evidence.insert(1, "source_play_id", classified["id"].astype("string"))
    evidence.insert(2, "play_index", game_ids.groupby(game_ids).cumcount() + 1)
    return evidence[EVIDENCE_COLUMNS].reset_index(drop=True)


def build_momentum_states(plays: pd.DataFrame) -> pd.DataFrame:
    """Play states joined with process evidence for the extended leakage replay."""
    states = build_game_states(plays)
    evidence = build_process_evidence(plays)
    merged = states.merge(
        evidence,
        on=["game_id", "source_play_id", "play_index"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(states):
        raise ValueError("process evidence does not align with play states")
    return merged


def leakage_problems(
    plays: pd.DataFrame, game_ids, checkpoints: int = 4, builder=build_game_states
) -> list[str]:
    """Prove states are prefix-stable: rebuilding from only the first N plays
    must reproduce states 1..N exactly, so no state can depend on later plays.
    """
    problems = []
    for game_id in game_ids:
        game_plays = plays[plays["game_id"].eq(game_id)]
        full = builder(game_plays)
        total = len(full)
        if total == 0:
            problems.append(f"game {game_id}: no plays to verify")
            continue
        cuts = sorted(
            {
                max(1, round(total * step / checkpoints))
                for step in range(1, checkpoints)
            }
            | {1, total}
        )
        for cut in cuts:
            kept = full["source_play_id"].iloc[:cut]
            truncated = builder(
                game_plays[game_plays["id"].astype("string").isin(set(kept))]
            )
            if len(truncated) != cut or not truncated.equals(
                full.iloc[:cut].reset_index(drop=True)
            ):
                problems.append(
                    f"game {game_id}: states for plays 1..{cut} change when "
                    "later plays are removed"
                )
                break
    return problems
