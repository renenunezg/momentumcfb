"""Per-team-game observations for descriptive CFB unit ratings.

These features are companions to the joint-scoring engine and never inputs to
it. Rush and pass channels use competitive-play PPA. Pass blocking is a
sack-cost proxy, while run blocking uses adjusted line yards. Both line
channels describe the result shared by the line, skill players, and scheme.

The current CFBD play contract does not carry reliable PPA for routine punts,
kickoffs, or field-goal attempts, so this module does not manufacture a
special-teams channel from sparse scoring-event values.
"""

import numpy as np
import pandas as pd

from backend.features.possessions import classify_plays


def _adjusted_line_yards(yards: pd.Series) -> pd.Series:
    """Limit runner credit while retaining line responsibility for losses."""
    numeric = pd.to_numeric(yards, errors="coerce")
    return pd.Series(
        np.select(
            [numeric < 0, numeric <= 4, numeric <= 10],
            [1.2 * numeric, numeric, 4.0 + 0.5 * (numeric - 4.0)],
            default=7.0,
        ),
        index=yards.index,
        dtype=float,
    ).where(numeric.notna())


def build_unit_games(plays: pd.DataFrame) -> pd.DataFrame:
    """Build one pregame-safe historical observation per game and offense.

    All values come from competitive scrimmage plays. A game can therefore be
    used for a forecast only after the caller's chronological cutoff admits it.
    """
    classified = classify_plays(plays)
    work = classified[
        classified["is_competitive"]
        & classified["offense"].notna()
        & classified["defense"].notna()
    ].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "season_type",
                "game_id",
                "team",
                "opponent",
                "rush_ppa",
                "pass_ppa",
                "protection_ppa_allowed",
                "adjusted_line_yards",
                "rush_plays",
                "pass_plays",
                "sacks_allowed",
                "line_yard_carries",
            ]
        )

    rush = work["is_rush"]
    dropback = work["is_pass"]
    sack = dropback & work["is_sack"]
    valid_rush_yards = rush & work["yards_gained"].notna()

    work["rush_ppa_value"] = work["epa"].where(rush, 0.0)
    work["pass_ppa_value"] = work["epa"].where(dropback, 0.0)
    work["protection_ppa_value"] = work["epa"].where(sack, 0.0)
    work["adjusted_line_yards_value"] = _adjusted_line_yards(
        work["yards_gained"]
    ).where(valid_rush_yards, 0.0)
    work["rush_play"] = rush.astype(int)
    work["pass_play"] = dropback.astype(int)
    work["sack_allowed"] = sack.astype(int)
    work["line_yard_carry"] = valid_rush_yards.astype(int)

    keys = [
        "season",
        "week",
        "season_type",
        "game_id",
        "offense",
        "defense",
    ]
    out = (
        work.groupby(keys, sort=True, observed=True)
        .agg(
            rush_ppa=("rush_ppa_value", "sum"),
            pass_ppa=("pass_ppa_value", "sum"),
            protection_ppa_allowed=("protection_ppa_value", "sum"),
            adjusted_line_yards=("adjusted_line_yards_value", "sum"),
            rush_plays=("rush_play", "sum"),
            pass_plays=("pass_play", "sum"),
            sacks_allowed=("sack_allowed", "sum"),
            line_yard_carries=("line_yard_carry", "sum"),
        )
        .reset_index()
        .rename(columns={"offense": "team", "defense": "opponent"})
    )
    return out.sort_values(
        ["season", "week", "game_id", "team"],
        kind="stable",
        ignore_index=True,
    )
