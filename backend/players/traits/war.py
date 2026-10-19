"""Wins above replacement: player value v4's points turned into wins.

v4 already measures a player's points above positional replacement:
``value_above_replacement = games × (shrunk_per_game − replacement_per_game)``.
WAR converts those points at the league-average rate, the way baseball WAR
converts runs at a league runs-per-win: ``wins = points / POINTS_PER_WIN``.

The rate is context neutral on purpose. Weighting each game by its own pregame
leverage (the slope of win probability per point at the projected margin)
credited a player for his schedule rather than his play: a quarterback on a
heavy favorite could earn a sixth of the wins per point an underdog's did, and on
2020-2025 the leveraged WAR ranked a player's later same-season production no
better than the points themselves. Nothing is fitted.
"""

import pandas as pd

# 1 / mean pregame leverage φ(margin/sd)/sd over the 3,467 FBS games of the
# 2020-2024 calibration replays; the per-season values run from 48.9 to 51.4,
# and 2025 (out of sample) gives 49.5.
POINTS_PER_WIN = 49.98

WAR_COLUMNS = [
    "athlete_id",
    "week",
    "points_above_replacement",
    "wins_above_replacement",
]


def wins_above_replacement(season_values: pd.DataFrame) -> pd.DataFrame:
    """``WAR_COLUMNS`` for every v4 player-week of ``season_values`` (v4 ``week,
    athlete_id, value_above_replacement``)."""
    points = season_values["value_above_replacement"].astype(float)
    return season_values.astype({"athlete_id": str}).assign(
        points_above_replacement=points,
        wins_above_replacement=points / POINTS_PER_WIN,
    )[WAR_COLUMNS]
