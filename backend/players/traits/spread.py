"""Point spread rating (PSR): the points a team's margin loses if its starting
quarterback is ruled out of its next game, the exact move the forecast makes.

A quarterback's raw value is his v4 points above replacement per game through
the week, ``shrunk_per_game - replacement_per_game``, times his share: the
fraction of his team's completed games through the week he was credited in,
each game weighted as the team rating weights it (``rating_half_life_weeks``;
the served rating's half-life is infinite, so every game counts the same). A
starter who has missed most of the season is already mostly absent from the
rating and costs little more.

PSR is ``availability.qb_out_points`` of that raw value, the regular-season
quarterback-out adjustment, 0.36 + 0.72 x max(raw, 0) points, fit on how the
closing line prices realized starter absences (see ``backend.model.availability``).
The forecast reads the published value back as points, so the site's number
and the line's move are one number. Only each team's expected starter carries
a row, its quarterback with the most v4 plays through the week, the absence
the adjustment was fit on and the one a report names.

Other positions carry no PSR. On 2020-2023 the closing line moved -0.11 (SE
0.17) per point of their absent raw value, defense alone -0.27 (0.56), and the
realized excess came from 2021-2022 only and reversed on 2024-2025
(``research/spread_weight_20261119``).
"""

import numpy as np
import pandas as pd

from backend.model.availability import qb_out_points


def shares(
    played: pd.DataFrame, team_games: pd.DataFrame, week: int, half_life: float
) -> pd.DataFrame:
    """``athlete_id, share``: the recency-weighted fraction of his team's completed
    games through ``week`` that he played.

    ``played`` holds ``athlete_id, team, game_id``, one row per game a player was
    credited in; ``team_games`` is ``context.completed_team_games``. A game weighs
    ``0.5 ** (age / half_life)``, its age in weeks counted from the team's latest
    game, so an infinite half-life weighs every game the same.
    """
    games = team_games[team_games["model_week"].le(week)]
    latest = games.groupby("team")["model_week"].transform("max")
    games = games.assign(
        weight=np.power(0.5, (latest - games["model_week"]) / half_life)
    )
    totals = games.groupby("team")["weight"].sum()
    mine = played.merge(games[["team", "game_id", "weight"]], on=["team", "game_id"])
    out = mine.groupby(["athlete_id", "team"], as_index=False)["weight"].sum()
    return out.assign(share=out["weight"] / out["team"].map(totals))[
        ["athlete_id", "share"]
    ]


def spread_values(
    values: pd.DataFrame,
    played: pd.DataFrame,
    team_games: pd.DataFrame,
    half_life: float,
) -> pd.DataFrame:
    """``athlete_id, week, value``: after each v4 week, the points each team's
    margin gives up if its expected starter misses its next game.

    ``values`` are v4 rows (``week, athlete_id, team, position_group, plays,
    shrunk_per_game, replacement_per_game``); only quarterbacks are priced, and
    only the one with the most plays on his team that week. ``played`` and
    ``team_games`` are as in ``shares``.
    """
    quarterbacks = values[values["position_group"].eq("QB")]
    parts = []
    for week, rows in quarterbacks.groupby("week"):
        priced = rows.astype({"athlete_id": str}).merge(
            shares(played, team_games, int(week), half_life), on="athlete_id"
        )
        starters = priced.sort_values("plays", ascending=False).drop_duplicates("team")
        raw = (
            starters["shrunk_per_game"] - starters["replacement_per_game"]
        ) * starters["share"]
        parts.append(
            pd.DataFrame(
                {
                    "athlete_id": starters["athlete_id"],
                    "week": week,
                    "value": [qb_out_points(v, postseason=False) for v in raw],
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=["athlete_id", "week", "value"])
    return pd.concat(parts, ignore_index=True)
