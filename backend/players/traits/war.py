"""Wins above replacement: player value v4's points turned into wins.

v4 already measures a player's points above positional replacement:
``value_above_replacement = games × (shrunk_per_game − replacement_per_game)``.
A point is worth more wins in a close game than in a rout, so each game the
player appeared in is weighted by the win-probability slope per point at the
team model's pregame projection:

    leverage_g = φ(home_margin_g / margin_sd_g) / margin_sd_g

(the normal density of the projected margin over its sd, divided by the sd),
and ``wins_above_replacement = (shrunk_per_game − replacement_per_game) ×
Σ leverage_g`` over the player's credited games through the week. A game
without a projection takes the season's mean leverage. Nothing is fitted.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from backend.serving.anchors import load_serving_anchors

log = logging.getLogger(__name__)

WAR_COLUMNS = [
    "athlete_id",
    "week",
    "points_above_replacement",
    "wins_above_replacement",
    "leverage_games",
]


def game_leverage(projections: pd.DataFrame) -> pd.Series:
    """``φ(home_margin / margin_sd) / margin_sd`` per game id; NaN where the sd is
    not positive or either number is missing."""
    rows = projections.drop_duplicates("game_id")
    sd = rows["margin_sd"].astype(float)
    sd = sd.where(sd.gt(0))
    leverage = norm.pdf(rows["home_margin"].astype(float) / sd) / sd
    return pd.Series(leverage.to_numpy(float), index=rows["game_id"].astype("int64"))


def season_projections(season: int, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """``game_id, home_margin, margin_sd`` for ``season``.

    The calibration replay is used when it covers the season (the development
    and validation seasons, built locally). Otherwise the projection published
    for each game's week is used, the same source player value v4 reads for win
    probability, which a fresh weekly runner has (the frozen published
    projections, then the served anchors). An empty frame means no leverage
    and no WAR rows.
    """
    try:
        return load_serving_anchors(season=season)[["game_id", "home_margin", "margin_sd"]]
    except (ValueError, FileNotFoundError, OSError):
        pass
    from backend.model.weekly import load_weekly_games
    from backend.players.value import _projection_anchors

    try:
        anchors = _projection_anchors(season, games if games is not None else load_weekly_games(season))
    except FileNotFoundError:
        log.warning("wins above replacement %s: no pregame projections on disk", season)
        return pd.DataFrame(columns=["game_id", "home_margin", "margin_sd"])
    log.info(
        "wins above replacement %s: no calibration replay; published projections for %d games",
        season,
        len(anchors),
    )
    return anchors[["game_id", "home_margin", "margin_sd"]].drop_duplicates("game_id")


def wins_above_replacement(
    season_values: pd.DataFrame, credited_games: pd.DataFrame, projections: pd.DataFrame
) -> pd.DataFrame:
    """``WAR_COLUMNS`` for every v4 player-week.

    ``season_values`` holds v4 ``week, athlete_id, shrunk_per_game,
    replacement_per_game, value_above_replacement``; ``credited_games`` one row per
    player and credited game with its ``model_week``; ``projections`` ``game_id,
    home_margin, margin_sd``. ``leverage_games`` is the summed leverage of the
    player's credited games with model week at or before the snapshot week, the same
    games v4 counts at that week.
    """
    leverage = game_leverage(projections)
    mean = float(leverage.mean()) if leverage.notna().any() else np.nan
    games = (
        credited_games[["athlete_id", "game_id", "model_week"]]
        .astype({"athlete_id": str})
        .dropna(subset=["game_id"])
        .drop_duplicates(["athlete_id", "game_id"])
    )
    games["leverage"] = games["game_id"].astype("int64").map(leverage).fillna(mean)
    values = season_values.astype({"athlete_id": str})
    weeks = sorted(set(games["model_week"].astype(int)) | set(values["week"].astype(int)))
    cumulative = (
        games.groupby(["athlete_id", games["model_week"].astype(int)])["leverage"]
        .sum()
        .unstack()
        .reindex(columns=weeks)
        .fillna(0.0)
        .cumsum(axis=1)
    )
    cumulative.columns.name = "week"
    stacked = cumulative.stack().rename("leverage_games").reset_index()
    out = values.merge(stacked, on=["athlete_id", "week"], how="left")
    margin = out["shrunk_per_game"].astype(float) - out["replacement_per_game"].astype(float)
    out["points_above_replacement"] = out["value_above_replacement"].astype(float)
    out["wins_above_replacement"] = margin * out["leverage_games"]
    return out[WAR_COLUMNS]
