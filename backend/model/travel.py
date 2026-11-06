"""Visitor travel distance as a frozen margin term.

The engine's fitted home-field advantage is one number for every home game, so
it averages over short and cross-country trips. On the 2020-2022 walk-forward
(regular season, FBS versus FBS) the calibrated pure margin missed by 1.770
points for every 1,000 miles the visitor travelled beyond the home team's own
trip. The coefficient was chosen by leave-one-season-out squared error on
2020-2022 against a threshold, a power curve, a week-varying weight, time
zones crossed and early body-clock kickoffs; none beat the straight line.
Frozen and scored once on 2023-2025, it cut margin MAE from 12.536 to 12.502
(week-clustered se 0.021). The closing line already prices most of it, so it
is a model-accuracy term, not a pick signal.

On a true home game the term is centered on the 2020-2022 average trip, so the
fitted home-field advantage keeps its meaning as the average home edge. Neutral
sites have no fitted home edge to absorb the average and stay uncentered. A game
whose venue or either team's location is unknown gets no term.
"""

import logging

import numpy as np
import pandas as pd

from backend.etl import store

log = logging.getLogger(__name__)

TRAVEL_POINTS_PER_1000_MILES = 1.770
# 1.770 times the 539.7 mile mean visitor-minus-home trip on 2020-2022 true
# home games.
TRAVEL_CENTER_POINTS = 0.9553
EARTH_RADIUS_MILES = 3958.8


def _teams_snapshot(season: int) -> pd.DataFrame:
    for path in (
        store.RAW_DIR / "players" / str(season) / "teams.parquet",
        store.RAW_DIR / "preseason" / str(season) / "teams.parquet",
    ):
        if path.exists():
            return pd.read_parquet(path, columns=["id", "location"])
    raise FileNotFoundError(f"no {season} /teams snapshot for team locations")


def team_locations(season: int) -> tuple[dict[int, tuple], dict[int, tuple]]:
    """Home stadium coordinates by team id, and coordinates by venue id."""
    teams, venues = {}, {}
    for team_id, location in _teams_snapshot(season).itertuples(index=False):
        if not isinstance(location, dict):
            continue
        point = (location.get("latitude"), location.get("longitude"))
        if not all(isinstance(v, float) and np.isfinite(v) for v in point):
            continue
        teams[int(team_id)] = point
        venue = location.get("id")
        if venue is not None and np.isfinite(venue):
            venues[int(venue)] = point
    return teams, venues


def _miles(a: tuple, b: tuple) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (*a, *b))
    h = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return float(2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(h)))


def travel_margin_adjustments(
    projections: pd.DataFrame, games: pd.DataFrame
) -> pd.Series:
    """Home-margin points per projected game, indexed like ``projections``."""
    venue_by_game = games.drop_duplicates("game_id").set_index("game_id")["venue_id"]
    points = pd.Series(0.0, index=projections.index)
    unknown = []
    for season, rows in projections.groupby("season"):
        teams, venues = team_locations(int(season))
        for index, game in rows.iterrows():
            home = teams.get(int(game["home_team_id"]))
            away = teams.get(int(game["away_team_id"]))
            neutral = bool(game["neutral_site"])
            venue_id = venue_by_game.get(game["game_id"])
            venue = venues.get(int(venue_id)) if pd.notna(venue_id) else None
            if venue is None and not neutral:
                venue = home
            if home is None or away is None or venue is None:
                unknown.append(int(game["game_id"]))
                continue
            extra_miles = _miles(away, venue) - _miles(home, venue)
            term = TRAVEL_POINTS_PER_1000_MILES * extra_miles / 1000.0
            points[index] = term if neutral else term - TRAVEL_CENTER_POINTS
    if unknown:
        log.warning("no travel term for %d games without locations", len(unknown))
    return points
