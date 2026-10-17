"""Season inputs every trait builder reads.

``load_season_context`` runs the same load sequence as
``value.build_player_values`` up to the credited frame, so trait builders see
exactly the plays, opponent effects, and finalized credit the v4 player values
are built from.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

from backend.etl import store
from backend.features.possessions import classify_plays
from backend.features.units import build_unit_games
from backend.model.weekly import load_weekly_games
from backend.players.credit import assign_play_credit
from backend.players.ingest import read_season_source, read_weekly
from backend.players.value import (
    POSITION_GROUPS,
    _historical_win_probabilities,
    _live_win_probabilities,
    _position_groups,
    adjust_credit,
    box_disruptions,
    finalize_credit,
    load_opponent_prior,
    play_wpa,
    season_unit_effects,
)

CREDITED_COLUMNS = [
    "game_id",
    "athlete_id",
    "athlete_name",
    "team",
    "model_week",
    "side",
    "role",
    "plays",
    "raw_epa",
    "adjusted_epa",
    "wpa",
    "weight",
]
PRESENCE_COLUMNS = ["athlete_id", "team", "game_id", "model_week", "present"]

log = logging.getLogger(__name__)


@dataclass
class SeasonContext:
    season: int
    plays: pd.DataFrame
    games: pd.DataFrame
    effects: pd.DataFrame
    credit: pd.DataFrame
    box: pd.DataFrame
    roster: pd.DataFrame
    teams: pd.DataFrame
    positions: pd.DataFrame
    presence: pd.DataFrame
    # v4's credited frame (play credit plus box disruptions), the source of its per-game value.
    credited: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=CREDITED_COLUMNS)
    )


def player_presence(
    box: pd.DataFrame, credit: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    """One row per player and completed game of the player's team.

    A player is present in a game when a box row (any stat, including a
    zero) or a credit row exists for it. The team comes from the box rows and
    the credit rows together, so a player listed under two teams gets both
    teams' games and is present only where a row exists.
    """
    seen = pd.concat(
        [
            frame[["athlete_id", "team", "game_id"]].assign(
                athlete_id=lambda f: f["athlete_id"].astype(str)
            )
            for frame in (box, credit)
        ],
        ignore_index=True,
    ).drop_duplicates()

    completed = games[games["completed"].fillna(False).astype(bool)]
    team_games = pd.concat(
        [
            completed[["game_id", "model_week", side]].rename(columns={side: "team"})
            for side in ("home_team", "away_team")
        ],
        ignore_index=True,
    )
    pairs = seen[["athlete_id", "team"]].drop_duplicates()
    out = pairs.merge(team_games, on="team", how="inner")
    out = out.merge(
        seen.assign(present=True), on=["athlete_id", "team", "game_id"], how="left"
    )
    out["present"] = out["present"].eq(True)
    return out[PRESENCE_COLUMNS].reset_index(drop=True)


def box_positions(
    positions: pd.DataFrame, box: pd.DataFrame, roster: pd.DataFrame
) -> pd.DataFrame:
    """The v4 position groups plus every box-score player they miss.

    v4 groups only credited players, so a defender with tackles and no play
    credit has no group there. Such a player takes the group of his roster
    position; one with no roster row (and so no credit either) or a roster
    position outside ``POSITION_GROUPS`` stays ungrouped.
    """
    missing = set(box["athlete_id"].astype(str)) - set(
        positions["athlete_id"].astype(str)
    )
    rostered = roster[["id", "position"]].rename(columns={"id": "athlete_id"})
    rostered = rostered.assign(
        athlete_id=rostered["athlete_id"].astype(str)
    ).drop_duplicates("athlete_id")
    added = rostered[rostered["athlete_id"].isin(missing)]
    unrostered = len(missing) - len(added)
    if unrostered:
        log.info(
            "%d box-score players have no roster row and no credit; skipped", unrostered
        )
    added = added.assign(position_group=added["position"].map(POSITION_GROUPS)).dropna(
        subset=["position_group"]
    )
    return pd.concat([positions, added], ignore_index=True)


def load_season_context(season: int) -> SeasonContext:
    plays = classify_plays(store.read_season_pbp(season))
    games = load_weekly_games(season)
    unit_games = build_unit_games(plays)
    effects = season_unit_effects(games, unit_games, load_opponent_prior(season))

    play_stats = read_weekly(season, "play_stats")
    credit = adjust_credit(assign_play_credit(play_stats, plays), effects, games)

    states = _historical_win_probabilities(season)
    if states is None:
        states = _live_win_probabilities(season, plays, games)
    wpa = play_wpa(states)
    credit = finalize_credit(credit, wpa)

    box = read_weekly(season, "box")
    disruptions = box_disruptions(box, credit, plays, effects, games)
    credited = pd.concat(
        [credit[CREDITED_COLUMNS], disruptions[CREDITED_COLUMNS]], ignore_index=True
    )

    roster = read_season_source(season, "roster")
    teams = read_season_source(season, "teams")
    positions = _position_groups(roster, credited)

    play_wpa_by_id = wpa.assign(play_id=wpa["play_id"].astype(str)).drop_duplicates(
        "play_id"
    )
    plays = (
        plays.assign(play_id_key=plays["id"].astype(str))
        .merge(
            play_wpa_by_id.rename(columns={"play_id": "play_id_key"}),
            on="play_id_key",
            how="left",
        )
        .drop(columns="play_id_key")
    )

    # The raw text stays in stat_text for the non-numeric box entries
    # (completions over attempts, field goals made over tried).
    box = box.assign(
        stat_text=box["stat"], stat=pd.to_numeric(box["stat"], errors="coerce")
    )
    presence = player_presence(box, credit, games)
    positions = box_positions(positions, box, roster)
    return SeasonContext(
        season=season,
        plays=plays,
        games=games,
        effects=effects,
        credit=credit,
        box=box,
        roster=roster,
        teams=teams,
        positions=positions,
        presence=presence,
        credited=credited,
    )
