"""Offensive line trait builders.

The offensive line is rated as one unit per team: ``athlete_id`` is
``unit:{team}`` and there are no individual lineman rows. Team dropbacks and
carries come from the team's competitive offensive credit rows, weighted by
the credit row's FCS weight, and the opponent is the defense faced.

``ol.protection`` v1 expects the league sack rate per dropback through the
previous week. A follow-up should expect each passer's own sack rate from the
previous season instead, so a line is not credited or blamed for a quarterback
who holds the ball. The season's first week has no league rate: its rows keep
their dropbacks and have a NaN numerator, and they are left out of the
opponent adjustment.
"""

import pandas as pd

from backend.features.units import _adjusted_line_yards
from backend.players.traits import TraitSpec, register
from backend.players.traits.context import SeasonContext
from backend.players.traits.expectations import ExpectationTables
from backend.players.traits.offense import (
    GAME_KEYS,
    conversion_over_expected,
    dropbacks,
    is_short_yardage,
    offense_rows,
    opponent_adjusted,
    per_game,
    prior_totals,
    trait_rows,
)

LINE_TRAITS = {
    spec.name: register(spec)
    for spec in (
        TraitSpec("ol.protection", ("OL",), "inferred"),
        TraitSpec("ol.stuff_rate", ("OL",), "inferred"),
        TraitSpec("ol.short_yardage", ("OL",), "inferred"),
        TraitSpec("ol.line_yards", ("OL",), "inferred"),
    )
}


def _ol_protection(all_dropbacks: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Sacks avoided against the league sack rate through the previous week."""
    sacked = all_dropbacks["is_sack"] * all_dropbacks["weight"]
    team = per_game(all_dropbacks, sacked, all_dropbacks["weight"])
    team = team.rename(columns={"numerator": "sacks", "denominator": "dropbacks"})
    league = prior_totals(team, [], ["sacks", "dropbacks"])
    league["league_rate"] = league["sacks"] / league["dropbacks"].where(
        league["dropbacks"] > 0
    )
    merged = team.merge(
        league[["model_week", "league_rate"]], on="model_week", how="left"
    )
    merged["numerator"] = merged["dropbacks"] * merged["league_rate"] - merged["sacks"]
    merged["denominator"] = merged["dropbacks"]
    events = merged[[*GAME_KEYS, "numerator", "denominator"]]
    known = events["numerator"].notna()
    adjusted = opponent_adjusted(events[known], games)
    return trait_rows(
        pd.concat([adjusted, events[~known]], ignore_index=True),
        LINE_TRAITS["ol.protection"],
    )


def _ol_stuff_rate(carries: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """The negative of stuffed carries (no gain or a loss) per carry, so higher is better."""
    stuffed = carries["yards_gained"].le(0) * carries["weight"]
    events = per_game(carries, -stuffed, carries["weight"])
    return trait_rows(opponent_adjusted(events, games), LINE_TRAITS["ol.stuff_rate"])


def _ol_short_yardage(
    carries: pd.DataFrame, tables: ExpectationTables, games: pd.DataFrame
) -> pd.DataFrame:
    """Conversions over expected on the team's short-yardage carries."""
    plays = carries[is_short_yardage(carries)]
    events = per_game(plays, conversion_over_expected(plays, tables), plays["weight"])
    return trait_rows(opponent_adjusted(events, games), LINE_TRAITS["ol.short_yardage"])


def _ol_line_yards(carries: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    line_yards = _adjusted_line_yards(carries["yards_gained"]) * carries["weight"]
    events = per_game(carries, line_yards, carries["weight"])
    return trait_rows(opponent_adjusted(events, games), LINE_TRAITS["ol.line_yards"])


def line_traits(ctx: SeasonContext, tables: ExpectationTables) -> pd.DataFrame:
    """The four offensive line traits, one unit row per team and game."""
    rows = offense_rows(ctx)
    rows = rows.assign(athlete_id="unit:" + rows["team"].astype(str))
    carries = rows[rows["role"].eq("rusher")]
    measured = carries[carries["yards_gained"].notna()]
    frames = [
        _ol_protection(dropbacks(rows), ctx.games),
        _ol_stuff_rate(measured, ctx.games),
        _ol_short_yardage(carries, tables, ctx.games),
        _ol_line_yards(measured, ctx.games),
    ]
    return pd.concat(frames, ignore_index=True)
