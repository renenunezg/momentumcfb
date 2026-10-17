"""Offensive line and kicking trait builders.

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

Kicking comes from the box score and the field goal plays. The feed does not
tag the kicker on a field goal play, so a team-game's field goals go to the one
kicker whose box ``FG`` line shows attempts; a team-game with no such kicker or
several is skipped and counted. Field goals are not scrimmage plays and the
feed never marks one competitive, so neither kicking trait is gated on
competitive plays (the expectation table is not either). Both are FCS weighted
by the opponent's classification, and punt averages are judged against the
league average through the previous week, with no row in the season's first
week.
"""

import logging

import pandas as pd

from backend.features.units import _adjusted_line_yards
from backend.players.traits import TraitSpec, register
from backend.players.traits.context import SeasonContext
from backend.players.traits.defense import opponents
from backend.players.traits.expectations import (
    FIELD_GOAL_TYPES,
    ExpectationTables,
    expected_field_goal,
)
from backend.players.traits.offense import (
    GAME_KEYS,
    conversion_over_expected,
    dropbacks,
    is_short_yardage,
    offense_rows,
    opponent_adjusted,
    per_game,
    position_ids,
    prior_totals,
    trait_rows,
)

log = logging.getLogger(__name__)

LINE_TRAITS = {
    spec.name: register(spec)
    for spec in (
        TraitSpec("ol.protection", ("OL",), "inferred"),
        TraitSpec("ol.stuff_rate", ("OL",), "inferred"),
        TraitSpec("ol.short_yardage", ("OL",), "inferred"),
        TraitSpec("ol.line_yards", ("OL",), "inferred"),
    )
}

KICKING_TRAITS = {
    spec.name: register(spec)
    for spec in (
        TraitSpec("k.field_goal", ("K",), "counted"),
        TraitSpec("p.punt_average", ("P",), "counted"),
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


def _schedule(games: pd.DataFrame) -> pd.DataFrame:
    """``game_id, team, opponent, weight, model_week`` for every team-game."""
    weeks = games[["game_id", "model_week"]].drop_duplicates("game_id")
    return opponents(games).merge(weeks, on="game_id", how="left")


def _lone_kickers(box: pd.DataFrame) -> pd.DataFrame:
    """``game_id, team, athlete_id`` for each team-game with exactly one kicker with attempts."""
    fg = box[box["category"].eq("kicking") & box["stat_name"].eq("FG")]
    attempts = pd.to_numeric(fg["stat_text"].str.extract(r"\d+\s*/\s*(\d+)")[0])
    kickers = fg.loc[
        attempts.gt(0), ["game_id", "team", "athlete_id"]
    ].drop_duplicates()
    kickers = kickers.assign(athlete_id=kickers["athlete_id"].astype(str))
    counts = kickers.groupby(["game_id", "team"])["athlete_id"].transform("size")
    return kickers[counts.eq(1)]


def _k_field_goal(ctx: SeasonContext, tables: ExpectationTables) -> pd.DataFrame:
    """Field goals made over the situation's expectation, per kicker and game.

    Every field goal play goes to the lone kicker with attempts in the box
    score; ambiguous team-games are skipped and logged once for the season.
    Field goals are not scrimmage plays, so the feed never marks one
    competitive and there is no competitive gate, as in the expectation table.
    """
    plays = ctx.plays
    kicks = plays[plays["play_type"].isin(FIELD_GOAL_TYPES)]
    if kicks.empty:
        return trait_rows(
            pd.DataFrame(columns=[*GAME_KEYS, "numerator", "denominator"]),
            KICKING_TRAITS["k.field_goal"],
        )
    over = kicks["play_type"].eq("Field Goal Good").astype(float) - expected_field_goal(
        tables, kicks
    )
    kicks = kicks.assign(over=over).rename(columns={"offense": "team"})[
        ["game_id", "team", "over"]
    ]
    kicks = kicks.merge(_schedule(ctx.games), on=["game_id", "team"], how="inner")
    kicks = kicks.merge(_lone_kickers(ctx.box), on=["game_id", "team"], how="left")
    skipped = kicks.loc[
        kicks["athlete_id"].isna(), ["game_id", "team"]
    ].drop_duplicates()
    log.info("%d team-games skipped: ambiguous kicker", len(skipped))
    kicks = kicks[kicks["athlete_id"].isin(position_ids(ctx.positions, "K"))]
    events = per_game(kicks, kicks["over"] * kicks["weight"], kicks["weight"])
    return trait_rows(events, KICKING_TRAITS["k.field_goal"])


def _punt_lines(box: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Punts (``no``) and punt yards (``yds``) per punter and game, with the game's schedule keys."""
    punting = box[box["category"].eq("punting") & box["stat_name"].isin(["NO", "YDS"])]
    wide = punting.pivot_table(
        index=["athlete_id", "team", "game_id"],
        columns="stat_name",
        values="stat",
        aggfunc="sum",
    )
    wide = wide.reindex(columns=["NO", "YDS"]).fillna(0.0).reset_index()
    wide = wide.rename(columns={"NO": "no", "YDS": "yds"}).assign(
        athlete_id=wide["athlete_id"].astype(str)
    )
    return wide.merge(_schedule(games), on=["game_id", "team"], how="inner")


def _p_punt_average(ctx: SeasonContext) -> pd.DataFrame:
    """Punt yards above the league average through the previous week, per punt.

    The league average is every punter's yards over every punt in earlier
    weeks; the season's first week has none and gets no row.
    """
    lines = _punt_lines(ctx.box, ctx.games)
    league = prior_totals(lines, [], ["yds", "no"])
    league["league_average"] = league["yds"] / league["no"].where(league["no"] > 0)
    lines = lines.merge(
        league[["model_week", "league_average"]], on="model_week", how="left"
    )
    punters = lines[lines["athlete_id"].isin(position_ids(ctx.positions, "P"))]
    punters = punters[punters["league_average"].notna() & punters["no"].gt(0)]
    numerator = (punters["yds"] - punters["no"] * punters["league_average"]) * punters[
        "weight"
    ]
    events = per_game(punters, numerator, punters["no"] * punters["weight"])
    return trait_rows(events, KICKING_TRAITS["p.punt_average"])


def kicking_traits(ctx: SeasonContext, tables: ExpectationTables) -> pd.DataFrame:
    """Field goal and punt traits per kicker and game, in the trait frame contract."""
    return pd.concat(
        [_k_field_goal(ctx, tables), _p_punt_average(ctx)], ignore_index=True
    )
