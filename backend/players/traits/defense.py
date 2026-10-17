"""Defensive trait builders for DL, LB and DB.

Per-game traits have one row for every game a defender played, which is every
game with a box row (any stat, a zero included) or a credit row, as
``player_presence`` records it; a game without an event still scores zero.
Counts come from the box score, apart from forced fumbles and sack win
probability, which come from competitive credit rows (the box carries neither).
Every numerator and denominator carries the FCS weight: 0.5 in a game against
an FCS opponent, as v4 weights its credit.

The opponent of a defender is the offense he faced, so ``adjust_rate`` nets
out how many such events that offense allowed in earlier weeks.
"""

import numpy as np
import pandas as pd

from backend.features.units import build_unit_games
from backend.players.traits import TraitSpec, register
from backend.players.traits.absence import with_without
from backend.players.traits.context import SeasonContext
from backend.players.traits.offense import (
    GAME_KEYS,
    attach_plays,
    opponent_adjusted,
    position_ids,
    trait_rows,
)
from backend.players.value import FCS_OPPONENT_WEIGHT

# Prior games of shrinkage for the with-without impact estimates. The trait
# builder sets it from split-half reliability on 2020 through 2024 and the
# build report documents the choice; 6.0 is the starting value.
WITH_WITHOUT_SHRINK_GAMES = 6.0

DEFENSE_GROUPS = ("DL", "LB", "DB")
FRONT_GROUPS = ("DL", "LB")
# Box score (category, stat name) to the column a trait reads.
BOX_STATS = {
    ("defensive", "TOT"): "tackles",
    ("defensive", "SOLO"): "solo",
    ("defensive", "TFL"): "tfl",
    ("defensive", "SACKS"): "sacks",
    ("defensive", "QB HUR"): "hurries",
    ("defensive", "PD"): "passes_defended",
    ("interceptions", "INT"): "interceptions",
}

DEF_TRAITS = {
    spec.name: register(spec)
    for spec in (
        TraitSpec("def.havoc", DEFENSE_GROUPS, "counted"),
        TraitSpec("def.pass_rush", FRONT_GROUPS, "counted"),
        TraitSpec("def.finishing", FRONT_GROUPS, "counted"),
        TraitSpec("def.run_stop", FRONT_GROUPS, "counted"),
        TraitSpec(
            "def.every_week", DEFENSE_GROUPS, "counted", aggregate="share_of_games"
        ),
        TraitSpec("def.tackle_share", ("LB",), "counted"),
        TraitSpec("def.coverage", ("LB",), "counted"),
        TraitSpec("def.ball_production", ("DB",), "counted"),
        TraitSpec("def.run_support", ("DB",), "counted", higher_is_better=None),
        TraitSpec("def.impact", DEFENSE_GROUPS, "inferred", aggregate="latest"),
        TraitSpec("def.pass_impact", ("DB",), "inferred", aggregate="latest"),
    )
}


def group_ids(positions: pd.DataFrame, groups: tuple[str, ...]) -> set[str]:
    return set().union(*(position_ids(positions, group) for group in groups))


def opponents(games: pd.DataFrame) -> pd.DataFrame:
    """Each team's opponent and FCS weight per game: ``game_id, team, opponent, weight``."""
    sides = []
    for team, opponent in (("home", "away"), ("away", "home")):
        side = games[
            [
                "game_id",
                f"{team}_team",
                f"{opponent}_team",
                f"{opponent}_classification",
            ]
        ]
        sides.append(
            side.set_axis(["game_id", "team", "opponent", "classification"], axis=1)
        )
    out = pd.concat(sides, ignore_index=True)
    is_fcs = out.pop("classification").fillna("").str.lower().eq("fcs")
    return out.assign(weight=np.where(is_fcs, FCS_OPPONENT_WEIGHT, 1.0))


def box_lines(box: pd.DataFrame) -> pd.DataFrame:
    """The defensive box stats per player, team and game, one column per stat."""
    keys = list(zip(box["category"], box["stat_name"]))
    rows = box.assign(column=[BOX_STATS.get(key) for key in keys]).dropna(
        subset=["column"]
    )
    wide = rows.pivot_table(
        index=["athlete_id", "team", "game_id"],
        columns="column",
        values="stat",
        aggfunc="sum",
    )
    wide = wide.reindex(columns=list(BOX_STATS.values())).fillna(0.0).reset_index()
    wide.columns.name = None
    return wide.assign(athlete_id=wide["athlete_id"].astype(str))


def defender_games(
    ctx: SeasonContext, rows: pd.DataFrame, defenders: set[str]
) -> pd.DataFrame:
    """One row per defender and game played with box counts, forced fumbles, and the FCS weight."""
    played = ctx.presence[
        ctx.presence["present"] & ctx.presence["athlete_id"].isin(defenders)
    ]
    played = played.drop(columns="present").merge(
        opponents(ctx.games), on=["game_id", "team"], how="inner"
    )
    lines = box_lines(ctx.box)
    out = played.merge(lines, on=["athlete_id", "team", "game_id"], how="left")
    forced = rows[rows["role"].eq("fumble_forcer")]
    forced = (
        forced.groupby(["athlete_id", "team", "game_id"])
        .size()
        .rename("forced_fumbles")
    )
    out = out.merge(
        forced.reset_index(), on=["athlete_id", "team", "game_id"], how="left"
    )
    counts = [*BOX_STATS.values(), "forced_fumbles"]
    out[counts] = out[counts].fillna(0.0)
    team_tackles = lines.groupby(["team", "game_id"], as_index=False)["tackles"].sum()
    return out.merge(
        team_tackles.rename(columns={"tackles": "team_tackles"}),
        on=["team", "game_id"],
        how="left",
    )


def defense_rows(ctx: SeasonContext) -> pd.DataFrame:
    """Competitive defensive credit rows with the play's situation attached."""
    credit = ctx.credit
    keep = credit["side"].eq("defense") & credit["is_competitive"].eq(True)
    rows = attach_plays(credit[keep], ctx.plays)
    return rows.assign(athlete_id=rows["athlete_id"].astype(str))


def events(
    frame: pd.DataFrame, numerator: pd.Series, denominator: pd.Series
) -> pd.DataFrame:
    """FCS-weighted per-game numerator and denominator under the game keys."""
    weight = frame["weight"]
    return frame[GAME_KEYS].assign(
        numerator=numerator * weight, denominator=denominator * weight
    )


def in_groups(
    frame: pd.DataFrame, positions: pd.DataFrame, groups: tuple[str, ...]
) -> pd.DataFrame:
    return frame[frame["athlete_id"].isin(group_ids(positions, groups))]


def _def_havoc(games: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    havoc = (
        games["tfl"]
        + games["forced_fumbles"]
        + games["passes_defended"]
        + games["interceptions"]
    )
    adjusted = opponent_adjusted(events(games, havoc, 1.0), schedule)
    return trait_rows(adjusted, DEF_TRAITS["def.havoc"])


def sack_prices(sacks: pd.DataFrame, weeks: pd.Series) -> pd.Series:
    """Median absolute WPA of every tagged sack before each week, indexed by week."""
    plays = sacks.drop_duplicates("play_id")
    value = plays["play_wpa"].fillna(0.0).abs()
    return pd.Series(
        {
            week: value[plays["model_week"].lt(week)].median()
            for week in sorted(weeks.unique())
        },
        dtype=float,
    )


def _def_pass_rush(games: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """The WPA of the player's own sacks plus his hurries priced at the median sack.

    A sack is worth its play's absolute WPA, split equally among its sackers.
    A hurry is priced at the median absolute WPA of every tagged sack through
    the previous week; with no earlier sack (the season's first week) a game
    with hurries has a NaN numerator. Missing WPA counts as zero.
    """
    sacks = rows[rows["role"].eq("sacker")]
    sack_wpa = sacks["play_wpa"].fillna(0.0).abs() * sacks["share"]
    own = (
        sacks.assign(own=sack_wpa)
        .groupby(["athlete_id", "team", "game_id"])["own"]
        .sum()
    )
    out = games.merge(
        own.reset_index(), on=["athlete_id", "team", "game_id"], how="left"
    )
    price = out["model_week"].map(sack_prices(sacks, out["model_week"]))
    hurry_value = (out["hurries"] * price).where(out["hurries"] > 0, 0.0)
    numerator = out["own"].fillna(0.0) + hurry_value
    return trait_rows(events(out, numerator, 1.0), DEF_TRAITS["def.pass_rush"])


def _def_finishing(games: pd.DataFrame) -> pd.DataFrame:
    """Sacks per sack chance (sacks plus hurries); a game with no chance gets no row."""
    chances = games["sacks"] + games["hurries"]
    out = events(games, games["sacks"], chances)
    return trait_rows(out[chances > 0], DEF_TRAITS["def.finishing"])


def _def_run_stop(games: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Tackles for loss that were not sacks."""
    run_stops = (games["tfl"] - games["sacks"]).clip(lower=0.0)
    adjusted = opponent_adjusted(events(games, run_stops, 1.0), schedule)
    return trait_rows(adjusted, DEF_TRAITS["def.run_stop"])


def _def_every_week(games: pd.DataFrame) -> pd.DataFrame:
    """1 in a game with a TFL, sack, pass defended, interception or forced fumble."""
    disruptions = ["tfl", "sacks", "passes_defended", "interceptions", "forced_fumbles"]
    disrupted = games[disruptions].gt(0).any(axis=1).astype(float)
    return trait_rows(events(games, disrupted, 1.0), DEF_TRAITS["def.every_week"])


def _def_tackle_share(games: pd.DataFrame) -> pd.DataFrame:
    """The player's tackles over his team's in the game; a game with none gets no row."""
    out = events(games, games["tackles"], games["team_tackles"])
    return trait_rows(out[games["team_tackles"] > 0], DEF_TRAITS["def.tackle_share"])


def _def_coverage(games: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    plays = games["passes_defended"] + games["interceptions"]
    adjusted = opponent_adjusted(events(games, plays, 1.0), schedule)
    return trait_rows(adjusted, DEF_TRAITS["def.coverage"])


def _def_ball_production(games: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    plays = games["interceptions"] + games["passes_defended"]
    adjusted = opponent_adjusted(events(games, plays, 1.0), schedule)
    return trait_rows(adjusted, DEF_TRAITS["def.ball_production"])


def _def_run_support(games: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Solo tackles; whether more is better for a defensive back is left to the fit."""
    adjusted = opponent_adjusted(events(games, games["solo"], 1.0), schedule)
    return trait_rows(adjusted, DEF_TRAITS["def.run_support"])


def epa_allowed(plays: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Each defense's competitive EPA allowed per play, all and passing, per game.

    ``team, game_id, model_week, epa_allowed, pass_epa_allowed``; the offense's
    unit game row is the defense's game.
    """
    units = build_unit_games(plays)
    total_plays = units["rush_plays"] + units["pass_plays"]
    out = pd.DataFrame(
        {
            "team": units["opponent"],
            "game_id": units["game_id"],
            "epa_allowed": (units["rush_ppa"] + units["pass_ppa"])
            / total_plays.where(total_plays > 0),
            "pass_epa_allowed": units["pass_ppa"]
            / units["pass_plays"].where(units["pass_plays"] > 0),
        }
    )
    weeks = schedule[["game_id", "model_week"]].drop_duplicates("game_id")
    return out.merge(weeks, on="game_id", how="inner")


def weekly_impact(
    presence: pd.DataFrame, unit_games: pd.DataFrame, value_column: str, spec: TraitSpec
) -> pd.DataFrame:
    """``with_without`` through each week over games of that week or earlier.

    One row per player and week with ``game_id`` missing, the impact as numerator
    and a denominator of 1.
    """
    frames = []
    for week in sorted(presence["model_week"].unique()):
        impact = with_without(
            presence[presence["model_week"].le(week)],
            unit_games[unit_games["model_week"].le(week)],
            value_column,
            WITH_WITHOUT_SHRINK_GAMES,
        )
        frames.append(
            impact[["athlete_id"]].assign(
                model_week=week,
                game_id=pd.NA,
                numerator=impact["impact"],
                denominator=1.0,
            )
        )
    return trait_rows(pd.concat(frames, ignore_index=True), spec)


def defense_traits(ctx: SeasonContext) -> pd.DataFrame:
    """The eleven DL, LB and DB traits in the trait frame contract.

    Per-game traits have one row per player and game; ``def.impact`` and
    ``def.pass_impact`` have one row per player and week, so ``game_id`` is a
    nullable integer that is missing on those rows.
    """
    rows = defense_rows(ctx)
    games = defender_games(ctx, rows, group_ids(ctx.positions, DEFENSE_GROUPS))
    front = in_groups(games, ctx.positions, FRONT_GROUPS)
    linebackers = in_groups(games, ctx.positions, ("LB",))
    backs = in_groups(games, ctx.positions, ("DB",))
    unit_games = epa_allowed(ctx.plays, ctx.games)
    presence = ctx.presence
    frames = [
        _def_havoc(games, ctx.games),
        _def_pass_rush(front, rows),
        _def_finishing(front),
        _def_run_stop(front, ctx.games),
        _def_every_week(games),
        _def_tackle_share(linebackers),
        _def_coverage(linebackers, ctx.games),
        _def_ball_production(backs, ctx.games),
        _def_run_support(backs, ctx.games),
        weekly_impact(
            in_groups(presence, ctx.positions, DEFENSE_GROUPS),
            unit_games,
            "epa_allowed",
            DEF_TRAITS["def.impact"],
        ),
        weekly_impact(
            in_groups(presence, ctx.positions, ("DB",)),
            unit_games,
            "pass_epa_allowed",
            DEF_TRAITS["def.pass_impact"],
        ),
    ]
    return pd.concat(frames, ignore_index=True).astype({"game_id": "Int64"})
