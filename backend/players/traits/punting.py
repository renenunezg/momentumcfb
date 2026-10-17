"""Punter trait builders.

A punter is graded on four traits, two of distance and two of placement:
``p.punt_average`` (yards per punt above the league average through the
previous week, from the box score), ``p.net`` (gross yards less return yards
and 20 per touchback, the NCAA net), ``p.fair_catch_share`` (punts fair
caught, the data's only hang-time signal) and ``p.no_return_share`` (punts the
returner never ran back: fair catch, downed, out of bounds or touchback). On
2020 through 2024 every one has a split-half reliability above the grade's
noise floor at eight games and holds year to year, and the placement traits
are near-independent of distance (research/kicker_punter_grade_2026-10-16.md).

The three play-level traits read the punt play text, which names the punter
("Will Karoll punt for 38 yds" or, from late 2025, "(02:25) #40 A.Pulkkinen
punt 44 yards to the ULM24"). The name is matched to the game's box score first
and the roster second; a name two players share is skipped. Blocked punts carry
no distance and are left out. Field goal kicking has no measurable skill at
college volumes, so there is no kicker trait.
"""

import re

import pandas as pd

from backend.players.traits import TraitSpec, register
from backend.players.traits.context import SeasonContext
from backend.players.traits.defense import opponents
from backend.players.traits.offense import (
    per_game,
    position_ids,
    prior_totals,
    trait_rows,
)
from backend.players.traits.targets import normalize_name

PUNT_TYPES = frozenset(
    {"Punt", "Blocked Punt", "Punt Return Touchdown", "Blocked Punt Touchdown"}
)
TOUCHBACK_YARDS = 20.0
PUNT_TRAITS = {
    spec.name: register(spec)
    for spec in (
        TraitSpec("p.punt_average", ("P",), "counted"),
        TraitSpec("p.net", ("P",), "counted"),
        TraitSpec("p.fair_catch_share", ("P",), "counted"),
        TraitSpec("p.no_return_share", ("P",), "counted"),
    )
}

_CLOCK = re.compile(r"^\(\d+:\d+\)\s*")
_JERSEY = re.compile(r"^#(\d+)\s+([A-Za-z])\.\s*(.+)$")
_PUNTER = re.compile(r"^(.+?)\s+punt\b", re.IGNORECASE)
_GROSS = re.compile(r"punt (?:for )?(-?\d+) y(?:ar)?ds", re.IGNORECASE)
_RETURN = re.compile(r"returns? (?:for )?(-?\d+) y(?:ar)?ds", re.IGNORECASE)
_RETURN_ALT = re.compile(r"(\d+) Yd Punt Return")
_LOSS = re.compile(r"returns? for a loss of (\d+)", re.IGNORECASE)


def _schedule(games: pd.DataFrame) -> pd.DataFrame:
    """``game_id, team, opponent, weight, model_week`` for every team-game."""
    weeks = games[["game_id", "model_week"]].drop_duplicates("game_id")
    return opponents(games).merge(weeks, on="game_id", how="left")


def _initial_key(name: str) -> str:
    """``i:last`` for a normalized full name, the key the late-2025 text matches on."""
    parts = name.split(" ", 1)
    return f"{parts[0][:1]}:{parts[1]}" if len(parts) == 2 and parts[0] else ""


def punter_name(text: object) -> tuple[str, str | None]:
    """The punter named before ``punt`` in a play text.

    Returns ``(key, jersey)``: a normalized full name with no jersey, or
    ``initial:last`` with the jersey number for the ``#40 A.Pulkkinen`` form.
    Empty when the text names nobody.
    """
    if not isinstance(text, str):
        return "", None
    match = _PUNTER.match(_CLOCK.sub("", text))
    if not match:
        return "", None
    raw = match.group(1).strip()
    jersey = _JERSEY.match(raw)
    if jersey:
        key = f"{jersey.group(2).lower()}:{normalize_name(jersey.group(3))}"
        return key, jersey.group(1)
    return normalize_name(raw), None


def _lookups(
    box: pd.DataFrame, roster: pd.DataFrame
) -> tuple[dict[tuple, dict[str, set[str]]], dict[str, dict[str, set[str]]]]:
    """Name keys to athlete ids, per (game, team) from the box and per team from the roster."""
    by_game: dict[tuple, dict[str, set[str]]] = {}
    names = box[["game_id", "team", "athlete_id", "athlete_name"]].drop_duplicates()
    for (game_id, team), frame in names.groupby(["game_id", "team"]):
        table: dict[str, set[str]] = {}
        for name, athlete in zip(frame["athlete_name"], frame["athlete_id"]):
            key = normalize_name(name)
            table.setdefault(key, set()).add(str(athlete))
            if initial := _initial_key(key):
                table.setdefault(initial, set()).add(str(athlete))
        by_game[(game_id, team)] = table
    by_team: dict[str, dict[str, set[str]]] = {}
    full = (roster["first_name"].fillna("") + " " + roster["last_name"].fillna("")).map(
        normalize_name
    )
    for team, frame in roster.assign(key=full).groupby("team"):
        table = {}
        for key, athlete, jersey in zip(frame["key"], frame["id"], frame["jersey"]):
            table.setdefault(key, set()).add(str(athlete))
            if initial := _initial_key(key):
                table.setdefault(initial, set()).add(str(athlete))
            if pd.notna(jersey):
                table.setdefault(f"#{int(jersey)}", set()).add(str(athlete))
        by_team[team] = table
    return by_game, by_team


def _resolve(
    by_game: dict[tuple, dict[str, set[str]]],
    by_team: dict[str, dict[str, set[str]]],
    game_id: object,
    team: str,
    key: str,
    jersey: str | None,
) -> str | None:
    """One athlete id for a punter key, or None when unnamed, unknown or shared."""
    if not key:
        return None
    hit = by_game.get((game_id, team), {}).get(key)
    if not hit:
        roster = by_team.get(team, {})
        hit = roster.get(key, set())
        if jersey is not None and len(hit) > 1:
            hit = hit & roster.get(f"#{int(jersey)}", set()) or hit
    return next(iter(hit)) if hit and len(hit) == 1 else None


def punt_plays(ctx: SeasonContext) -> pd.DataFrame:
    """Unblocked punts with a parsed distance, attributed to a punter.

    Columns: the trait game keys, ``weight``, ``gross``, ``returned_yards``,
    ``touchback``, ``fair_catch``, ``no_return``.
    """
    plays = ctx.plays[ctx.plays["play_type"].isin(PUNT_TYPES)]
    text = plays["play_text"]
    named = [punter_name(value) for value in text]
    gross = text.str.extract(_GROSS)[0].astype(float)
    blocked = text.str.contains("blocked", case=False, na=False)
    touchback = text.str.contains("touchback", case=False, na=False)
    fair_catch = text.str.contains("fair catch", case=False, na=False)
    downed = text.str.contains("downed", case=False, na=False)
    out_of_bounds = text.str.contains(
        "out-of-bounds|out of bounds", case=False, na=False
    )
    returned = text.str.extract(_RETURN)[0].astype(float)
    returned = returned.fillna(text.str.extract(_RETURN_ALT)[0].astype(float))
    returned = returned.where(
        ~text.str.contains("returns? for no gain", case=False, na=False), 0.0
    )
    returned = returned.fillna(-text.str.extract(_LOSS)[0].astype(float)).fillna(0.0)
    by_game, by_team = _lookups(ctx.box, ctx.roster)
    athlete = [
        _resolve(by_game, by_team, game_id, team, key, jersey)
        for game_id, team, (key, jersey) in zip(
            plays["game_id"], plays["offense"], named
        )
    ]
    out = pd.DataFrame(
        {
            "game_id": plays["game_id"].to_numpy(),
            "team": plays["offense"].to_numpy(),
            "athlete_id": athlete,
            "gross": gross.to_numpy(),
            "returned_yards": returned.to_numpy(),
            "touchback": touchback.to_numpy(float),
            "fair_catch": fair_catch.to_numpy(float),
            "no_return": (fair_catch | downed | out_of_bounds | touchback).to_numpy(
                float
            ),
            "blocked": blocked.to_numpy(),
        }
    )
    out = out[out["athlete_id"].notna() & out["gross"].notna() & ~out["blocked"]]
    out = out[out["athlete_id"].isin(position_ids(ctx.positions, "P"))]
    return out.merge(_schedule(ctx.games), on=["game_id", "team"], how="inner").drop(
        columns="blocked"
    )


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
    return trait_rows(events, PUNT_TRAITS["p.punt_average"])


def _per_punt(punts: pd.DataFrame, values: pd.Series, spec: TraitSpec) -> pd.DataFrame:
    events = per_game(punts, values * punts["weight"], punts["weight"])
    return trait_rows(events, spec)


def punting_traits(ctx: SeasonContext) -> pd.DataFrame:
    """The four punter traits per punter and game, in the trait frame contract."""
    punts = punt_plays(ctx)
    net = (
        punts["gross"] - punts["returned_yards"] - TOUCHBACK_YARDS * punts["touchback"]
    )
    return pd.concat(
        [
            _p_punt_average(ctx),
            _per_punt(punts, net, PUNT_TRAITS["p.net"]),
            _per_punt(punts, punts["fair_catch"], PUNT_TRAITS["p.fair_catch_share"]),
            _per_punt(punts, punts["no_return"], PUNT_TRAITS["p.no_return_share"]),
        ],
        ignore_index=True,
    )
