"""Players named in play text, resolved to athlete ids.

CFBD play text names players as a full name ("Will Karoll punt for 38 yds") or,
from late 2025, as a jersey and initial ("(02:25) #40 A.Pulkkinen punt 44
yards"). A name is matched to the game's box score first and the team's roster
second; a name two players share resolves to nobody.
"""

import re

import pandas as pd

from backend.players.traits.targets import normalize_name

CLOCK = re.compile(r"^\(\d+:\d+\)\s*")
_JERSEY = re.compile(r"^#(\d+)\s+([A-Za-z])\.\s*(.+)$")


def _initial_key(name: str) -> str:
    """``i:last`` for a normalized full name, the key the late-2025 text matches on."""
    parts = name.split(" ", 1)
    return f"{parts[0][:1]}:{parts[1]}" if len(parts) == 2 and parts[0] else ""


def name_key(raw: str) -> tuple[str, str | None]:
    """``(key, jersey)`` for a name as the text writes it.

    A normalized full name with no jersey, or ``initial:last`` with the jersey
    number for the ``#40 A.Pulkkinen`` form. Empty when ``raw`` is blank.
    """
    raw = raw.strip()
    jersey = _JERSEY.match(raw)
    if jersey:
        key = f"{jersey.group(2).lower()}:{normalize_name(jersey.group(3))}"
        return key, jersey.group(1)
    return normalize_name(raw), None


def name_lookups(
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


def resolve_name(
    by_game: dict[tuple, dict[str, set[str]]],
    by_team: dict[str, dict[str, set[str]]],
    game_id: object,
    team: str,
    key: str,
    jersey: str | None,
) -> str | None:
    """One athlete id for a name key, or None when unnamed, unknown or shared."""
    if not key:
        return None
    hit = by_game.get((game_id, team), {}).get(key)
    if not hit:
        roster = by_team.get(team, {})
        hit = roster.get(key, set())
        if jersey is not None and len(hit) > 1:
            hit = hit & roster.get(f"#{int(jersey)}", set()) or hit
    return next(iter(hit)) if hit and len(hit) == 1 else None
