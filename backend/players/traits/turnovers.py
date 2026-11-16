"""Quarterback ball security, measured by the turnovers a passer risks.

Who recovers a fumble is luck (a player's lost share of his fumbles does not
repeat year to year, r -0.03 over 2019-2025) and an interception is a small
share of the throws a defender gets a hand on. So both traits count the
risk itself (research 2026-11-16).

``qb.ball_security`` is expected turnovers per pass and run from the box
score: ``FUMBLE_LOST_SHARE`` of every fumble, recovered by either side, plus
``INTERCEPTION_SHARE`` of interceptions and passes defended, the opposing
defenders' passes defended in the game split by his share of the team's
attempts. A game whose opponent's box carries no passes defended (most FCS
box scores) gets no row. Pooled over 2020-2024 its split-half reliability at
eight games is 0.24; the win probability of actual turnovers it replaced read
0.08.

``qb.sack_ball_security`` is fumbles per sack, whoever recovered, from the
competitive play text that names the passer sacked. Plays whose text records
an overturned call are skipped, because their text quotes the original play.
At eight games its reliability is 0.07, so the grade leaves it descriptive.

Both are negated, so higher is better, and adjusted for the defense faced.
CFBD has no pressure data, so a throw a defender got a hand on cannot be split
into clean and pressured pockets.
"""

import re

import pandas as pd

from backend.players.traits import TraitSpec, register
from backend.players.traits.context import SeasonContext
from backend.players.traits.defense import opponents
from backend.players.traits.offense import (
    GAME_KEYS,
    offense_rows,
    opponent_adjusted,
    per_game,
    position_ids,
    trait_rows,
)
from backend.players.traits.play_names import (
    CLOCK,
    name_key,
    name_lookups,
    resolve_name,
)

# League shares on 2019-2024 FBS games: lost fumbles over all fumbles, and
# interceptions over interceptions plus passes defended. Base rates, not fitted
# to any outcome.
FUMBLE_LOST_SHARE = 0.486
INTERCEPTION_SHARE = 0.199

TURNOVER_TRAITS = {
    spec.name: register(spec)
    for spec in (
        TraitSpec("qb.ball_security", ("QB",), "counted"),
        TraitSpec("qb.sack_ball_security", ("QB",), "counted"),
    )
}

_SACKED = re.compile(r"^(.+?)\s+sacked\b", re.IGNORECASE)
_FORMATION = re.compile(
    r"^(?:no huddle-?\s*)?(?:shotgun|pistol|under center)?\s*", re.IGNORECASE
)
BOX_LINES = {
    ("passing", "C/ATT"): "attempts",
    ("passing", "INT"): "interceptions",
    ("rushing", "CAR"): "carries",
    ("fumbles", "FUM"): "fumbles",
    ("defensive", "PD"): "passes_defended",
}


def _schedule(games: pd.DataFrame) -> pd.DataFrame:
    weeks = games[["game_id", "model_week"]].drop_duplicates("game_id")
    return opponents(games).merge(weeks, on="game_id", how="left")


def box_turnover_lines(box: pd.DataFrame) -> pd.DataFrame:
    """Per player and game: attempts, interceptions, carries, fumbles, passes defended."""
    keys = list(zip(box["category"], box["stat_name"]))
    rows = box.assign(column=[BOX_LINES.get(key) for key in keys]).dropna(
        subset=["column"]
    )
    # A row the feed repeats verbatim within a game is counted once.
    rows = rows.drop_duplicates(
        ["game_id", "team", "athlete_id", "category", "stat_name", "stat_text"]
    )
    attempts = rows["column"].eq("attempts")
    stat = rows["stat_text"].astype(str)
    stat = stat.where(~attempts, stat.str.split("/").str[-1])
    rows = rows.assign(value=pd.to_numeric(stat, errors="coerce").fillna(0.0))
    wide = rows.pivot_table(
        index=["game_id", "team", "athlete_id"],
        columns="column",
        values="value",
        aggfunc="sum",
    )
    wide = wide.reindex(columns=list(BOX_LINES.values())).fillna(0.0).reset_index()
    reported = (
        rows[rows["column"].eq("passes_defended")]
        .groupby(["game_id", "team"])
        .size()
        .rename("reports_passes_defended")
    )
    return (
        wide.assign(athlete_id=wide["athlete_id"].astype(str))
        .merge(reported.reset_index(), on=["game_id", "team"], how="left")
        .fillna({"reports_passes_defended": 0})
    )


def _qb_ball_security(ctx: SeasonContext, qbs: set[str]) -> pd.DataFrame:
    lines = box_turnover_lines(ctx.box)
    teams = lines.groupby(["game_id", "team"], as_index=False).agg(
        team_attempts=("attempts", "sum"),
        passes_defended=("passes_defended", "sum"),
        reported=("reports_passes_defended", "max"),
    )
    against = teams.rename(
        columns={
            "team": "opponent",
            "passes_defended": "passes_defended_against",
            "reported": "opponent_reported",
        }
    )[["game_id", "opponent", "passes_defended_against", "opponent_reported"]]
    passers = lines[lines["athlete_id"].isin(qbs)].merge(
        _schedule(ctx.games), on=["game_id", "team"], how="inner"
    )
    passers = passers.merge(
        teams[["game_id", "team", "team_attempts"]], on=["game_id", "team"]
    ).merge(against, on=["game_id", "opponent"], how="left")
    passers = passers[
        passers["opponent_reported"].gt(0) & passers["attempts"].gt(0)
    ].copy()
    defended = (
        passers["passes_defended_against"]
        * passers["attempts"]
        / passers["team_attempts"]
    )
    expected = (
        INTERCEPTION_SHARE * (passers["interceptions"] + defended)
        + FUMBLE_LOST_SHARE * passers["fumbles"]
    )
    plays = passers["attempts"] + passers["carries"]
    events = per_game(passers, -expected * passers["weight"], plays * passers["weight"])
    return trait_rows(
        opponent_adjusted(events, ctx.games), TURNOVER_TRAITS["qb.ball_security"]
    )


def sacked_name(text: object) -> tuple[str, str | None]:
    """The passer named before ``sacked``, as ``play_names.name_key``; empty when none."""
    if not isinstance(text, str):
        return "", None
    match = _SACKED.match(CLOCK.sub("", text))
    if not match:
        return "", None
    raw = match.group(1)
    if "#" in raw:
        raw = raw[raw.rindex("#") :]
    else:
        raw = _FORMATION.sub("", raw)
    return name_key(raw)


def sack_plays(ctx: SeasonContext, qbs: set[str]) -> pd.DataFrame:
    """Competitive sacks of a named quarterback, with ``fumbled`` and the game keys."""
    plays = ctx.plays
    text = plays["play_text"].fillna("")
    sacked = text.str.contains(r"\bsacked\b", case=False)
    keep = (
        plays["is_competitive"].eq(True)
        & plays["is_scrimmage"].eq(True)
        & sacked
        & ~text.str.contains("overturned", case=False)
    )
    plays, text = plays[keep], text[keep]
    by_game, by_team = name_lookups(ctx.box, ctx.roster)
    athlete = [
        resolve_name(by_game, by_team, game_id, team, *sacked_name(value))
        for game_id, team, value in zip(plays["game_id"], plays["offense"], text)
    ]
    out = pd.DataFrame(
        {
            "game_id": plays["game_id"].to_numpy(),
            "team": plays["offense"].to_numpy(),
            "athlete_id": athlete,
            "fumbled": text.str.contains("fumble", case=False).to_numpy(float),
        }
    )
    out = out[out["athlete_id"].isin(qbs)]
    return out.merge(_schedule(ctx.games), on=["game_id", "team"], how="inner")


def _qb_sack_ball_security(ctx: SeasonContext, qbs: set[str]) -> pd.DataFrame:
    sacks = sack_plays(ctx, qbs)
    events = per_game(sacks, -sacks["fumbled"] * sacks["weight"], sacks["weight"])
    return trait_rows(
        opponent_adjusted(events[[*GAME_KEYS, "numerator", "denominator"]], ctx.games),
        TURNOVER_TRAITS["qb.sack_ball_security"],
    )


def turnover_traits(ctx: SeasonContext) -> pd.DataFrame:
    """The two quarterback ball security traits per passer and game, in the trait frame contract.

    Only games in which the passer has a competitive dropback in the credit frame
    get a row, the games the other passing traits see, so the box score adds no
    game to a player's count and no player to the snapshots.
    """
    qbs = position_ids(ctx.positions, "QB")
    rows = offense_rows(ctx)
    passed = rows.loc[
        rows["role"].eq("passer") & rows["athlete_id"].isin(qbs),
        ["athlete_id", "game_id"],
    ].drop_duplicates()
    frames = pd.concat(
        [_qb_ball_security(ctx, qbs), _qb_sack_ball_security(ctx, qbs)],
        ignore_index=True,
    )
    keys = passed.astype({"athlete_id": str, "game_id": "int64"})
    return frames.merge(
        keys, on=["athlete_id", "game_id"], how="inner", validate="many_to_one"
    )[frames.columns]
