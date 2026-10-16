"""Offensive trait builders.

Every builder reads the finalized v4 credit frame with the play's situation
attached, keeps competitive plays only (the same plays v4 counts), and weights
each play by the credit row's FCS weight, so numerators and denominators are
FCS-weighted as the trait frame contract requires.

A dropback is a passer credit row: completions, incompletions, sacks taken and
interceptions thrown. A fumble play charges only the fumbler in the credit
frame, so a strip sack is a fumble play and not a dropback.
"""

import numpy as np
import pandas as pd

from backend.players.traits import TraitSpec, register
from backend.players.traits.adjust import adjust_rate
from backend.players.traits.context import SeasonContext
from backend.players.traits.expectations import ExpectationTables, expected_conversion

# A season whose share of incompletions without an identified receiver exceeds
# this is "poor coverage": target-denominated traits count receptions instead.
POOR_TARGET_COVERAGE = 0.5

TRAIT_COLUMNS = ["athlete_id", "model_week", "game_id", "trait", "numerator", "denominator", "kind"]
GAME_KEYS = ["athlete_id", "game_id", "model_week", "team", "opponent"]
PLAY_COLUMNS = [
    "down",
    "distance",
    "yards_to_goal",
    "yards_gained",
    "period",
    "clock_seconds_remaining",
    "is_success",
    "play_type",
    "wpa",
]

LATE_DOWNS = (3, 4)
TWO_MINUTE_PERIODS = (2, 4)
TWO_MINUTE_SECONDS = 120
EXPLOSIVE_PASS_YARDS = 20
FLOOR_MIN_DROPBACKS = 10
PRESSURE_MIN_OTHER_DROPBACKS = 30
RECEIVER_MIN_OTHER_TARGETS = 10

QB_TRAITS = {
    spec.name: register(spec)
    for spec in (
        TraitSpec("qb.drive_engine", ("QB",), "counted"),
        TraitSpec("qb.explosive", ("QB",), "counted"),
        TraitSpec("qb.ball_security", ("QB",), "counted"),
        TraitSpec("qb.floor", ("QB",), "counted", aggregate="quantile20"),
        TraitSpec("qb.legs", ("QB",), "counted"),
        TraitSpec("qb.pressure_handling", ("QB",), "inferred"),
        TraitSpec("qb.receiver_independence", ("QB",), "inferred"),
    )
}


def attach_plays(credit: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """Credit rows with the play's situation; the play's own WPA is ``play_wpa``."""
    situation = plays[["id", *PLAY_COLUMNS]].rename(columns={"wpa": "play_wpa"})
    situation = situation.assign(play_id=situation.pop("id").astype(str))
    out = credit.assign(play_id=credit["play_id"].astype(str)).merge(
        situation, on="play_id", how="left"
    )
    out["is_success"] = out["is_success"].eq(True)
    return out


def offense_rows(ctx: SeasonContext) -> pd.DataFrame:
    """Competitive offensive credit rows with the play's situation attached."""
    credit = ctx.credit
    keep = credit["side"].eq("offense") & credit["is_competitive"].eq(True)
    rows = attach_plays(credit[keep], ctx.plays)
    return rows.assign(athlete_id=rows["athlete_id"].astype(str))


def position_ids(positions: pd.DataFrame, group: str) -> set[str]:
    return set(positions.loc[positions["position_group"].eq(group), "athlete_id"].astype(str))


def dropbacks(rows: pd.DataFrame) -> pd.DataFrame:
    """Passer rows flagged as sack, interception, incompletion, or completion."""
    out = rows[rows["role"].eq("passer")].copy()
    play_type = out["play_type"].fillna("")
    out["is_sack"] = play_type.eq("Sack")
    out["is_interception"] = play_type.str.contains("Interception")
    out["is_incompletion"] = play_type.eq("Pass Incompletion")
    out["is_completion"] = ~(out["is_sack"] | out["is_interception"] | out["is_incompletion"])
    return out


def is_late_down(rows: pd.DataFrame) -> pd.Series:
    return rows["down"].isin(LATE_DOWNS)


def is_two_minute(rows: pd.DataFrame) -> pd.Series:
    return rows["period"].isin(TWO_MINUTE_PERIODS) & rows["clock_seconds_remaining"].le(
        TWO_MINUTE_SECONDS
    )


def conversion_over_expected(rows: pd.DataFrame, tables: ExpectationTables) -> pd.Series:
    """FCS-weighted ``is_success - expected_conversion`` for each row's situation."""
    expected = expected_conversion(tables, rows)
    return (rows["is_success"].astype(float) - expected) * rows["weight"]


def per_game(rows: pd.DataFrame, numerator: pd.Series, denominator: pd.Series) -> pd.DataFrame:
    """Sum a numerator and denominator per player and game."""
    frame = rows[GAME_KEYS].assign(numerator=numerator, denominator=denominator)
    return frame.groupby(GAME_KEYS, as_index=False)[["numerator", "denominator"]].sum()


def prior_totals(frame: pd.DataFrame, keys: list[str], columns: list[str]) -> pd.DataFrame:
    """Sums of ``columns`` per ``keys`` over model weeks strictly before each week."""
    weekly = (
        frame.groupby([*keys, "model_week"], as_index=False)[columns]
        .sum()
        .sort_values([*keys, "model_week"])
    )
    running = weekly.groupby(keys)[columns].cumsum() if keys else weekly[columns].cumsum()
    weekly[columns] = running - weekly[columns]
    return weekly


def opponent_adjusted(events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """``adjust_rate`` by the defense faced; it takes the week from ``games``."""
    adjusted = adjust_rate(events.drop(columns="model_week"), games)
    return adjusted.assign(model_week=events["model_week"].to_numpy())


def trait_rows(frame: pd.DataFrame, spec: TraitSpec) -> pd.DataFrame:
    """Rows in the trait frame contract for one trait."""
    out = frame.assign(trait=spec.name, kind=spec.kind)
    out = out.astype({"numerator": float, "denominator": float})
    return out[TRAIT_COLUMNS]


def play_receivers(
    rows: pd.DataFrame, recovered_targets: pd.DataFrame, poor_coverage: bool
) -> pd.DataFrame:
    """One receiver per play: ``play_id, receiver_id``.

    A target is a credit ``receiver`` or ``target`` row, else a recovered
    target. Under poor coverage only receptions (``receiver`` rows) count.
    """
    roles = ["receiver"] if poor_coverage else ["receiver", "target"]
    credited = rows.loc[rows["role"].isin(roles), ["play_id", "athlete_id"]]
    sources = [credited]
    if not poor_coverage:
        sources.append(
            recovered_targets[["play_id", "athlete_id"]].astype({"play_id": str, "athlete_id": str})
        )
    receivers = pd.concat(sources, ignore_index=True).drop_duplicates("play_id")
    return receivers.rename(columns={"athlete_id": "receiver_id"})


def _qb_drive_engine(qb_rows: pd.DataFrame, tables: ExpectationTables, games: pd.DataFrame) -> pd.DataFrame:
    plays = qb_rows[qb_rows["role"].isin(["passer", "rusher"])]
    plays = plays[is_late_down(plays) | is_two_minute(plays)]
    events = per_game(plays, conversion_over_expected(plays, tables), plays["weight"])
    return trait_rows(opponent_adjusted(events, games), QB_TRAITS["qb.drive_engine"])


def _qb_explosive(qb_dropbacks: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    explosive = qb_dropbacks["is_completion"] & qb_dropbacks["yards_gained"].ge(EXPLOSIVE_PASS_YARDS)
    events = per_game(qb_dropbacks, explosive * qb_dropbacks["weight"], qb_dropbacks["weight"])
    return trait_rows(opponent_adjusted(events, games), QB_TRAITS["qb.explosive"])


def _qb_ball_security(
    qb_rows: pd.DataFrame, qb_dropbacks: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    """Offense-view WPA of the passer's interceptions and fumbles per dropback.

    Turnovers lose win probability, so the sum is usually negative and a
    clean game scores zero: higher is better. FCS games carry no stored win
    probability and their missing WPA counts as zero.
    """
    interceptions = qb_dropbacks[qb_dropbacks["is_interception"]]
    fumbles = qb_rows[qb_rows["role"].eq("fumbler")]
    turnovers = pd.concat([interceptions, fumbles], ignore_index=True)
    cost = per_game(turnovers, turnovers["play_wpa"].fillna(0.0) * turnovers["weight"], 0.0)
    exposure = per_game(qb_dropbacks, 0.0, qb_dropbacks["weight"])
    events = exposure.drop(columns="numerator").merge(
        cost.drop(columns="denominator"), on=GAME_KEYS, how="left"
    )
    events["numerator"] = events["numerator"].fillna(0.0)
    return trait_rows(opponent_adjusted(events, games), QB_TRAITS["qb.ball_security"])


def _qb_floor(qb_dropbacks: pd.DataFrame) -> pd.DataFrame:
    games = qb_dropbacks.groupby(GAME_KEYS, as_index=False).agg(
        epa=("adjusted_epa", "sum"), weight=("weight", "sum"), count=("play_id", "size")
    )
    per_dropback = games["epa"] / games["weight"].where(games["weight"] > 0)
    games["numerator"] = per_dropback.where(games["count"] >= FLOOR_MIN_DROPBACKS)
    games["denominator"] = 1.0
    return trait_rows(games, QB_TRAITS["qb.floor"])


def _qb_legs(qb_rows: pd.DataFrame, tables: ExpectationTables, games: pd.DataFrame) -> pd.DataFrame:
    rushes = qb_rows[qb_rows["role"].eq("rusher")]
    events = per_game(rushes, conversion_over_expected(rushes, tables), rushes["weight"])
    return trait_rows(opponent_adjusted(events, games), QB_TRAITS["qb.legs"])


def _qb_pressure_handling(
    all_dropbacks: pd.DataFrame, qbs: set[str], games: pd.DataFrame
) -> pd.DataFrame:
    """Sacks avoided against the protection baseline the passer inherited.

    The baseline is the team's sack rate with its other passers through the
    previous week when they have at least 30 dropbacks, else the league rate
    through the previous week. A game with neither (the season's first week)
    gets no row.
    """
    counts = ["dropbacks", "sacks"]
    passers = per_game(
        all_dropbacks, all_dropbacks["is_sack"] * all_dropbacks["weight"], all_dropbacks["weight"]
    ).rename(columns={"numerator": "sacks", "denominator": "dropbacks"})

    def prior(keys: list[str], prefix: str) -> pd.DataFrame:
        return prior_totals(passers, keys, counts).rename(columns={c: f"{prefix}_{c}" for c in counts})

    merged = (
        passers.merge(prior([], "league"), on="model_week", how="left")
        .merge(prior(["team"], "team"), on=["team", "model_week"], how="left")
        .merge(prior(["team", "athlete_id"], "own"), on=["team", "athlete_id", "model_week"], how="left")
    )
    other_dropbacks = merged["team_dropbacks"] - merged["own_dropbacks"]
    other_sacks = merged["team_sacks"] - merged["own_sacks"]
    league_rate = merged["league_sacks"] / merged["league_dropbacks"].where(merged["league_dropbacks"] > 0)
    baseline = np.where(
        other_dropbacks >= PRESSURE_MIN_OTHER_DROPBACKS, other_sacks / other_dropbacks, league_rate
    )
    merged["numerator"] = merged["dropbacks"] * baseline - merged["sacks"]
    merged["denominator"] = merged["dropbacks"]
    events = merged[merged["athlete_id"].isin(qbs) & merged["numerator"].notna()]
    events = events[[*GAME_KEYS, "numerator", "denominator"]]
    return trait_rows(opponent_adjusted(events, games), QB_TRAITS["qb.pressure_handling"])


def _qb_receiver_independence(
    all_dropbacks: pd.DataFrame,
    receivers: pd.DataFrame,
    qbs: set[str],
) -> pd.DataFrame:
    """Passer EPA on each target above the receiver's rate with other passers.

    The receiver's rate is the passer-row adjusted EPA per target his other
    passers produced through the previous week. Only targets whose receiver has
    at least ten such targets count; a game with none has a NaN numerator.
    """
    throws = all_dropbacks[~all_dropbacks["is_sack"]].merge(receivers, on="play_id", how="inner")
    throws = throws.assign(epa=throws["adjusted_epa"], targets=throws["weight"])
    sums = ["epa", "targets"]
    receiver_prior = prior_totals(throws, ["receiver_id"], sums)
    pair_prior = prior_totals(throws, ["receiver_id", "athlete_id"], sums)
    qb_throws = throws[throws["athlete_id"].isin(qbs)]
    merged = qb_throws.merge(
        receiver_prior, on=["receiver_id", "model_week"], how="left", suffixes=("", "_receiver")
    ).merge(
        pair_prior, on=["receiver_id", "athlete_id", "model_week"], how="left", suffixes=("", "_pair")
    )
    other_targets = merged["targets_receiver"] - merged["targets_pair"]
    other_epa = merged["epa_receiver"] - merged["epa_pair"]
    qualifies = other_targets >= RECEIVER_MIN_OTHER_TARGETS
    rate = other_epa / other_targets.where(qualifies)
    value = (merged["epa"] - merged["weight"] * rate).where(qualifies)
    events = merged[GAME_KEYS].assign(value=value, denominator=merged["weight"].where(qualifies, 0.0))
    events = events.groupby(GAME_KEYS, as_index=False).agg(
        numerator=("value", lambda v: v.sum(min_count=1)), denominator=("denominator", "sum")
    )
    return trait_rows(events, QB_TRAITS["qb.receiver_independence"])


def qb_traits(
    ctx: SeasonContext,
    tables: ExpectationTables,
    recovered_targets: pd.DataFrame,
    *,
    poor_coverage: bool = False,
) -> pd.DataFrame:
    """The seven quarterback traits per player and game, in the trait frame contract.

    ``poor_coverage`` (decided per season by the caller from ``target_coverage``
    against ``POOR_TARGET_COVERAGE``) makes receiver independence use receptions
    instead of targets.
    """
    rows = offense_rows(ctx)
    qbs = position_ids(ctx.positions, "QB")
    qb_rows = rows[rows["athlete_id"].isin(qbs)]
    all_dropbacks = dropbacks(rows)
    qb_dropbacks = all_dropbacks[all_dropbacks["athlete_id"].isin(qbs)]
    receivers = play_receivers(rows, recovered_targets, poor_coverage)
    frames = [
        _qb_drive_engine(qb_rows, tables, ctx.games),
        _qb_explosive(qb_dropbacks, ctx.games),
        _qb_ball_security(qb_rows, qb_dropbacks, ctx.games),
        _qb_floor(qb_dropbacks),
        _qb_legs(qb_rows, tables, ctx.games),
        _qb_pressure_handling(all_dropbacks, qbs, ctx.games),
        _qb_receiver_independence(all_dropbacks, receivers, qbs),
    ]
    return pd.concat(frames, ignore_index=True)
