"""Raw season stat lines for the position grade cards (``cfb.player_season_stats``).

One row per published athlete with the season's numbers through the trait
snapshot week, in the key order and scale ``STAT_KEYS`` gives per position
group; a key ending ``*10`` is stored as the value times ten, rounded.

Every number is raw: no opponent adjustment, no FCS weighting, garbage time
included. Box counts come from ESPN's box score through CFBD, play counts and
EPA (CFBD's expected points added) from the play-by-play and the play stat
feed. Two values come from the trait code so they cannot disagree with it:
target share is the ``rec.target_share`` trait's own computation (competitive,
FCS-weighted plays) and rush yards over expected use the frozen expectation
table behind ``rb.yards_over_expected``. ``games`` is the trait rows' count.

CFBD's play-by-play carries no air yards, completion probability, QB hits,
pressures or punt EPA, the season's Total QBR is not ingested, and there is no
coverage data, so those keys are null.
"""

import numpy as np
import pandas as pd

from backend.features.possessions import PASS_TYPES, RUSH_TYPES
from backend.features.units import _adjusted_line_yards
from backend.players.credit import PASSER_STATS, untagged_interceptions
from backend.players.traits.context import SeasonContext
from backend.players.traits.expectations import ExpectationTables, expected_rush_yards
from backend.players.traits.offense import (
    RECEIVER_GROUPS,
    offense_rows,
    position_ids,
    rec_target_share,
    target_rows,
)
from backend.players.traits.punting import PUNT_TYPES, TOUCHBACK_YARDS, punt_plays
from backend.players.traits.snapshots import UNIT_PREFIX
from backend.players.traits.targets import athlete_ids, recover_targets

SMALLINT_MAX = 32767
STAT_LINE_COLUMNS = ["season", "athlete_id", "position_group", "week", "stats"]
EXPLOSIVE_RUSH_YARDS = 10


def _line(*keys: str) -> tuple[tuple[str, int], ...]:
    return tuple(
        (key.removesuffix("*10"), 10 if key.endswith("*10") else 1) for key in keys
    )


DEFENSE_LINE = _line(
    "games", "tackles", "solo", "tfl*10", "sacks*10", "qb_hits", "pressures", "pbu",
    "int", "ff", "epa_credit*10", "cov_targets", "cov_cmp", "cov_yds", "cov_td",
    "cov_int",
)  # fmt: skip
RECEIVER_LINE = _line(
    "games", "targets", "rec", "rec_yds", "rec_td", "yac", "target_share*10",
    "adot*10", "epa_total*10",
)  # fmt: skip
STAT_KEYS: dict[str, tuple[tuple[str, int], ...]] = {
    "QB": _line(
        "games", "att", "cmp", "pass_yds", "pass_td", "int", "sacks", "rush_att",
        "rush_yds", "rush_td", "epa_plays", "epa_total*10", "cpoe*10", "adot*10",
        "espn_qbr*10",
    ),
    "RB": _line(
        "games", "rush_att", "rush_yds", "rush_td", "fumbles_lost", "targets", "rec",
        "rec_yds", "rec_td", "rush_success", "rush_explosive", "ryoe_total*10",
        "rush_epa*10", "epa_total*10",
    ),
    "WR": RECEIVER_LINE,
    "TE": RECEIVER_LINE,
    "DL": DEFENSE_LINE,
    "LB": DEFENSE_LINE,
    "DB": DEFENSE_LINE,
    "OL": _line(
        "games", "rush_att", "stuffs", "line_yds*10", "dropbacks", "sacks_allowed",
        "pressures_allowed",
    ),
    "P": _line(
        "games", "punts", "punt_yds", "net_yds", "inside_20", "touchbacks",
        "punt_epa*10",
    ),
}  # fmt: skip
# Keys the CFB sources cannot fill, and ratios, which have no value without a
# denominator. Every other key is a count or a sum, zero for a player without
# the event.
RATIOS = frozenset({"target_share"})
UNAVAILABLE = frozenset(
    {
        "cpoe", "adot", "espn_qbr", "yac", "qb_hits", "pressures", "cov_targets",
        "cov_cmp", "cov_yds", "cov_td", "cov_int", "pressures_allowed", "punt_epa",
    }
)  # fmt: skip
BOX_STATS = {
    ("passing", "YDS"): "pass_yds",
    ("passing", "TD"): "pass_td",
    ("passing", "INT"): "int_thrown",
    ("rushing", "CAR"): "rush_att",
    ("rushing", "YDS"): "rush_yds",
    ("rushing", "TD"): "rush_td",
    ("receiving", "REC"): "rec",
    ("receiving", "YDS"): "rec_yds",
    ("receiving", "TD"): "rec_td",
    ("fumbles", "LOST"): "fumbles_lost",
    ("defensive", "TOT"): "tackles",
    ("defensive", "SOLO"): "solo",
    ("defensive", "TFL"): "tfl",
    ("defensive", "SACKS"): "sacks_made",
    ("defensive", "PD"): "pbu",
    ("interceptions", "INT"): "int_caught",
    ("punting", "NO"): "punts",
    ("punting", "YDS"): "punt_yds",
    ("punting", "In 20"): "inside_20",
    ("punting", "TB"): "touchbacks",
}


def _box_totals(box: pd.DataFrame) -> pd.DataFrame:
    """Season box totals per athlete, one column per ``BOX_STATS`` value plus ``cmp, att``.

    A row the feed repeats verbatim within a game is counted once.
    """
    box = box.drop_duplicates(
        ["game_id", "team", "athlete_id", "category", "stat_name", "stat_text"]
    )
    keys = list(zip(box["category"], box["stat_name"]))
    rows = box.assign(column=[BOX_STATS.get(key) for key in keys]).dropna(
        subset=["column"]
    )
    totals = rows.pivot_table(
        index="athlete_id", columns="column", values="stat", aggfunc="sum"
    ).reindex(columns=list(BOX_STATS.values()))
    passing = box[box["category"].eq("passing") & box["stat_name"].eq("C/ATT")]
    split = passing["stat_text"].str.extract(r"^(\d+)/(\d+)$").astype(float)
    attempts = (
        split.set_axis(["cmp", "att"], axis=1)
        .assign(athlete_id=passing["athlete_id"])
        .groupby("athlete_id")[["cmp", "att"]]
        .sum()
    )
    out = totals.join(attempts, how="outer")
    out.index = out.index.astype(str)
    return out.fillna(0.0)


def _feed_plays(
    ctx: SeasonContext, play_stats: pd.DataFrame, game_ids: set
) -> pd.DataFrame:
    """Play stat feed rows (with charged untagged picks) on scrimmage plays, with the play.

    Columns: ``play_id, athlete_id, team, stat_type`` plus the play's ``epa``,
    ``yards_gained`` and the situation the expectation tables bucket on.
    """
    stats = pd.concat(
        [play_stats, untagged_interceptions(play_stats, ctx.plays)], ignore_index=True
    )
    stats = stats.assign(
        play_id=stats["play_id"].astype(str),
        athlete_id=athlete_ids(stats["athlete_id"]),
    ).dropna(subset=["athlete_id"])
    stats = stats.astype({"athlete_id": str})
    plays = ctx.plays[ctx.plays["is_scrimmage"] & ctx.plays["game_id"].isin(game_ids)]
    plays = plays[["id", "epa", "yards_gained", "down", "distance", "yards_to_goal"]]
    plays = plays.assign(play_id=plays.pop("id").astype(str))
    rows = stats[["play_id", "athlete_id", "team", "stat_type"]].merge(
        plays, on="play_id", how="inner"
    )
    return rows.drop_duplicates(["play_id", "athlete_id", "stat_type"])


def _per_athlete(rows: pd.DataFrame, **sums: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"athlete_id": rows["athlete_id"], **sums})
    return frame.groupby("athlete_id").sum()


def _quarterbacks(feed: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """Box passing and rushing, sacks taken, and EPA over dropbacks plus runs.

    Every dropback (completion, incompletion, interception, sack) and every
    carry counts: CFBD does not tell a scramble from a designed run, so the two
    are both in. A play without EPA is not one of the ``epa_plays``.
    """
    sacks = feed[feed["stat_type"].eq("Sack Taken")]
    sacks = sacks.drop_duplicates(["play_id", "athlete_id"])
    plays = feed[feed["stat_type"].isin([*PASSER_STATS, "Rush"]) & feed["epa"].notna()]
    plays = plays.drop_duplicates(["play_id", "athlete_id"])
    out = box[["att", "cmp", "pass_yds", "pass_td", "int_thrown"]].rename(
        columns={"int_thrown": "int"}
    )
    out = out.join(box[["rush_att", "rush_yds", "rush_td"]])
    out = out.join(
        _per_athlete(sacks, sacks=pd.Series(1.0, index=sacks.index)), how="outer"
    )
    epa = _per_athlete(
        plays, epa_plays=pd.Series(1.0, index=plays.index), epa_total=plays["epa"]
    )
    return out.join(epa, how="outer")


def _targets(feed: pd.DataFrame, recovered: pd.DataFrame) -> pd.DataFrame:
    """One row per target: a reception, a feed target, or a target recovered from the text."""
    keys = ["play_id", "athlete_id"]
    receptions = feed.loc[feed["stat_type"].eq("Reception"), keys]
    recovered = recovered[keys].astype({"play_id": str, "athlete_id": str})
    targets = pd.concat([receptions, recovered], ignore_index=True).drop_duplicates(
        "play_id"
    )
    plays = feed.drop_duplicates("play_id")[["play_id", "epa"]]
    return targets.merge(plays, on="play_id", how="inner")


def _target_epa(targets: pd.DataFrame) -> pd.DataFrame:
    return _per_athlete(
        targets,
        targets=pd.Series(1.0, index=targets.index),
        target_epa=targets["epa"].fillna(0.0),
    )


def _running_backs(
    feed: pd.DataFrame,
    box: pd.DataFrame,
    targets: pd.DataFrame,
    tables: ExpectationTables,
) -> pd.DataFrame:
    carries = feed[feed["stat_type"].eq("Rush")].drop_duplicates(
        ["play_id", "athlete_id"]
    )
    over = carries["yards_gained"] - expected_rush_yards(tables, carries).to_numpy()
    rushing = _per_athlete(
        carries,
        rush_success=carries["epa"].gt(0).astype(float),
        rush_explosive=carries["yards_gained"].ge(EXPLOSIVE_RUSH_YARDS).astype(float),
        ryoe_total=over.fillna(0.0),
        rush_epa=carries["epa"].fillna(0.0),
    )
    out = box[
        ["rush_att", "rush_yds", "rush_td", "fumbles_lost", "rec", "rec_yds", "rec_td"]
    ].join(rushing, how="outer")
    out = out.join(_target_epa(targets), how="outer").fillna(0.0)
    return out.assign(epa_total=out["rush_epa"] + out.pop("target_epa"))


def _receivers(
    ctx: SeasonContext,
    box: pd.DataFrame,
    targets: pd.DataFrame,
    recovered: pd.DataFrame,
    poor_coverage: bool,
    game_ids: set,
) -> pd.DataFrame:
    """Box receiving, raw targets and their EPA, and the trait's target share in percent."""
    rows = offense_rows(ctx)
    rows = rows[rows["game_id"].isin(game_ids)]
    receivers = set().union(
        *(position_ids(ctx.positions, group) for group in RECEIVER_GROUPS)
    )
    share = rec_target_share(
        rows, target_rows(rows, recovered, poor_coverage), receivers
    )
    sums = share.groupby("athlete_id")[["numerator", "denominator"]].sum()
    percent = (
        100 * sums["numerator"] / sums["denominator"].where(sums["denominator"] > 0)
    )
    out = box[["rec", "rec_yds", "rec_td"]].join(_target_epa(targets), how="outer")
    out = out.fillna(0.0).rename(columns={"target_epa": "epa_total"})
    return out.join(percent.rename("target_share"), how="outer")


def _defenders(
    feed: pd.DataFrame, box: pd.DataFrame, ctx: SeasonContext, game_ids: set
) -> pd.DataFrame:
    """Box defense, forced fumbles from the feed, and v4's EPA credit unadjusted and unweighted.

    The credit is each tagged disruption's share of the play's EPA plus the box
    tackles for loss and passes defended v4 values, on competitive plays.
    """
    forced = feed[feed["stat_type"].eq("Fumble Forced")]
    forced = forced.drop_duplicates(["play_id", "athlete_id"])
    credited = ctx.credited[
        ctx.credited["side"].eq("defense") & ctx.credited["game_id"].isin(game_ids)
    ]
    out = box[["tackles", "solo", "tfl", "sacks_made", "pbu", "int_caught"]].rename(
        columns={"sacks_made": "sacks", "int_caught": "int"}
    )
    out = out.join(
        _per_athlete(forced, ff=pd.Series(1.0, index=forced.index)), how="outer"
    )
    credit = _per_athlete(
        credited.assign(athlete_id=credited["athlete_id"].astype(str)),
        epa_credit=credited["raw_epa"] / credited["weight"],
    )
    return out.join(credit, how="outer")


def _lines(ctx: SeasonContext, game_ids: set) -> pd.DataFrame:
    """Team carries, stuffs, line yards, dropbacks and sacks per ``unit:{team}``."""
    plays = ctx.plays[ctx.plays["is_scrimmage"] & ctx.plays["game_id"].isin(game_ids)]
    carries = plays[plays["play_type"].isin(RUSH_TYPES)]
    measured = carries[carries["yards_gained"].notna()]
    dropbacks = plays[plays["play_type"].isin(PASS_TYPES)]
    out = pd.concat(
        [
            carries.groupby("offense").size().rename("rush_att"),
            measured["yards_gained"]
            .le(0)
            .groupby(measured["offense"])
            .sum()
            .rename("stuffs"),
            _adjusted_line_yards(measured["yards_gained"])
            .groupby(measured["offense"])
            .sum()
            .rename("line_yds"),
            dropbacks.groupby("offense").size().rename("dropbacks"),
            dropbacks["play_type"]
            .eq("Sack")
            .groupby(dropbacks["offense"])
            .sum()
            .rename("sacks_allowed"),
        ],
        axis=1,
    )
    out.index = UNIT_PREFIX + out.index.astype(str)
    return out.astype(float)


def _punters(ctx: SeasonContext, box: pd.DataFrame, game_ids: set) -> pd.DataFrame:
    """Box punting, with net yards the NCAA way: gross less returns less 20 per touchback.

    Return yards come from the punt play text (``punt_plays``), returned punts
    included; it skips blocked punts and punts whose punter it cannot name.
    """
    punts = punt_plays(ctx, PUNT_TYPES | {"Punt Return"})
    punts = punts[punts["game_id"].isin(game_ids)]
    returned = punts.groupby(punts["athlete_id"].astype(str))["returned_yards"].sum()
    out = box[["punts", "punt_yds", "inside_20", "touchbacks"]].join(
        returned, how="outer"
    )
    out = out.fillna(0.0)
    net = (
        out["punt_yds"]
        - out.pop("returned_yards")
        - TOUCHBACK_YARDS * out["touchbacks"]
    )
    return out.assign(net_yds=net)


def encode(line: pd.DataFrame, group: str) -> list[list[int | None]]:
    """Each row's ``STAT_KEYS[group]`` values, scaled and rounded; NaN becomes null.

    A value outside a smallint is a contract break and raises.
    """
    keys = STAT_KEYS[group]
    scaled = np.rint(
        np.column_stack(
            [line[name].to_numpy(dtype=float) * scale for name, scale in keys]
        )
    )
    too_big = np.abs(np.nan_to_num(scaled)) > SMALLINT_MAX
    if too_big.any():
        row, column = np.argwhere(too_big)[0]
        raise ValueError(
            f"{group} {keys[column][0]} of {line.index[row]} is {scaled[row, column]}, "
            "outside a smallint"
        )
    return [
        [None if np.isnan(value) else int(value) for value in row] for row in scaled
    ]


def season_stat_lines(
    ctx: SeasonContext,
    tables: ExpectationTables,
    play_stats: pd.DataFrame,
    published: pd.DataFrame,
    poor_coverage: bool,
) -> pd.DataFrame:
    """One ``STAT_LINE_COLUMNS`` row per athlete in ``published``.

    ``published`` is the snapshot week's published trait rows; they give the
    population, each athlete's group and games, and the week. Only games of
    model weeks through that week count.
    """
    if published.empty:
        return pd.DataFrame(columns=STAT_LINE_COLUMNS)
    week = int(published["week"].max())
    games = ctx.games
    game_ids = set(games.loc[games["model_week"].le(week), "game_id"])
    box = _box_totals(ctx.box[ctx.box["game_id"].isin(game_ids)])
    play_stats = play_stats[play_stats["game_id"].isin(game_ids)]
    feed = _feed_plays(ctx, play_stats, game_ids)
    recovered = recover_targets(ctx.plays, play_stats, ctx.roster)
    targets = _targets(feed, recovered)
    by_group = {
        "QB": _quarterbacks(feed, box),
        "RB": _running_backs(feed, box, targets, tables),
        "WR": _receivers(ctx, box, targets, recovered, poor_coverage, game_ids),
        "DL": _defenders(feed, box, ctx, game_ids),
        "OL": _lines(ctx, game_ids),
        "P": _punters(ctx, box, game_ids),
    }
    for alias, source in (("TE", "WR"), ("LB", "DL"), ("DB", "DL")):
        by_group[alias] = by_group[source]

    people = published.drop_duplicates("athlete_id").astype({"athlete_id": str})
    unknown = set(people["position_group"]) - set(STAT_KEYS)
    if unknown:
        raise ValueError(f"no stat line for position groups {sorted(unknown)}")
    parts = []
    for group, members in people.groupby("position_group"):
        names = [name for name, _ in STAT_KEYS[group]]
        line = by_group[group].reindex(members["athlete_id"])
        counted = [n for n in names if n in line.columns]
        counted = [n for n in counted if n not in UNAVAILABLE | RATIOS]
        line[counted] = line[counted].fillna(0.0)
        line["games"] = members["games"].to_numpy(dtype=float)
        line = line.reindex(columns=names)
        parts.append(
            pd.DataFrame(
                {
                    "season": ctx.season,
                    "athlete_id": members["athlete_id"].to_numpy(),
                    "position_group": group,
                    "week": week,
                    "stats": encode(line, group),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)[STAT_LINE_COLUMNS]
