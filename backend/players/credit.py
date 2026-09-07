"""Split each play's EPA among the players the stat feed credits on it.

Shares are fixed, documented assumptions rather than fitted values: there is
no ground truth for who was responsible on a play, so no dataset can fit
them directly. They live in one table so a future tuning run against the
split-half reliability diagnostic can change them without touching code.
Offensive shares sum to at most one, with the remainder attributed to the
unit. Defensive credit exists only on the disruption plays the feed tags
(sacks, interceptions, pass breakups, forced and recovered fumbles); the
feed never tags tacklers, so every other defensive play belongs to the unit.
"""

import numpy as np
import pandas as pd

# (play kind, stat type) -> (role, share of the play's EPA). A play's kind is
# the highest-precedence kind among its stat rows; rows that do not belong to
# that kind carry no credit, so a fumbled reception charges only the fumbler.
OFFENSE_SHARES: dict[tuple[str, str], tuple[str, float]] = {
    ("fumble", "Fumble"): ("fumbler", 1.0),
    ("interception", "Interception Thrown"): ("passer", 0.8),
    ("sack", "Sack Taken"): ("passer", 0.5),
    ("completion", "Completion"): ("passer", 0.6),
    ("completion", "Reception"): ("receiver", 0.4),
    ("incompletion", "Incompletion"): ("passer", 0.7),
    ("incompletion", "Target"): ("target", 0.3),
    ("rush", "Rush"): ("rusher", 0.7),
    ("return", "Kickoff Return"): ("returner", 1.0),
    ("return", "Punt Return"): ("returner", 1.0),
}

OFFENSE_KIND_PRECEDENCE = (
    "fumble",
    "interception",
    "sack",
    "completion",
    "incompletion",
    "rush",
    "return",
)

# Defensive stat type -> role. Every tagged defender on a play shares the
# whole defensive EPA equally, so two players on a sack take half each.
DEFENSE_ROLES: dict[str, str] = {
    "Sack": "sacker",
    "Interception": "interceptor",
    "Pass Breakup": "pass_defender",
    "Fumble Forced": "fumble_forcer",
    "Fumble Recovered": "fumble_recoverer",
}

CREDIT_COLUMNS = [
    "season",
    "week",
    "season_type",
    "game_id",
    "play_id",
    "athlete_id",
    "athlete_name",
    "team",
    "opponent",
    "side",
    "role",
    "share",
    "channel",
    "epa",
    "credit_epa",
    "is_competitive",
]

_KIND_BY_STAT = {
    stat: kind for (kind, stat), _ in OFFENSE_SHARES.items() if kind != "return"
}
_KIND_BY_STAT.update({"Kickoff Return": "return", "Punt Return": "return"})


def _offense_credit(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows[rows["stat_type"].isin(_KIND_BY_STAT)].copy()
    if rows.empty:
        return rows.assign(role=pd.Series(dtype=str), share=pd.Series(dtype=float))
    rows["row_kind"] = rows["stat_type"].map(_KIND_BY_STAT)
    rank = {kind: index for index, kind in enumerate(OFFENSE_KIND_PRECEDENCE)}
    rows["kind_rank"] = rows["row_kind"].map(rank)
    rows["play_kind_rank"] = rows.groupby("play_id")["kind_rank"].transform("min")
    rows = rows[rows["kind_rank"].eq(rows["play_kind_rank"])].copy()
    keys = list(zip(rows["row_kind"], rows["stat_type"]))
    rows["role"] = [OFFENSE_SHARES[key][0] for key in keys]
    rows["share"] = [OFFENSE_SHARES[key][1] for key in keys]
    # A lateral or a duplicate feed row can repeat one role on one play; the
    # role's share is split rather than paid twice.
    duplicates = rows.groupby(["play_id", "role"])["athlete_id"].transform("size")
    rows["share"] = rows["share"] / duplicates
    return rows


def _defense_credit(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows[rows["stat_type"].isin(DEFENSE_ROLES)].copy()
    if rows.empty:
        return rows.assign(role=pd.Series(dtype=str), share=pd.Series(dtype=float))
    rows["role"] = rows["stat_type"].map(DEFENSE_ROLES)
    rows["share"] = 1.0 / rows.groupby("play_id")["athlete_id"].transform("size")
    return rows


def assign_play_credit(play_stats: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """Return one credited row per (play, player) with the share applied.

    ``plays`` is the classified play frame (``classify_plays``) and provides
    the EPA, the competitive flag, and the rush or pass channel.
    ``play_stats`` is the raw CFBD per-play stat feed for the same games.
    """
    play_columns = plays[
        [
            "id",
            "season",
            "week",
            "season_type",
            "game_id",
            "offense",
            "defense",
            "epa",
            "is_competitive",
            "is_rush",
            "is_pass",
        ]
    ].rename(columns={"id": "play_id"})
    stats = play_stats[
        ["play_id", "athlete_id", "athlete_name", "team", "stat_type", "stat"]
    ].copy()
    stats["play_id"] = stats["play_id"].astype(str)
    play_columns["play_id"] = play_columns["play_id"].astype(str)
    merged = stats.merge(play_columns, on="play_id", how="inner")
    merged = merged[merged["epa"].notna()]
    merged["side"] = np.select(
        [merged["team"].eq(merged["offense"]), merged["team"].eq(merged["defense"])],
        ["offense", "defense"],
        default="",
    )
    merged = merged[merged["side"].ne("")]

    offense = _offense_credit(merged[merged["side"].eq("offense")])
    defense = _defense_credit(merged[merged["side"].eq("defense")])
    credit = pd.concat([offense, defense], ignore_index=True)
    if credit.empty:
        return pd.DataFrame(columns=CREDIT_COLUMNS)

    sign = np.where(credit["side"].eq("offense"), 1.0, -1.0)
    credit["credit_epa"] = credit["share"] * credit["epa"] * sign
    credit["opponent"] = np.where(
        credit["side"].eq("offense"), credit["defense"], credit["offense"]
    )
    credit["channel"] = np.select(
        [credit["is_rush"], credit["is_pass"]], ["rush", "pass"], default="other"
    )
    credit["athlete_id"] = credit["athlete_id"].astype(str)
    return credit[CREDIT_COLUMNS].reset_index(drop=True)
