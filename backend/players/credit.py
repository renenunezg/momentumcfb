"""Split each play's EPA among the players the stat feed credits on it.

There is no ground truth for who was responsible on a play, so the shares
are conventions checked against two things the data can measure: whether
the resulting per-game rates repeat (odd/even split-half and year-to-year on
2020-2023) and whether they agree with the next NFL draft. The 2026-10-26
run showed that reliability alone is degenerate (it zeroes rare costly
events such as interceptions, which repeat poorly per game but are real
value), so the structure is fixed by hand and only its free parts were
fitted: the passer takes one share of every pass attempt whatever its
outcome, and the receiver's share of an incompletion is zero because that
credit repeated worse than noise and the feed's target tagging varied from
19% to 90% of incompletions by season. The passer-versus-receiver level and
the rush share were flat on every criterion and keep their original values.
Offensive shares sum to at most one, with the remainder attributed to the
unit. Defensive credit exists only on the disruption plays the feed tags
(sacks, interceptions, pass breakups, forced and recovered fumbles); the
feed never tags tacklers, so every other defensive play belongs to the unit.

The feed also drops interceptions. Relative to completions it tags
"Interception Thrown" at 1.2-2.0% in 2021-2023 and 3.0-3.9% from 2024, while
play-by-play has picks on about 4.7% of receptions every season; in 2022-2023
two thirds of the picks in games the feed covers carry no stat row at all. An
untagged pick would cost the passer nothing, so in every covered game each
play-by-play interception without a thrown row is charged to the passer that
the play text names, or else to the offense's passer on the nearest pass.
"""

import re

import numpy as np
import pandas as pd

# (play kind, stat type) -> (role, share of the play's EPA). A play's kind is
# the highest-precedence kind among its stat rows; rows that do not belong to
# that kind carry no credit, so a fumbled reception charges only the fumbler.
OFFENSE_SHARES: dict[tuple[str, str], tuple[str, float]] = {
    ("fumble", "Fumble"): ("fumbler", 1.0),
    ("interception", "Interception Thrown"): ("passer", 0.6),
    ("sack", "Sack Taken"): ("passer", 0.5),
    ("completion", "Completion"): ("passer", 0.6),
    ("completion", "Reception"): ("receiver", 0.4),
    ("incompletion", "Incompletion"): ("passer", 0.6),
    # A target row carries the receiver's presence and play count, not credit.
    ("incompletion", "Target"): ("target", 0.0),
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

PASSER_STATS = ("Completion", "Incompletion", "Sack Taken", "Interception Thrown")
_INTERCEPTED_BY = re.compile(r"^\s*(.+?)\s+pass\s+intercepted", re.IGNORECASE)
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
STAT_COLUMNS = ["play_id", "athlete_id", "athlete_name", "team", "stat_type", "stat"]


def _name_key(name) -> str:
    letters = re.sub(r"[^a-z ]", "", str(name).lower().replace("-", " "))
    return " ".join(t for t in letters.split() if t not in _NAME_SUFFIXES)


def untagged_interceptions(
    play_stats: pd.DataFrame, plays: pd.DataFrame
) -> pd.DataFrame:
    """Interception Thrown rows for play-by-play picks the feed left untagged.

    Only games the feed covers are filled, so a passer is never charged for a
    pick in a game where none of his other throws carry credit. The passer is
    the offense's passer in that game whose name the play text gives, or the
    offense's passer on the nearest tagged pass when the name does not match.
    """
    empty = pd.DataFrame(columns=STAT_COLUMNS)
    if play_stats.empty or not {"play_type", "play_text"}.issubset(plays.columns):
        return empty
    stats = play_stats[STAT_COLUMNS].assign(play_id=lambda d: d["play_id"].astype(str))
    frame = plays[["id", "game_id", "offense", "play_type", "play_text"]].assign(
        play_id=lambda d: d["id"].astype(str),
        order=lambda d: pd.to_numeric(d["id"], errors="coerce"),
    )
    covered = frame.loc[frame["play_id"].isin(set(stats["play_id"])), "game_id"]
    thrown = set(stats.loc[stats["stat_type"].eq("Interception Thrown"), "play_id"])
    picks = frame[
        frame["play_type"].astype(str).str.contains("Interception")
        & frame["game_id"].isin(set(covered))
        & ~frame["play_id"].isin(thrown)
    ]
    if picks.empty:
        return empty
    passers = stats[stats["stat_type"].isin(PASSER_STATS)].merge(
        frame[["play_id", "game_id", "offense", "order"]], on="play_id"
    )
    passers = passers[passers["team"].eq(passers["offense"])].assign(
        key=lambda d: d["athlete_name"].map(_name_key)
    )
    by_offense = {key: group for key, group in passers.groupby(["game_id", "offense"])}
    rows = []
    for pick in picks.itertuples(index=False):
        candidates = by_offense.get((pick.game_id, pick.offense))
        if candidates is None:
            continue
        named = _INTERCEPTED_BY.match(str(pick.play_text))
        matched = (
            candidates[candidates["key"].eq(_name_key(named.group(1)))]
            if named
            else candidates.iloc[0:0]
        )
        if matched["athlete_id"].nunique() == 1:
            passer = matched.iloc[0]
        else:
            distance = (candidates["order"] - pick.order).abs()
            passer = candidates.loc[distance.idxmin()]
        rows.append(
            {
                "play_id": pick.play_id,
                "athlete_id": passer["athlete_id"],
                "athlete_name": passer["athlete_name"],
                "team": pick.offense,
                "stat_type": "Interception Thrown",
                "stat": "1",
            }
        )
    return pd.DataFrame(rows, columns=STAT_COLUMNS)


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
    stats = pd.concat(
        [play_stats[STAT_COLUMNS], untagged_interceptions(play_stats, plays)],
        ignore_index=True,
    )
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
