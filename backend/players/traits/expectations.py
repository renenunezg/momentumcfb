"""Situation-based expectations fitted on earlier seasons.

A player's rushing yards, conversions, and field goals are judged against what
an average play in the same situation produced. The tables are fit only on
seasons before the one being rated, so a season never sets its own baseline.
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

DOWN_KEYS = ["down_bucket", "distance_bucket", "field_bucket"]
FG_KEYS = ["fg_bucket"]
FIELD_GOAL_TYPES = frozenset({
    "Field Goal Good",
    "Field Goal Missed",
    "Blocked Field Goal",
    "Missed Field Goal Return",
    "Blocked Field Goal Touchdown",
})
_FG_DISTANCE = re.compile(r"(\d+)\s*(?:yd|yard)s?\s*(?:FG|field goal)", re.IGNORECASE)


@dataclass
class ExpectationTables:
    rush_yards: pd.DataFrame
    conversion: pd.DataFrame
    field_goal: pd.DataFrame
    source_seasons: list[int]


def _fg_distance(play_text: object) -> float:
    match = _FG_DISTANCE.search(play_text) if isinstance(play_text, str) else None
    return float(match.group(1)) if match else np.nan


def bucket(plays: pd.DataFrame) -> pd.DataFrame:
    """A copy of ``plays`` with the down, distance, field, and field goal buckets."""
    out = plays.copy()
    down = pd.to_numeric(out["down"], errors="coerce")
    distance = pd.to_numeric(out["distance"], errors="coerce")
    yards_to_goal = pd.to_numeric(out["yards_to_goal"], errors="coerce")
    out["down_bucket"] = down.where(down.isin([1, 2, 3, 4])).astype("Int64")
    out["distance_bucket"] = np.select(
        [distance.le(2), distance.le(6), distance.le(10), distance.gt(10)],
        ["1-2", "3-6", "7-10", "11+"],
        default=None,
    )
    out["field_bucket"] = np.select(
        [yards_to_goal.ge(80), yards_to_goal.ge(40), yards_to_goal.ge(20), yards_to_goal.lt(20)],
        ["own", "mid", "plus", "red"],
        default=None,
    )
    if "play_text" in out:
        feet = out["play_text"].map(_fg_distance)
        out["fg_bucket"] = (5 * (feet // 5)).astype("Int64")
    return out


def _table(frame: pd.DataFrame, keys: list[str], value: pd.Series) -> pd.DataFrame:
    grouped = value.groupby([frame[key] for key in keys], observed=True)
    table = grouped.agg(expected="mean", plays="size").reset_index()
    table.columns = [*keys, "expected", "plays"]
    return table


def fit_expectation_tables(
    plays_by_season: dict[int, pd.DataFrame], before_season: int
) -> ExpectationTables:
    seasons = sorted(season for season in plays_by_season if season < before_season)
    if not seasons:
        raise ValueError(f"no season before {before_season} to fit expectations on")
    plays = bucket(pd.concat([plays_by_season[season] for season in seasons], ignore_index=True))

    rushes = plays[plays["is_rush"].astype(bool) & plays["is_competitive"].astype(bool)]
    scrimmage = plays[plays["is_competitive"].astype(bool)]
    kicks = plays[plays["play_type"].isin(FIELD_GOAL_TYPES) & plays["fg_bucket"].notna()]
    return ExpectationTables(
        rush_yards=_table(rushes, DOWN_KEYS, rushes["yards_gained"].astype(float)),
        conversion=_table(scrimmage, DOWN_KEYS, scrimmage["is_success"].astype(float)),
        field_goal=_table(kicks, FG_KEYS, kicks["play_type"].eq("Field Goal Good").astype(float)),
        source_seasons=seasons,
    )


def _lookup(table: pd.DataFrame, plays: pd.DataFrame, keys: list[str]) -> pd.Series:
    """The table's mean for each play's bucket; the table's global mean where unseen."""
    if not set(keys) <= set(plays.columns):
        plays = bucket(plays)
    merged = plays[keys].reset_index(drop=True).merge(
        table[[*keys, "expected"]], on=keys, how="left"
    )
    mean = (table["expected"] * table["plays"]).sum() / table["plays"].sum() if len(table) else np.nan
    return pd.Series(merged["expected"].fillna(mean).to_numpy(float), index=plays.index)


def expected_rush_yards(tables: ExpectationTables, plays: pd.DataFrame) -> pd.Series:
    return _lookup(tables.rush_yards, plays, DOWN_KEYS)


def expected_conversion(tables: ExpectationTables, plays: pd.DataFrame) -> pd.Series:
    return _lookup(tables.conversion, plays, DOWN_KEYS)


def expected_field_goal(tables: ExpectationTables, plays: pd.DataFrame) -> pd.Series:
    return _lookup(tables.field_goal, plays, FG_KEYS)
