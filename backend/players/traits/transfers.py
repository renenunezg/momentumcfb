"""Portal transfers matched to rosters, and how well each trait travels.

A portal entry carries only a name and two schools, so it is tied to an athlete
id by finding the same name on the destination school's roster in the portal
season and on the origin school's roster the season before. Both lookups must
return exactly one player and the two ids must agree; anything else is dropped
and counted rather than guessed.

``portability`` then asks, trait by trait, whether last season's value predicts
this season's as well for players who changed schools as for those who stayed.
The ratio of the two slopes is the share of the trait that travels.
"""

import logging

import numpy as np
import pandas as pd

from backend.players.traits.targets import athlete_ids, normalize_name

log = logging.getLogger(__name__)

TRANSFER_COLUMNS = ["athlete_id", "season", "origin", "destination"]
PORTABILITY_COLUMNS = [
    "position_group",
    "trait",
    "stayers",
    "movers",
    "stay_slope",
    "move_slope",
    "move_slope_se",
    "portable_share",
]
DROP_REASONS = (
    "no_destination_match",
    "ambiguous_destination",
    "no_origin_match",
    "ambiguous_origin",
    "id_mismatch",
)
MIN_MOVERS = 20


def _full_name(frame: pd.DataFrame) -> pd.Series:
    names = (
        frame["first_name"].fillna("").astype(str)
        + " "
        + frame["last_name"].fillna("").astype(str)
    )
    return names.map(normalize_name)


def _roster_lookup(roster: pd.DataFrame, team_column: str) -> pd.DataFrame:
    """One row per (name, team): how many distinct ids share it, and the first."""
    keyed = pd.DataFrame(
        {
            "key": _full_name(roster),
            team_column: roster["team"],
            "athlete_id": athlete_ids(roster["id"]),
        }
    )
    keyed = keyed[keyed["key"].ne("") & keyed["athlete_id"].notna()]
    return keyed.groupby(["key", team_column], as_index=False)["athlete_id"].agg(
        candidates="nunique", athlete_id="first"
    )


def _empty(dropped: dict[str, int]) -> pd.DataFrame:
    out = pd.DataFrame({column: pd.Series(dtype=object) for column in TRANSFER_COLUMNS})
    out.attrs["dropped"] = dropped
    return out


def match_transfers(
    portal: pd.DataFrame, roster_to: pd.DataFrame, roster_from: pd.DataFrame
) -> pd.DataFrame:
    """Portal rows tied to one athlete id by both rosters.

    ``roster_to`` is the portal season's roster and ``roster_from`` the season
    before. The counts of rows dropped, by reason, are in ``out.attrs["dropped"]``.
    """
    dropped = dict.fromkeys(DROP_REASONS, 0)
    if portal.empty or not {
        "season",
        "first_name",
        "last_name",
        "origin",
        "destination",
    } <= set(portal.columns):
        log.info("transfers matched 0 of 0 portal rows")
        return _empty(dropped)

    rows = portal.assign(key=_full_name(portal)).reset_index(drop=True)
    sides = (
        (
            "destination",
            roster_to,
            "no_destination_match",
            "ambiguous_destination",
            "to_id",
        ),
        ("origin", roster_from, "no_origin_match", "ambiguous_origin", "from_id"),
    )
    for column, roster, missing, ambiguous, id_column in sides:
        lookup = _roster_lookup(roster, column).rename(
            columns={"athlete_id": id_column}
        )
        rows = rows.merge(lookup, on=["key", column], how="left")
        absent = rows["candidates"].isna()
        crowded = rows["candidates"].gt(1)
        dropped[missing] = int(absent.sum())
        dropped[ambiguous] = int(crowded.sum())
        rows = rows[~absent & ~crowded].drop(columns="candidates")

    mismatch = rows["to_id"].ne(rows["from_id"])
    dropped["id_mismatch"] = int(mismatch.sum())
    rows = rows[~mismatch]

    out = (
        rows.rename(columns={"to_id": "athlete_id"})[TRANSFER_COLUMNS]
        .astype({"athlete_id": object})
        .drop_duplicates()
        .reset_index(drop=True)
    )
    out.attrs["dropped"] = dropped
    log.info(
        "transfers matched %d of %d portal rows; dropped %s",
        len(out),
        len(portal),
        dropped,
    )
    return out


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope of ``y`` on ``x`` with an intercept, and the slope's standard error."""
    if len(x) < 2:
        return np.nan, np.nan
    dx = x - x.mean()
    sxx = float(dx @ dx)
    if sxx <= 0:
        return np.nan, np.nan
    slope = float(dx @ (y - y.mean())) / sxx
    if len(x) < 3:
        return slope, np.nan
    resid = (y - y.mean()) - slope * dx
    return slope, float(np.sqrt(float(resid @ resid) / (len(x) - 2) / sxx))


def _final_week(frame: pd.DataFrame, final_week_only: bool) -> pd.DataFrame:
    if final_week_only and not frame.empty:
        frame = frame[frame["week"].eq(frame["week"].max())]
    frame = frame[frame["qualified"].astype(bool)]
    return frame[["athlete_id", "trait", "week", "position_group", "value"]]


def portability(
    snapshots_by_season: dict[int, pd.DataFrame],
    transfers: pd.DataFrame,
    final_week_only: bool = True,
) -> pd.DataFrame:
    """Stayer and mover year-over-year slopes per position group and trait.

    Consecutive seasons ``(t-1, t)`` are pooled into one regression of the
    season-``t`` value on the season-``(t-1)`` value. A player is a mover when
    ``transfers`` lists them for season ``t``. With ``final_week_only`` each
    season contributes its last week; with ``False`` every week is used and a
    week-``w`` value in ``t-1`` is paired only with the week-``w`` value in ``t``.
    Pairs where either value is NaN are dropped. ``portable_share`` is the mover
    slope over the stayer slope, NaN when the stayer slope is not positive or
    fewer than ``MIN_MOVERS`` players moved.
    """
    pairs = []
    seasons = sorted(snapshots_by_season)
    for prior, season in zip(seasons, seasons[1:]):
        if season - prior != 1:
            continue
        before = _final_week(snapshots_by_season[prior], final_week_only)
        after = _final_week(snapshots_by_season[season], final_week_only)
        keys = ["athlete_id", "trait"] + ([] if final_week_only else ["week"])
        both = before.merge(after, on=keys, suffixes=("_before", ""))
        both = both.dropna(subset=["value_before", "value"])
        moved = (
            set(transfers.loc[transfers["season"].eq(season), "athlete_id"])
            if len(transfers)
            else set()
        )
        pairs.append(both.assign(mover=both["athlete_id"].isin(moved)))
    if not pairs:
        return pd.DataFrame(columns=PORTABILITY_COLUMNS)

    pooled = pd.concat(pairs, ignore_index=True)
    rows = []
    for (group, trait), part in pooled.groupby(["position_group", "trait"], sort=True):
        stay, move = part[~part["mover"]], part[part["mover"]]
        stay_slope, _ = _ols(
            stay["value_before"].to_numpy(float), stay["value"].to_numpy(float)
        )
        move_slope, move_se = _ols(
            move["value_before"].to_numpy(float), move["value"].to_numpy(float)
        )
        usable = len(move) >= MIN_MOVERS and stay_slope > 0
        rows.append(
            {
                "position_group": group,
                "trait": trait,
                "stayers": len(stay),
                "movers": len(move),
                "stay_slope": stay_slope,
                "move_slope": move_slope,
                "move_slope_se": move_se,
                "portable_share": move_slope / stay_slope if usable else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=PORTABILITY_COLUMNS)
