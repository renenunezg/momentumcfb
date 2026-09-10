"""Explicit pregame quarterback availability as a documented adjustment.

CFBD carries no availability data and play-by-play text must never be read
as an injury signal, so the only sanctioned source is the ``cfb.qb_availability``
table: one row per season, week and team naming the starting quarterback's
status, its source and the time it was reported. Only rows reported before the
forecast or decision cutoff count, so a frozen forecast can always be replayed.

The adjustment sizes are fixed documented assumptions, not fitted parameters.
On the 2020 through 2025 historical-carryover walk-forward (3,982 FBS games
with a closing spread) a team whose previous starter took no snaps was priced
1.5 points lower by the closing market and finished 1.5 points below the
model; in bowl games a new starter moved the close 4.4 points. Outcomes agree
with the market on the regular-season size and are too few to size the
postseason separately, so the market's reaction is used for both.
"""

import numpy as np
import pandas as pd

from backend.etl.store import RAW_DIR

QB_OUT_POINTS = 1.5
QB_OUT_POSTSEASON_POINTS = 4.0
ADJUSTING_STATUSES = ("out", "doubtful")
AVAILABILITY_COLUMNS = ["home_qb_out", "away_qb_out", "qb_availability_points"]


def pregame_qb_outs(
    availability: pd.DataFrame | None, season: int, week: int, as_of
) -> set[str]:
    """Teams whose starting quarterback was reported out or doubtful for the
    week before ``as_of``."""
    if availability is None or availability.empty:
        return set()
    reported = pd.to_datetime(availability["reported_at"], utc=True)
    rows = availability[
        availability["season"].eq(season)
        & availability["week"].eq(week)
        & availability["status"].str.lower().isin(ADJUSTING_STATUSES)
        & reported.lt(pd.Timestamp(as_of))
    ]
    return set(rows["team"])


def apply_qb_availability(
    projections: pd.DataFrame,
    outs: set[str],
    postseason_game_ids: set[int] = frozenset(),
) -> pd.DataFrame:
    """Lower each affected team's expected points and every derived margin.

    Pure margins move by the full net adjustment. A market-informed margin
    moves only by its model share because the market already priced the
    absence. A team already flagged on the frame (a published forecast being
    re-decided) is not adjusted twice. The columns in AVAILABILITY_COLUMNS
    are always present so the published record shows when nothing applied.
    """
    out = projections.copy()
    flagged = {}
    fresh = {}
    for side in ("home", "away"):
        column = f"{side}_qb_out"
        flagged[side] = (
            out[column].fillna(False).astype(bool)
            if column in out
            else pd.Series(False, index=out.index)
        )
        fresh[side] = out[f"{side}_team"].isin(outs) & ~flagged[side]
        out[column] = flagged[side] | fresh[side]
    points = np.where(
        out["game_id"].isin(postseason_game_ids),
        QB_OUT_POSTSEASON_POINTS,
        QB_OUT_POINTS,
    )
    home_loss = np.where(fresh["home"], points, 0.0)
    away_loss = np.where(fresh["away"], points, 0.0)
    net = away_loss - home_loss
    previous = (
        pd.to_numeric(out["qb_availability_points"], errors="coerce").fillna(0.0)
        if "qb_availability_points" in out
        else 0.0
    )
    out["qb_availability_points"] = previous + net
    if not (fresh["home"].any() or fresh["away"].any()):
        return out
    out["expected_home_points"] = (out["expected_home_points"] - home_loss).clip(
        lower=0.0
    )
    out["expected_away_points"] = (out["expected_away_points"] - away_loss).clip(
        lower=0.0
    )
    for margin, spread in (
        ("home_margin", "home_spread"),
        ("pure_home_margin", "pure_home_spread"),
    ):
        if margin in out:
            out[margin] = out[margin] + net
            out[spread] = -out[margin]
    if "market_informed_home_margin" in out:
        share = 1.0 - pd.to_numeric(out["market_weight"], errors="coerce").fillna(0.0)
        out["market_informed_home_margin"] = (
            out["market_informed_home_margin"] + share * net
        )
        out["market_informed_home_spread"] = -out["market_informed_home_margin"]
    out["model_total"] = out["model_total"] - home_loss - away_loss
    return out


def observed_starters(season: int) -> pd.DataFrame:
    """Each team's most recent starting quarterback from completed box scores.

    The starter is the passer with the most attempts in the game. This is an
    observed fact about played games, never an availability inference: it
    exists to point at teams whose last starter differs from their usual one
    so a report can be checked and entered with its source.
    """
    files = sorted((RAW_DIR / "players" / str(season)).glob("box_*.parquet"))
    if not files:
        return pd.DataFrame(
            columns=[
                "team",
                "last_game_id",
                "last_starter",
                "usual_starter",
                "starts",
                "changed",
            ]
        )
    box = pd.concat(pd.read_parquet(path) for path in files)
    passing = box[box["category"].eq("passing") & box["stat_name"].eq("C/ATT")].copy()
    passing["attempts"] = pd.to_numeric(
        passing["stat"].astype(str).str.split("/").str[-1], errors="coerce"
    ).fillna(0)
    starters = (
        passing.sort_values("attempts", ascending=False)
        .drop_duplicates(["game_id", "team"])
        .sort_values(["season_type", "week", "game_id"])
    )
    usual = (
        starters.groupby(["team", "athlete_name"]).size().rename("starts").reset_index()
    )
    usual = usual.sort_values("starts", ascending=False).drop_duplicates("team")
    last = starters.drop_duplicates("team", keep="last")[
        ["team", "game_id", "athlete_name"]
    ]
    out = last.rename(
        columns={"game_id": "last_game_id", "athlete_name": "last_starter"}
    ).merge(usual.rename(columns={"athlete_name": "usual_starter"}), on="team")
    out["changed"] = out["last_starter"].ne(out["usual_starter"])
    return out.sort_values(["changed", "team"], ascending=[False, True]).reset_index(
        drop=True
    )
