"""Outcome-free serving anchor loading.

The streaming serving path scores games from pregame anchor projections in
the serving contract shape (game_id, model_week, home_margin, margin_sd).
Historical replays anchor on the calibration backtest table, which only
exists for completed seasons; forward-looking serving anchors come from a
stored projection artifact (preseason projections now, weekly ``fit``
projections later). Every source emits exactly the contract columns and
validates them at load time: one row per game_id, finite home_margin,
positive finite margin_sd. Artifact reads are column-restricted, so no
outcome column is ever read even when one is stored alongside the anchors.
"""

import numpy as np
import pandas as pd

from backend.etl import store
from backend.model.ingame import SERVING_ANCHOR_COLUMNS

CALIBRATION_SOURCE = "calibration"
PRESEASON_SOURCE = "preseason"
SERVING_SOURCE = "serving"
ANCHOR_SOURCES = (CALIBRATION_SOURCE, PRESEASON_SOURCE, SERVING_SOURCE)

CALIBRATION_ARTIFACT = ("calibration", "joint_scoring_predictions.parquet")


def serving_anchor_artifact(season: int, week: int) -> tuple[str, str]:
    """Processed-store location of a stored serving anchor artifact."""
    return ("serving", f"anchors_{season}_{week:02d}.parquet")


def load_serving_anchors(
    source: str = CALIBRATION_SOURCE,
    *,
    season: int,
    week: int | None = None,
) -> pd.DataFrame:
    """Load pregame serving anchors from the selected source.

    ``calibration`` anchors a completed season's replay on the backtest rows
    it was originally scored with and covers the whole season, so ``week``
    must be omitted. ``preseason`` and ``serving`` read one stored projection
    artifact and require the projection ``week`` that names it.
    """
    if source == CALIBRATION_SOURCE:
        if week is not None:
            raise ValueError(
                "calibration anchors cover a whole season; do not pass week"
            )
        return _calibration_anchors(season)
    if source not in ANCHOR_SOURCES:
        raise ValueError(
            f"unknown anchor source {source!r}; expected one of "
            + ", ".join(ANCHOR_SOURCES)
        )
    if week is None:
        raise ValueError(f"anchor source {source!r} requires a projection week")
    if source == PRESEASON_SOURCE:
        parts = ("preseason", "projections", f"{season}_{week:02d}.parquet")
    else:
        parts = serving_anchor_artifact(season, week)
    return _projection_anchors(parts, season=season, week=week)


def _calibration_anchors(season: int) -> pd.DataFrame:
    location = "/".join(CALIBRATION_ARTIFACT)
    needed = ["season", *SERVING_ANCHOR_COLUMNS]
    _require_columns(location, store.processed_columns(*CALIBRATION_ARTIFACT), needed)
    table = store.read_processed(*CALIBRATION_ARTIFACT, columns=needed)
    anchors = table[table["season"].eq(season)]
    if anchors.empty:
        raise ValueError(f"no pregame anchors for season {season} in {location}")
    return _validated(anchors, f"{location} season {season}")


def _projection_anchors(
    parts: tuple[str, ...], *, season: int, week: int
) -> pd.DataFrame:
    location = "/".join(parts)
    available = store.processed_columns(*parts)
    week_column = "model_week" if "model_week" in available else "week"
    needed = ["game_id", week_column, "home_margin", "margin_sd"]
    if "season" in available:
        needed.append("season")
    _require_columns(location, available, needed)
    frame = store.read_processed(*parts, columns=needed)
    if frame.empty:
        raise ValueError(f"{location} contains no projections")
    if "season" in needed and not frame["season"].eq(season).all():
        raise ValueError(f"{location} contains seasons other than {season}")
    if week_column == "week":
        # A projection artifact records the schedule week it targets; the
        # serving contract's model_week is derived from that projection week.
        # Regular-season weeks map one-to-one; postseason projections need
        # their own offset mapping and have no serving source yet.
        if not frame["week"].eq(week).all():
            raise ValueError(f"{location} targets weeks other than {week}")
        frame = frame.rename(columns={"week": "model_week"})
    return _validated(frame, location)


def _require_columns(
    location: str, available: list[str], needed: list[str]
) -> None:
    missing = sorted(set(needed) - set(available))
    if missing:
        raise ValueError(
            f"{location} is missing anchor columns: {', '.join(missing)}"
        )


def _validated(anchors: pd.DataFrame, origin: str) -> pd.DataFrame:
    """Enforce the serving anchor contract on one loaded source."""
    duplicated = anchors["game_id"].duplicated()
    if duplicated.any():
        raise ValueError(
            f"{origin}: expected one projection per game_id; "
            f"{int(duplicated.sum())} duplicated"
        )
    model_week = anchors["model_week"].to_numpy(float)
    if not (np.isfinite(model_week) & (model_week >= 1)).all():
        raise ValueError(f"{origin}: model_week must be a chronological week >= 1")
    if not np.isfinite(anchors["home_margin"].to_numpy(float)).all():
        raise ValueError(f"{origin}: home_margin must be finite")
    margin_sd = anchors["margin_sd"].to_numpy(float)
    if not (np.isfinite(margin_sd) & (margin_sd > 0)).all():
        raise ValueError(f"{origin}: margin_sd must be positive and finite")
    out = anchors[SERVING_ANCHOR_COLUMNS].reset_index(drop=True)
    out["model_week"] = out["model_week"].astype(int)
    return out
