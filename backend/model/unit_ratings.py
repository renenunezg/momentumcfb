"""Opponent-adjusted descriptive unit ratings for college football.

The ratings are companions to the joint-scoring engine and never feed it.
Every channel is expressed as PPA per game above an average FBS team. The run
block channel converts adjusted line yards with a training-window relationship
to rushing PPA. Pass block and run block are public-data proxies, not isolated
line grades.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backend.model.joint_scoring import _solve_ridge, _team_catalog

MODEL_VERSION = "cfb_unit_ratings_v1"
CHANNEL_PRIOR_SD = 3.0

COLUMNS = (
    "rush_offense",
    "pass_offense",
    "rush_defense",
    "pass_defense",
    "pass_block",
    "run_block",
)


def _two_sided_ridge(
    observations: pd.DataFrame,
    teams: list[str],
    classifications: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit value = unit(team) - counter(opponent) with class-aware priors."""
    if observations.empty:
        raise ValueError("unit channel has no eligible observations")

    index = {team: position for position, team in enumerate(teams)}
    n_teams = len(teams)
    design = np.zeros((len(observations), 2 * n_teams))
    for row, record in enumerate(observations.itertuples(index=False)):
        design[row, index[str(record.team)]] = 1.0
        design[row, n_teams + index[str(record.opponent)]] = -1.0

    target = observations["value"].to_numpy(float)
    weights = observations["weight"].to_numpy(float)
    center = float(np.average(target, weights=weights))
    centered_target = target - center
    prior_mean = np.zeros(2 * n_teams)
    prior_sd = np.full(2 * n_teams, CHANNEL_PRIOR_SD)

    initial, _ = _solve_ridge(
        design,
        centered_target,
        weights,
        prior_mean,
        prior_sd,
    )
    fbs = classifications == "fbs"
    fcs = classifications == "fcs"
    if fbs.any() and fcs.any():
        fcs_indices = np.flatnonzero(fcs)
        prior_mean[fcs_indices] = (
            initial[:n_teams][fcs].mean() - initial[:n_teams][fbs].mean()
        )
        prior_mean[n_teams + fcs_indices] = (
            initial[n_teams:][fcs].mean() - initial[n_teams:][fbs].mean()
        )

    parameters, _ = _solve_ridge(
        design,
        centered_target,
        weights,
        prior_mean,
        prior_sd,
    )
    center_mask = fbs if fbs.any() else np.ones(n_teams, dtype=bool)
    unit = parameters[:n_teams]
    counter = parameters[n_teams:]
    unit = unit - unit[center_mask].mean()
    counter = counter - counter[center_mask].mean()
    return unit, counter


def _game_weights(
    game_ids: pd.Series,
    recency_by_game: Mapping[int | str, float] | None,
) -> pd.Series:
    if recency_by_game is None:
        return pd.Series(1.0, index=game_ids.index, dtype=float)

    def lookup(game_id: object) -> float:
        direct = recency_by_game.get(game_id)
        if direct is None:
            direct = recency_by_game.get(str(game_id), 0.0)
        return float(direct)

    return game_ids.map(lookup).astype(float)


def _pass_block_values(frame: pd.DataFrame) -> pd.Series:
    """Sack PPA relative to the weighted per-dropback baseline."""
    valid = frame["pass_plays"].gt(0)
    weighted_dropbacks = frame.loc[valid, "pass_plays"] * frame.loc[
        valid, "weight"
    ]
    denominator = float(weighted_dropbacks.sum())
    if denominator <= 0:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    baseline = float(
        (
            frame.loc[valid, "protection_ppa_allowed"]
            * frame.loc[valid, "weight"]
        ).sum()
        / denominator
    )
    values = (
        frame["protection_ppa_allowed"] - baseline * frame["pass_plays"]
    )
    return values.where(valid)


def _run_block_values(frame: pd.DataFrame) -> pd.Series:
    """Convert adjusted line yards over baseline to PPA-equivalent value."""
    valid = frame["line_yard_carries"].gt(0)
    weighted_carries = frame.loc[valid, "line_yard_carries"] * frame.loc[
        valid, "weight"
    ]
    denominator = float(weighted_carries.sum())
    if denominator <= 0:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    baseline = float(
        (
            frame.loc[valid, "adjusted_line_yards"]
            * frame.loc[valid, "weight"]
        ).sum()
        / denominator
    )
    line_yard_residual = (
        frame["adjusted_line_yards"]
        - baseline * frame["line_yard_carries"]
    )
    weighted_rushes = frame.loc[valid, "rush_plays"] * frame.loc[valid, "weight"]
    rush_denominator = float(weighted_rushes.sum())
    if rush_denominator <= 0:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    rush_baseline = float(
        (frame.loc[valid, "rush_ppa"] * frame.loc[valid, "weight"]).sum()
        / rush_denominator
    )
    rush_ppa_residual = frame["rush_ppa"] - rush_baseline * frame["rush_plays"]
    scale_denominator = float(
        (
            frame.loc[valid, "weight"]
            * np.square(line_yard_residual.loc[valid])
        ).sum()
    )
    if scale_denominator <= 0:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    ppa_per_line_yard = float(
        (
            frame.loc[valid, "weight"]
            * line_yard_residual.loc[valid]
            * rush_ppa_residual.loc[valid]
        ).sum()
        / scale_denominator
    )
    if not np.isfinite(ppa_per_line_yard) or ppa_per_line_yard <= 0:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return (line_yard_residual * ppa_per_line_yard).where(valid)


def _channel_frame(
    window: pd.DataFrame,
    column: str,
    teams: set[str],
) -> pd.DataFrame:
    frame = window[["game_id", "team", "opponent", column, "weight"]].rename(
        columns={column: "value"}
    )
    numeric = ["value", "weight"]
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame[
        frame["team"].isin(teams)
        & frame["opponent"].isin(teams)
        & frame["value"].notna()
        & np.isfinite(frame["value"])
        & frame["weight"].gt(0)
        & np.isfinite(frame["weight"])
    ]
    return frame.reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class UnitRatings:
    season: int
    week: int
    as_of: datetime
    frame: pd.DataFrame

    def to_records(self) -> list[dict[str, object]]:
        records = []
        as_of = self.as_of.astimezone(timezone.utc).isoformat()
        for row in self.frame.itertuples(index=False):
            record: dict[str, object] = {
                "season": self.season,
                "week": self.week,
                "as_of": as_of,
                "model_version": MODEL_VERSION,
                "team_id": int(row.team_id),
                "team": str(row.team),
                "classification": str(row.classification),
            }
            for column in COLUMNS:
                record[column] = float(getattr(row, column))
            records.append(record)
        return records


def fit_unit_ratings(
    unit_games: pd.DataFrame,
    games: pd.DataFrame,
    forecast_week: int,
    as_of: datetime,
    recency_by_game: Mapping[int | str, float] | None = None,
) -> UnitRatings:
    """Fit each unit channel using games strictly before ``forecast_week``."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    if games["season"].nunique() != 1:
        raise ValueError("unit ratings require exactly one season")

    training = games[games["model_week"] < forecast_week].copy()
    if "completed" in training:
        training = training[training["completed"].fillna(False).astype(bool)]
    if training.empty:
        raise ValueError("at least one prior model week is required")
    if "start_date" in training:
        latest_start = pd.to_datetime(training["start_date"], utc=True).max()
        if latest_start.to_pydatetime() >= as_of:
            raise ValueError("training games must start before as_of")

    catalog = _team_catalog(games).sort_values("team_id").reset_index(drop=True)
    catalog["classification"] = catalog["classification"].fillna("").str.lower()
    teams = catalog["team"].astype(str).tolist()
    team_set = set(teams)
    classifications = catalog["classification"].to_numpy(str)

    game_ids = set(training["game_id"])
    window = unit_games[unit_games["game_id"].isin(game_ids)].copy()
    window["weight"] = _game_weights(window["game_id"], recency_by_game)
    window["rush_value"] = window["rush_ppa"].where(
        window["rush_plays"].gt(0)
    )
    window["pass_value"] = window["pass_ppa"].where(
        window["pass_plays"].gt(0)
    )
    window["pass_block_value"] = _pass_block_values(window)
    window["run_block_value"] = _run_block_values(window)

    ratings: dict[str, np.ndarray] = {}
    for name, column in {
        "rush": "rush_value",
        "pass": "pass_value",
        "pass_block": "pass_block_value",
        "run_block": "run_block_value",
    }.items():
        observations = _channel_frame(window, column, team_set)
        unit, counter = _two_sided_ridge(
            observations,
            teams,
            classifications,
        )
        if name == "rush":
            ratings["rush_offense"] = unit
            ratings["rush_defense"] = counter
        elif name == "pass":
            ratings["pass_offense"] = unit
            ratings["pass_defense"] = counter
        else:
            ratings[name] = unit

    frame = catalog[["team_id", "team", "classification"]].copy()
    for column in COLUMNS:
        frame[column] = ratings[column]
    return UnitRatings(
        season=int(games["season"].iloc[0]),
        week=int(forecast_week),
        as_of=as_of,
        frame=frame,
    )
