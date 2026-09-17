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

MODEL_VERSION = "cfb_unit_ratings_v2"
# Prior sd of a unit effect as a share of the channel's game-to-game sd. With
# the fitted noise this shrinks each unit like roughly ten games of evidence.
CHANNEL_PRIOR_SHARE = 0.3

COLUMNS = (
    "rush_offense",
    "pass_offense",
    "rush_defense",
    "pass_defense",
    "pass_block",
    "run_block",
)


def _centered_units(
    parameters: np.ndarray, n_teams: int, fbs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    center_mask = fbs if fbs.any() else np.ones(n_teams, dtype=bool)
    unit = parameters[:n_teams]
    counter = parameters[n_teams:]
    return unit - unit[center_mask].mean(), counter - counter[center_mask].mean()


def _two_sided_ridge(
    observations: pd.DataFrame,
    teams: list[str],
    classifications: np.ndarray,
    prior_units: np.ndarray | None = None,
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
    prior_mean = np.zeros(2 * n_teams) if prior_units is None else prior_units
    # With preseason evidence, estimate the intercept after accounting for
    # schedule strength; a weak opponent is not an average-unit observation.
    center = float(np.average(target - design @ prior_mean, weights=weights))
    centered_target = target - center
    # Game totals are far noisier than team effects. Both the prior width and
    # the observation noise come from the channel itself, so every channel is
    # shrunk on its own scale instead of being fitted as near-exact evidence.
    variance = float(
        np.average(np.square(centered_target - design @ prior_mean), weights=weights)
    )
    if not np.isfinite(variance):
        raise ValueError("unit channel has nonfinite observations")
    if variance <= 0:
        # Every observation already equals its prior expectation.
        return _centered_units(prior_mean, n_teams, classifications == "fbs")
    prior_sd = np.full(2 * n_teams, CHANNEL_PRIOR_SHARE * np.sqrt(variance))

    def solve(noise: float) -> np.ndarray:
        parameters, _ = _solve_ridge(
            design,
            centered_target,
            weights / noise,
            prior_mean,
            prior_sd,
        )
        return parameters

    initial = solve(variance)
    fbs = classifications == "fbs"
    fcs = classifications == "fcs"
    if prior_units is None and fbs.any() and fcs.any():
        fcs_indices = np.flatnonzero(fcs)
        prior_mean[fcs_indices] = (
            initial[:n_teams][fcs].mean() - initial[:n_teams][fbs].mean()
        )
        prior_mean[n_teams + fcs_indices] = (
            initial[n_teams:][fcs].mean() - initial[n_teams:][fbs].mean()
        )

    residual = centered_target - design @ initial
    noise = float(np.average(np.square(residual), weights=weights))
    parameters = solve(noise if noise > 0 else variance)
    return _centered_units(parameters, n_teams, fbs)


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
    weighted_dropbacks = frame.loc[valid, "pass_plays"] * frame.loc[valid, "weight"]
    denominator = float(weighted_dropbacks.sum())
    if denominator <= 0:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    baseline = float(
        (frame.loc[valid, "protection_ppa_allowed"] * frame.loc[valid, "weight"]).sum()
        / denominator
    )
    values = frame["protection_ppa_allowed"] - baseline * frame["pass_plays"]
    return values.where(valid)


def _run_block_values(frame: pd.DataFrame) -> pd.Series:
    """Convert adjusted line yards over baseline to PPA-equivalent value."""
    valid = frame["line_yard_carries"].gt(0)
    weighted_carries = (
        frame.loc[valid, "line_yard_carries"] * frame.loc[valid, "weight"]
    )
    denominator = float(weighted_carries.sum())
    if denominator <= 0:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    baseline = float(
        (frame.loc[valid, "adjusted_line_yards"] * frame.loc[valid, "weight"]).sum()
        / denominator
    )
    line_yard_residual = (
        frame["adjusted_line_yards"] - baseline * frame["line_yard_carries"]
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
        (frame.loc[valid, "weight"] * np.square(line_yard_residual.loc[valid])).sum()
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
    channel_priors: pd.DataFrame | None = None,
    per_play: bool = False,
) -> UnitRatings:
    """Fit pregame units; per-play mode returns only rush/pass player channels."""
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
    window["rush_value"] = window["rush_ppa"].where(window["rush_plays"].gt(0))
    window["pass_value"] = window["pass_ppa"].where(window["pass_plays"].gt(0))
    window["pass_block_value"] = _pass_block_values(window)
    window["run_block_value"] = _run_block_values(window)

    ratings: dict[str, np.ndarray] = {}
    for name, column in {
        "rush": "rush_value",
        "pass": "pass_value",
        "pass_block": "pass_block_value",
        "run_block": "run_block_value",
    }.items():
        if per_play and name not in {"rush", "pass"}:
            continue
        channel_window = window
        if per_play and name in {"rush", "pass"}:
            # Player credit is per play: normalize game totals to common
            # exposure and weight the rate by its actual sample size.
            channel_window = window.copy()
            counts = channel_window[f"{name}_plays"]
            exposure = float(counts.mean())
            if not np.isfinite(exposure) or exposure <= 0:
                raise ValueError("per-play unit fit requires positive exposure")
            channel_window[column] = (channel_window[column] / counts * exposure).where(
                counts.gt(0)
            )
            channel_window["weight"] = channel_window["weight"] * counts / exposure
        observations = _channel_frame(channel_window, column, team_set)
        prior_units = None
        if channel_priors is not None and name in {"rush", "pass"}:
            aligned = channel_priors.set_index("team").reindex(teams)
            per_game = float(window[f"{name}_plays"].mean())
            prior_units = (
                np.concatenate(
                    [
                        aligned[f"{name}_offense"].to_numpy(float),
                        aligned[f"{name}_defense"].to_numpy(float),
                    ]
                )
                * per_game
            )
            if not np.isfinite(prior_units).all() or per_game <= 0:
                raise ValueError(
                    "unit channel priors must cover every team with finite effects"
                )
        unit, counter = _two_sided_ridge(
            observations,
            teams,
            classifications,
            prior_units,
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
    for column, rating in ratings.items():
        frame[column] = rating
    return UnitRatings(
        season=int(games["season"].iloc[0]),
        week=int(forecast_week),
        as_of=as_of,
        frame=frame,
    )
