"""Fixed-configuration replay of the production preseason and weekly paths.

Archived preseason forecasts supply the opening slate and its rich priors.
Later periods use the weekly input checks and joint scoring fit.
This is a retrospective replay of cached final data, not a live grading ledger.
"""

import logging
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import PROCESSED_DIR
from backend.features.scoring import SCORING_COLUMNS
from backend.model.calibration import _add_proper_scores, _evaluation_rows
from backend.model.joint_scoring import DEFAULT_CONFIG, fit_joint_scoring
from backend.model.preseason import load_score_noise_prior, scoring_priors_from_ratings
from backend.model.weekly import _validate_weekly_inputs, load_weekly_games

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PregameSnapshot:
    ratings: pd.DataFrame
    projections: pd.DataFrame
    as_of: pd.Timestamp
    path: Path
    digest: str
    score_noise_prior: pd.DataFrame | None = None


def load_pregame_snapshots(season: int, directory: Path) -> list[PregameSnapshot]:
    snapshots = []
    for path in sorted(directory.glob(f"{season}_01_*/ratings.parquet")):
        projections_path = path.with_name("projections.parquet")
        if not projections_path.exists():
            continue
        ratings = pd.read_parquet(path)
        projections = pd.read_parquet(projections_path)
        timestamps = []
        valid = True
        for frame in (ratings, projections):
            for column in ("as_of", "source_snapshot_as_of"):
                if column not in frame or frame.empty:
                    valid = False
                    continue
                dates = pd.to_datetime(frame[column], utc=True, errors="coerce")
                if dates.isna().any():
                    valid = False
                timestamps.append(dates.max())
        if not valid:
            log.warning("Skipping snapshot with incomplete provenance: %s", path.parent)
            continue
        noise_path = path.with_name("score_noise_prior.parquet")
        noise_prior = (
            pd.read_parquet(noise_path)
            if noise_path.exists()
            else load_score_noise_prior(season)
        )
        noise_as_of = pd.to_datetime(noise_prior["as_of"], utc=True, errors="coerce")
        if noise_as_of.isna().any() or noise_as_of.max() > max(timestamps):
            raise ValueError(
                "score noise prior was not available at the snapshot cutoff"
            )
        snapshots.append(
            PregameSnapshot(
                ratings,
                projections,
                max(timestamps),
                path.parent,
                sha256(
                    path.read_bytes()
                    + projections_path.read_bytes()
                    + noise_prior.to_json(orient="records").encode()
                ).hexdigest(),
                noise_prior,
            )
        )
    if not snapshots:
        raise FileNotFoundError(
            f"{season}: no archived rich preseason snapshots in {directory}; "
            "historical carryover is not a substitute for production evaluation"
        )
    return sorted(snapshots, key=lambda item: item.as_of)


def replay_production_season(
    season: int,
    *,
    games: pd.DataFrame | None = None,
    snapshots: list[PregameSnapshot] | None = None,
) -> pd.DataFrame:
    games = load_weekly_games(season) if games is None else games.copy()
    snapshots = (
        load_pregame_snapshots(season, PROCESSED_DIR / "preseason" / "forecast_log")
        if snapshots is None
        else snapshots
    )
    complete = games["completed"].fillna(False).astype(bool)
    frames = []
    for week in sorted(games.loc[complete, "model_week"].unique()):
        target = games[games["model_week"].eq(week)]
        cutoff = target["start_date"].min() - pd.Timedelta(microseconds=1)
        available = [snapshot for snapshot in snapshots if snapshot.as_of <= cutoff]
        if not available:
            raise ValueError(
                f"{season} week {week}: no preseason snapshot before cutoff"
            )
        snapshot = max(available, key=lambda item: item.as_of)
        priors = scoring_priors_from_ratings(
            snapshot.ratings, snapshot.score_noise_prior
        )
        prior_games = games[games["model_week"].lt(week)]
        if prior_games.empty:
            # The opening forecast was already frozen with its preseason model.
            projected = snapshot.projections[
                snapshot.projections["game_id"].isin(target["game_id"])
            ].copy()
            stage = "frozen_preseason"
        else:
            # Reconstruct the schedule state, withholding all target/future results.
            known = games.copy()
            future = known["model_week"].ge(week)
            known.loc[future, "completed"] = False
            known.loc[future, SCORING_COLUMNS] = np.nan
            forecast_target, _ = _validate_weekly_inputs(known, int(week), cutoff)
            fitted = fit_joint_scoring(
                known, int(week), cutoff.to_pydatetime(), DEFAULT_CONFIG, priors=priors
            )
            projected = pd.DataFrame(
                p.to_record() for p in fitted.project(forecast_target)
            )
            stage = "weekly_replay"
        if projected["game_id"].duplicated().any() or set(projected["game_id"]) != set(
            target["game_id"]
        ):
            raise ValueError(f"{season} week {week}: incomplete forecast coverage")
        actual = target.loc[
            complete.loc[target.index],
            [
                "game_id",
                "start_date",
                "model_week",
                "home_points",
                "away_points",
                "home_classification",
                "away_classification",
            ],
        ].rename(
            columns={
                "home_points": "actual_home_points",
                "away_points": "actual_away_points",
            }
        )
        projected = projected.drop(
            columns=[c for c in actual if c != "game_id" and c in projected]
        )
        out = projected.merge(actual, on="game_id", validate="one_to_one")
        out["actual_margin"] = out["actual_home_points"] - out["actual_away_points"]
        out["actual_total"] = out["actual_home_points"] + out["actual_away_points"]
        for metric, prediction in (
            ("home_score", "expected_home_points"),
            ("away_score", "expected_away_points"),
            ("margin", "home_margin"),
            ("total", "model_total"),
        ):
            actual_column = {
                "home_score": "actual_home_points",
                "away_score": "actual_away_points",
            }.get(metric, f"actual_{metric}")
            out[f"{metric}_error"] = out[prediction] - out[actual_column]
        out["evaluation_stage"] = stage
        out["evaluation_contract"] = "production_replay_v1"
        out["source_contract"] = "cached_final_game_data_with_pregame_priors"
        out["forecast_cutoff"] = cutoff
        out["prior_snapshot_as_of"] = snapshot.as_of
        out["prior_snapshot_path"] = str(snapshot.path)
        out["prior_snapshot_sha256"] = snapshot.digest
        out["prior_model_version"] = str(snapshot.ratings["model_version"].iloc[0])
        out["training_games"] = int(
            prior_games[SCORING_COLUMNS].notna().all(axis=1).sum()
        )
        frames.append(_add_proper_scores(out))
    if not frames:
        raise ValueError(f"{season}: no completed games to evaluate")
    return pd.concat(frames, ignore_index=True)


def summarize_production_replay(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Separate frozen opening predictions from fits using the current weekly code.
    for (season, stage, week), frame in predictions.groupby(
        ["season", "evaluation_stage", "model_week"], sort=True
    ):
        for row in _evaluation_rows(
            frame,
            partition="retrospective_replay",
            scope="season_week",
            group_value=f"{season}:{week}",
            assess_status=False,
        ):
            row["evaluation_contract"] = "production_replay_v1"
            row["evaluation_stage"] = stage
            row["thin_sample"] = len(frame) < 30
            rows.append(row)
    return pd.DataFrame(rows)
