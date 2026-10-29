"""The core joint scoring model's walk-forward backtest, as published to
cfb.backtest_predictions.

Rows come from the joint scoring walk-forward stored at
``calibration/joint_scoring_predictions.parquet`` (``CALIBRATION_ARTIFACT``):
one FBS vs FBS prediction per game, each fitted only on games that started
before it. The benchmark is the CFBD median closing spread, resolved exactly
as live grading resolves it.

The rows are ``joint_scoring_v13``, the core fit the weekly forecast runs
(``joint_scoring.MODEL_VERSION``). Live forecasts are stamped
``joint_scoring_v14`` after the frozen final-mean calibration in
``backend/model/forecast_calibration.py`` adds a process margin adjustment and
a median total offset, which this walk-forward does not apply. The backtest
and the live record are two populations from two versions, not one series.
The artifact must carry the current core version, so a stale walk-forward
cannot be published under the current model's name.
"""

import pandas as pd

from backend.etl import store
from backend.grading import opponent_classification
from backend.model.joint_scoring import MODEL_VERSION
from backend.serving.anchors import CALIBRATION_ARTIFACT
from backend.serving.market import flatten_closing_lines

BACKTEST_COLUMNS = [
    "game_id",
    "season",
    "week",
    "week_index",
    "season_type",
    "home_team",
    "away_team",
    "neutral_site",
    "home_points",
    "away_points",
    "margin",
    "closing_spread",
    "model_margin",
    "actual_margin",
    "model_version",
    "classification",
]
SOURCE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "model_week",
    "season_type",
    "start_date",
    "home_team",
    "away_team",
    "neutral_site",
    "home_margin",
    "actual_home_points",
    "actual_away_points",
    "model_version",
    "training_latest_start",
]


def build_backtest_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Validate the walk-forward and attach classifications and the close."""
    versions = set(predictions["model_version"].dropna().unique())
    if versions != {MODEL_VERSION}:
        raise ValueError(
            f"walk-forward carries {sorted(versions)}, the served model is "
            f"{MODEL_VERSION}; rerun the calibration before publishing"
        )
    if predictions["game_id"].duplicated().any():
        raise ValueError("walk-forward has more than one prediction for a game")
    trained = pd.to_datetime(predictions["training_latest_start"], utc=True)
    started = pd.to_datetime(predictions["start_date"], utc=True)
    if not trained.lt(started).all():
        raise ValueError(
            "a walk-forward prediction was fitted on a game that had not started"
        )
    frames = []
    for season, rows in predictions.groupby("season", sort=True):
        games = store.read_games(int(season))[
            ["id", "home_classification", "away_classification"]
        ]
        closing = flatten_closing_lines(store.read_lines(int(season)))
        frames.append(
            rows.merge(
                games,
                left_on="game_id",
                right_on="id",
                how="left",
                validate="one_to_one",
            ).merge(closing[["game_id", "closing_spread"]], on="game_id", how="left")
        )
    out = pd.concat(frames, ignore_index=True)
    if out[["home_classification", "away_classification"]].isna().any().any():
        raise ValueError("walk-forward games are missing from the cached schedule")
    margin = out["actual_home_points"] - out["actual_away_points"]
    return pd.DataFrame(
        {
            "game_id": out["game_id"].astype(int),
            "season": out["season"].astype(int),
            "week": out["week"].astype(int),
            "week_index": out["model_week"].astype(int),
            "season_type": out["season_type"],
            "home_team": out["home_team"],
            "away_team": out["away_team"],
            "neutral_site": out["neutral_site"].astype(bool),
            "home_points": out["actual_home_points"].astype(int),
            "away_points": out["actual_away_points"].astype(int),
            "margin": margin.astype(float),
            "closing_spread": out["closing_spread"],
            "model_margin": out["home_margin"],
            "actual_margin": margin.astype(float),
            "model_version": out["model_version"],
            "classification": opponent_classification(out),
        },
        columns=BACKTEST_COLUMNS,
    ).sort_values(["season", "week_index", "game_id"], ignore_index=True)


def load_backtest_predictions() -> pd.DataFrame:
    """The publishable backtest built from the stored walk-forward."""
    predictions = store.read_processed(*CALIBRATION_ARTIFACT, columns=SOURCE_COLUMNS)
    return build_backtest_predictions(predictions)
