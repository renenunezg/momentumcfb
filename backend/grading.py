"""Grade frozen published projections against final scores and closing lines.

Inputs are records frozen before the result was known: the projection row
published before kickoff, the CFBD closing line, and the final score. A game
is graded once; later runs keep stored rows and add newly completed games.
The closing line is a benchmark graded alongside the model, never an input.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backend.config import PROCESSED_DIR
from backend.etl import store
from backend.model.distributions import marginal_cdf, marginal_interval_half_width
from backend.recommendations import PRICING_MARGIN_SD
from backend.serving.market import flatten_closing_lines, flatten_closing_totals

CLOSING_SOURCE = "cfbd_lines_median"
SCORE_SOURCE = "cfbd_games"
LEGACY_PROBABILITY_METHOD = (
    "P(home margin > 0) from the frozen pregame margin marginal: "
    "t.cdf(pure_home_margin / margin_sd, degrees_of_freedom)"
)
PROBABILITY_METHOD = (
    "P(home margin > 0) from the frozen pregame margin marginal: "
    "t.cdf(pure_home_margin / (margin_sd * sqrt((df - 2) / df)), df); "
    "normal limit when df is missing or infinite"
)

# Every model source is graded against the same actual margin; the closing
# market is graded as a source so its own error sits next to the model's.
PREDICTION_SOURCES = ("pure_model", "market_informed", "closing_market")
COVERAGE_LEVELS = (0.5, 0.8, 0.9)
THIN_SAMPLE_GAMES = 30
# Paired bootstrap over games for the model-minus-market MAE gap. The seed is
# fixed so a rerun on the same graded rows publishes the same interval.
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20261029

GRADED_GAME_COLUMNS = [
    "game_id",
    "season",
    "week",
    "season_type",
    "forecast_week",
    "start_date",
    "neutral_site",
    "conference_game",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "home_classification",
    "away_classification",
    "home_missing_input_count",
    "away_missing_input_count",
    "model_version",
    "forecast_as_of",
    "forecast_published_at",
    "pure_home_margin",
    "market_informed_home_margin",
    "market_weight",
    "forecast_market_home_spread",
    "model_total",
    "market_informed_total",
    "margin_sd",
    "total_sd",
    "distribution",
    "degrees_of_freedom",
    "home_win_probability",
    "probability_method",
    "closing_spread",
    "closing_total",
    "n_spread_offers",
    "n_total_offers",
    "closing_source",
    "home_points",
    "away_points",
    "actual_margin",
    "actual_total",
    "score_source",
    "source_ingested_at",
    "graded_at",
]

PERFORMANCE_METRIC_COLUMNS = [
    "season",
    "prediction_source",
    "segment_kind",
    "segment",
    "segment_order",
    "games",
    "thin_sample",
    "margin_mae",
    "margin_rmse",
    "margin_bias",
    "total_games",
    "total_mae",
    "total_rmse",
    "total_bias",
    "coverage_50",
    "coverage_80",
    "coverage_90",
    "games_with_market",
    "market_mae",
    "model_minus_market_mae",
    "model_minus_market_mae_lo_95",
    "model_minus_market_mae_hi_95",
    "closer_than_market_share",
    "probability_games",
    "brier_score",
    "log_loss",
    "computed_at",
]


def grading_artifact(season: int, name: str) -> tuple[str, str]:
    return ("grading", f"{name}_{season}.parquet")


def _home_win_probability(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        marginal_cdf(
            frame["pure_home_margin"], frame["margin_sd"], frame["degrees_of_freedom"]
        ),
        index=frame.index,
    )


def _correct_legacy_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    """Repair derived values using frozen parameters without changing forecasts.

    Rows graded before 2026-09-07 stored a home win probability that read the
    frozen margin_sd as the Student t scale instead of its standard deviation,
    which overstated confidence. The frozen inputs (pure_home_margin, margin_sd,
    degrees_of_freedom) are kept as published; only the derived probability and
    its probability_method are recomputed with the corrected scale, so Brier
    and log loss for those rows reflect the corrected formula, not the value
    that was stored at the time. No forecast input is ever rewritten here.
    """
    out = frame.copy()
    legacy = out["probability_method"].eq(LEGACY_PROBABILITY_METHOD)
    if legacy.any():
        out.loc[legacy, "home_win_probability"] = _home_win_probability(out[legacy])
        out.loc[legacy, "probability_method"] = PROBABILITY_METHOD
    return out


def build_graded_games(
    season: int,
    projections: pd.DataFrame,
    existing: pd.DataFrame | None = None,
    graded_at: datetime | None = None,
) -> pd.DataFrame:
    """Grade every completed game with a pregame published projection.

    ``projections`` is the published cfb.game_projections record for the
    season. Existing frozen inputs are preserved; the legacy probability
    scale calculation is repaired from those inputs when present.
    """
    graded_at = graded_at or datetime.now(timezone.utc)
    games = store.read_games(season)
    lines = store.read_lines(season)
    ingested_at = datetime.fromtimestamp(
        store.raw_path("games", season).stat().st_mtime, tz=timezone.utc
    )

    completed = games[
        games["completed"].fillna(False).astype(bool)
        & games["home_points"].notna()
        & games["away_points"].notna()
    ]
    if existing is not None and not existing.empty:
        completed = completed[~completed["id"].isin(existing["game_id"])]

    frozen = projections.copy()
    frozen["start_date"] = pd.to_datetime(frozen["start_date"], utc=True)
    frozen["as_of"] = pd.to_datetime(frozen["as_of"], utc=True)
    # Only a projection published before kickoff is a grade-able forecast.
    # Rows published before the database stamped publication have a null
    # published_at; for them the forecast's as_of is the evidence. A record
    # without the column at all is not the published contract.
    if "published_at" not in frozen:
        raise ValueError("published projections lack published_at")
    published = pd.to_datetime(frozen["published_at"], utc=True)
    frozen["published_at"] = published
    frozen = frozen[
        frozen["as_of"].lt(frozen["start_date"])
        & (published.isna() | published.lt(frozen["start_date"]))
    ]
    frozen = frozen.drop_duplicates("game_id", keep="last")

    merged = completed.merge(
        frozen,
        left_on="id",
        right_on="game_id",
        suffixes=("_schedule", ""),
    )
    if merged.empty:
        return _graded_frame(pd.DataFrame(), existing)

    closing = flatten_closing_lines(lines).merge(
        flatten_closing_totals(lines), on="game_id", how="outer"
    )
    merged = merged.merge(closing, on="game_id", how="left")

    out = pd.DataFrame(
        {
            "game_id": merged["game_id"].astype(int),
            "season": season,
            "week": merged["week_schedule"].astype(int),
            "season_type": merged["season_type"],
            "forecast_week": merged["week"].astype(int),
            "start_date": merged["start_date"],
            "neutral_site": merged["neutral_site"],
            "conference_game": merged["conference_game"],
            "home_team_id": merged["home_team_id"],
            "home_team": merged["home_team"],
            "away_team_id": merged["away_team_id"],
            "away_team": merged["away_team"],
            "home_classification": merged["home_classification"],
            "away_classification": merged["away_classification"],
            "home_missing_input_count": merged["home_missing_input_count"],
            "away_missing_input_count": merged["away_missing_input_count"],
            "model_version": merged["model_version"],
            "forecast_as_of": merged["as_of"],
            "forecast_published_at": merged["published_at"],
            "pure_home_margin": merged["pure_home_margin"].fillna(
                merged["home_margin"]
            ),
            "market_informed_home_margin": merged["market_informed_home_margin"],
            "market_weight": merged["market_weight"],
            "forecast_market_home_spread": merged["market_home_spread"],
            "model_total": merged["model_total"],
            "market_informed_total": merged.get("market_informed_total"),
            "margin_sd": merged["margin_sd"],
            "total_sd": merged["total_sd"],
            "distribution": merged["distribution"],
            "degrees_of_freedom": merged["degrees_of_freedom"],
            "closing_spread": merged["closing_spread"],
            "closing_total": merged["closing_total"],
            "n_spread_offers": merged["n_spread_offers"],
            "n_total_offers": merged["n_total_offers"],
            "closing_source": np.where(
                merged["closing_spread"].notna() | merged["closing_total"].notna(),
                CLOSING_SOURCE,
                None,
            ),
            "home_points": merged["home_points"].astype(int),
            "away_points": merged["away_points"].astype(int),
            "score_source": SCORE_SOURCE,
            "source_ingested_at": ingested_at,
            "graded_at": graded_at,
        }
    )
    out["actual_margin"] = out["home_points"] - out["away_points"]
    out["actual_total"] = out["home_points"] + out["away_points"]
    out["home_win_probability"] = _home_win_probability(out)
    out["probability_method"] = PROBABILITY_METHOD
    return _graded_frame(out, existing)


def _graded_frame(new: pd.DataFrame, existing: pd.DataFrame | None) -> pd.DataFrame:
    parts = [
        frame.reindex(columns=GRADED_GAME_COLUMNS)
        for frame in (existing, new)
        if frame is not None and not frame.empty
    ]
    if not parts:
        return pd.DataFrame(columns=GRADED_GAME_COLUMNS)
    out = _correct_legacy_probabilities(pd.concat(parts, ignore_index=True))
    for column in (
        "start_date",
        "forecast_as_of",
        "forecast_published_at",
        "source_ingested_at",
        "graded_at",
    ):
        out[column] = pd.to_datetime(out[column], utc=True)
    return out.sort_values(["start_date", "game_id"], kind="stable", ignore_index=True)


def _favorite_size(spread: pd.Series) -> pd.Series:
    size = spread.abs()
    bins = [-np.inf, 7, 14, 21, np.inf]
    labels = ["under 7", "7 to 14", "14 to 21", "21 or more"]
    return pd.cut(size, bins=bins, labels=labels, right=False).astype(object)


def _missing_inputs(frame: pd.DataFrame) -> pd.Series:
    count = frame["home_missing_input_count"].fillna(0) + frame[
        "away_missing_input_count"
    ].fillna(0)
    bins = [-np.inf, 1, 4, np.inf]
    labels = ["none", "1 to 3", "4 or more"]
    return pd.cut(count, bins=bins, labels=labels, right=False).astype(object)


def opponent_classification(frame: pd.DataFrame) -> pd.Series:
    """FBS vs FBS, FBS vs FCS or FCS vs FCS from the two classifications."""
    fbs_home = frame["home_classification"].eq("fbs")
    fbs_away = frame["away_classification"].eq("fbs")
    return pd.Series(
        np.select(
            [fbs_home & fbs_away, fbs_home | fbs_away],
            ["FBS vs FBS", "FBS vs FCS"],
            default="FCS vs FCS",
        ),
        index=frame.index,
    )


def _model_version_order(graded: pd.DataFrame) -> list[str]:
    """Model versions in the order they first graded a game."""
    first = (
        graded.dropna(subset=["model_version"])
        .groupby("model_version")["start_date"]
        .min()
    )
    return [str(v) for v in first.sort_values(kind="stable").index]


def _segment_frames(graded: pd.DataFrame) -> list[tuple[str, str, int, pd.Series]]:
    """Yield (kind, label, order, mask) for every segment with any games."""
    segments: list[tuple[str, str, int, pd.Series]] = [
        ("overall", "All graded games", 0, pd.Series(True, index=graded.index))
    ]
    keyed = {
        "week": graded["week"].astype(int).astype(str),
        "opponent_classification": opponent_classification(graded),
        "model_favorite_size": _favorite_size(-graded["pure_home_margin"]),
        "missing_inputs": _missing_inputs(graded),
        "model_version": graded["model_version"].astype(object),
    }
    orders = {
        "opponent_classification": ["FBS vs FBS", "FBS vs FCS", "FCS vs FCS"],
        "model_favorite_size": ["under 7", "7 to 14", "14 to 21", "21 or more"],
        "missing_inputs": ["none", "1 to 3", "4 or more"],
        "model_version": _model_version_order(graded),
    }
    for kind, labels in keyed.items():
        present = [label for label in labels.dropna().unique()]
        if kind == "week":
            ordered = sorted(present, key=int)
        else:
            ordered = [label for label in orders[kind] if label in present]
        for order, label in enumerate(ordered, start=1):
            segments.append((kind, str(label), order, labels.eq(label)))
    return segments


def _source_columns(graded: pd.DataFrame, source: str) -> tuple[pd.Series, pd.Series]:
    if source == "pure_model":
        return graded["pure_home_margin"], graded["model_total"]
    if source == "market_informed":
        # Games published before the blended total existed have none to grade.
        return graded["market_informed_home_margin"], graded["market_informed_total"]
    if source == "closing_market":
        return -graded["closing_spread"], graded["closing_total"]
    raise ValueError(f"unknown prediction source {source!r}")


def _margin_sd(frame: pd.DataFrame, source: str) -> pd.Series:
    """The pure model's frozen predictive sd, or for the market-informed line
    and the closing line the dispersion picks are priced with, so the published
    line and its benchmark share one probability scale."""
    if source == "pure_model":
        return frame["margin_sd"]
    if source in ("market_informed", "closing_market"):
        return pd.Series(PRICING_MARGIN_SD, index=frame.index)
    raise ValueError(f"unknown prediction source {source!r}")


def _paired_interval(difference: np.ndarray) -> tuple[float, float]:
    """95 percent percentile interval of the mean of per-game differences,
    resampling games with replacement."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(difference), size=(BOOTSTRAP_DRAWS, len(difference)))
    means = difference[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _coverage(
    frame: pd.DataFrame, prediction: pd.Series, sd: pd.Series, level: float
) -> float | None:
    width = marginal_interval_half_width(level, sd, frame["degrees_of_freedom"])
    error = (prediction - frame["actual_margin"]).abs()
    return float((error <= width).mean())


def _error_stats(prediction: pd.Series, actual: pd.Series) -> dict[str, float | None]:
    error = (prediction - actual).dropna()
    if error.empty:
        return {"mae": None, "rmse": None, "bias": None, "n": 0}
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "bias": float(error.mean()),
        "n": int(len(error)),
    }


def compute_performance_metrics(
    graded: pd.DataFrame, computed_at: datetime | None = None
) -> pd.DataFrame:
    """Aggregate graded games per prediction source and segment."""
    computed_at = computed_at or datetime.now(timezone.utc)
    if graded.empty:
        return pd.DataFrame(columns=PERFORMANCE_METRIC_COLUMNS)
    # A fixed game order makes the bootstrap interval independent of how the
    # rows arrive (database, artifact or public ledger).
    graded = _correct_legacy_probabilities(graded).sort_values(
        "game_id", kind="stable", ignore_index=True
    )
    season = int(graded["season"].iloc[0])
    rows = []
    for source in PREDICTION_SOURCES:
        margin_prediction, total_prediction = _source_columns(graded, source)
        for kind, label, order, mask in _segment_frames(graded):
            frame = graded[mask & margin_prediction.notna()]
            if frame.empty:
                continue
            margin = _error_stats(
                margin_prediction[frame.index], frame["actual_margin"]
            )
            total = _error_stats(total_prediction[frame.index], frame["actual_total"])
            row: dict[str, object] = {
                "season": season,
                "prediction_source": source,
                "segment_kind": kind,
                "segment": label,
                "segment_order": order,
                "games": margin["n"],
                "thin_sample": margin["n"] < THIN_SAMPLE_GAMES,
                "margin_mae": margin["mae"],
                "margin_rmse": margin["rmse"],
                "margin_bias": margin["bias"],
                "total_games": total["n"],
                "total_mae": total["mae"],
                "total_rmse": total["rmse"],
                "total_bias": total["bias"],
                "coverage_50": None,
                "coverage_80": None,
                "coverage_90": None,
                "games_with_market": None,
                "market_mae": None,
                "model_minus_market_mae": None,
                "model_minus_market_mae_lo_95": None,
                "model_minus_market_mae_hi_95": None,
                "closer_than_market_share": None,
                "probability_games": None,
                "brier_score": None,
                "log_loss": None,
                "computed_at": computed_at,
            }
            prediction = margin_prediction[frame.index]
            sd = _margin_sd(frame, source)
            if source != "closing_market":
                for level in COVERAGE_LEVELS:
                    row[f"coverage_{int(level * 100)}"] = _coverage(
                        frame, prediction, sd, level
                    )
            # The closing line is scored on the published line's scale, over
            # the games it has a line for, so the two rows compare directly.
            probability = (
                frame["home_win_probability"]
                if source == "pure_model"
                else pd.Series(
                    marginal_cdf(prediction, sd, frame["degrees_of_freedom"]),
                    index=frame.index,
                )
            ).dropna()
            if not probability.empty:
                won = (frame.loc[probability.index, "actual_margin"] > 0).astype(float)
                clipped = probability.clip(1e-6, 1 - 1e-6)
                row["probability_games"] = int(len(probability))
                row["brier_score"] = float(((probability - won) ** 2).mean())
                row["log_loss"] = float(
                    -(won * np.log(clipped) + (1 - won) * np.log(1 - clipped)).mean()
                )
            if source != "closing_market":
                benchmarked = frame[frame["closing_spread"].notna()]
                row["games_with_market"] = int(len(benchmarked))
                if not benchmarked.empty:
                    model_error = (
                        margin_prediction[benchmarked.index]
                        - benchmarked["actual_margin"]
                    ).abs()
                    market_error = (
                        -benchmarked["closing_spread"] - benchmarked["actual_margin"]
                    ).abs()
                    difference = (model_error - market_error).to_numpy(float)
                    row["market_mae"] = float(market_error.mean())
                    row["model_minus_market_mae"] = float(difference.mean())
                    (
                        row["model_minus_market_mae_lo_95"],
                        row["model_minus_market_mae_hi_95"],
                    ) = _paired_interval(difference)
                    # A tie is not closer: ties stay in the denominator.
                    row["closer_than_market_share"] = float(
                        (model_error < market_error).mean()
                    )
            rows.append(row)
    return pd.DataFrame(rows, columns=PERFORMANCE_METRIC_COLUMNS)


def write_grading_artifacts(
    season: int, graded: pd.DataFrame, metrics: pd.DataFrame
) -> None:
    store.write_processed(graded, *grading_artifact(season, "graded_games"))
    store.write_processed(metrics, *grading_artifact(season, "performance_metrics"))


def read_grading_artifacts(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    graded = pd.read_parquet(
        PROCESSED_DIR.joinpath(*grading_artifact(season, "graded_games"))
    )
    metrics = pd.read_parquet(
        PROCESSED_DIR.joinpath(*grading_artifact(season, "performance_metrics"))
    )
    return graded, metrics
