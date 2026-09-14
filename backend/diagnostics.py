"""Read-only diagnostics of frozen forecasts and executable-price decisions.

Prediction error is predicted minus actual. Closing lines are benchmarks only.
Unknown pace and availability stay unknown; no probabilities are reconstructed
from results, and no diagnostic changes a forecast or recommendation threshold.
"""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from backend.grading import THIN_SAMPLE_GAMES, _classification
from backend.model.distributions import marginal_interval_half_width
from backend.model.scoring_calibration import bounded_model_total


def scoring_diagnostics(
    graded: pd.DataFrame,
    forecast_details: pd.DataFrame | None = None,
    games: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = graded.copy()
    if out.empty:
        return out, pd.DataFrame()
    if out.duplicated(["season", "game_id"]).any():
        raise ValueError("duplicate frozen grades")
    forecast_time = pd.to_datetime(out.forecast_as_of, utc=True)
    if not forecast_time.lt(pd.to_datetime(out.start_date, utc=True)).all():
        raise ValueError("diagnostics require pregame frozen forecasts")
    out["matchup"] = _classification(out)
    out["margin_error"] = out.pure_home_margin - out.actual_margin
    out["total_error"] = out.model_total - out.actual_total
    out["home_score_error"] = (out.total_error + out.margin_error) / 2
    out["away_score_error"] = (out.total_error - out.margin_error) / 2
    out["favorite_error"] = out.margin_error * np.sign(out.pure_home_margin)
    out["expected_game_possessions"] = np.nan
    if forecast_details is not None and "expected_game_possessions" in forecast_details:
        detail = forecast_details[
            ["game_id", "as_of", "expected_game_possessions"]
        ].copy()
        detail["forecast_as_of"] = pd.to_datetime(detail.pop("as_of"), utc=True)
        out["forecast_as_of"] = forecast_time
        out = out.drop(columns="expected_game_possessions").merge(
            detail, on=["game_id", "forecast_as_of"], how="left", validate="one_to_one"
        )
    out["game_possessions"] = np.nan
    if games is not None:
        out = out.drop(columns="game_possessions").merge(
            games[["game_id", "game_possessions"]],
            on="game_id",
            how="left",
            validate="one_to_one",
        )
    # Exact decomposition: total error = pace effect + scoring-rate effect.
    pace = out.expected_game_possessions
    actual_pace = out.game_possessions.where(out.game_possessions.gt(0))
    expected_ppp = out.model_total / pace.where(pace.gt(0))
    out["pace_error_points"] = (pace - actual_pace) * expected_ppp
    out["efficiency_error_points"] = actual_pace * (
        expected_ppp - out.actual_total / actual_pace
    )
    rows = []
    for key, frame in out.groupby(["season", "week", "model_version", "matchup"]):
        row = dict(zip(["season", "week", "model_version", "matchup"], key))
        row.update(games=len(frame), thin_sample=len(frame) < THIN_SAMPLE_GAMES)
        for metric in ("margin", "total", "home_score", "away_score", "favorite"):
            error = frame[f"{metric}_error"]
            row[f"{metric}_bias"] = error.mean()
            row[f"{metric}_mae"] = error.abs().mean()
        for metric in ("margin", "total"):
            width = marginal_interval_half_width(
                0.8, frame[f"{metric}_sd"], frame.degrees_of_freedom
            )
            row[f"{metric}_coverage_80"] = (
                frame[f"{metric}_error"].abs() <= width
            ).mean()
        row["pace_decomposition_games"] = int(frame.pace_error_points.notna().sum())
        row["pace_error_points"] = frame.pace_error_points.mean()
        row["efficiency_error_points"] = frame.efficiency_error_points.mean()
        rows.append(row)
    return out, pd.DataFrame(rows)


def recommendation_calibration(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()
    selected = recommendations[recommendations.status.eq("recommended")].copy()
    if selected.duplicated(["season", "game_id", "market"]).any():
        raise ValueError("duplicate executable recommendations")
    settled = selected[selected.outcome.isin(["win", "loss", "push"])].copy()
    if settled.empty:
        return pd.DataFrame()
    if (
        not pd.to_datetime(settled.published_at, utc=True)
        .lt(pd.to_datetime(settled.start_date, utc=True))
        .all()
    ):
        raise ValueError("recommendation calibration requires pregame publication")
    win = settled.win_probability
    push = settled.push_probability
    if (win.isna() | push.isna() | (win < 0) | (push < 0) | (win + push > 1)).any():
        raise ValueError("invalid frozen win/push probabilities")
    settled["conditional_win_probability"] = win / (1 - push).replace(0, np.nan)
    settled["probability_bucket"] = pd.cut(
        settled.conditional_win_probability,
        [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1],
        include_lowest=True,
    ).astype(str)
    rows = []
    keys = ["season", "week", "policy_version", "market", "reason"]
    for key, group in settled.groupby(keys, dropna=False):
        for bucket, frame in [
            ("all", group),
            *list(group.groupby("probability_bucket")),
        ]:
            decisive = frame[frame.outcome.ne("push")]
            observed = decisive.outcome.eq("win").astype(float)
            probability = decisive.conditional_win_probability
            loss = 1 - frame.win_probability - frame.push_probability
            probability_of_result = np.select(
                [frame.outcome.eq("win"), frame.outcome.eq("push")],
                [frame.win_probability, frame.push_probability],
                default=loss,
            )
            row = dict(zip(keys, key))
            row.update(
                probability_bucket=bucket,
                selections=len(frame),
                unique_games=frame.game_id.nunique(),
                decisive=len(decisive),
                wins=int(frame.outcome.eq("win").sum()),
                losses=int(frame.outcome.eq("loss").sum()),
                pushes=int(frame.outcome.eq("push").sum()),
                thin_sample=frame.game_id.nunique() < THIN_SAMPLE_GAMES,
                predicted_decisive_win_rate=probability.mean(),
                realized_decisive_win_rate=observed.mean(),
                conditional_brier=((probability - observed) ** 2).mean(),
                outcome_log_loss=-np.log(
                    np.clip(probability_of_result, 1e-12, 1)
                ).mean(),
                predicted_push_rate=frame.push_probability.mean(),
                realized_push_rate=frame.outcome.eq("push").mean(),
                clv_games=int(frame.clv_points.notna().sum()),
                mean_clv_points=frame.clv_points.mean(),
            )
            rows.append(row)
    return pd.DataFrame(rows)


def disagreement_audit(
    projections: pd.DataFrame,
    ratings: pd.DataFrame | None = None,
    availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = projections.copy()
    out["matchup"] = _classification(out)
    out["market_gap"] = out.pure_home_margin + out.market_home_spread
    out["absolute_market_gap"] = out.market_gap.abs()
    for side in ("home", "away"):
        out[f"{side}_availability_status"] = "unknown"
        if availability is not None and not availability.empty:
            for index, game in out.iterrows():
                reports = availability[
                    availability.season.eq(game.season)
                    & availability.week.eq(game.week)
                    & availability.team.eq(game[f"{side}_team"])
                    & pd.to_datetime(availability.reported_at, utc=True).lt(
                        pd.Timestamp(game.as_of)
                    )
                ]
                if not reports.empty:
                    latest = (
                        reports.assign(
                            _time=pd.to_datetime(reports.reported_at, utc=True)
                        )
                        .sort_values("_time")
                        .iloc[-1]
                    )
                    out.loc[index, f"{side}_availability_status"] = latest.status
        if ratings is not None and not ratings.empty:
            changes = []
            for game in out.itertuples():
                history = ratings[
                    ratings.season.eq(game.season)
                    & ratings.team_id.eq(getattr(game, f"{side}_team_id"))
                    & pd.to_datetime(ratings.as_of, utc=True).le(
                        pd.Timestamp(game.as_of)
                    )
                ].sort_values("week")
                current = history[history.week.eq(game.week)]
                previous = history[history.week.lt(game.week)]
                changes.append(
                    float(
                        current.iloc[-1].power_rating - previous.iloc[-1].power_rating
                    )
                    if not current.empty and not previous.empty
                    else np.nan
                )
            out[f"{side}_power_change"] = changes
    return out.sort_values("absolute_market_gap", ascending=False, na_position="last")


def write_diagnostics(destination: Path, tables: dict[str, pd.DataFrame]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(destination / f"{name}.parquet", index=False)


def shadow_totals(
    projections: pd.DataFrame, calibration_directory: Path, created_at: pd.Timestamp
) -> pd.DataFrame:
    """Freeze a prospective research forecast without repricing any selection."""
    manifest_path = calibration_directory / "manifest.json"
    offsets_path = calibration_directory / "totals_offsets.parquet"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["contract"] != "fall_d1_historical_carryover_v1"
        or manifest["development_seasons"] != [2020, 2021, 2022]
        or manifest["validation_seasons"] != [2023, 2024, 2025]
    ):
        raise ValueError("shadow totals require the reviewed historical split contract")
    if created_at.tzinfo is None:
        raise ValueError("shadow creation time must have a timezone")
    out = projections[
        pd.to_datetime(projections.start_date, utc=True).gt(created_at)
        & pd.to_datetime(projections.as_of, utc=True).le(created_at)
    ].copy()
    if not out.season.gt(2025).all():
        raise ValueError(
            "shadow forecasts must follow the historical validation seasons"
        )
    if out.model_version.eq("joint_scoring_v10").any() or (
        "total_calibration_adjustment" in out
        and out.total_calibration_adjustment.fillna(0).ne(0).any()
    ):
        raise ValueError("forecasts already include matchup total calibration")
    offsets = pd.read_parquet(offsets_path)
    out["matchup"] = _classification(out)
    adjustment = out.matchup.map(offsets.total_adjustment)
    if adjustment.isna().any() or not np.isfinite(adjustment).all():
        raise ValueError("missing or nonfinite matchup calibration")
    out["shadow_created_at"] = created_at
    out["shadow_candidate"] = "matchup_total_offset_v1"
    out["shadow_requested_adjustment"] = adjustment
    out["shadow_total"] = bounded_model_total(
        out.model_total, out.pure_home_margin, adjustment
    )
    out["shadow_total_adjustment"] = out.shadow_total - out.model_total
    out["shadow_home_points"] = (out.shadow_total + out.pure_home_margin) / 2
    out["shadow_away_points"] = (out.shadow_total - out.pure_home_margin) / 2
    if (out[["shadow_home_points", "shadow_away_points"]] < 0).any().any():
        raise ValueError("shadow correction would imply negative points")
    out["calibration_sha256"] = sha256(
        manifest_path.read_bytes() + offsets_path.read_bytes()
    ).hexdigest()
    out["evaluation_status"] = "research_shadow_not_a_published_forecast_or_pick"
    return out


def grade_shadow_totals(shadows: pd.DataFrame, graded: pd.DataFrame) -> pd.DataFrame:
    """Compare the earliest frozen candidate per game with subsequent final scores."""
    frozen = shadows.copy()
    frozen["shadow_created_at"] = pd.to_datetime(frozen.shadow_created_at, utc=True)
    if not frozen.shadow_created_at.lt(
        pd.to_datetime(frozen.start_date, utc=True)
    ).all():
        raise ValueError("shadow evaluation requires pregame creation")
    frozen = frozen.sort_values("shadow_created_at").drop_duplicates(
        ["season", "game_id", "shadow_candidate", "calibration_sha256"], keep="first"
    )
    out = frozen.merge(
        graded[["season", "game_id", "actual_total", "graded_at"]],
        on=["season", "game_id"],
        how="left",
        validate="many_to_one",
    )
    out["shadow_absolute_error"] = (out.shadow_total - out.actual_total).abs()
    out["parent_absolute_error"] = (out.model_total - out.actual_total).abs()
    out["shadow_minus_parent_error"] = (
        out.shadow_absolute_error - out.parent_absolute_error
    )
    out["result_status"] = np.where(out.actual_total.notna(), "graded", "pending")
    return out
