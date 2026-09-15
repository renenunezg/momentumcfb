"""Observed-weather residual diagnostic using a fixed chronological split.

Historical API weather fetched after kickoff is NOT an archived pregame
forecast. Even a favorable diagnostic cannot activate weather coefficients.
Prospective weather snapshots must establish forecast-time skill separately.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from backend.cfbd.snapshots import receipts


def evaluate_weather(predictions, destination, *, root=None):
    records = {}
    fetched_by_game = {}
    for _, receipt in receipts("/games/weather", root=root):
        fetched = pd.Timestamp(receipt["fetched_at"])
        for row in receipt["payload"]:
            kickoff = pd.to_datetime(row.get("startTime"), utc=True, errors="coerce")
            game_id = int(row["id"])
            if pd.isna(kickoff) or fetched < kickoff:
                continue
            if game_id not in fetched_by_game or fetched > fetched_by_game[game_id]:
                records[game_id] = row
                fetched_by_game[game_id] = fetched
    weather = pd.DataFrame(records.values())
    if weather.empty:
        raise ValueError("no cached weather")
    forecasts = predictions.copy()
    if "candidate" in forecasts:
        forecasts = forecasts[forecasts.candidate.eq(100)]
    if forecasts.duplicated(["season", "game_id"]).any():
        raise ValueError("one forecast per game required")
    forecasts = forecasts[forecasts.season.between(2020, 2025)]
    joined = forecasts.merge(
        weather,
        left_on="game_id",
        right_on="id",
        validate="one_to_one",
        suffixes=("", "_weather"),
    )
    for col in ("windSpeed", "precipitation", "temperature"):
        joined[col] = pd.to_numeric(joined[col], errors="coerce")
    joined = joined.dropna(
        subset=[
            "windSpeed",
            "precipitation",
            "temperature",
            "gameIndoors",
            "actual_total",
        ]
    )
    joined = joined[(joined.windSpeed >= 0) & (joined.precipitation >= 0)].copy()
    for years in ((2020, 2021), (2022,), (2023, 2024, 2025)):
        if not joined.season.isin(years).any():
            raise ValueError(f"weather missing chronological split {years}")
    outdoors = (~joined.gameIndoors.astype(bool)).astype(float).to_numpy()
    raw = np.column_stack(
        [
            np.log1p(joined.windSpeed) * outdoors,
            np.log1p(joined.precipitation) * outdoors,
            joined.temperature * outdoors,
            outdoors,
        ]
    )
    residual = (joined.actual_total - joined.model_total).to_numpy()
    fit = joined.season.le(2021).to_numpy()
    select = joined.season.eq(2022).to_numpy()

    def train(mask, ridge):
        center, scale = raw[mask].mean(0), raw[mask].std(0)
        scale[scale == 0] = 1
        design = np.column_stack([np.ones(len(raw)), (raw - center) / scale])
        penalty = np.diag([0.0, ridge, ridge, ridge, ridge])
        coef = np.linalg.solve(
            design[mask].T @ design[mask] + penalty, design[mask].T @ residual[mask]
        )
        return design @ coef

    choices = {
        ridge: np.abs(train(fit, ridge)[select] - residual[select]).mean()
        for ridge in (10.0, 100.0, 1000.0)
    }
    ridge = min(choices, key=choices.get)
    # Include no change in selection; do not force a weather effect.
    selected = choices[ridge] < np.abs(residual[select]).mean()
    joined["weather_adjustment"] = (
        train(joined.season.le(2022).to_numpy(), ridge) if selected else 0.0
    )
    joined["evidence_kind"] = "observed_weather_only_not_pregame_validation"
    summaries = []
    for split, subset in (
        ("development", joined[joined.season.le(2022)]),
        ("validation", joined[joined.season.ge(2023)]),
    ):
        for group, frame in [("all", subset), *subset.groupby("matchup")]:
            error = frame.model_total - frame.actual_total
            adjusted = error + frame.weather_adjustment
            summaries.append(
                dict(
                    split=split,
                    matchup=group,
                    games=len(frame),
                    baseline_mae=error.abs().mean(),
                    weather_mae=adjusted.abs().mean(),
                    baseline_bias=error.mean(),
                    weather_bias=adjusted.mean(),
                    ridge=ridge,
                    selected=selected,
                    production_eligible=False,
                    evidence_kind="observed_weather_only_not_pregame_validation",
                )
            )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(destination / "weather_diagnostic.parquet", index=False)
    result = pd.DataFrame(summaries)
    result.to_csv(destination / "weather_summary.csv", index=False)
    return result
