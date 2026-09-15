"""Pregame weather context, with observation time separate from game time.

Raw CFBD units are retained without assuming a conversion. Historical weather
retrieved after kickoff is diagnostic evidence, never an archived forecast.
No weather coefficient is applied to the production scoring model.
"""

import numpy as np
import pandas as pd

from backend.cfbd.snapshots import receipts

WEATHER_FIELDS = (
    "temperature",
    "dewPoint",
    "humidity",
    "precipitation",
    "snowfall",
    "windDirection",
    "windSpeed",
    "pressure",
)


def weather_context(schedule, as_of, *, root=None):
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("weather cutoff must be timezone-aware")
    target = schedule[["game_id", "start_date"]].copy()
    if target.game_id.duplicated().any():
        raise ValueError("weather schedule must have unique games")
    starts = pd.to_datetime(target.start_date, utc=True)
    kickoff = dict(zip(target.game_id, starts))
    latest = {}
    for path, receipt in receipts("/games/weather", root=root):
        fetched = pd.Timestamp(receipt["fetched_at"])
        if fetched > cutoff:
            continue
        payload = receipt["payload"]
        ids = [int(row["id"]) for row in payload]
        if len(ids) != len(set(ids)):
            raise ValueError("weather snapshot contains duplicate game IDs")
        for row in payload:
            game_id = int(row["id"])
            if game_id not in kickoff:
                continue
            provider_start = pd.to_datetime(
                row.get("startTime"), utc=True, errors="coerce"
            )
            if pd.isna(provider_start) or fetched >= min(
                kickoff[game_id], provider_start
            ):
                continue
            if game_id in latest and latest[game_id][0] >= fetched:
                continue
            latest[game_id] = (fetched, row, str(path), receipt["payload_sha256"])
    rows = []
    for game_id, start in kickoff.items():
        row = dict(
            game_id=game_id,
            start_date=start,
            weather_missing=True,
            weather_source="cfbd",
            weather_units="provider_native",
            weather_fetched_at=None,
            weather_snapshot=None,
            weather_sha256=None,
            game_indoors=None,
            weather_age_hours=np.nan,
        )
        row.update({f"weather_{field}": np.nan for field in WEATHER_FIELDS})
        if game_id in latest:
            fetched, source, path, digest = latest[game_id]
            row.update(
                weather_fetched_at=fetched,
                weather_snapshot=path,
                weather_sha256=digest,
                game_indoors=source.get("gameIndoors"),
                weather_age_hours=(cutoff - fetched).total_seconds() / 3600,
            )
            for field in WEATHER_FIELDS:
                value = pd.to_numeric(source.get(field), errors="coerce")
                row[f"weather_{field}"] = value if np.isfinite(value) else np.nan
            row["weather_missing"] = any(
                pd.isna(row[f"weather_{f}"])
                for f in ("temperature", "windSpeed", "precipitation")
            )
        rows.append(row)
    return pd.DataFrame(rows)
