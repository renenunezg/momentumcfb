"""Validate and atomically replace the current compact CFP snapshot."""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import text

from backend.playoffs.serving_payload import GAME_COLUMNS, TEAM_COLUMNS


def validate(payload):
    if payload["schema_version"] != 2 or payload["season"] != 2026:
        raise ValueError("Only the reviewed 2026 compact contract is supported")
    if (
        payload["team_columns"] != TEAM_COLUMNS
        or payload["game_columns"] != GAME_COLUMNS
    ):
        raise ValueError("Unexpected payload columns")
    teams = [dict(zip(TEAM_COLUMNS, row, strict=True)) for row in payload["teams"]]
    ids = {t["id"] for t in teams}
    if len(teams) != 138 or len(ids) != len(teams):
        raise ValueError("Incomplete or duplicate FBS field")
    for t in teams:
        for key in TEAM_COLUMNS[8:13] + TEAM_COLUMNS[15:]:
            if not np.isfinite(t[key]) or not 0 <= t[key] <= 1:
                raise ValueError(f"Invalid probability: {key}")
        if not (
            t["champion"] <= t["playoff"]
            and t["bye"] <= t["playoff"]
            and t["conference_champion"] <= t["conference_title_game"]
        ):
            raise ValueError("Inconsistent nested probabilities")
    for key, total in [("playoff", 12), ("bye", 4), ("champion", 1)]:
        np.testing.assert_allclose(
            sum(t[key] for t in teams), total, atol=1e-10, rtol=0
        )
    for conference in {t["conference"] for t in teams}:
        members = [t for t in teams if t["conference"] == conference]
        for key, total in [("conference_champion", 1), ("conference_title_game", 2)]:
            np.testing.assert_allclose(
                sum(t[key] for t in members),
                0 if conference == "Independent" else total,
                atol=1e-10,
                rtol=0,
            )
    bracket = payload["bracket"]
    if (
        sorted(s["seed"] for s in bracket) != list(range(1, 13))
        or len({s["team_id"] for s in bracket}) != 12
        or not ids.issuperset(s["team_id"] for s in bracket)
    ):
        raise ValueError("Invalid bracket")
    names = ids | {int(i) for i in payload["opponent_names"]}
    games = [dict(zip(GAME_COLUMNS, row, strict=True)) for row in payload["games"]]
    if len({g["id"] for g in games}) != len(games):
        raise ValueError("Duplicate game")
    for g in games:
        if (
            not {g["home_id"], g["away_id"]} <= names
            or g["home_id"] == g["away_id"]
            or not 0 <= g["home_win_probability"] <= 1
        ):
            raise ValueError("Invalid matchup")
    now = datetime.now(timezone.utc)
    model = datetime.fromisoformat(payload["model_as_of"])
    results = datetime.fromisoformat(payload["results_as_of"])
    if not now - timedelta(days=8) <= model <= results <= now + timedelta(
        minutes=5
    ) or results < now - timedelta(hours=24):
        raise ValueError("Stale or future-dated forecast; regenerate before publishing")
    json.dumps(payload, allow_nan=False)
    return payload


def publish(path):
    payload = validate(json.loads(path.read_text()))
    if payload["status"] != "published":
        raise ValueError("Only validated publication artifacts may be served")
    from backend.db import engine

    with engine.begin() as conn:
        result = conn.execute(
            text("""
            insert into cfb.playoff_forecasts(season,payload,published_at)
            values(:season,cast(:payload as jsonb),now())
            on conflict(season) do update set payload=excluded.payload,published_at=excluded.published_at
            where (cfb.playoff_forecasts.payload->>'results_as_of')::timestamptz <= (excluded.payload->>'results_as_of')::timestamptz
              and cfb.playoff_forecasts.payload is distinct from excluded.payload
            returning season
        """),
            dict(
                season=payload["season"],
                payload=json.dumps(payload, separators=(",", ":"), allow_nan=False),
            ),
        )
        changed = result.scalar_one_or_none() is not None
    print(
        json.dumps(
            dict(
                published=changed,
                season=payload["season"],
                results_as_of=payload["results_as_of"],
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    publish(parser.parse_args().path)
