"""Prove served events match the stored batch baseline, after the fact.

This is the only serving module allowed to open
``ingame/baseline_predictions.parquet``. It runs strictly after serving, over
artifacts already written to ``serving/served/``, so the equality proof for
completed seasons costs the serving path nothing and adds no outcome-bearing
read to it. Seasons with no stored batch run (2026) have nothing to compare
against and are reported as unverifiable rather than as failures.
"""

import pandas as pd

from backend.etl import store
from backend.model.ingame import MODEL_VERSION
from backend.serving.replay import stream_problems
from backend.serving.serve import served_events_artifact

STORED_ARTIFACT = ("ingame", "baseline_predictions.parquet")

STORED_COLUMNS = [
    "season",
    "game_id",
    "play_index",
    "source_play_id",
    "win_probability",
    "model_version",
]


def served_games(season: int) -> list[int]:
    """Game ids served for a season, in the order they will be verified."""
    names = store.processed_names("serving", "served", str(season))
    return sorted(int(name) for name in names)


def verify_served(
    season: int, game_ids: list[int] | None = None
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """Compare each served game against the stored baseline predictions."""
    served = served_games(season)
    if game_ids is not None:
        missing = sorted(set(game_ids) - set(served))
        if missing:
            raise ValueError(
                "no served events for game"
                f"{'s' if len(missing) > 1 else ''} "
                + ", ".join(str(game_id) for game_id in missing)
                + f" in season {season}; run `python -m backend serve-game` first"
            )
        served = [game_id for game_id in served if game_id in set(game_ids)]
    if not served:
        raise ValueError(
            f"no served events stored for season {season}; run "
            "`python -m backend serve-game` first"
        )

    try:
        stored = store.read_processed(*STORED_ARTIFACT, columns=STORED_COLUMNS)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"no stored baseline predictions at {'/'.join(STORED_ARTIFACT)}; "
            "run `python -m backend ingame-baseline` first"
        ) from exc
    stored = stored[stored["season"].eq(season)]

    problems: list[str] = []
    rows = []
    for game_id in served:
        events = store.read_processed(*served_events_artifact(season, game_id))
        versions = sorted(events["model_version"].unique())
        if versions != [MODEL_VERSION]:
            problems.append(
                f"game {game_id}: served events carry model versions "
                f"{versions}; verification covers {MODEL_VERSION}"
            )
            continue
        batch = stored[stored["game_id"].eq(game_id)].reset_index(drop=True)
        if batch.empty:
            rows.append(
                {
                    "game_id": game_id,
                    "served_rows": int(events["emitted"].sum()),
                    "stored_rows": 0,
                    "status": "unverifiable",
                }
            )
            continue
        game_problems = stream_problems(events, batch)
        problems.extend(game_problems)
        rows.append(
            {
                "game_id": game_id,
                "served_rows": int(events["emitted"].sum()),
                "stored_rows": len(batch),
                "status": "exact_match" if not game_problems else "mismatch",
            }
        )

    games = pd.DataFrame(
        rows, columns=["game_id", "served_rows", "stored_rows", "status"]
    )
    compared = games[games["status"].ne("unverifiable")]
    unverifiable = int(games["status"].eq("unverifiable").sum())
    summary = pd.DataFrame(
        [
            {
                "summary_type": "verification",
                "season": season,
                "model_version": MODEL_VERSION,
                "served_games": len(served),
                "compared_games": len(compared),
                "unverifiable_games": unverifiable,
                "served_rows": int(compared["served_rows"].sum()),
                "stored_rows": int(compared["stored_rows"].sum()),
                "problem_count": len(problems),
                "status": "exact_match" if not problems else "mismatch",
                "diagnostic": (
                    "every served probability equals the stored baseline "
                    "prediction row"
                    if not problems
                    else "served outputs diverge from stored baseline "
                    "predictions"
                ),
            }
        ]
    )
    return problems, {"summary": summary, "games": games}
