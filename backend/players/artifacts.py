"""Portable, validated Heisman coefficients and chronological evaluation."""

import hashlib
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from backend.config import PROCESSED_DIR
from backend.players import heisman
from backend.players import value as player_value
from backend.players.traits import expectations, grade

ARTIFACT_VERSION = 1
EVALUATION_KIND = heisman.EVALUATION_KIND


def model_path(cutoff_season: int) -> Path:
    return PROCESSED_DIR / "players" / "models" / f"heisman_{cutoff_season}.json"


def _seed_digest() -> str:
    return hashlib.sha256(heisman.SEED_PATH.read_bytes()).hexdigest()


def save_heisman_model(
    model: heisman.ShareModel,
    training_seasons: list[int],
    history: pd.DataFrame,
    *,
    cutoff_season: int,
    created_at: datetime,
    source_provenance: dict[str, dict[str, str]],
) -> Path:
    """Write data-only JSON; inference never unpickles remote executable objects."""
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "model_version": heisman.MODEL_VERSION,
        "cutoff_season": cutoff_season,
        "training_seasons": sorted(int(season) for season in training_seasons),
        "created_at": created_at.isoformat(),
        "seed_sha256": _seed_digest(),
        "source_provenance": source_provenance,
        "features": heisman.FEATURES,
        "evaluation_kind": EVALUATION_KIND,
        "mean": model.mean.tolist(),
        "scale": model.scale.tolist(),
        "beta": model.beta.tolist(),
        "history": json.loads(history.to_json(orient="table", index=False)),
    }
    _validate(payload, cutoff_season)
    path = model_path(cutoff_season)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False))
    temporary.replace(path)
    return path


def _validate(payload: dict, cutoff_season: int) -> None:
    expected = {
        "artifact_version": ARTIFACT_VERSION,
        "model_version": heisman.MODEL_VERSION,
        "cutoff_season": cutoff_season,
        "seed_sha256": _seed_digest(),
        "features": heisman.FEATURES,
        "evaluation_kind": EVALUATION_KIND,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"Heisman model artifact has incompatible {name}")
    seasons = payload.get("training_seasons", [])
    if not seasons or any(int(season) >= cutoff_season for season in seasons):
        raise ValueError("Heisman model must train only on seasons before the forecast")
    provenance = payload.get("source_provenance", {})
    if set(provenance) != {str(season) for season in seasons}:
        raise ValueError(
            "Heisman model requires source provenance for every training season"
        )
    for sources in provenance.values():
        for name in ("manifest_sha256", "games_sha256"):
            digest = sources.get(name, "")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"Heisman model has invalid {name} provenance")
    for name in ("mean", "scale", "beta"):
        values = np.asarray(payload[name], dtype=float)
        if values.shape != (len(heisman.FEATURES),) or not np.isfinite(values).all():
            raise ValueError(f"Heisman model artifact has invalid {name}")
        if name == "scale" and np.any(values <= 0):
            raise ValueError("Heisman model artifact has nonpositive scale")
    history = pd.DataFrame(payload["history"]["data"])
    required = {
        *heisman.HISTORY_COLUMNS,
        "season",
        "week",
        "training_seasons",
        "evaluation_kind",
        "candidate_count",
        "winner_in_pool",
        "actual_winner_predicted_rank",
        "actual_winner_predicted_share",
        "ballot_share_covered",
    }
    if history.empty or not required.issubset(history.columns):
        raise ValueError("Heisman model requires chronological weekly evaluation")
    for row in history.itertuples(index=False):
        if (
            int(row.season) >= cutoff_season
            or int(row.week) < 0
            or int(row.candidate_count) < 1
            or row.evaluation_kind != EVALUATION_KIND
            or not row.training_seasons
            or any(int(year) >= int(row.season) for year in row.training_seasons)
        ):
            raise ValueError(
                "Heisman evaluation violates chronological weekly provenance"
            )
        if not row.winner_in_pool and (
            not pd.isna(row.actual_winner_predicted_rank)
            or row.actual_winner_predicted_share != 0
        ):
            raise ValueError("Heisman evaluation must count missing winners as misses")


def load_heisman_model(
    cutoff_season: int,
) -> tuple[heisman.ShareModel, list[int], pd.DataFrame]:
    path = model_path(cutoff_season)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing trained Heisman model {path}. Run heisman-train --season "
            f"{cutoff_season} where the historical player snapshots already exist, "
            "then restore its versioned model artifact on this runner."
        )
    payload = json.loads(path.read_text())
    _validate(payload, cutoff_season)
    history = pd.DataFrame(payload["history"]["data"])
    model = heisman.ShareModel(
        *(np.asarray(payload[name], dtype=float) for name in ("mean", "scale", "beta"))
    )
    return model, payload["training_seasons"], history


def _validate_trait_pair(
    model: grade.GradeModel,
    tables: expectations.ExpectationTables,
    model_cutoff: int,
    cutoff_season: int,
) -> None:
    """The position grade model and its expectation tables may only serve a later season."""
    if model_cutoff >= cutoff_season:
        raise ValueError(
            f"position grade model with cutoff {model_cutoff} cannot serve {cutoff_season}"
        )
    grade.validate_grade_model(model, cutoff_season)
    if not tables.source_seasons or any(
        int(season) >= cutoff_season for season in tables.source_seasons
    ):
        raise ValueError(
            f"position grade expectation tables from {tables.source_seasons} "
            f"cannot serve {cutoff_season}"
        )


def trait_model_paths(cutoff_season: int) -> list[Path]:
    """The latest position grade model before ``cutoff_season`` and its expectation
    tables, validated, or ``[]`` when no model exists. A model without its tables is
    an error."""
    models = PROCESSED_DIR / "players" / "models"
    try:
        model_path = grade.latest_grade_model_path(models, cutoff_season)
    except FileNotFoundError:
        return []
    model_cutoff = grade.grade_cutoff(model_path)
    tables_path = expectations.expectation_tables_path(models, model_cutoff)
    if not tables_path.exists():
        raise FileNotFoundError(
            f"{model_path.name} has no {tables_path.name}; rerun position-traits-fit"
        )
    _validate_trait_pair(
        grade.load_grade_model(model_path),
        expectations.load_expectation_tables(tables_path),
        model_cutoff,
        cutoff_season,
    )
    return [model_path, tables_path]


def export_runtime_bundle(cutoff_season: int, destination: Path) -> Path:
    """Package validated coefficients and existing frozen WPA parameters for CI, plus the
    latest position grade model and expectation tables before the season when they exist."""
    from backend.model.ingame import load_baseline_params

    load_heisman_model(cutoff_season)
    player_value.load_opponent_prior(cutoff_season)
    prior = PROCESSED_DIR.joinpath(*player_value.opponent_prior_artifact(cutoff_season))
    baseline = PROCESSED_DIR / "ingame" / "baseline_summary.parquet"
    load_baseline_params(pd.read_parquet(baseline))
    traits = trait_model_paths(cutoff_season)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as bundle:
        for source in (model_path(cutoff_season), baseline, prior, *traits):
            bundle.write(source, str(source.relative_to(PROCESSED_DIR)))
    return destination


def _trait_members(extra: set[str]) -> tuple[str, str, int] | None:
    """``(model name, tables name, model cutoff)`` when ``extra`` is exactly one grade pair."""
    if not extra:
        return None
    models = [
        name
        for name in extra
        if name.startswith("players/models/")
        and grade.GRADE_NAME.fullmatch(name.removeprefix("players/models/"))
    ]
    if len(models) != 1:
        raise ValueError("player runtime bundle contains unexpected files")
    model_cutoff = grade.grade_cutoff(Path(models[0]))
    tables_name = (
        "players/models/"
        + expectations.expectation_tables_path(Path("."), model_cutoff).name
    )
    if extra != {models[0], tables_name}:
        raise ValueError("player runtime bundle contains unexpected files")
    return models[0], tables_name, model_cutoff


def install_runtime_bundle(bundle: bytes, cutoff_season: int) -> None:
    """Restore only the expected data files after validating every one.

    The Heisman model, the WPA baseline and the opponent prior are required. The
    position grade model and its expectation tables are optional but travel as a
    pair whose model cutoff and table seasons precede ``cutoff_season``.
    """
    from backend.model.ingame import load_baseline_params

    model_name = f"players/models/heisman_{cutoff_season}.json"
    baseline_name = "ingame/baseline_summary.parquet"
    prior_name = "/".join(player_value.opponent_prior_artifact(cutoff_season))
    required = {model_name, baseline_name, prior_name}
    with ZipFile(BytesIO(bundle)) as archive:
        names = set(archive.namelist())
        if not required <= names or len(names) != len(archive.namelist()):
            raise ValueError("player runtime bundle contains unexpected files")
        traits = _trait_members(names - required)
        if any(item.file_size > 10_000_000 for item in archive.infolist()):
            raise ValueError("player runtime bundle exceeds the data size limit")
        members = {name: archive.read(name) for name in sorted(names)}
    _validate(json.loads(members[model_name]), cutoff_season)
    load_baseline_params(pd.read_parquet(BytesIO(members[baseline_name])))
    player_value.validate_opponent_prior(
        pd.read_parquet(BytesIO(members[prior_name])), cutoff_season
    )
    if traits:
        grade_name, tables_name, model_cutoff = traits
        _validate_trait_pair(
            grade.grade_model_from_payload(json.loads(members[grade_name])),
            expectations.load_expectation_tables(BytesIO(members[tables_name])),
            model_cutoff,
            cutoff_season,
        )
    for name, data in members.items():
        path = PROCESSED_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
