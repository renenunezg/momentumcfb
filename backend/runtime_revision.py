"""Reject mixed historical data vintages before production inference."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.etl import store

MANIFEST_DIR = Path(__file__).parent / "model"
RECEIPT_COLUMNS = ["revision_id", "season", "artifact", "sha256", "validated_at"]


def validate_installed_runtime(
    season: int,
    *,
    processed_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    """Verify every member of the reviewed player/team runtime before use.

    The manifest lives with deployed code, independently of cache and ZIP files.
    Changing either runtime without updating the reviewed manifest fails closed.
    Published forecast archives are never changed by this check.
    """
    root = processed_dir or store.PROCESSED_DIR
    path = manifest_path or MANIFEST_DIR / f"data_revision_{season}.json"
    manifest = json.loads(path.read_text())
    if manifest["season"] != season or not manifest["artifacts"]:
        raise ValueError("Historical data revision has incompatible season or no files")
    expected_grade = manifest.get("grade_model")
    if expected_grade:
        from backend.players.traits.grade import latest_grade_model_path

        selected = latest_grade_model_path(root / "players" / "models", season)
        if str(selected.relative_to(root)) != expected_grade:
            raise ValueError(
                "A cached grade model overrides the reviewed data revision"
            )
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for name, expected in sorted(manifest["artifacts"].items()):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Historical data revision contains an invalid path")
        source = root / relative
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Historical data revision {manifest['revision_id']} mismatch: {name}; "
                "restore both reviewed runtime bundles before forecasting"
            )
        rows.append(
            dict(
                revision_id=manifest["revision_id"],
                season=season,
                artifact=name,
                sha256=actual,
                validated_at=now,
            )
        )
    return pd.DataFrame(rows, columns=RECEIPT_COLUMNS)


def forecast_revision_receipt(season: int) -> pd.DataFrame:
    """Production opts in explicitly; offline historical replay remains independent."""
    if os.getenv("CFB_REQUIRE_DATA_REVISION") != "1":
        return pd.DataFrame(columns=RECEIPT_COLUMNS)
    return validate_installed_runtime(season)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    season = parser.parse_args().season
    receipt = validate_installed_runtime(season)
    store.write_processed(receipt, "runtime", f"data_revision_{season}.parquet")
    print(receipt.to_string(index=False))


if __name__ == "__main__":
    main()
