"""Production must reject mismatched historical runtime bundles."""

import hashlib
import json

import pytest

from backend.runtime_revision import validate_installed_runtime


def test_paired_runtime_revision_accepts_reviewed_files_and_rejects_stale_member(
    tmp_path,
):
    artifacts = {}
    for name, data in [
        ("preseason/ratings/2026_01.parquet", b"revised-team"),
        ("players/opponent_priors/2026.parquet", b"revised-player"),
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
        artifacts[name] = hashlib.sha256(data).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"season": 2026, "revision_id": "reviewed", "artifacts": artifacts})
    )
    receipt = validate_installed_runtime(
        2026, processed_dir=tmp_path, manifest_path=manifest
    )
    assert set(receipt.artifact) == set(artifacts)
    assert receipt.revision_id.eq("reviewed").all()
    (tmp_path / "players/opponent_priors/2026.parquet").write_bytes(b"stale-player")
    with pytest.raises(ValueError, match="restore both reviewed runtime bundles"):
        validate_installed_runtime(2026, processed_dir=tmp_path, manifest_path=manifest)


def test_runtime_revision_refuses_missing_artifact(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "season": 2026,
                "revision_id": "reviewed",
                "artifacts": {"missing.parquet": "0" * 64},
            }
        )
    )
    with pytest.raises(FileNotFoundError):
        validate_installed_runtime(2026, processed_dir=tmp_path, manifest_path=manifest)
