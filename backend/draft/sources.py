"""Explicit, bounded source snapshots for reproducible draft research."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from backend.cfbd.client import CFBDClient


def receipt(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def snapshot(root: Path, season: int, years: list[int]) -> dict:
    """Reuse cached files; no refresh or production writes are implicit."""
    root.mkdir(parents=True, exist_ok=True)
    assets = [
        ("draft_picks", "draft_picks"),
        ("combine", "combine"),
        ("contracts", "historical_contracts"),
        ("depth_charts", f"depth_charts_{season}"),
        ("snap_counts", f"snap_counts_{season}"),
        ("rosters", f"roster_{season}"),
    ]
    for tag, name in assets:
        path = root / f"{name}.parquet"
        if path.exists():
            continue
        url = (
            "https://github.com/nflverse/nflverse-data/releases/download/"
            f"{tag}/{name}.parquet"
        )
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(response.content)
        temporary.replace(path)
        path.with_suffix(".source.json").write_text(
            json.dumps(
                {
                    "source": url,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    **receipt(path),
                },
                indent=2,
            )
        )

    missing = [
        y for y in sorted(set(years)) if not (root / f"cfbd_picks_{y}.json").exists()
    ]
    if len(missing) > 100:
        raise ValueError("Draft snapshots are limited to 100 CFBD requests")
    client = None
    if missing:
        client = CFBDClient(max_calls=len(missing), min_remaining=3000)
        client.ensure_budget(len(missing))
    for year in missing:
        rows = client.get("/draft/picks", {"year": year}, retries=1, timeout=45)
        path = root / f"cfbd_picks_{year}.json"
        path.write_text(
            json.dumps(
                {
                    "source": "https://api.collegefootballdata.com/draft/picks",
                    "params": {"year": year},
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "rows": rows,
                },
                indent=2,
            )
        )
    return {
        "cfbd_calls": client.calls_used if client else 0,
        "remaining": client.remaining if client else None,
        "cache": str(root.resolve()),
    }
