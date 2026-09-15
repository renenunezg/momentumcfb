"""Append-only paid-source receipts, shared by weather, benchmarks and live feeds."""

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import pandas as pd

from backend.config import RAW_DIR


def snapshot_root() -> Path:
    return RAW_DIR / "tier2"


def save_snapshot(
    root,
    endpoint,
    params,
    payload,
    requested_at,
    fetched_at,
    *,
    request_seconds=0.0,
    remaining=None,
):
    requested, fetched = pd.Timestamp(requested_at), pd.Timestamp(fetched_at)
    if requested.tzinfo is None or fetched.tzinfo is None or fetched < requested:
        raise ValueError("snapshot timestamps must be ordered and timezone-aware")
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    digest = sha256(encoded.encode()).hexdigest()
    query = sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    directory = Path(root) / endpoint.strip("/").replace("/", "_") / query
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{fetched.strftime('%Y%m%dT%H%M%S%fZ')}_{digest[:12]}.json"
    receipt = dict(
        endpoint=endpoint,
        params=params,
        requested_at=requested.isoformat(),
        fetched_at=fetched.isoformat(),
        request_seconds=request_seconds,
        remaining_calls=remaining,
        payload_sha256=digest,
        payload=payload,
    )
    # Repeated same receipt is safe; a different receipt never overwrites history.
    content = json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError("conflicting snapshot receipt")
    else:
        with path.open("x") as handle:
            handle.write(content)
    return path


def read_snapshot(path):
    receipt = json.loads(Path(path).read_text())
    encoded = json.dumps(
        receipt["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if sha256(encoded.encode()).hexdigest() != receipt["payload_sha256"]:
        raise ValueError("snapshot payload checksum mismatch")
    return receipt


def capture(client, endpoint, params, *, root=None, object_response=False):
    requested = datetime.now(timezone.utc)
    started = perf_counter()
    method = client.get_object if object_response else client.get
    # Bound individual probes; the caller reserves the whole sequential job.
    payload = method(endpoint, params, retries=1, timeout=30)
    return save_snapshot(
        root or snapshot_root(),
        endpoint,
        params,
        payload,
        requested,
        datetime.now(timezone.utc),
        request_seconds=perf_counter() - started,
        remaining=client.remaining,
    )


def receipts(endpoint, *, root=None):
    directory = (root or snapshot_root()) / endpoint.strip("/").replace("/", "_")
    for path in sorted(directory.glob("*/*.json")):
        yield path, read_snapshot(path)
