"""Run the weekly CFP refresh after a complete, verified weekly model archive."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from backend.cfbd.client import CFBDClient
from backend.etl.store import completed_forecast_run
from backend.playoffs.forecast import build
from backend.playoffs.publish import publish, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/cfb_playoffs"))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    if args.season != 2026 or now >= datetime(2026, 12, 7, tzinfo=timezone.utc):
        print("No refresh: outside the reviewed 2026 pre-selection window")
        return
    if args.archive is None:
        from backend.config import PROCESSED_DIR

        pointers = list(
            (PROCESSED_DIR / "weekly" / "forecast_log").glob(f"{args.season}_*.json")
        )
        if not pointers:
            raise ValueError("No completed weekly model archive")
        latest = max(
            pointers,
            key=lambda p: json.loads(p.read_text())["directory"].split("_")[-1],
        )
        week = int(latest.stem.split("_")[1])
        args.archive = completed_forecast_run("weekly", args.season, week)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.inputs is None:
        args.inputs = args.output / now.strftime("inputs-%Y%m%dT%H%M%SZ")
        args.inputs.mkdir()
        client = CFBDClient(
            max_calls=int(os.getenv("CFBD_MAX_CALLS", "2")),
            min_remaining=3000,
            usage_file=Path(os.environ["CFBD_USAGE_FILE"])
            if os.getenv("CFBD_USAGE_FILE")
            else args.inputs / "usage.json",
        )
        client.ensure_budget(2)
        for name, endpoint, params in [
            ("games", "/games", {"year": args.season, "seasonType": "regular"}),
            ("rankings", "/rankings", {"year": args.season}),
        ]:
            data = client.get(endpoint, params, retries=1, timeout=30)
            (args.inputs / f"{name}.json").write_text(json.dumps(data))
        (args.inputs / "retrieved.json").write_text(
            json.dumps(
                {
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "source": "CFBD",
                    "calls": client.calls_used,
                }
            )
        )
    build(
        SimpleNamespace(
            archive=args.archive,
            inputs=args.inputs,
            output=args.output,
            n=20000,
            sensitivity_n=5000,
            seed=20261018,
        )
    )
    path = args.output / "forecast.json"
    validate(json.loads(path.read_text()))
    if args.publish:
        publish(path)


if __name__ == "__main__":
    main()
