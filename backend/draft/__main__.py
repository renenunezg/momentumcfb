"""Research entry point: python -m backend.draft {snapshot,build}."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.config import DATA_DIR
from backend.draft.research import build
from backend.draft.sources import snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser(
        "snapshot", help="cache public NFL and bounded CFBD draft data"
    )
    fetch.add_argument("--season", type=int, required=True)
    fetch.add_argument("--draft-years", type=int, nargs="+", required=True)
    fetch.add_argument("--cache", type=Path, default=DATA_DIR / "raw/draft")
    research = sub.add_parser("build", help="audit local evidence; never publishes")
    research.add_argument("--season", type=int, required=True)
    research.add_argument("--cfb-data", type=Path, default=DATA_DIR)
    research.add_argument("--nfl-data", type=Path, required=True)
    research.add_argument("--cache", type=Path, default=DATA_DIR / "raw/draft")
    research.add_argument("--output", type=Path, required=True)
    research.add_argument("--as-of", default=datetime.now(timezone.utc).isoformat())
    args = parser.parse_args()
    if args.command == "snapshot":
        result = snapshot(args.cache, args.season, args.draft_years)
    else:
        result = build(
            args.cfb_data,
            args.nfl_data,
            args.cache,
            args.output,
            args.season,
            args.as_of,
        )
        result = {k: v for k, v in result.items() if k != "sources"}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
