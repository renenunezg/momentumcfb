import logging
from argparse import Namespace

log = logging.getLogger(__name__)


def handle_qb_availability_reports(args: Namespace) -> None:
    """Collect Power 4 availability reports into cfb.qb_availability.

    Dry by default: prints the quarterback rows that would be written. With
    --write, inserts new rows and updates rows whose player or status changed.
    """
    from datetime import datetime, timezone

    import pandas as pd

    from backend.etl.availability_reports import (
        changed_rows,
        collect_quarterback_reports,
        write_rows,
    )

    legacy = args.legacy or args.season < datetime.now(timezone.utc).year
    rows, quarterbacks = collect_quarterback_reports(args.season, legacy=legacy)
    log.info(
        "%s: %d quarterback report rows across %d team-weeks; %d name the team's "
        "previous starter with a storable status",
        args.season,
        len(quarterbacks),
        quarterbacks[["team", "week"]].drop_duplicates().shape[0],
        len(rows),
    )
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        log.info(
            "\n%s",
            rows[["week", "team", "player", "status", "notes"]].to_string(index=False),
        )
    if not args.write:
        log.info("dry run; pass --write to insert or update the table")
        return
    from backend.publish import fetch_qb_availability

    pending = changed_rows(rows, fetch_qb_availability(args.season))
    written = write_rows(pending)
    log.info("%s: wrote %d new or changed rows", args.season, written)
