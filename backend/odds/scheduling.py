"""Rearm the existing kickoff dispatcher from a published weekly forecast."""

import json
import re
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from backend import db

# A capture run starts this long before its first kickoff.
DISPATCH_LEAD = pd.Timedelta(minutes=30)
# A fresh runner needs about this long to install and validate before polling,
# so a kickoff closer than this to a chained rearm is left to the next window.
RUNNER_SETUP = pd.Timedelta(minutes=12)


def _priced_starts(projections: pd.DataFrame) -> pd.Series:
    priced = projections[projections["market_home_spread"].notna()]
    return pd.to_datetime(priced["start_date"], utc=True)


def schedule_weekly_kickoff(projections: pd.DataFrame, season: int) -> str:
    """Capture the first priced kickoff cluster, thirty minutes before kickoff."""
    dispatch_at = _priced_starts(projections).min() - DISPATCH_LEAD
    if pd.isna(dispatch_at) or dispatch_at <= datetime.now(timezone.utc):
        raise ValueError(
            "No priced future kickoff has thirty minutes of scheduling lead"
        )
    return _arm(dispatch_at, season)


def schedule_next_kickoff(
    projections: pd.DataFrame, season: int, now: datetime | None = None
) -> str | None:
    """After a capture run, arm the next priced kickoff a fresh runner can reach.

    Each run captures one kickoff window and then moves the same dispatcher to
    the next, so every window of the week is captured, not only the first.
    Returns None once the week's last priced kickoff has passed; the weekly
    update arms the next week.
    """
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    starts = _priced_starts(projections)
    upcoming = starts[starts > now_ts + RUNNER_SETUP]
    if upcoming.empty:
        return None
    earliest = (now_ts + pd.Timedelta(minutes=2)).ceil("min")
    return _arm(max(upcoming.min() - DISPATCH_LEAD, earliest), season)


def _arm(dispatch_at: pd.Timestamp, season: int) -> str:
    if not db.writes_allowed():
        raise RuntimeError("Kickoff scheduling requires MOMENTUMCFB_DB_WRITES=1")
    schedule = dispatch_at.strftime("%M %H %d %m *")
    payload = json.dumps(
        {"event_type": "kickoff-capture", "client_payload": {"season": str(season)}}
    )
    with db.engine.begin() as connection:
        jobs = (
            connection.execute(
                text(
                    "SELECT jobid, command FROM cron.job WHERE jobname IN "
                    "('cfb-kickoff-capture', 'cfb-kickoff-capture-sep5')"
                )
            )
            .mappings()
            .all()
        )
        if len(jobs) != 1:
            raise ValueError("Expected exactly one existing CFB kickoff dispatcher")
        job = jobs[0]
        # Keep the existing endpoint and secret-backed authorization unchanged.
        # The payload follows the headers as the final http_post argument.
        command = str(job["command"])
        match = re.search(r"\bbody\s*:=.*\)\s*(?:WHERE\s+.*)?;\s*$", command, re.DOTALL)
        if match is None or "net.http_post" not in command[: match.start()]:
            raise ValueError("Kickoff dispatcher has an unsupported command shape")
        command = command[: match.start()] + (
            f"body := '{payload}'::jsonb) "
            f"WHERE (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::date = '{dispatch_at.date().isoformat()}'::date;"
        )
        # The date guard prevents a five-field cron from dispatching next year.
        # The weekly update arms the first window; each capture run arms the next.
        connection.exec_driver_sql("EXPLAIN " + command)
        connection.execute(
            text(
                "SELECT cron.alter_job(:job_id, schedule := :schedule, "
                "command := :command, active := true)"
            ),
            {"job_id": job["jobid"], "schedule": schedule, "command": command},
        )
    return dispatch_at.isoformat()
