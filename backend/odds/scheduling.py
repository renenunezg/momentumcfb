"""Rearm the existing kickoff dispatcher from a published weekly forecast."""

import json
import re
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from backend import db


def schedule_weekly_kickoff(projections: pd.DataFrame, season: int) -> str:
    """Capture the first priced kickoff cluster, thirty minutes before kickoff."""
    if not db.writes_allowed():
        raise RuntimeError("Kickoff scheduling requires MOMENTUMCFB_DB_WRITES=1")
    priced = projections[projections["market_home_spread"].notna()]
    starts = pd.to_datetime(priced["start_date"], utc=True)
    dispatch_at = starts.min() - pd.Timedelta(minutes=30)
    if pd.isna(dispatch_at) or dispatch_at <= datetime.now(timezone.utc):
        raise ValueError(
            "No priced future kickoff has thirty minutes of scheduling lead"
        )
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
        # Each successful weekly publication rearms this same job for its games.
        connection.exec_driver_sql("EXPLAIN " + command)
        connection.execute(
            text(
                "SELECT cron.alter_job(:job_id, schedule := :schedule, "
                "command := :command, active := true)"
            ),
            {"job_id": job["jobid"], "schedule": schedule, "command": command},
        )
    return dispatch_at.isoformat()
