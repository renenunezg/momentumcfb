"""Publish live win probabilities across the Supabase boundary.

One CFBD scoreboard call covers the whole slate. None is made before a tracked
game's scheduled kickoff or after every tracked game is terminal, and the
worker exits as soon as nothing is left to watch. The frozen in-game baseline
scores each state against the projection published before kickoff; pregame
tables are never written here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import pandas as pd

from backend.cfbd.client import CFBDError
from backend.model.ingame import (
    MODEL_VERSION,
    REGULATION_SECONDS,
    IngameBaselineParams,
    build_serving_inputs,
    win_probability,
)
from backend.serving.cfbd_live import ACTIVE, FINAL

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PROBABILITY_SOURCE = "pregame_anchored_game_state"
CALLED_OFF = {"postponed", "canceled", "cancelled", "forfeit", "suspended"}
TERMINAL_STATES = {"Final", "Off"}
# The dispatcher treats a row newer than four minutes as a living worker.
HEARTBEAT_SECONDS = 120
SCHEDULE_REFRESH_SECONDS = 300
KICKOFF_LEAD = timedelta(minutes=15)
# Matches the dispatcher window, so a recovered worker resumes the same games.
KICKOFF_LOOKBACK = timedelta(hours=6)
# A game still unstarted this long after kickoff is abandoned, not polled.
NO_START_LIMIT = timedelta(hours=3)
VOLATILE_FIELDS = {"fetched_at", "worker_expires_at"}


@dataclass(frozen=True, slots=True)
class Game:
    game_id: int
    season: int
    week: int
    start: datetime
    home_team_id: int
    away_team_id: int
    home_team: str
    away_team: str
    home_margin: float
    margin_sd: float
    as_of: datetime
    model_version: str

    def anchor(self) -> dict:
        return {
            "home_margin": self.home_margin,
            "margin_sd": self.margin_sd,
            "as_of": self.as_of.isoformat(),
            "model_version": self.model_version,
        }


def _clock_seconds(clock) -> int:
    minutes, seconds = (int(part) for part in str(clock).split(":")[-2:])
    if not 0 <= seconds < 60 or not 0 <= minutes * 60 + seconds <= 900:
        raise ValueError("Invalid game clock")
    return minutes * 60 + seconds


def _points(team: dict) -> int:
    value = team.get("points")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Missing or invalid score")
    return value


def _possession(row: dict) -> str | None:
    value = row.get("possession")
    for side in ("home", "away"):
        if value in (side, row[f"{side}Team"].get("name")):
            return side
    return None


def _probability(
    game: Game,
    params: IngameBaselineParams,
    *,
    seconds_remaining: float,
    overtime: bool,
    margin: int,
    possession: str | None,
) -> float:
    state = pd.DataFrame(
        [
            {
                "game_id": game.game_id,
                "seconds_remaining": seconds_remaining,
                "is_overtime": overtime,
                # Unknown possession carries no possession value.
                "play_category": "scrimmage" if possession else "unknown",
                "offense_is_home": possession == "home",
                "yards_to_goal": float("nan"),
                "home_margin": margin,
            }
        ]
    )
    anchor = pd.DataFrame(
        [
            {
                "game_id": game.game_id,
                "model_week": game.week,
                "home_margin": game.home_margin,
                "margin_sd": game.margin_sd,
            }
        ]
    )
    return float(win_probability(build_serving_inputs(state, anchor), params)[0])


def _point(elapsed: int, home: int, away: int, probability: float) -> dict:
    return {"s": elapsed, "h": home, "a": away, "p": round(probability, 4)}


def score_game(
    game: Game,
    row: dict | None,
    params: IngameBaselineParams,
    history: list[dict],
    now: datetime,
    *,
    board_fetched: bool,
) -> dict:
    """One game's serving payload from its scoreboard row and frozen anchor."""
    pregame = _probability(
        game,
        params,
        seconds_remaining=REGULATION_SECONDS,
        overtime=False,
        margin=0,
        possession=None,
    )
    history = history or [_point(0, 0, 0, pregame)]
    result = {
        "schema_version": SCHEMA_VERSION,
        "game_id": game.game_id,
        "season": game.season,
        "week": game.week,
        "abstract_state": "Pre",
        "status": "scheduled",
        "home_team": game.home_team,
        "away_team": game.away_team,
        "fetched_at": now.isoformat(),
        "probability_source": PROBABILITY_SOURCE,
        "model_version": MODEL_VERSION,
        "anchor": game.anchor(),
        "home_win_probability": pregame,
        "away_win_probability": 1 - pregame,
        "history": history,
    }

    def unavailable(reason: str) -> dict:
        result.update(
            home_win_probability=None,
            away_win_probability=None,
            unavailable_reason=reason,
        )
        return result

    if game.as_of >= game.start:
        return unavailable("Pregame projection was not frozen before kickoff")
    if not board_fetched:
        return result
    status = str((row or {}).get("status", "missing")).lower()
    result["status"] = status
    if status in CALLED_OFF:
        result["abstract_state"] = "Off"
        return unavailable("Game was called off")
    if row is None or status not in ACTIVE | FINAL:
        if now - game.start > NO_START_LIMIT:
            result["abstract_state"] = "Off"
            return unavailable("Game did not start")
        return result
    result["abstract_state"] = "Final" if status in FINAL else "Live"
    try:
        home_team, away_team = row["homeTeam"], row["awayTeam"]
        if (home_team["id"], away_team["id"]) != (
            game.home_team_id,
            game.away_team_id,
        ):
            raise ValueError("Scoreboard teams differ from the projection")
        home, away = _points(home_team), _points(away_team)
        result.update(home_score=home, away_score=away)
        if status in FINAL:
            if home == away:
                raise ValueError("Tied final has no winner")
            probability, elapsed = float(home > away), int(REGULATION_SECONDS)
        else:
            period = row.get("period")
            if isinstance(period, bool) or not isinstance(period, int) or period < 1:
                raise ValueError("Missing game period")
            overtime = period > 4
            remaining = (
                0 if overtime else (4 - period) * 900 + _clock_seconds(row.get("clock"))
            )
            possession = _possession(row)
            probability = _probability(
                game,
                params,
                seconds_remaining=remaining,
                overtime=overtime,
                margin=home - away,
                possession=possession,
            )
            elapsed = int(REGULATION_SECONDS) - remaining
            result.update(
                period=period,
                clock=row.get("clock"),
                possession=possession,
                situation=row.get("situation"),
            )
    except (KeyError, TypeError, ValueError) as exc:
        return unavailable(str(exc))
    result.update(
        home_win_probability=probability, away_win_probability=1 - probability
    )
    point = _point(elapsed, home, away, probability)
    if history[-1] != point:
        result["history"] = [*history, point]
    return result


def _signature(payload: dict) -> str:
    content = {k: v for k, v in payload.items() if k not in VOLATILE_FIELDS}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class LivePublisher:
    """Tracks the games in the kickoff window and writes only what changed."""

    def __init__(
        self,
        params: IngameBaselineParams,
        *,
        load_games,
        load_saved,
        fetch_board,
        write=None,
        expires_at: datetime | None = None,
    ):
        self.params = params
        self.load_games = load_games
        self.load_saved = load_saved
        self.fetch_board = fetch_board
        self.write = write
        self.expires_at = expires_at.isoformat() if expires_at else None
        self.games: dict[int, Game] = {}
        self.history: dict[int, list[dict]] = {}
        self.written: dict[int, tuple[str, datetime]] = {}
        self.done: set[int] = set()
        self.next_schedule: datetime | None = None

    def _refresh_schedule(self, now: datetime) -> None:
        if self.next_schedule and now < self.next_schedule:
            return
        window = {
            game.game_id: game
            for game in self.load_games(now - KICKOFF_LOOKBACK, now + KICKOFF_LEAD)
        }
        unseen = [gid for gid in window if gid not in self.games]
        saved = self.load_saved(unseen) if unseen and self.write else {}
        for game_id in unseen:
            game, payload = window[game_id], saved.get(game_id)
            if payload:
                # A projection republished mid-game must not move the anchor.
                anchor = payload["anchor"]
                game = replace(
                    game,
                    home_margin=float(anchor["home_margin"]),
                    margin_sd=float(anchor["margin_sd"]),
                    as_of=datetime.fromisoformat(anchor["as_of"]),
                    model_version=anchor["model_version"],
                )
                self.history[game_id] = payload["history"]
                if payload["abstract_state"] in TERMINAL_STATES:
                    self.done.add(game_id)
            self.games[game_id] = game
        for game_id in set(self.games) - set(window):
            del self.games[game_id]
            self.history.pop(game_id, None)
            self.written.pop(game_id, None)
            self.done.discard(game_id)
        self.next_schedule = now + timedelta(seconds=SCHEDULE_REFRESH_SECONDS)

    def poll(self, now: datetime) -> bool:
        """Score the open games once; False when nothing is left to watch."""
        self._refresh_schedule(now)
        open_games = [g for g in self.games.values() if g.game_id not in self.done]
        if not open_games:
            return False
        # Before the first scheduled kickoff the database alone is enough.
        board_fetched = any(game.start <= now for game in open_games)
        board = (
            {int(row["id"]): row for row in self.fetch_board()} if board_fetched else {}
        )
        changed, finished = [], []
        for game in open_games:
            payload = score_game(
                game,
                board.get(game.game_id),
                self.params,
                self.history.get(game.game_id, []),
                now,
                board_fetched=board_fetched and game.start <= now,
            )
            payload["worker_expires_at"] = self.expires_at
            self.history[game.game_id] = payload["history"]
            terminal = payload["abstract_state"] in TERMINAL_STATES
            signature = _signature(payload)
            previous = self.written.get(game.game_id)
            if (
                previous
                and previous[0] == signature
                and (now - previous[1]).total_seconds() < HEARTBEAT_SECONDS
            ):
                continue
            changed.append((payload, signature))
            if terminal:
                finished.append(game.game_id)
        if changed and self.write:
            self.write([payload for payload, _ in changed])
        # Acknowledge only after the batch commits, so a failed final retries.
        for payload, signature in changed:
            self.written[payload["game_id"]] = (signature, now)
            log.info(
                "%s %s @ %s: %s, home WP %s%s",
                payload["game_id"],
                payload["away_team"],
                payload["home_team"],
                payload["abstract_state"],
                payload["home_win_probability"],
                " (published)" if self.write else " (read only)",
            )
        self.done.update(finished)
        return len(self.done) < len(self.games)


def run(
    publisher: LivePublisher,
    *,
    watch: bool,
    interval: int,
    duration: int | None = None,
    sleep=time.sleep,
    monotonic=time.monotonic,
    now=lambda: datetime.now(timezone.utc),
) -> None:
    deadline = monotonic() + duration if duration else float("inf")
    while True:
        started = monotonic()
        try:
            more = publisher.poll(now())
        except CFBDError:
            # A spent or refused quota must stop the worker, not loop on it.
            raise
        except Exception:
            if not watch:
                raise
            log.exception("Refresh failed; previous snapshots age visibly on the site")
            more = True
        if not more:
            log.info("No live or imminent games; stopping until the next dispatch")
            return
        if not watch or monotonic() >= deadline:
            return
        sleep(max(1, min(deadline - monotonic(), interval - (monotonic() - started))))
        if monotonic() >= deadline:
            return


def load_games(start: datetime, end: datetime) -> list[Game]:
    from sqlalchemy import text

    from backend.db import CFB_SCHEMA, engine

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT game_id, season, week, start_date, home_team_id, "
                "away_team_id, home_team, away_team, home_margin, margin_sd, "
                f"as_of, model_version FROM {CFB_SCHEMA}.game_projections "
                "WHERE start_date BETWEEN :start AND :end"
            ),
            {"start": start, "end": end},
        ).mappings()
        return [
            Game(
                game_id=int(row["game_id"]),
                season=int(row["season"]),
                week=int(row["week"]),
                start=row["start_date"],
                home_team_id=int(row["home_team_id"]),
                away_team_id=int(row["away_team_id"]),
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_margin=float(row["home_margin"]),
                margin_sd=float(row["margin_sd"]),
                as_of=row["as_of"],
                model_version=row["model_version"],
            )
            for row in rows
        ]


def load_saved(game_ids: list[int]) -> dict[int, dict]:
    from sqlalchemy import text

    from backend.db import CFB_SCHEMA, engine

    with engine.connect() as conn:
        return dict(
            conn.execute(
                text(
                    f"SELECT game_id, payload FROM {CFB_SCHEMA}.live_win_probability "
                    "WHERE game_id = ANY(:game_ids)"
                ),
                {"game_ids": game_ids},
            )
            .tuples()
            .all()
        )


def write_snapshots(payloads: list[dict]) -> None:
    from sqlalchemy import text

    from backend.db import CFB_SCHEMA, engine

    rows = [
        {"game_id": p["game_id"], "updated_at": p["fetched_at"], "payload": p}
        for p in payloads
    ]
    with engine.begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO {CFB_SCHEMA}.live_win_probability "
                "(game_id, updated_at, payload) "
                "SELECT game_id, updated_at, payload "
                "FROM jsonb_to_recordset(CAST(:snapshots AS jsonb)) "
                "AS incoming(game_id bigint, updated_at timestamptz, payload jsonb) "
                "ON CONFLICT (game_id) DO UPDATE SET "
                "updated_at = EXCLUDED.updated_at, payload = EXCLUDED.payload "
                f"WHERE {CFB_SCHEMA}.live_win_probability.updated_at "
                "< EXCLUDED.updated_at"
            ),
            {"snapshots": json.dumps(rows, allow_nan=False)},
        )
