"""Append-only capture and replay of live sportsbook lines.

Each poll stores three timestamped parquet files under
``raw/live_odds/<season>/{polls,events,offers}/<snapshot_id>.parquet``.
Snapshot ids are derived from the odds fetch time, files are never
rewritten, and game phase comes only from The Odds API scores feed so
line movement alone can never mark a pregame offer as live.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from backend.config import RAW_DIR
from backend.etl.ingest import write_parquet
from backend.odds.client import (
    OddsAPIClient,
    OddsAPIError,
    OddsSnapshot,
    ScoreboardSnapshot,
)
from backend.odds.markets import match_event

PHASES = ("pregame", "live", "final", "unknown")
DIVISION_ONE = {"fbs", "fcs"}
LIVE_MARKETS = ("spreads",)
CLOSING_MARKETS = ("spreads", "totals")
SUPPORTED_LIVE_MARKETS = frozenset(CLOSING_MARKETS)
SCORES_COST = 1

POLL_COLUMNS = [
    "snapshot_id",
    "season",
    "poll_status",
    "error",
    "odds_fetched_at",
    "scores_fetched_at",
    "commence_from",
    "commence_to",
    "days_from",
    "configured_bookmakers",
    "requested_markets",
    "odds_request_cost",
    "odds_requests_used",
    "odds_requests_remaining",
    "scores_request_cost",
    "scores_requests_used",
    "scores_requests_remaining",
    "event_count",
    "pregame_event_count",
    "live_event_count",
    "final_event_count",
    "unknown_event_count",
    "unmatched_event_count",
    "offer_count",
    "live_offer_count",
    "dropped_unmatched_offer_count",
    "dropped_untimestamped_offer_count",
    "max_offer_staleness_seconds",
    "median_offer_staleness_seconds",
    "events_sha256",
    "offers_sha256",
]

EVENT_COLUMNS = [
    "snapshot_id",
    "fetched_at",
    "odds_api_event_id",
    "commence_time",
    "odds_home_team",
    "odds_away_team",
    "phase",
    "status_source",
    "status_completed",
    "scoreboard_last_update",
    "scoreboard_staleness_seconds",
    "home_score",
    "away_score",
    "game_id",
    "match_score",
    "matched",
    "has_odds",
]

OFFER_COLUMNS = [
    "snapshot_id",
    "fetched_at",
    "game_id",
    "match_score",
    "odds_api_event_id",
    "phase",
    "commence_time",
    "provider_key",
    "provider",
    "market",
    "selection",
    "point",
    "price",
    "provider_last_update",
    "staleness_seconds",
    "event_link",
    "market_link",
    "bet_link",
    "execution_eligibility_verified",
]

# Explicit dtypes keep every snapshot's parquet schema identical, so
# multi-snapshot concatenation during replay never has to guess.
POLL_DTYPES = {
    "snapshot_id": "string",
    "season": "Int64",
    "poll_status": "string",
    "error": "string",
    "odds_fetched_at": "datetime64[ns, UTC]",
    "scores_fetched_at": "datetime64[ns, UTC]",
    "commence_from": "datetime64[ns, UTC]",
    "commence_to": "datetime64[ns, UTC]",
    "days_from": "Int64",
    "configured_bookmakers": "string",
    "requested_markets": "string",
    "odds_request_cost": "Int64",
    "odds_requests_used": "Int64",
    "odds_requests_remaining": "Int64",
    "scores_request_cost": "Int64",
    "scores_requests_used": "Int64",
    "scores_requests_remaining": "Int64",
    "event_count": "Int64",
    "pregame_event_count": "Int64",
    "live_event_count": "Int64",
    "final_event_count": "Int64",
    "unknown_event_count": "Int64",
    "unmatched_event_count": "Int64",
    "offer_count": "Int64",
    "live_offer_count": "Int64",
    "dropped_unmatched_offer_count": "Int64",
    "dropped_untimestamped_offer_count": "Int64",
    "max_offer_staleness_seconds": "float64",
    "median_offer_staleness_seconds": "float64",
    "events_sha256": "string",
    "offers_sha256": "string",
}

EVENT_DTYPES = {
    "snapshot_id": "string",
    "fetched_at": "datetime64[ns, UTC]",
    "odds_api_event_id": "string",
    "commence_time": "string",
    "odds_home_team": "string",
    "odds_away_team": "string",
    "phase": "string",
    "status_source": "string",
    "status_completed": "boolean",
    "scoreboard_last_update": "string",
    "scoreboard_staleness_seconds": "float64",
    "home_score": "float64",
    "away_score": "float64",
    "game_id": "Int64",
    "match_score": "float64",
    "matched": "boolean",
    "has_odds": "boolean",
}

OFFER_DTYPES = {
    "snapshot_id": "string",
    "fetched_at": "datetime64[ns, UTC]",
    "game_id": "Int64",
    "match_score": "float64",
    "odds_api_event_id": "string",
    "phase": "string",
    "commence_time": "string",
    "provider_key": "string",
    "provider": "string",
    "market": "string",
    "selection": "string",
    "point": "float64",
    "price": "float64",
    "provider_last_update": "string",
    "staleness_seconds": "float64",
    "event_link": "string",
    "market_link": "string",
    "bet_link": "string",
    "execution_eligibility_verified": "boolean",
}


@dataclass(frozen=True, slots=True)
class LiveSnapshot:
    snapshot_id: str
    poll: pd.DataFrame
    events: pd.DataFrame
    offers: pd.DataFrame


class LivePollSkipped(RuntimeError):
    def __init__(self, reason: str, requests_remaining: int | None):
        super().__init__(reason)
        self.requests_remaining = requests_remaining


class QuotaFloorReached(RuntimeError):
    def __init__(self, reason: str, odds: OddsSnapshot | None = None):
        super().__init__(reason)
        self.odds = odds


class PartialPaidPollError(OddsAPIError):
    def __init__(
        self,
        reason: str,
        odds: OddsSnapshot,
        scoreboard: ScoreboardSnapshot | None = None,
    ):
        super().__init__(reason)
        self.odds = odds
        self.scoreboard = scoreboard


def live_odds_dir(season: int):
    return RAW_DIR / "live_odds" / str(season)


def snapshot_id_for(fetched_at: datetime) -> str:
    return fetched_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def normalize_live_markets(markets: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(market) for market in markets))
    if not normalized:
        raise ValueError("live odds capture requires at least one market")
    unsupported = sorted(set(normalized) - SUPPORTED_LIVE_MARKETS)
    if unsupported:
        raise ValueError("unsupported live odds markets: " + ", ".join(unsupported))
    if "spreads" not in normalized:
        raise ValueError("live odds capture requires the spreads market")
    return normalized


def live_poll_quota_cost(markets: tuple[str, ...], days_from: int | None = None) -> int:
    """Maximum one-region cost for one odds and scores capture cycle."""
    scores_cost = 2 if days_from is not None else SCORES_COST
    return len(normalize_live_markets(markets)) + scores_cost


def resolve_phase(commence_time, completed, as_of: datetime) -> str:
    """Provider-backed phase; events absent from the scores feed stay unknown."""
    if completed is None:
        return "unknown"
    if bool(completed):
        return "final"
    if pd.to_datetime(commence_time, utc=True) <= pd.Timestamp(as_of):
        return "live"
    return "pregame"


def load_division_one_schedule(season: int) -> pd.DataFrame:
    """Load the freshest local schedule for game-id joining."""
    from backend.etl import store

    games_path = RAW_DIR / "games" / f"{season}.parquet"
    preseason_path = RAW_DIR / "preseason" / str(season) / "games.parquet"
    candidates = [path for path in (games_path, preseason_path) if path.exists()]
    if not candidates:
        raise FileNotFoundError(f"season {season} has no stored schedule")
    freshest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    if freshest == games_path:
        games = store.read_games(season)
    else:
        games = store.read_preseason_source(season, "games")
    return division_one_schedule(games)


def division_one_schedule(games: pd.DataFrame) -> pd.DataFrame:
    home = games["home_classification"].astype("string").str.lower().isin(DIVISION_ONE)
    away = games["away_classification"].astype("string").str.lower().isin(DIVISION_ONE)
    schedule = games[home.fillna(False) | away.fillna(False)].copy()
    schedule = schedule.rename(columns={"id": "game_id"})
    return schedule[["game_id", "start_date", "home_team", "away_team"]].reset_index(
        drop=True
    )


def frame_digest(frame: pd.DataFrame) -> str:
    """Content hash of a snapshot frame, stable across parquet round-trips."""
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def _staleness_seconds(fetched_at: datetime, provider_timestamp) -> float | None:
    stamp = pd.to_datetime(provider_timestamp, utc=True, errors="coerce")
    if pd.isna(stamp):
        return None
    return float((pd.Timestamp(fetched_at) - stamp).total_seconds())


def _team_score(scores, team: str) -> float | None:
    for entry in scores or []:
        if entry.get("name") == team:
            value = pd.to_numeric(entry.get("score"), errors="coerce")
            return None if pd.isna(value) else float(value)
    return None


def build_live_snapshot(
    season: int,
    odds: OddsSnapshot,
    scoreboard: ScoreboardSnapshot,
    schedule: pd.DataFrame,
    query_window: tuple[datetime, datetime],
    days_from: int | None,
    requested_markets: tuple[str, ...] = LIVE_MARKETS,
) -> LiveSnapshot:
    requested_markets = normalize_live_markets(requested_markets)
    fetched_at = odds.fetched_at
    snapshot_id = snapshot_id_for(fetched_at)
    scoreboard_by_id = {event.get("id"): event for event in scoreboard.events}
    odds_ids = {event.get("id") for event in odds.events}
    execution_verified = bool(odds.configured_bookmakers)

    event_rows = []
    offer_rows = []
    dropped_unmatched = 0
    dropped_untimestamped = 0

    merged_events: dict[str, dict] = {}
    for event in scoreboard.events:
        merged_events[event.get("id")] = event
    for event in odds.events:
        merged_events.setdefault(event.get("id"), event)

    for event_id, event in merged_events.items():
        status = scoreboard_by_id.get(event_id)
        completed = None if status is None else bool(status.get("completed"))
        phase = resolve_phase(event.get("commence_time"), completed, fetched_at)
        scoreboard_last_update = None if status is None else status.get("last_update")
        game_id, match_score = match_event(
            event.get("commence_time"),
            event.get("home_team", ""),
            event.get("away_team", ""),
            schedule,
        )
        event_rows.append(
            {
                "snapshot_id": snapshot_id,
                "fetched_at": fetched_at,
                "odds_api_event_id": event_id,
                "commence_time": event.get("commence_time"),
                "odds_home_team": event.get("home_team"),
                "odds_away_team": event.get("away_team"),
                "phase": phase,
                "status_source": "scores" if status is not None else "missing",
                "status_completed": completed,
                "scoreboard_last_update": scoreboard_last_update,
                "scoreboard_staleness_seconds": _staleness_seconds(
                    fetched_at, scoreboard_last_update
                ),
                "home_score": None
                if status is None
                else _team_score(status.get("scores"), status.get("home_team")),
                "away_score": None
                if status is None
                else _team_score(status.get("scores"), status.get("away_team")),
                "game_id": game_id,
                "match_score": match_score,
                "matched": game_id is not None,
                "has_odds": event_id in odds_ids,
            }
        )

    phase_by_id = {row["odds_api_event_id"]: row["phase"] for row in event_rows}
    game_by_id = {row["odds_api_event_id"]: row["game_id"] for row in event_rows}
    score_by_id = {row["odds_api_event_id"]: row["match_score"] for row in event_rows}

    for event in odds.events:
        event_id = event.get("id")
        game_id = game_by_id.get(event_id)
        for bookmaker in event.get("bookmakers") or []:
            for market in bookmaker.get("markets") or []:
                market_key = market.get("key")
                if market_key not in {"spreads", "totals"}:
                    continue
                for outcome in market.get("outcomes") or []:
                    if market_key == "spreads":
                        if outcome.get("name") == event.get("home_team"):
                            selection = "home"
                        elif outcome.get("name") == event.get("away_team"):
                            selection = "away"
                        else:
                            continue
                    else:
                        selection = str(outcome.get("name", "")).lower()
                        if selection not in {"over", "under"}:
                            continue
                    if game_id is None:
                        dropped_unmatched += 1
                        continue
                    provider_last_update = market.get("last_update") or bookmaker.get(
                        "last_update"
                    )
                    staleness = _staleness_seconds(fetched_at, provider_last_update)
                    if staleness is None:
                        dropped_untimestamped += 1
                        continue
                    offer_rows.append(
                        {
                            "snapshot_id": snapshot_id,
                            "fetched_at": fetched_at,
                            "game_id": game_id,
                            "match_score": score_by_id.get(event_id),
                            "odds_api_event_id": event_id,
                            "phase": phase_by_id.get(event_id, "unknown"),
                            "commence_time": event.get("commence_time"),
                            "provider_key": bookmaker.get("key"),
                            "provider": bookmaker.get("title"),
                            "market": market_key,
                            "selection": selection,
                            "point": outcome.get("point"),
                            "price": outcome.get("price"),
                            "provider_last_update": provider_last_update,
                            "staleness_seconds": staleness,
                            "event_link": bookmaker.get("link"),
                            "market_link": market.get("link"),
                            "bet_link": outcome.get("link"),
                            "execution_eligibility_verified": execution_verified,
                        }
                    )

    events_frame = pd.DataFrame(event_rows, columns=EVENT_COLUMNS).astype(EVENT_DTYPES)
    offers_frame = pd.DataFrame(offer_rows, columns=OFFER_COLUMNS).astype(OFFER_DTYPES)

    phase_counts = events_frame["phase"].value_counts()
    staleness = offers_frame["staleness_seconds"]
    poll_frame = pd.DataFrame(
        [
            {
                "snapshot_id": snapshot_id,
                "season": season,
                "poll_status": "ok",
                "error": None,
                "odds_fetched_at": fetched_at,
                "scores_fetched_at": scoreboard.fetched_at,
                "commence_from": query_window[0],
                "commence_to": query_window[1],
                "days_from": days_from,
                "configured_bookmakers": json.dumps(list(odds.configured_bookmakers)),
                "requested_markets": json.dumps(list(requested_markets)),
                "odds_request_cost": odds.request_cost,
                "odds_requests_used": odds.requests_used,
                "odds_requests_remaining": odds.requests_remaining,
                "scores_request_cost": scoreboard.request_cost,
                "scores_requests_used": scoreboard.requests_used,
                "scores_requests_remaining": scoreboard.requests_remaining,
                "event_count": len(events_frame),
                "pregame_event_count": int(phase_counts.get("pregame", 0)),
                "live_event_count": int(phase_counts.get("live", 0)),
                "final_event_count": int(phase_counts.get("final", 0)),
                "unknown_event_count": int(phase_counts.get("unknown", 0)),
                "unmatched_event_count": int((~events_frame["matched"]).sum()),
                "offer_count": len(offers_frame),
                "live_offer_count": int(offers_frame["phase"].eq("live").sum()),
                "dropped_unmatched_offer_count": dropped_unmatched,
                "dropped_untimestamped_offer_count": dropped_untimestamped,
                "max_offer_staleness_seconds": (
                    float(staleness.max()) if not offers_frame.empty else None
                ),
                "median_offer_staleness_seconds": (
                    float(staleness.median()) if not offers_frame.empty else None
                ),
                "events_sha256": frame_digest(events_frame),
                "offers_sha256": frame_digest(offers_frame),
            }
        ],
        columns=POLL_COLUMNS,
    ).astype(POLL_DTYPES)
    return LiveSnapshot(
        snapshot_id=snapshot_id,
        poll=poll_frame,
        events=events_frame,
        offers=offers_frame,
    )


def write_live_snapshot(season: int, snapshot: LiveSnapshot) -> None:
    base = live_odds_dir(season)
    parts = (
        ("polls", snapshot.poll),
        ("events", snapshot.events),
        ("offers", snapshot.offers),
    )
    for name, frame in parts:
        path = base / name / f"{snapshot.snapshot_id}.parquet"
        if path.exists():
            raise FileExistsError(
                f"live snapshot {snapshot.snapshot_id} already exists at {path}"
            )
        write_parquet(frame, path)


def record_failed_poll(
    season: int,
    error: str,
    query_window: tuple[datetime, datetime],
    days_from: int | None,
    requested_markets: tuple[str, ...] = LIVE_MARKETS,
    odds: OddsSnapshot | None = None,
    scoreboard: ScoreboardSnapshot | None = None,
) -> str:
    requested_markets = normalize_live_markets(requested_markets)
    failed_at = datetime.now(timezone.utc)
    snapshot_id = snapshot_id_for(failed_at)
    row = {column: None for column in POLL_COLUMNS}
    row.update(
        {
            "snapshot_id": snapshot_id,
            "season": season,
            "poll_status": "error",
            "error": error[:500],
            "odds_fetched_at": failed_at,
            "commence_from": query_window[0],
            "commence_to": query_window[1],
            "days_from": days_from,
            "requested_markets": json.dumps(list(requested_markets)),
        }
    )
    if odds is not None:
        row.update(
            {
                "odds_fetched_at": odds.fetched_at,
                "configured_bookmakers": json.dumps(list(odds.configured_bookmakers)),
                "odds_request_cost": odds.request_cost,
                "odds_requests_used": odds.requests_used,
                "odds_requests_remaining": odds.requests_remaining,
            }
        )
    if scoreboard is not None:
        row.update(
            {
                "scores_fetched_at": scoreboard.fetched_at,
                "scores_request_cost": scoreboard.request_cost,
                "scores_requests_used": scoreboard.requests_used,
                "scores_requests_remaining": scoreboard.requests_remaining,
            }
        )
    path = live_odds_dir(season) / "polls" / f"{snapshot_id}.parquet"
    if path.exists():
        raise FileExistsError(f"live snapshot {snapshot_id} already exists at {path}")
    write_parquet(pd.DataFrame([row], columns=POLL_COLUMNS).astype(POLL_DTYPES), path)
    return snapshot_id


def capture_live_snapshot(
    client: OddsAPIClient,
    season: int,
    schedule: pd.DataFrame,
    lookback_hours: float,
    lookahead_hours: float,
    days_from: int | None,
    min_quota: int,
    required_game_ids: tuple[int, ...] | None = None,
    markets: tuple[str, ...] = LIVE_MARKETS,
    future_poll_markets: tuple[tuple[str, ...], ...] = (),
) -> LiveSnapshot:
    markets = normalize_live_markets(markets)
    future_poll_markets = tuple(
        normalize_live_markets(planned) for planned in future_poll_markets
    )
    now = datetime.now(timezone.utc)
    window = (
        now - timedelta(hours=lookback_hours),
        now + timedelta(hours=lookahead_hours),
    )
    discovery = client.get_ncaaf_events(*window)
    if discovery.request_cost != 0:
        raise OddsAPIError(
            "free event discovery returned missing or nonzero request cost"
        )
    if discovery.requests_remaining is None:
        raise OddsAPIError("free event discovery lacked quota provenance")

    event_id_by_game = {}
    commence_by_event_id = {}
    for event in discovery.events:
        game_id, _ = match_event(
            event.get("commence_time"),
            event.get("home_team", ""),
            event.get("away_team", ""),
            schedule,
        )
        event_id = event.get("id")
        if game_id is not None and event_id:
            event_id = str(event_id)
            event_id_by_game.setdefault(int(game_id), event_id)
            commence_by_event_id[event_id] = event.get("commence_time")
    required = tuple(dict.fromkeys(required_game_ids or ()))
    missing_discovery = [
        game_id for game_id in required if game_id not in event_id_by_game
    ]
    if missing_discovery:
        raise LivePollSkipped(
            "free event discovery is missing target games: "
            + ", ".join(str(game_id) for game_id in missing_discovery),
            discovery.requests_remaining,
        )
    relevant_event_ids = (
        [event_id_by_game[game_id] for game_id in required]
        if required
        else list(event_id_by_game.values())
    )
    if not relevant_event_ids:
        raise LivePollSkipped(
            "free event discovery found no matching Division I games",
            discovery.requests_remaining,
        )

    scores_cost = 2 if days_from is not None else SCORES_COST
    current_odds_cost = len(markets)
    future_cost = sum(
        live_poll_quota_cost(planned, days_from) for planned in future_poll_markets
    )
    remaining = discovery.requests_remaining
    reserved_cost = current_odds_cost + scores_cost + future_cost
    if remaining - reserved_cost < min_quota:
        raise QuotaFloorReached(
            f"{remaining} credits remain; {1 + len(future_poll_markets)} "
            "remaining live "
            f"cycles reserve {reserved_cost} credits and would cross the "
            f"{min_quota}-credit floor"
        )

    odds = client.get_ncaaf_odds(
        *window,
        markets=markets,
        event_ids=tuple(relevant_event_ids),
    )
    if odds.request_cost is None or not 0 <= odds.request_cost <= current_odds_cost:
        raise PartialPaidPollError(
            "odds response returned missing or unexpected request cost", odds
        )
    if odds.requests_remaining is None:
        raise PartialPaidPollError("odds response lacked quota provenance", odds)

    def has_spread(event: dict) -> bool:
        return any(
            market.get("key") == "spreads"
            for bookmaker in event.get("bookmakers") or []
            for market in bookmaker.get("markets") or []
        )

    selected_ids = set(relevant_event_ids)
    returned = [
        event
        for event in odds.events
        if event.get("id") in selected_ids and has_spread(event)
    ]
    returned_ids = {str(event.get("id")) for event in returned}
    missing_odds = [
        event_id for event_id in relevant_event_ids if event_id not in returned_ids
    ]
    if missing_odds:
        reason = "spread odds are missing discovered target events: " + ", ".join(
            missing_odds
        )
        missing_have_started = all(
            pd.to_datetime(
                commence_by_event_id.get(event_id), utc=True, errors="coerce"
            )
            <= pd.Timestamp(now)
            for event_id in missing_odds
        )
        if not required or not missing_have_started:
            if odds.request_cost:
                raise PartialPaidPollError(reason, odds)
            raise LivePollSkipped(reason, odds.requests_remaining)
    if not returned and not missing_odds:
        raise LivePollSkipped(
            "no spread odds were returned for the discovered games",
            odds.requests_remaining,
        )
    odds = OddsSnapshot(
        events=returned,
        fetched_at=odds.fetched_at,
        requests_remaining=odds.requests_remaining,
        requests_used=odds.requests_used,
        request_cost=odds.request_cost,
        configured_bookmakers=odds.configured_bookmakers,
    )
    future_reserved_cost = scores_cost + future_cost
    if odds.requests_remaining - future_reserved_cost < min_quota:
        raise QuotaFloorReached(
            f"{odds.requests_remaining} credits remain after odds; a "
            f"{future_reserved_cost}-credit reserve for scores and future "
            f"cycles would cross the {min_quota}-credit floor",
            odds,
        )

    try:
        scoreboard = client.get_ncaaf_scores(days_from)
    except (OddsAPIError, requests.RequestException) as exc:
        raise PartialPaidPollError(str(exc), odds) from exc
    if scoreboard.request_cost != scores_cost or scoreboard.requests_remaining is None:
        raise PartialPaidPollError(
            "scores response returned missing or unexpected quota provenance",
            odds,
            scoreboard,
        )
    if scoreboard.requests_remaining - future_cost < min_quota:
        raise PartialPaidPollError(
            f"{scoreboard.requests_remaining} credits remain after scores; "
            f"{future_cost} credits are reserved for future cycles and would "
            f"cross the {min_quota}-credit floor",
            odds,
            scoreboard,
        )
    relevant = set(relevant_event_ids)
    scoreboard = ScoreboardSnapshot(
        events=[event for event in scoreboard.events if event.get("id") in relevant],
        fetched_at=scoreboard.fetched_at,
        requests_remaining=scoreboard.requests_remaining,
        requests_used=scoreboard.requests_used,
        request_cost=scoreboard.request_cost,
    )
    snapshot = build_live_snapshot(
        season,
        odds,
        scoreboard,
        schedule,
        window,
        days_from,
        requested_markets=markets,
    )
    write_live_snapshot(season, snapshot)
    return snapshot


def _poll_summary(snapshot: LiveSnapshot) -> str:
    poll = snapshot.poll.iloc[0]
    staleness = (
        "n/a"
        if pd.isna(poll["median_offer_staleness_seconds"])
        else (
            f"{poll['median_offer_staleness_seconds']:.0f}s median / "
            f"{poll['max_offer_staleness_seconds']:.0f}s max"
        )
    )
    return (
        f"{snapshot.snapshot_id}: {poll['live_event_count']} live, "
        f"{poll['pregame_event_count']} pregame, "
        f"{poll['final_event_count']} final, "
        f"{poll['unknown_event_count']} unknown | "
        f"{poll['offer_count']} offers ({poll['live_offer_count']} live, "
        f"{poll['dropped_unmatched_offer_count']} dropped unmatched, "
        f"{poll['dropped_untimestamped_offer_count']} dropped untimestamped) | "
        f"staleness {staleness} | "
        f"markets {poll['requested_markets']} | "
        f"quota cost {poll['odds_request_cost']}+{poll['scores_request_cost']}, "
        f"{poll['odds_requests_remaining']} remaining"
    )


def run_live_polling(
    season: int,
    schedule: pd.DataFrame,
    polls: int,
    interval_seconds: float,
    lookback_hours: float,
    lookahead_hours: float,
    days_from: int | None,
    min_quota: int,
    max_failures: int,
    required_game_ids: tuple[int, ...] | None = None,
    poll_markets: tuple[tuple[str, ...], ...] | None = None,
    progress=print,
    sleep=time.sleep,
    monotonic=time.monotonic,
) -> int:
    if polls < 1:
        raise ValueError("--polls must be at least 1")
    if max_failures < 1:
        raise ValueError("--max-failures must be at least 1")
    if min_quota < 0:
        raise ValueError("--min-quota must be nonnegative")
    if poll_markets is None:
        market_plan = (LIVE_MARKETS,) * polls
    else:
        if len(poll_markets) != polls:
            raise ValueError("poll market plan must contain one entry per poll")
        market_plan = tuple(normalize_live_markets(markets) for markets in poll_markets)
    client = OddsAPIClient()
    client.ensure_single_quota_region()
    consecutive_failures = 0
    completed = 0
    started_at = monotonic()
    for poll_index in range(polls):
        if poll_index:
            due_at = started_at + poll_index * interval_seconds
            sleep(max(0.0, due_at - monotonic()))
        markets = market_plan[poll_index]
        try:
            snapshot = capture_live_snapshot(
                client,
                season,
                schedule,
                lookback_hours,
                lookahead_hours,
                days_from,
                min_quota,
                required_game_ids=required_game_ids,
                markets=markets,
                future_poll_markets=market_plan[poll_index + 1 :],
            )
        except QuotaFloorReached as exc:
            if exc.odds is None:
                progress(f"stopping: {exc}")
            else:
                now = datetime.now(timezone.utc)
                window = (
                    now - timedelta(hours=lookback_hours),
                    now + timedelta(hours=lookahead_hours),
                )
                snapshot_id = record_failed_poll(
                    season,
                    str(exc),
                    window,
                    days_from,
                    requested_markets=markets,
                    odds=exc.odds,
                )
                progress(f"{snapshot_id}: stopping after paid odds: {exc}")
            break
        except LivePollSkipped as exc:
            consecutive_failures = 0
            remaining = (
                ""
                if exc.requests_remaining is None
                else f" ({exc.requests_remaining} credits remain)"
            )
            progress(f"skipping paid capture: {exc}{remaining}")
        except PartialPaidPollError as exc:
            consecutive_failures += 1
            now = datetime.now(timezone.utc)
            window = (
                now - timedelta(hours=lookback_hours),
                now + timedelta(hours=lookahead_hours),
            )
            snapshot_id = record_failed_poll(
                season,
                str(exc),
                window,
                days_from,
                requested_markets=markets,
                odds=exc.odds,
                scoreboard=exc.scoreboard,
            )
            progress(
                f"{snapshot_id}: paid poll failed "
                f"({consecutive_failures}/{max_failures} consecutive): {exc}"
            )
            if consecutive_failures >= max_failures:
                progress("stopping: consecutive failure limit reached")
                break
        except (OddsAPIError, requests.RequestException) as exc:
            consecutive_failures += 1
            now = datetime.now(timezone.utc)
            window = (
                now - timedelta(hours=lookback_hours),
                now + timedelta(hours=lookahead_hours),
            )
            snapshot_id = record_failed_poll(
                season,
                str(exc),
                window,
                days_from,
                requested_markets=markets,
            )
            progress(
                f"{snapshot_id}: poll failed "
                f"({consecutive_failures}/{max_failures} consecutive): {exc}"
            )
            if consecutive_failures >= max_failures:
                progress("stopping: consecutive failure limit reached")
                break
        else:
            consecutive_failures = 0
            completed += 1
            progress(_poll_summary(snapshot))
    return completed


def read_live_frames(season: int) -> dict[str, pd.DataFrame]:
    base = live_odds_dir(season)
    frames = {}
    for name in ("polls", "events", "offers"):
        directory = base / name
        files = sorted(directory.glob("*.parquet")) if directory.exists() else []
        frames[name] = (
            pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
            if files
            else pd.DataFrame()
        )
    return frames


def offers_available_at(
    polls: pd.DataFrame, offers: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Offers from the latest successful snapshot fetched at or before as_of."""
    successful = polls[polls["poll_status"].eq("ok")].copy()
    successful["odds_fetched_at"] = pd.to_datetime(
        successful["odds_fetched_at"], utc=True
    )
    eligible = successful[successful["odds_fetched_at"].le(as_of)]
    if eligible.empty:
        return offers.iloc[0:0]
    latest = eligible.sort_values("odds_fetched_at").iloc[-1]
    return offers[offers["snapshot_id"].eq(latest["snapshot_id"])].copy()


def verify_live_snapshots(season: int) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """Check the append-only invariants; returns (problems, frames)."""
    frames = read_live_frames(season)
    polls, events, offers = frames["polls"], frames["events"], frames["offers"]
    problems: list[str] = []
    if polls.empty:
        return ["no live odds polls stored for this season"], frames

    if polls["snapshot_id"].duplicated().any():
        problems.append("duplicate snapshot ids in polls")
    fetched = pd.to_datetime(polls["odds_fetched_at"], utc=True)
    if not fetched.is_monotonic_increasing:
        problems.append("poll fetch times are not monotonically increasing")

    successful = polls[polls["poll_status"].eq("ok")]
    for poll in successful.itertuples():
        stored_offers = (
            offers[offers["snapshot_id"].eq(poll.snapshot_id)]
            if not offers.empty
            else offers
        )
        stored_events = (
            events[events["snapshot_id"].eq(poll.snapshot_id)]
            if not events.empty
            else events
        )
        if len(stored_offers) != int(poll.offer_count):
            problems.append(
                f"{poll.snapshot_id}: stored offers {len(stored_offers)} != "
                f"recorded offer_count {int(poll.offer_count)}"
            )
        elif frame_digest(stored_offers.reset_index(drop=True)) != poll.offers_sha256:
            problems.append(
                f"{poll.snapshot_id}: stored offers do not match the "
                "offers_sha256 recorded at capture time"
            )
        if len(stored_events) != int(poll.event_count):
            problems.append(
                f"{poll.snapshot_id}: stored events {len(stored_events)} != "
                f"recorded event_count {int(poll.event_count)}"
            )
        elif frame_digest(stored_events.reset_index(drop=True)) != poll.events_sha256:
            problems.append(
                f"{poll.snapshot_id}: stored events do not match the "
                "events_sha256 recorded at capture time"
            )
        replayed = offers_available_at(
            polls, offers, pd.to_datetime(poll.odds_fetched_at, utc=True)
        )
        if not replayed["snapshot_id"].eq(poll.snapshot_id).all() or len(
            replayed
        ) != len(stored_offers):
            problems.append(
                f"{poll.snapshot_id}: replay at its own fetch time does not "
                "reproduce the stored snapshot"
            )

    if not offers.empty:
        required = [
            "game_id",
            "phase",
            "provider_key",
            "market",
            "selection",
            "price",
            "provider_last_update",
            "fetched_at",
            "staleness_seconds",
        ]
        for column in required:
            missing = int(offers[column].isna().sum())
            if missing:
                problems.append(f"{missing} offers missing {column}")
        bad_phase = offers[~offers["phase"].isin(PHASES)]
        if not bad_phase.empty:
            problems.append(f"{len(bad_phase)} offers with unrecognized phase")
        if not events.empty:
            event_phase = events.set_index(["snapshot_id", "odds_api_event_id"])[
                "phase"
            ]
            offer_phase = offers.set_index(["snapshot_id", "odds_api_event_id"])[
                "phase"
            ]
            joined = offer_phase.to_frame("offer_phase").join(
                event_phase.to_frame("event_phase"), how="left"
            )
            mismatched = joined[
                joined["event_phase"].notna()
                & joined["offer_phase"].ne(joined["event_phase"])
            ]
            if not mismatched.empty:
                problems.append(
                    f"{len(mismatched)} offers whose phase disagrees with the "
                    "stored provider game status"
                )
    return problems, frames
