"""DraftKings pregame odds from ESPN's public scoreboard feed.

Fallback for when The Odds API monthly quota is exhausted. Selected with
ODDS_SOURCE=espn; unset that variable to return to The Odds API. Events,
odds, and game states are returned in The Odds API's shapes, so offer
flattening, kickoff capture, and grading are unchanged.

ESPN differs from The Odds API in three ways callers rely on:
- It carries one book, so the consensus is DraftKings alone.
- It has no quote timestamp, so each price is stamped with the read time.
- It is unmetered, so every quota field is None.
"""

from datetime import date, datetime, timedelta, timezone

import requests

from backend.odds.client import (
    EventsSnapshot,
    OddsAPIError,
    OddsSnapshot,
    ScoreboardSnapshot,
)

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
)
# ESPN scoreboard groups: 80 is FBS, 81 is FCS.
ESPN_GROUPS = (80, 81)
# ESPN's provider id is stable while its display name is not ("DraftKings" and
# "Draft Kings" both appear). Map the id to The Odds API's key and title.
ESPN_BOOKMAKERS = {"100": ("draftkings", "DraftKings")}
# Events, odds, and scores of one capture poll come from a single read.
SCOREBOARD_REUSE_SECONDS = 30


def _price(quote: dict) -> int | None:
    raw = str(quote.get("odds", "")).strip().upper()
    if raw == "EVEN":
        return 100
    try:
        return int(raw)
    except ValueError:
        return None


def _point(quote: dict) -> float | None:
    # Totals arrive as "o48.5" / "u48.5", spreads as "-9.5" / "+9.5".
    try:
        return float(str(quote.get("line", "")).strip().lstrip("ou"))
    except ValueError:
        return None


def _markets(odds: dict, home_team: str, away_team: str, updated: str) -> list[dict]:
    def close(market: str, side: str) -> dict:
        return (odds.get(market) or {}).get(side, {}).get("close") or {}

    sides = (("home", home_team), ("away", away_team))
    outcomes = {
        "h2h": [
            {"name": team, "price": _price(close("moneyline", side))}
            for side, team in sides
        ],
        "spreads": [
            {
                "name": team,
                "price": _price(close("pointSpread", side)),
                "point": _point(close("pointSpread", side)),
            }
            for side, team in sides
        ],
        "totals": [
            {
                "name": side.title(),
                "price": _price(close("total", side)),
                "point": _point(close("total", side)),
            }
            for side in ("over", "under")
        ],
    }
    return [
        {"key": key, "last_update": updated, "outcomes": priced}
        for key, priced in outcomes.items()
        if all(outcome["price"] is not None for outcome in priced)
    ]


def _commence(event: dict) -> datetime:
    return datetime.fromisoformat(event["date"].replace("Z", "+00:00"))


def _teams(event: dict) -> dict[str, str]:
    return {
        competitor["homeAway"]: competitor["team"]["displayName"]
        for competitor in event["competitions"][0]["competitors"]
    }


def _started(event: dict, as_of: datetime) -> bool:
    return event["status"]["type"]["state"] != "pre" or _commence(event) <= as_of


def _bookmakers(event: dict, updated: str) -> list[dict]:
    teams = _teams(event)
    bookmakers = []
    for odds in event["competitions"][0].get("odds") or []:
        provider = str((odds.get("provider") or {}).get("id"))
        if provider not in ESPN_BOOKMAKERS:
            continue
        key, title = ESPN_BOOKMAKERS[provider]
        bookmakers.append(
            {
                "key": key,
                "title": title,
                "last_update": updated,
                "markets": _markets(odds, teams["home"], teams["away"], updated),
            }
        )
    return bookmakers


def _identity(event: dict) -> dict:
    teams = _teams(event)
    return {
        "id": f"espn-{event['id']}",
        "commence_time": _commence(event).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "home_team": teams["home"],
        "away_team": teams["away"],
    }


class EspnOddsClient:
    def __init__(self):
        self._reads: dict[date, tuple[datetime, list[dict]]] = {}

    def _day(self, day: date) -> tuple[datetime, list[dict]]:
        cached = self._reads.get(day)
        if cached is not None:
            age = (datetime.now(timezone.utc) - cached[0]).total_seconds()
            if age < SCOREBOARD_REUSE_SECONDS:
                return cached
        events = []
        for group in ESPN_GROUPS:
            try:
                response = requests.get(
                    ESPN_SCOREBOARD_URL,
                    params={
                        "dates": day.strftime("%Y%m%d"),
                        "groups": group,
                        "limit": 300,
                    },
                    timeout=60,
                )
            except requests.RequestException as error:
                raise OddsAPIError(
                    f"The ESPN odds request failed: {type(error).__name__}"
                ) from None
            if response.status_code != 200:
                raise OddsAPIError(f"ESPN returned {response.status_code}")
            try:
                events.extend(response.json()["events"])
            except (ValueError, KeyError, TypeError):
                raise OddsAPIError(
                    "ESPN returned an unexpected scoreboard body"
                ) from None
        self._reads[day] = (datetime.now(timezone.utc), events)
        return self._reads[day]

    def _scoreboard(
        self, first: datetime, last: datetime
    ) -> tuple[datetime, list[dict]]:
        """Every event from the days spanning a window, with the oldest read time.

        ESPN serves one US date per request and rejects date ranges, so the
        span starts a day early and callers filter to their exact window.
        """
        day = first.astimezone(timezone.utc).date() - timedelta(days=1)
        fetched_at = None
        events: dict[str, dict] = {}
        while day <= last.astimezone(timezone.utc).date():
            read_at, day_events = self._day(day)
            fetched_at = read_at if fetched_at is None else min(fetched_at, read_at)
            events.update({event["id"]: event for event in day_events})
            day += timedelta(days=1)
        return fetched_at, list(events.values())

    def get_ncaaf_events(
        self, commence_from: datetime, commence_to: datetime
    ) -> EventsSnapshot:
        """Games in the window that are priced or already started.

        An unpriced future game is not a capture target. A started game stays
        listed after ESPN stops pricing it so its game state is still recorded.
        """
        if commence_from.tzinfo is None or commence_to.tzinfo is None:
            raise ValueError("odds query timestamps must be timezone-aware")
        fetched_at, scoreboard = self._scoreboard(commence_from, commence_to)
        updated = fetched_at.isoformat()
        events = [
            _identity(event)
            for event in scoreboard
            if commence_from <= _commence(event) <= commence_to
            and (
                _started(event, fetched_at)
                or any(
                    market["key"] == "spreads"
                    for bookmaker in _bookmakers(event, updated)
                    for market in bookmaker["markets"]
                )
            )
        ]
        return EventsSnapshot(
            events=events,
            fetched_at=fetched_at,
            requests_remaining=None,
            requests_used=None,
            request_cost=None,
        )

    def get_ncaaf_odds(
        self,
        commence_from: datetime,
        commence_to: datetime,
        *,
        markets: tuple[str, ...] = ("h2h", "spreads", "totals"),
        event_ids: tuple[str, ...] = (),
    ) -> OddsSnapshot:
        if commence_from.tzinfo is None or commence_to.tzinfo is None:
            raise ValueError("odds query timestamps must be timezone-aware")
        if not markets:
            raise ValueError("at least one odds market is required")
        fetched_at, scoreboard = self._scoreboard(commence_from, commence_to)
        updated = fetched_at.isoformat()
        events = []
        for event in scoreboard:
            identity = _identity(event)
            if event_ids and identity["id"] not in event_ids:
                continue
            # Once a game starts ESPN shows its closing line, never a live offer.
            if _started(event, fetched_at):
                continue
            if not commence_from <= _commence(event) <= commence_to:
                continue
            bookmakers = _bookmakers(event, updated)
            for bookmaker in bookmakers:
                bookmaker["markets"] = [
                    market
                    for market in bookmaker["markets"]
                    if market["key"] in markets
                ]
            events.append({**identity, "bookmakers": bookmakers})
        return OddsSnapshot(
            events=events,
            fetched_at=fetched_at,
            requests_remaining=None,
            requests_used=None,
            request_cost=None,
            configured_bookmakers=tuple(key for key, _ in ESPN_BOOKMAKERS.values()),
        )

    def get_ncaaf_scores(self, days_from: int | None = None) -> ScoreboardSnapshot:
        """Game states (upcoming, in-play, completed) with scores."""
        now = datetime.now(timezone.utc)
        fetched_at, scoreboard = self._scoreboard(
            now - timedelta(days=days_from or 0), now + timedelta(days=1)
        )
        events = []
        for event in scoreboard:
            status = event["status"]["type"]
            competitors = event["competitions"][0]["competitors"]
            events.append(
                {
                    **_identity(event),
                    # A postponed game is "post" without being completed.
                    "completed": bool(status.get("completed")),
                    "scores": None
                    if status["state"] == "pre"
                    else [
                        {
                            "name": competitor["team"]["displayName"],
                            "score": competitor.get("score"),
                        }
                        for competitor in competitors
                    ],
                    "last_update": fetched_at.isoformat(),
                }
            )
        return ScoreboardSnapshot(
            events=events,
            fetched_at=fetched_at,
            requests_remaining=None,
            requests_used=None,
            request_cost=None,
        )
