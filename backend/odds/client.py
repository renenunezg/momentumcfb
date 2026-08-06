import os
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
NCAAF_SPORT_KEY = "americanfootball_ncaaf"


class OddsAPIError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OddsSnapshot:
    events: list[dict]
    fetched_at: datetime
    requests_remaining: int | None
    requests_used: int | None
    request_cost: int | None
    configured_bookmakers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScoreboardSnapshot:
    events: list[dict]
    fetched_at: datetime
    requests_remaining: int | None
    requests_used: int | None
    request_cost: int | None


def _header_int(response: requests.Response, name: str) -> int | None:
    value = response.headers.get(name)
    return int(value) if value is not None else None


class OddsAPIClient:
    def __init__(
        self,
        api_key: str | None = None,
        regions: str | None = None,
        bookmakers: str | None = None,
    ):
        key = api_key or os.getenv("ODDS_API_KEY")
        if not key:
            raise OddsAPIError("ODDS_API_KEY is not set")
        self.api_key = key
        self.regions = regions or os.getenv("ODDS_API_REGIONS", "us")
        configured = bookmakers or os.getenv("ODDS_API_BOOKMAKERS", "")
        self.bookmakers = tuple(
            value.strip() for value in configured.split(",") if value.strip()
        )
        self.session = requests.Session()

    def _get_events(self, path: str, params: dict) -> tuple[list[dict], requests.Response]:
        response = self.session.get(
            f"{ODDS_API_BASE_URL}{path}", params=params, timeout=60
        )
        if response.status_code != 200:
            raise OddsAPIError(
                f"The Odds API returned {response.status_code}: "
                f"{response.text[:300]}"
            )
        events = response.json()
        if not isinstance(events, list):
            raise OddsAPIError("The Odds API returned a non-list payload")
        return events, response

    def get_ncaaf_odds(
        self, commence_from: datetime, commence_to: datetime
    ) -> OddsSnapshot:
        if commence_from.tzinfo is None or commence_to.tzinfo is None:
            raise ValueError("odds query timestamps must be timezone-aware")
        params = {
            "apiKey": self.api_key,
            "markets": "spreads,totals",
            "oddsFormat": "american",
            "dateFormat": "iso",
            "commenceTimeFrom": commence_from.astimezone(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "commenceTimeTo": commence_to.astimezone(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "includeLinks": "true",
        }
        if self.bookmakers:
            params["bookmakers"] = ",".join(self.bookmakers)
        else:
            params["regions"] = self.regions
        events, response = self._get_events(
            f"/sports/{NCAAF_SPORT_KEY}/odds", params
        )
        return OddsSnapshot(
            events=events,
            fetched_at=datetime.now(timezone.utc),
            requests_remaining=_header_int(response, "x-requests-remaining"),
            requests_used=_header_int(response, "x-requests-used"),
            request_cost=_header_int(response, "x-requests-last"),
            configured_bookmakers=self.bookmakers,
        )

    def get_ncaaf_scores(self, days_from: int | None = None) -> ScoreboardSnapshot:
        """Fetch provider game states (upcoming, in-play, completed) with scores."""
        if days_from is not None and not 1 <= days_from <= 3:
            raise ValueError("days_from must be between 1 and 3")
        params = {"apiKey": self.api_key, "dateFormat": "iso"}
        if days_from is not None:
            params["daysFrom"] = days_from
        events, response = self._get_events(
            f"/sports/{NCAAF_SPORT_KEY}/scores", params
        )
        return ScoreboardSnapshot(
            events=events,
            fetched_at=datetime.now(timezone.utc),
            requests_remaining=_header_int(response, "x-requests-remaining"),
            requests_used=_header_int(response, "x-requests-used"),
            request_cost=_header_int(response, "x-requests-last"),
        )
