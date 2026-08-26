import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from backend.odds import kickoff, live
from backend.odds.client import (
    EventsSnapshot,
    OddsAPIClient,
    OddsAPIError,
    OddsSnapshot,
    ScoreboardSnapshot,
)

NOW = datetime(2026, 9, 5, 20, 0, 0, tzinfo=timezone.utc)


def _odds_event(event_id, home, away, commence, last_update):
    return {
        "id": event_id,
        "commence_time": commence,
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": last_update,
                "markets": [
                    {
                        "key": "spreads",
                        "last_update": last_update,
                        "outcomes": [
                            {"name": home, "point": -3.5, "price": -110},
                            {"name": away, "point": 3.5, "price": -110},
                        ],
                    },
                    {
                        "key": "totals",
                        "last_update": last_update,
                        "outcomes": [
                            {"name": "Over", "point": 51.5, "price": -105},
                            {"name": "Under", "point": 51.5, "price": -115},
                        ],
                    },
                ],
            }
        ],
    }


def _schedule():
    return pd.DataFrame(
        {
            "game_id": [101, 102],
            "start_date": [
                (NOW - timedelta(hours=1)).isoformat(),
                (NOW + timedelta(hours=3)).isoformat(),
            ],
            "home_team": ["Ohio State", "Montana"],
            "away_team": ["Rutgers", "Idaho"],
        }
    )


def _snapshot():
    live_commence = (NOW - timedelta(hours=1)).isoformat()
    pregame_commence = (NOW + timedelta(hours=3)).isoformat()
    update = (NOW - timedelta(seconds=30)).isoformat()
    odds = OddsSnapshot(
        events=[
            _odds_event("ev-live", "Ohio State", "Rutgers", live_commence, update),
            _odds_event("ev-pre", "Montana", "Idaho", pregame_commence, update),
            _odds_event(
                "ev-unmatched", "Nobody State", "Anywhere Tech", live_commence, update
            ),
        ],
        fetched_at=NOW,
        requests_remaining=400,
        requests_used=100,
        request_cost=2,
        configured_bookmakers=("draftkings",),
    )
    scoreboard = ScoreboardSnapshot(
        events=[
            {
                "id": "ev-live",
                "commence_time": live_commence,
                "completed": False,
                "home_team": "Ohio State",
                "away_team": "Rutgers",
                "scores": [
                    {"name": "Ohio State", "score": "14"},
                    {"name": "Rutgers", "score": "7"},
                ],
                "last_update": update,
            },
            {
                "id": "ev-pre",
                "commence_time": pregame_commence,
                "completed": False,
                "home_team": "Montana",
                "away_team": "Idaho",
                "scores": None,
                "last_update": None,
            },
        ],
        fetched_at=NOW,
        requests_remaining=401,
        requests_used=99,
        request_cost=1,
    )
    return live.build_live_snapshot(
        2026,
        odds,
        scoreboard,
        _schedule(),
        (NOW - timedelta(hours=8), NOW + timedelta(hours=6)),
        None,
        requested_markets=live.CLOSING_MARKETS,
    )


def test_phase_comes_only_from_provider_status():
    assert live.resolve_phase((NOW + timedelta(hours=3)).isoformat(), False, NOW) == "pregame"
    assert live.resolve_phase((NOW - timedelta(hours=1)).isoformat(), False, NOW) == "live"
    assert live.resolve_phase((NOW - timedelta(hours=4)).isoformat(), True, NOW) == "final"
    # An event absent from the scores feed must never be guessed live,
    # even if it commenced in the past and its lines are moving.
    assert live.resolve_phase((NOW - timedelta(hours=1)).isoformat(), None, NOW) == "unknown"


def test_snapshot_offers_carry_required_provenance():
    snapshot = _snapshot()
    offers = snapshot.offers

    assert set(offers["phase"]) == {"live", "pregame"}
    assert offers[offers["game_id"].eq(101)]["phase"].eq("live").all()
    assert offers[offers["game_id"].eq(102)]["phase"].eq("pregame").all()
    for column in (
        "game_id",
        "phase",
        "provider_key",
        "provider_last_update",
        "fetched_at",
        "staleness_seconds",
    ):
        assert offers[column].notna().all()
    assert (offers["staleness_seconds"] == 30.0).all()

    poll = snapshot.poll.iloc[0]
    assert poll["offer_count"] == len(offers) == 8
    assert poll["dropped_unmatched_offer_count"] == 4
    unmatched = snapshot.events[snapshot.events["odds_api_event_id"].eq("ev-unmatched")]
    assert unmatched["phase"].eq("unknown").all()
    assert not unmatched["matched"].any()


class _QuotaAwareClient:
    def __init__(self, discovery, odds=None, scoreboard=None):
        self.discovery = discovery
        self.odds = odds
        self.scoreboard = scoreboard
        self.calls = []

    def ensure_single_quota_region(self):
        return None

    def get_ncaaf_events(self, commence_from, commence_to):
        self.calls.append(("events", commence_from, commence_to))
        return self.discovery

    def get_ncaaf_odds(self, commence_from, commence_to, *, markets, event_ids):
        self.calls.append(("odds", markets, event_ids))
        if self.odds is None:
            raise AssertionError("paid odds should not have been requested")
        return self.odds

    def get_ncaaf_scores(self, days_from):
        self.calls.append(("scores", days_from))
        if self.scoreboard is None:
            raise AssertionError("scores should not have been requested")
        return self.scoreboard


def _events_snapshot(events, remaining=78):
    return EventsSnapshot(
        events=events,
        fetched_at=NOW,
        requests_remaining=remaining,
        requests_used=422,
        request_cost=0,
    )


def test_live_capture_discovers_for_free_then_buys_requested_markets(
    monkeypatch,
):
    commence = (NOW + timedelta(hours=3)).isoformat()
    update = (NOW - timedelta(seconds=30)).isoformat()
    event = _odds_event("ev-pre", "Montana", "Idaho", commence, update)
    odds = OddsSnapshot(
        events=[event],
        fetched_at=NOW,
        requests_remaining=76,
        requests_used=424,
        request_cost=2,
        configured_bookmakers=("draftkings",),
    )
    scoreboard = ScoreboardSnapshot(
        events=[
            {
                "id": "ev-pre",
                "commence_time": commence,
                "completed": False,
                "home_team": "Montana",
                "away_team": "Idaho",
                "scores": None,
                "last_update": None,
            },
            {
                "id": "unrelated",
                "commence_time": commence,
                "completed": False,
                "home_team": "Other",
                "away_team": "Else",
            },
        ],
        fetched_at=NOW,
        requests_remaining=75,
        requests_used=425,
        request_cost=1,
    )
    client = _QuotaAwareClient(_events_snapshot([event]), odds, scoreboard)
    written = []
    monkeypatch.setattr(
        live,
        "write_live_snapshot",
        lambda season, snapshot: written.append(snapshot),
    )

    snapshot = live.capture_live_snapshot(
        client,
        2026,
        _schedule(),
        lookback_hours=8,
        lookahead_hours=6,
        days_from=None,
        min_quota=50,
        markets=live.CLOSING_MARKETS,
    )

    assert [call[0] for call in client.calls] == ["events", "odds", "scores"]
    assert client.calls[1][1:] == (live.CLOSING_MARKETS, ("ev-pre",))
    assert snapshot.events["odds_api_event_id"].tolist() == ["ev-pre"]
    assert set(snapshot.offers["market"]) == {"spreads", "totals"}
    assert snapshot.poll.iloc[0]["requested_markets"] == '["spreads", "totals"]'
    assert snapshot.poll.iloc[0]["odds_request_cost"] == 2
    assert snapshot.poll.iloc[0]["scores_request_cost"] == 1
    assert written == [snapshot]

    spread_only_event = _odds_event(
        "ev-pre", "Montana", "Idaho", commence, update
    )
    spread_only_event["bookmakers"][0]["markets"] = spread_only_event[
        "bookmakers"
    ][0]["markets"][:1]
    spread_only_odds = OddsSnapshot(
        events=[spread_only_event],
        fetched_at=NOW,
        requests_remaining=77,
        requests_used=423,
        request_cost=1,
        configured_bookmakers=("draftkings",),
    )
    partial = live.capture_live_snapshot(
        _QuotaAwareClient(_events_snapshot([event]), spread_only_odds, scoreboard),
        2026,
        _schedule(),
        lookback_hours=8,
        lookahead_hours=6,
        days_from=None,
        min_quota=50,
        markets=live.CLOSING_MARKETS,
    )
    assert set(partial.offers["market"]) == {"spreads"}
    assert partial.poll.iloc[0]["requested_markets"] == '["spreads", "totals"]'
    assert written[-1] is partial


def test_live_capture_spends_nothing_without_relevant_events():
    commence = (NOW + timedelta(hours=3)).isoformat()
    irrelevant = {
        "id": "other",
        "commence_time": commence,
        "home_team": "Nobody State",
        "away_team": "Anywhere Tech",
    }
    client = _QuotaAwareClient(_events_snapshot([irrelevant]))

    with pytest.raises(live.LivePollSkipped, match="free event discovery"):
        live.capture_live_snapshot(
            client,
            2026,
            _schedule(),
            lookback_hours=8,
            lookahead_hours=6,
            days_from=None,
            min_quota=50,
        )

    assert [call[0] for call in client.calls] == ["events"]


def test_live_capture_requires_the_requested_kickoff_target():
    commence = (NOW + timedelta(hours=3)).isoformat()
    other_game = {
        "id": "ev-pre",
        "commence_time": commence,
        "home_team": "Montana",
        "away_team": "Idaho",
    }
    client = _QuotaAwareClient(_events_snapshot([other_game]))

    with pytest.raises(live.LivePollSkipped, match="missing target games: 101"):
        live.capture_live_snapshot(
            client,
            2026,
            _schedule(),
            lookback_hours=8,
            lookahead_hours=6,
            days_from=None,
            min_quota=50,
            required_game_ids=(101,),
        )

    assert [call[0] for call in client.calls] == ["events"]


def test_postkick_score_is_captured_when_books_pull_the_spread(monkeypatch):
    current = datetime.now(timezone.utc)
    commence = (current - timedelta(minutes=1)).isoformat()
    event = {
        "id": "ev-live",
        "commence_time": commence,
        "home_team": "Montana",
        "away_team": "Idaho",
    }
    odds = OddsSnapshot(
        events=[],
        fetched_at=current,
        requests_remaining=78,
        requests_used=422,
        request_cost=0,
        configured_bookmakers=("draftkings",),
    )
    scoreboard = ScoreboardSnapshot(
        events=[
            {
                **event,
                "completed": False,
                "scores": [
                    {"name": "Montana", "score": "0"},
                    {"name": "Idaho", "score": "0"},
                ],
                "last_update": current.isoformat(),
            }
        ],
        fetched_at=current,
        requests_remaining=77,
        requests_used=423,
        request_cost=1,
    )
    schedule = pd.DataFrame(
        {
            "game_id": [501],
            "start_date": [commence],
            "home_team": ["Montana"],
            "away_team": ["Idaho"],
        }
    )
    client = _QuotaAwareClient(
        _events_snapshot([event], remaining=78), odds, scoreboard
    )
    monkeypatch.setattr(live, "write_live_snapshot", lambda season, snapshot: None)

    snapshot = live.capture_live_snapshot(
        client,
        2026,
        schedule,
        lookback_hours=1,
        lookahead_hours=1,
        days_from=None,
        min_quota=50,
        required_game_ids=(501,),
    )

    assert [call[0] for call in client.calls] == ["events", "odds", "scores"]
    assert snapshot.offers.empty
    assert snapshot.events["phase"].tolist() == ["live"]
    assert snapshot.poll.iloc[0]["odds_request_cost"] == 0
    assert snapshot.poll.iloc[0]["scores_request_cost"] == 1


def test_live_capture_checks_shared_quota_before_paid_calls(monkeypatch):
    commence = (NOW + timedelta(hours=3)).isoformat()
    event = {
        "id": "ev-pre",
        "commence_time": commence,
        "home_team": "Montana",
        "away_team": "Idaho",
    }
    client = _QuotaAwareClient(_events_snapshot([event], remaining=62))
    future_markets = (
        live.LIVE_MARKETS,
        live.CLOSING_MARKETS,
        live.LIVE_MARKETS,
        live.LIVE_MARKETS,
        live.LIVE_MARKETS,
    )

    with pytest.raises(live.QuotaFloorReached, match="would cross"):
        live.capture_live_snapshot(
            client,
            2026,
            _schedule(),
            lookback_hours=8,
            lookahead_hours=6,
            days_from=None,
            min_quota=50,
            future_poll_markets=future_markets,
        )

    assert [call[0] for call in client.calls] == ["events"]

    update = (NOW - timedelta(seconds=30)).isoformat()
    priced_event = _odds_event(
        "ev-pre", "Montana", "Idaho", commence, update
    )
    priced_event["bookmakers"][0]["markets"] = priced_event["bookmakers"][0][
        "markets"
    ][:1]
    odds = OddsSnapshot(
        events=[priced_event],
        fetched_at=NOW,
        requests_remaining=62,
        requests_used=438,
        request_cost=1,
        configured_bookmakers=("draftkings",),
    )
    scoreboard = ScoreboardSnapshot(
        events=[{**event, "completed": False, "scores": None}],
        fetched_at=NOW,
        requests_remaining=61,
        requests_used=439,
        request_cost=1,
    )
    exact_floor = _QuotaAwareClient(
        _events_snapshot([event], remaining=63), odds, scoreboard
    )
    monkeypatch.setattr(live, "write_live_snapshot", lambda season, snapshot: None)
    snapshot = live.capture_live_snapshot(
        exact_floor,
        2026,
        _schedule(),
        lookback_hours=8,
        lookahead_hours=6,
        days_from=None,
        min_quota=50,
        future_poll_markets=future_markets,
    )
    assert snapshot.poll.iloc[0]["scores_requests_remaining"] == 61
    assert [call[0] for call in exact_floor.calls] == ["events", "odds", "scores"]


def test_paid_partial_failure_preserves_quota_provenance(tmp_path, monkeypatch):
    commence = (NOW + timedelta(hours=3)).isoformat()
    update = (NOW - timedelta(seconds=30)).isoformat()
    event = _odds_event("ev-pre", "Montana", "Idaho", commence, update)
    event["bookmakers"][0]["markets"] = event["bookmakers"][0]["markets"][:1]
    odds = OddsSnapshot(
        events=[event],
        fetched_at=NOW,
        requests_remaining=77,
        requests_used=423,
        request_cost=1,
        configured_bookmakers=("draftkings",),
    )

    class Client(_QuotaAwareClient):
        def get_ncaaf_scores(self, days_from):
            self.calls.append(("scores", days_from))
            raise OddsAPIError("score provider unavailable")

    client = Client(_events_snapshot([event]), odds)
    monkeypatch.setattr(live, "OddsAPIClient", lambda: client)
    monkeypatch.setattr(live, "RAW_DIR", tmp_path)

    completed = live.run_live_polling(
        2026,
        _schedule(),
        polls=1,
        interval_seconds=0,
        lookback_hours=8,
        lookahead_hours=6,
        days_from=None,
        min_quota=50,
        max_failures=1,
        progress=lambda message: None,
    )

    polls = live.read_live_frames(2026)["polls"]
    assert completed == 0
    assert polls["poll_status"].tolist() == ["error"]
    assert polls["odds_request_cost"].tolist() == [1]
    assert polls["odds_requests_remaining"].tolist() == [77]
    assert polls["scores_request_cost"].isna().all()


def test_post_scores_quota_change_fails_closed(monkeypatch):
    commence = (NOW + timedelta(hours=3)).isoformat()
    update = (NOW - timedelta(seconds=30)).isoformat()
    event = _odds_event("ev-pre", "Montana", "Idaho", commence, update)
    event["bookmakers"][0]["markets"] = event["bookmakers"][0]["markets"][:1]
    odds = OddsSnapshot(
        events=[event],
        fetched_at=NOW,
        requests_remaining=69,
        requests_used=431,
        request_cost=1,
        configured_bookmakers=("draftkings",),
    )
    scoreboard = ScoreboardSnapshot(
        events=[],
        fetched_at=NOW,
        requests_remaining=59,
        requests_used=441,
        request_cost=1,
    )
    client = _QuotaAwareClient(
        _events_snapshot([event], remaining=70), odds, scoreboard
    )
    monkeypatch.setattr(live, "write_live_snapshot", lambda season, snapshot: None)

    with pytest.raises(live.PartialPaidPollError, match="reserved for future"):
        live.capture_live_snapshot(
            client,
            2026,
            _schedule(),
            lookback_hours=8,
            lookahead_hours=6,
            days_from=None,
            min_quota=50,
            future_poll_markets=(
                live.LIVE_MARKETS,
                live.CLOSING_MARKETS,
                live.LIVE_MARKETS,
                live.LIVE_MARKETS,
                live.LIVE_MARKETS,
            ),
        )


def test_polling_passes_full_window_reserve_and_required_targets(monkeypatch):
    client = _QuotaAwareClient(_events_snapshot([]))
    monkeypatch.setattr(live, "OddsAPIClient", lambda: client)
    calls = []
    clock = [0.0]
    sleeps = []

    def capture(
        client,
        season,
        schedule,
        lookback_hours,
        lookahead_hours,
        days_from,
        min_quota,
        required_game_ids,
        markets,
        future_poll_markets,
    ):
        calls.append((markets, future_poll_markets, required_game_ids))
        clock[0] += 0.5
        return _snapshot()

    def advance(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(live, "capture_live_snapshot", capture)

    market_plan = (
        live.LIVE_MARKETS,
        live.CLOSING_MARKETS,
        live.LIVE_MARKETS,
    )
    completed = live.run_live_polling(
        2026,
        _schedule(),
        polls=3,
        interval_seconds=2,
        lookback_hours=8,
        lookahead_hours=6,
        days_from=None,
        min_quota=50,
        max_failures=1,
        required_game_ids=(101,),
        poll_markets=market_plan,
        progress=lambda message: None,
        sleep=advance,
        monotonic=lambda: clock[0],
    )

    assert completed == 3
    assert calls == [
        (live.LIVE_MARKETS, market_plan[1:], (101,)),
        (live.CLOSING_MARKETS, market_plan[2:], (101,)),
        (live.LIVE_MARKETS, (), (101,)),
    ]
    assert sleeps == [1.5, 1.5]

    kickoff_at = pd.Timestamp("2026-08-29T16:00:00Z")
    target = kickoff.KickoffTarget(
        game_ids=(101,),
        first_kickoff=kickoff_at,
        last_kickoff=kickoff_at,
        labels=("North Carolina at TCU (101)",),
        kickoffs=(kickoff_at,),
    )
    window = kickoff.plan_kickoff_window(
        target,
        as_of=kickoff_at - pd.Timedelta(minutes=5),
        interval_seconds=120,
    )
    planned = kickoff.plan_kickoff_poll_markets(target, window, 120)
    assert planned == (
        live.LIVE_MARKETS,
        live.LIVE_MARKETS,
        live.CLOSING_MARKETS,
        live.LIVE_MARKETS,
        live.LIVE_MARKETS,
        live.LIVE_MARKETS,
    )
    assert sum(live.live_poll_quota_cost(markets) for markets in planned) == 13

    boundary_window = kickoff.plan_kickoff_window(
        target,
        as_of=kickoff_at - pd.Timedelta(minutes=5),
        interval_seconds=300,
    )
    assert kickoff.plan_kickoff_poll_markets(target, boundary_window, 300) == (
        live.CLOSING_MARKETS,
        live.LIVE_MARKETS,
        live.LIVE_MARKETS,
    )

    same_time = kickoff.KickoffTarget(
        game_ids=(101, 102),
        first_kickoff=kickoff_at,
        last_kickoff=kickoff_at,
        labels=("Game one", "Game two"),
        kickoffs=(kickoff_at, kickoff_at),
    )
    same_time_plan = kickoff.plan_kickoff_poll_markets(
        same_time, window, 120
    )
    assert sum("totals" in markets for markets in same_time_plan) == 1

    later = kickoff_at + pd.Timedelta(minutes=5)
    staggered = kickoff.KickoffTarget(
        game_ids=(101, 102),
        first_kickoff=kickoff_at,
        last_kickoff=later,
        labels=("Game one", "Game two"),
        kickoffs=(kickoff_at, later),
    )
    staggered_window = kickoff.plan_kickoff_window(
        staggered,
        as_of=kickoff_at - pd.Timedelta(minutes=5),
        interval_seconds=120,
    )
    staggered_plan = kickoff.plan_kickoff_poll_markets(
        staggered, staggered_window, 120
    )
    assert [
        index for index, markets in enumerate(staggered_plan)
        if "totals" in markets
    ] == [2, 4]


def test_live_odds_rejects_multi_region_quota_configuration():
    assert OddsAPIClient(api_key="test", regions="us").quota_region_units == 1
    client = OddsAPIClient(api_key="test", regions="us,us2")

    with pytest.raises(OddsAPIError, match="would charge 2x"):
        client.ensure_single_quota_region()


def test_events_endpoint_uses_the_free_filtered_discovery_route():
    class Response:
        status_code = 200
        text = ""
        headers = {
            "x-requests-remaining": "78",
            "x-requests-used": "422",
            "x-requests-last": "0",
        }

        @staticmethod
        def json():
            return []

    class Session:
        def __init__(self):
            self.request = None

        def get(self, url, params, timeout):
            self.request = (url, params, timeout)
            return Response()

    client = OddsAPIClient(api_key="test", regions="us")
    client.session = Session()
    snapshot = client.get_ncaaf_events(
        NOW - timedelta(hours=1), NOW + timedelta(hours=1)
    )

    url, params, timeout = client.session.request
    assert url.endswith("/sports/americanfootball_ncaaf/events")
    assert params["commenceTimeFrom"] == "2026-09-05T19:00:00Z"
    assert params["commenceTimeTo"] == "2026-09-05T21:00:00Z"
    assert "markets" not in params
    assert timeout == 60
    assert snapshot.request_cost == 0
    assert snapshot.requests_remaining == 78


def test_snapshots_are_append_only_and_replayable(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "RAW_DIR", tmp_path)
    snapshot = _snapshot()
    live.write_live_snapshot(2026, snapshot)

    with pytest.raises(FileExistsError):
        live.write_live_snapshot(2026, snapshot)

    problems, frames = live.verify_live_snapshots(2026)
    assert problems == []
    available = live.offers_available_at(
        frames["polls"], frames["offers"], pd.Timestamp(NOW)
    )
    assert len(available) == 8
    before = live.offers_available_at(
        frames["polls"], frames["offers"], pd.Timestamp(NOW) - pd.Timedelta(hours=1)
    )
    assert before.empty

    # Tampering with a stored snapshot must be detected by the replay check,
    # even when the mutation keeps the row count unchanged.
    offers_path = next((tmp_path / "live_odds" / "2026" / "offers").glob("*.parquet"))
    tampered = pd.read_parquet(offers_path)
    tampered.loc[0, "point"] = tampered.loc[0, "point"] + 1.0
    tampered.to_parquet(offers_path, index=False)
    problems, _ = live.verify_live_snapshots(2026)
    assert any("offers_sha256" in problem for problem in problems)


def test_live_schedule_uses_the_freshest_stored_snapshot(tmp_path, monkeypatch):
    raw_path = tmp_path / "games" / "2026.parquet"
    preseason_path = tmp_path / "preseason" / "2026" / "games.parquet"
    raw_path.parent.mkdir(parents=True)
    preseason_path.parent.mkdir(parents=True)
    raw_path.touch()
    preseason_path.touch()
    os.utime(raw_path, (1, 1))
    os.utime(preseason_path, (2, 2))
    raw = pd.DataFrame(
        {
            "id": [1],
            "start_date": [NOW],
            "home_team": ["Old State"],
            "away_team": ["Old Tech"],
            "home_classification": ["fbs"],
            "away_classification": ["fbs"],
        }
    )
    preseason = raw.assign(
        id=2, home_team="Fresh State", away_team="Fresh Tech"
    )
    monkeypatch.setattr(live, "RAW_DIR", tmp_path)
    monkeypatch.setattr("backend.etl.store.read_games", lambda season: raw)
    monkeypatch.setattr(
        "backend.etl.store.read_preseason_source",
        lambda season, source: preseason,
    )

    schedule = live.load_division_one_schedule(2026)
    assert schedule["game_id"].tolist() == [2]

    os.utime(raw_path, (3, 3))
    schedule = live.load_division_one_schedule(2026)
    assert schedule["game_id"].tolist() == [1]


def test_kickoff_window_requires_pregame_anchor_and_postkick_provider_state(
    monkeypatch,
):
    # This is the live acceptance boundary: a window is ready only after a
    # provider-backed postkick state exists, while the selected anchor still
    # points to a fresh multi-provider snapshot captured before kickoff.
    starts_at = pd.Timestamp("2026-08-29T16:00:00Z")
    target = kickoff.KickoffTarget(
        game_ids=(101,),
        first_kickoff=starts_at,
        last_kickoff=starts_at,
        labels=("North Carolina at TCU (101)",),
    )
    monkeypatch.setattr(
        kickoff,
        "load_division_one_schedule",
        lambda season: pd.DataFrame(
            {
                "game_id": [101],
                "start_date": [starts_at],
                "home_team": ["TCU"],
                "away_team": ["North Carolina"],
            }
        ),
    )
    pregame = starts_at - pd.Timedelta(minutes=1)
    live_at = starts_at + pd.Timedelta(minutes=1)
    frames = {
        "events": pd.DataFrame(
            {
                "game_id": [101, 101],
                "fetched_at": [pregame, live_at],
                "phase": ["pregame", "live"],
            }
        ),
        "offers": pd.DataFrame(
            {
                "game_id": [101] * 7,
                "snapshot_id": ["close"] * 6 + ["live"],
                "fetched_at": [pregame] * 6 + [live_at],
                "phase": ["pregame"] * 6 + ["live"],
                "market": [
                    "spreads",
                    "spreads",
                    "totals",
                    "totals",
                    "totals",
                    "totals",
                    "spreads",
                ],
                "selection": [
                    "home",
                    "home",
                    "over",
                    "under",
                    "over",
                    "under",
                    "home",
                ],
                "provider_key": ["a", "b", "a", "a", "b", "b", "a"],
                "provider_last_update": [
                    starts_at - pd.Timedelta(seconds=30),
                    starts_at - pd.Timedelta(seconds=45),
                    starts_at - pd.Timedelta(seconds=25),
                    starts_at - pd.Timedelta(seconds=25),
                    starts_at - pd.Timedelta(seconds=40),
                    starts_at - pd.Timedelta(seconds=40),
                    live_at,
                ],
                "point": [-3.5, -3.5, 51.5, 51.5, 51.5, 51.5, -4.0],
                "price": [-110, -110, -105, -115, -108, -112, -110],
            }
        ),
    }
    anchors = pd.DataFrame(
        {
            "game_id": [101],
            "season": [2026],
            "closing_snapshot_id": ["close"],
            "closing_fetched_at": [pregame],
            "latest_provider_update": [
                starts_at - pd.Timedelta(seconds=30)
            ],
        }
    )

    problems, details = kickoff.validate_completed_window(
        target, frames, anchors
    )
    assert problems == []
    assert "ignored 1 live offers" in details[0]

    without_totals = {
        **frames,
        "offers": frames["offers"][frames["offers"]["market"].ne("totals")],
    }
    problems, _ = kickoff.validate_completed_window(
        target, without_totals, anchors
    )
    assert any("0 paired spread/total providers" in problem for problem in problems)

    postkick_provider = {**frames, "offers": frames["offers"].copy()}
    postkick_provider["offers"].loc[
        postkick_provider["offers"]["market"].eq("totals")
        & postkick_provider["offers"]["provider_key"].eq("b"),
        "provider_last_update",
    ] = live_at
    problems, _ = kickoff.validate_completed_window(
        target, postkick_provider, anchors
    )
    assert any("total provider update at or after" in problem for problem in problems)

    split_pair = {**frames, "offers": frames["offers"].copy()}
    split_pair["offers"].loc[
        split_pair["offers"]["market"].eq("totals")
        & split_pair["offers"]["provider_key"].eq("a")
        & split_pair["offers"]["selection"].eq("under"),
        "provider_last_update",
    ] = live_at
    problems, _ = kickoff.validate_completed_window(target, split_pair, anchors)
    assert any("total provider update at or after" in problem for problem in problems)

    leaked = anchors.assign(
        closing_snapshot_id="live", closing_fetched_at=live_at
    )
    problems, _ = kickoff.validate_completed_window(target, frames, leaked)
    assert any("post-kickoff snapshot" in problem for problem in problems)
    assert any("closing snapshot has 0 providers" in problem for problem in problems)


def test_flatten_offers_survives_parquet_nested_arrays(tmp_path):
    # Odds snapshots are read back from parquet, which turns every nested
    # list into a numpy array; truthiness checks on those raised as soon as
    # a bookmaker carried more than one market (spreads + totals).
    from backend.odds.markets import flatten_odds_api_offers

    events = pd.DataFrame(
        [
            {
                **_odds_event(
                    "ev-1",
                    "Alabama Crimson Tide",
                    "Georgia Bulldogs",
                    NOW.isoformat(),
                    NOW.isoformat(),
                ),
                "execution_eligibility_verified": False,
                "source_fetched_at": NOW.isoformat(),
            }
        ]
    )
    path = tmp_path / "odds_api.parquet"
    events.to_parquet(path, index=False)
    schedule = pd.DataFrame(
        {
            "game_id": [7],
            "start_date": [NOW.isoformat()],
            "home_team": ["Alabama"],
            "away_team": ["Georgia"],
        }
    )
    offers, coverage = flatten_odds_api_offers(pd.read_parquet(path), schedule)
    assert coverage["matched"].all()
    assert len(offers) == 4  # two spreads outcomes + over + under
    assert set(offers["market"]) == {"spreads", "totals"}


def test_odds_team_alias_matches_the_canonical_cfbd_game():
    from backend.odds.markets import match_event

    schedule = pd.DataFrame(
        {
            "game_id": [401856769],
            "start_date": ["2026-09-05T00:00:00Z"],
            "home_team": ["Kansas"],
            "away_team": ["Long Island University"],
        }
    )

    game_id, score = match_event(
        "2026-09-05T00:00:00Z",
        "Kansas Jayhawks",
        "LIU Sharks",
        schedule,
    )

    assert game_id == 401856769
    assert score == 1.0
