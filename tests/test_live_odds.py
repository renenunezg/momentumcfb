from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from backend.odds import live
from backend.odds.client import OddsSnapshot, ScoreboardSnapshot

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
