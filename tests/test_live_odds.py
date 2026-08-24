import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from backend.odds import kickoff, live
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
                "game_id": [101, 101, 101],
                "snapshot_id": ["close", "close", "live"],
                "fetched_at": [pregame, pregame, live_at],
                "phase": ["pregame", "pregame", "live"],
                "market": ["spreads", "spreads", "spreads"],
                "selection": ["home", "home", "home"],
                "provider_key": ["a", "b", "a"],
                "provider_last_update": [
                    starts_at - pd.Timedelta(seconds=30),
                    starts_at - pd.Timedelta(seconds=45),
                    live_at,
                ],
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
