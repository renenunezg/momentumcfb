from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from backend.cfbd.client import CFBDClient
from backend.model.ingame import IngameBaselineParams
from backend.serving.live_publish import Game, LivePublisher, run

KICKOFF = datetime(2026, 10, 3, 19, 0, tzinfo=timezone.utc)
PARAMS = IngameBaselineParams(1.4, 0.03, 3.0)


def _game(game_id, start=KICKOFF):
    return Game(
        game_id=game_id,
        season=2026,
        week=5,
        start=start,
        home_team_id=1,
        away_team_id=2,
        home_team="Home",
        away_team="Away",
        home_margin=3.0,
        margin_sd=16.0,
        as_of=KICKOFF - timedelta(days=2),
        model_version="joint_scoring_v13",
    )


def _row(game_id, status, home=None, away=None, period=None, clock=None):
    return {
        "id": game_id,
        "status": status,
        "period": period,
        "clock": clock,
        "possession": "home",
        "situation": None,
        "homeTeam": {"id": 1, "name": "Home", "points": home},
        "awayTeam": {"id": 2, "name": "Away", "points": away},
    }


def _publisher(games, board):
    calls, written = [], []

    def fetch_board():
        calls.append(1)
        return board

    publisher = LivePublisher(
        PARAMS,
        load_games=lambda start, end: [g for g in games if start <= g.start <= end],
        load_saved=lambda game_ids: {},
        fetch_board=fetch_board,
        write=written.extend,
    )
    return publisher, calls, written


def test_polling_starts_at_kickoff_and_stops_at_the_final():
    board = [_row(10, "scheduled")]
    publisher, calls, written = _publisher([_game(10)], board)

    # Inside the lead window the database alone answers; the quota is untouched.
    assert publisher.poll(KICKOFF - timedelta(minutes=10))
    assert calls == [] and written[-1]["abstract_state"] == "Pre"

    board[0] = _row(10, "in_progress", home=14, away=0, period=2, clock="07:30")
    assert publisher.poll(KICKOFF + timedelta(hours=1))
    live = written[-1]
    assert live["abstract_state"] == "Live" and live["home_win_probability"] > 0.8

    for minute, replacement in enumerate(
        [
            [],
            [_row(10, "completed", home=None, away=7)],
            [_row(10, "suspended", home=14, away=0)],
        ],
        1,
    ):
        board[:] = replacement
        assert publisher.poll(KICKOFF + timedelta(hours=3, minutes=minute))
        assert written[-1]["abstract_state"] == "Live"
        assert 10 not in publisher.done
    saved = written[-1]
    publisher, calls, written = _publisher([_game(10)], board)
    publisher.load_saved = lambda ids: {10: saved}
    board[:] = [_row(10, "completed", home=28, away=10, period=4, clock="00:00")]

    board[0] = _row(10, "completed", home=28, away=10, period=4, clock="00:00")
    assert not publisher.poll(KICKOFF + timedelta(hours=3, minutes=4))
    assert written[-1]["home_win_probability"] == 1.0
    assert [p["s"] for p in written[-1]["history"]] == [0, 1350, 3600]

    spent, rows = len(calls), len(written)
    assert not publisher.poll(KICKOFF + timedelta(hours=3, minutes=5))
    assert (len(calls), len(written)) == (spent, rows)


def test_a_game_that_never_plays_is_closed_instead_of_polled_forever():
    board = [_row(10, "postponed"), _row(11, "scheduled")]
    publisher, calls, written = _publisher([_game(10), _game(11)], board)

    assert publisher.poll(KICKOFF + timedelta(minutes=1))
    states = {p["game_id"]: p["abstract_state"] for p in written}
    assert states == {10: "Off", 11: "Pre"}

    assert not publisher.poll(KICKOFF + timedelta(hours=3, minutes=1))
    assert written[-1]["abstract_state"] == "Off" and len(calls) == 2


def test_a_gateway_error_page_costs_one_poll_and_the_worker_reaches_the_final(
    monkeypatch,
):
    # Oct 3: one Cloudflare error page, which carries no quota header, ended
    # ten workers. Only the HTTP call is stubbed; the client, the publisher
    # and the worker loop are the real ones.
    final = [_row(10, "completed", home=28, away=10, period=4, clock="00:00")]
    answers = iter(
        [
            SimpleNamespace(status_code=502, headers={}, text="Bad gateway"),
            SimpleNamespace(
                status_code=200,
                headers={"X-CallLimit-Remaining": "5000"},
                json=lambda: final,
            ),
        ]
    )
    client = CFBDClient("test")
    monkeypatch.setattr(client.session, "get", lambda *a, **k: next(answers))
    publisher, _, written = _publisher([_game(10)], [])
    publisher.fetch_board = lambda: client.get("/scoreboard", retries=1, timeout=15)

    run(
        publisher,
        watch=True,
        interval=30,
        sleep=lambda seconds: None,
        now=lambda: KICKOFF + timedelta(hours=3),
    )
    assert client.calls_used == 2 and written[-1]["abstract_state"] == "Final"
