from types import SimpleNamespace

import pandas as pd
import pytest

from backend import cli
from backend.etl import ingest, store
from backend.odds.client import OddsAPIError


class FakeCFBDClient:
    def __init__(self):
        self.calls = []

    def ensure_budget(self, calls):
        assert calls <= 100

    def get(self, path, params):
        self.calls.append((path, params))
        if path == "/games":
            return [{"id": 401752794, "season": 2026, "week": 1}]
        if path == "/plays" and params["seasonType"] == "regular":
            return [
                {
                    "id": 11,
                    "gameId": 401752794,
                    "driveId": "4017527941",
                    "ppa": 0.31,
                }
            ]
        if path == "/plays":
            return []
        if path == "/lines":
            return [{"id": 401752794, "lines": []}]
        if path == "/talent":
            return [{"year": 2026, "school": "Ohio State", "talent": 1000}]
        if path == "/player/returning":
            return [{"season": 2026, "team": "Ohio State"}]
        raise AssertionError(f"unexpected CFBD request: {path} {params}")


def test_ingest_season_fetches_and_labels_cfbd_plays(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)
    client = FakeCFBDClient()

    ingest.ingest_season(client, 2026, only_week=1)

    plays = pd.read_parquet(tmp_path / "pbp" / "2026" / "regular_01.parquet")
    assert plays["id"].tolist() == [11]
    assert plays["game_id"].tolist() == [401752794]
    assert plays["pbp_source"].tolist() == ["cfbd"]
    assert client.calls == [
        ("/games", {"year": 2026, "seasonType": "both"}),
        ("/plays", {"year": 2026, "week": 1, "seasonType": "regular"}),
        ("/plays", {"year": 2026, "week": 1, "seasonType": "postseason"}),
        ("/lines", {"year": 2026}),
        ("/talent", {"year": 2026}),
        ("/player/returning", {"year": 2026}),
    ]


def test_read_season_pbp_combines_only_cfbd_weekly_snapshots(tmp_path, monkeypatch):
    season_dir = tmp_path / "pbp" / "2025"
    season_dir.mkdir(parents=True)
    pd.DataFrame({"id": [1]}).to_parquet(season_dir / "regular_01.parquet", index=False)
    pd.DataFrame({"id": [2], "pbp_source": ["cfbd"]}).to_parquet(
        season_dir / "postseason_01.parquet", index=False
    )
    monkeypatch.setattr(store, "RAW_DIR", tmp_path)

    loaded = store.read_season_pbp(2025)

    assert loaded["id"].tolist() == [1, 2]
    assert loaded["pbp_source"].tolist() == ["cfbd", "cfbd"]


def test_preseason_weekly_commands_noop_without_cfbd_plays(
    tmp_path, monkeypatch, capsys
):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    games_dir = raw_dir / "games"
    games_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "id": 401752794,
                "season": 2026,
                "week": 1,
                "season_type": "regular",
                "start_date": "2026-08-29T18:00:00Z",
                "completed": False,
                "home_id": 194,
                "home_team": "Ohio State",
                "home_classification": "fbs",
                "away_id": 164,
                "away_team": "Rutgers",
                "away_classification": "fbs",
            }
        ]
    ).to_parquet(games_dir / "2026.parquet", index=False)
    monkeypatch.setattr(store, "RAW_DIR", raw_dir)
    monkeypatch.setattr(store, "PROCESSED_DIR", processed_dir)

    cli.main(["features", "--seasons", "2026"])
    cli.main(["weekly-update", "--season", "2026"])

    output = capsys.readouterr().out
    assert "2026: no CFBD play-by-play is available" in output
    assert "weekly update not ready: no completed D1 games are available" in output


def test_weekly_update_publishes_pure_model_when_odds_quota_is_exhausted(
    tmp_path, monkeypatch, capsys
):
    from backend import publish
    from backend.model import weekly
    from backend.odds import client as odds_client
    from backend.odds import scheduling

    calls = []
    scheduled = []
    schema_checks = []
    monkeypatch.setattr(
        publish, "ensure_recommendation_schema", lambda: schema_checks.append(True)
    )
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(
        scheduling,
        "schedule_weekly_kickoff",
        lambda projections, season: scheduled.append(season),
    )
    result = SimpleNamespace(
        week=1,
        ratings=[object()],
        projections=pd.DataFrame({"market_home_spread": [float("nan")]}),
        market_comparisons=[object()],
        log_directory=tmp_path,
    )

    def fake_run_weekly_forecast(*args, **kwargs):
        assert schema_checks
        calls.append(kwargs)
        if len(calls) == 1:
            raise OddsAPIError("OUT_OF_USAGE_CREDITS")
        return result

    monkeypatch.setattr(weekly, "resolve_ready_forecast_week", lambda *args: 1)
    monkeypatch.setattr(weekly, "run_weekly_forecast", fake_run_weekly_forecast)
    monkeypatch.setattr(publish, "weekly_forecast_is_published", lambda *args: False)
    monkeypatch.setattr(publish, "publish", lambda *args, **kwargs: {})
    monkeypatch.setattr(odds_client, "OddsAPIClient", lambda: object())

    cli.main(["weekly-update", "--season", "2026"])

    assert [call["require_market"] for call in calls] == [True, False]
    assert calls[0]["odds_client"] is not None
    assert calls[1]["odds_client"] is None
    assert "publishing the pure-model forecast" in capsys.readouterr().out
    assert scheduled == []

    result.projections["market_home_spread"] = -3.5
    cli.main(["weekly-update", "--season", "2026"])
    assert scheduled == [2026]


def test_cfbd_quota_gate_counts_retries_and_persists_across_commands(
    tmp_path, monkeypatch
):
    from backend.cfbd import client as cfbd

    requests = []

    def response(status, remaining):
        return SimpleNamespace(
            status_code=status,
            headers={"X-CallLimit-Remaining": str(remaining)},
            text="unavailable",
            json=lambda: [],
        )

    responses = iter([response(503, 99), response(200, 98)])

    def get(*args, **kwargs):
        requests.append(args)
        return next(responses)

    monkeypatch.setattr(cfbd.time, "sleep", lambda _: None)
    usage = tmp_path / "usage.json"
    client = cfbd.CFBDClient("test", max_calls=3, usage_file=usage)
    monkeypatch.setattr(client.session, "get", get)
    with pytest.raises(cfbd.CFBDError, match="session budget"):
        client.ensure_budget(4)
    assert requests == []
    client.ensure_budget(2)
    client.get("/teams", {})
    assert len(requests) == client.calls_used == 2

    resumed = cfbd.CFBDClient("test", max_calls=3, usage_file=usage)
    with pytest.raises(cfbd.CFBDError, match="session budget"):
        resumed.ensure_budget(2)
    assert resumed.remaining == 98

    for headers, error in [
        ({"X-CallLimit-Remaining": "41"}, "preserving a 40-call reserve"),
        ({}, "omitted X-CallLimit-Remaining"),
    ]:
        constrained = cfbd.CFBDClient("test")
        monkeypatch.setattr(
            constrained.session,
            "get",
            lambda *a, **k: SimpleNamespace(
                status_code=200,
                headers=headers,
                json=lambda: [],
            ),
        )
        constrained.ensure_budget(3)
        constrained.get("/teams", {})
        with pytest.raises(cfbd.CFBDError, match=error):
            constrained.get("/roster", {})
        assert constrained.calls_used == 1


def test_player_ingestion_resumes_completed_week_chunks_without_future_calls(
    tmp_path, monkeypatch
):
    from backend.players import ingest as players

    monkeypatch.setattr(players, "RAW_DIR", tmp_path)
    games = pd.DataFrame(
        [
            {
                "id": 1,
                "week": 1,
                "season_type": "regular",
                "completed": True,
                "home_team": "A",
                "away_team": "B",
            },
            {
                "id": 2,
                "week": 2,
                "season_type": "regular",
                "completed": False,
                "home_team": "A",
                "away_team": "C",
            },
            {
                "id": 3,
                "week": 1,
                "season_type": "postseason",
                "completed": False,
                "home_team": "A",
                "away_team": "C",
            },
        ]
    )
    ingest.write_parquet(games, tmp_path / "games" / "2026.parquet")
    teams = pd.DataFrame(
        [
            {"school": "A", "classification": "fbs", "conference": "Alpha"},
            {"school": "B", "classification": "fbs", "conference": "Beta"},
            {"school": "C", "classification": "fbs", "conference": "Future"},
        ]
    )
    ingest.write_parquet(teams, players.players_dir(2026) / "teams.parquet")
    calls = []
    estimates = []
    fail_beta = True

    class PlayerClient:
        def ensure_budget(self, estimated):
            estimates.append(estimated)

        def get(self, endpoint, params, **kwargs):
            nonlocal fail_beta
            calls.append((endpoint, params))
            if endpoint == "/roster":
                return [{"id": "athlete", "team": "A"}]
            if endpoint == "/rankings":
                return []
            if endpoint == "/conferences":
                return [
                    {"name": name, "abbreviation": name}
                    for name in ["Alpha", "Beta", "Future"]
                ]
            assert params["week"] == 1 and params["seasonType"] == "regular"
            if endpoint == "/games/players":
                return [
                    {
                        "id": 1,
                        "teams": [
                            {
                                "team": "A",
                                "categories": [
                                    {
                                        "name": "passing",
                                        "types": [
                                            {
                                                "name": "YDS",
                                                "athletes": [
                                                    {
                                                        "id": "athlete",
                                                        "name": "Quarterback",
                                                        "stat": "200",
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            assert endpoint == "/plays/stats"
            if params["conference"] == "Beta" and fail_beta:
                fail_beta = False
                raise RuntimeError("interrupted")
            return [
                {
                    "gameId": 1,
                    "playId": 10,
                    "athleteId": "athlete",
                    "statType": "passing",
                    "stat": 20,
                }
            ]

    with pytest.raises(RuntimeError, match="interrupted"):
        players.ingest_player_sources(PlayerClient(), 2026)
    before_resume = len(calls)
    manifest = players.ingest_player_sources(PlayerClient(), 2026)
    resumed_calls = calls[before_resume:]
    assert [
        params["conference"]
        for endpoint, params in resumed_calls
        if endpoint == "/plays/stats"
    ] == ["Beta"]
    assert not any(endpoint == "/games/players" for endpoint, _ in resumed_calls)
    assert estimates == [6, 1]
    assert set(manifest["source"]) >= {"box_regular_01", "play_stats_regular_01"}
    stats = pd.read_parquet(players.weekly_path(2026, "play_stats", "regular", 1))
    assert stats["game_id"].tolist() == [1]

    calls.clear()
    players.ingest_player_sources(PlayerClient(), 2026)
    assert calls == []
    # A pre-checkpoint cache is adopted from its game ids without refetching.
    (players.players_dir(2026) / "completed_games.json").unlink()
    players.ingest_player_sources(PlayerClient(), 2026)
    assert calls == []

    # A late completed game must not be marked covered when the provider
    # returns a nonempty but partial box or uncapped conference response.
    extra = games.iloc[[0]].assign(id=4)
    ingest.write_parquet(pd.concat([games, extra]), tmp_path / "games" / "2026.parquet")
    with pytest.raises(ValueError, match="Incomplete box snapshot"):
        players.ingest_player_sources(PlayerClient(), 2026)
    existing_box = pd.read_parquet(players.weekly_path(2026, "box", "regular", 1))
    assert existing_box["game_id"].tolist() == [1]

    original_get = PlayerClient.get

    def complete_box(self, endpoint, params, **kwargs):
        rows = original_get(self, endpoint, params, **kwargs)
        if endpoint == "/games/players":
            rows.append({**rows[0], "id": 4})
        return rows

    monkeypatch.setattr(PlayerClient, "get", complete_box)
    with pytest.raises(ValueError, match="Incomplete play stats"):
        players.ingest_player_sources(PlayerClient(), 2026)
    updated_games = pd.concat([games, extra]).query("completed and week == 1")
    incomplete_parts = players._parts_directory(2026, "regular", 1, updated_games)
    assert not list(incomplete_parts.glob("*.parquet"))

    # Explicit correction refresh still refetches the requested completed week.
    ingest.write_parquet(games, tmp_path / "games" / "2026.parquet")
    monkeypatch.setattr(PlayerClient, "get", original_get)
    calls.clear()
    players.ingest_player_sources(PlayerClient(), 2026, only_week=1, refresh=True)
    assert any(endpoint == "/games/players" for endpoint, _ in calls)
    assert [
        params["conference"] for endpoint, params in calls if endpoint == "/plays/stats"
    ] == ["Alpha", "Beta"]
