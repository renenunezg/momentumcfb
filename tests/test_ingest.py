import pandas as pd

from backend.etl import ingest, store


class FakeCFBDClient:
    def __init__(self):
        self.calls = []

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
    pd.DataFrame({"id": [1]}).to_parquet(
        season_dir / "regular_01.parquet", index=False
    )
    pd.DataFrame({"id": [2], "pbp_source": ["cfbd"]}).to_parquet(
        season_dir / "postseason_01.parquet", index=False
    )
    monkeypatch.setattr(store, "RAW_DIR", tmp_path)

    loaded = store.read_season_pbp(2025)

    assert loaded["id"].tolist() == [1, 2]
    assert loaded["pbp_source"].tolist() == ["cfbd", "cfbd"]
