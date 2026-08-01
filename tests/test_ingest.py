import pandas as pd

from backend.etl import store
from backend.sportsdataverse.pbp import CORE_SOURCE_COLUMNS, normalize_pbp


def _source_play(game_play_number: int, play_id: int) -> dict:
    row = {column: pd.NA for column in CORE_SOURCE_COLUMNS}
    row.update(
        {
            "season": 2025,
            "seasonType": 2,
            "week": 1,
            "game_id": 401752794,
            "game_play_number": game_play_number,
            "id": play_id,
            "period": 1,
            "clock.minutes": 14,
            "clock.seconds": 30,
            "start.TimeSecsRem": 3570,
            "homeTeamId": 194,
            "awayTeamId": 164,
            "homeTeamName": "Ohio State",
            "awayTeamName": "Rutgers",
            "start.pos_team.id": 194,
            "start.def_pos_team.id": 164,
            "start.pos_team.name": "Ohio State",
            "start.def_pos_team.name": "Rutgers",
            "start.pos_team_score": 0,
            "start.def_pos_team_score": 0,
            "end.pos_team_score": 0,
            "end.def_pos_team_score": 0,
            "down": 1,
            "distance": 10,
            "start.yardsToEndzone": 75,
            "end.down": 2,
            "end.distance": 4,
            "end.yardsToEndzone": 69,
            "statYardage": 6,
            "type.text": "Pass Completion",
            "text": "Pass complete for 6 yards",
            "drive.id": "4017527942",
            "scoringPlay": False,
            "isPenalty": False,
            "isTurnover": False,
            "scrimmage_play": True,
            "sp": False,
            "kneel_down": False,
            "EPA_scrimmage": 0.31,
            "pass": True,
            "cpoe": 4.5,
        }
    )
    return row


def test_sportsdataverse_normalization_preserves_play_contract():
    source = pd.DataFrame([_source_play(2, 22), _source_play(1, 11)])

    normalized = normalize_pbp(source)

    assert normalized["id"].tolist() == [11, 22]
    assert normalized["season_type"].tolist() == ["regular", "regular"]
    assert normalized["epa"].tolist() == [0.31, 0.31]
    assert normalized["ppa"].equals(normalized["epa"])
    assert normalized.loc[0, "clock"] == {"minutes": 14, "seconds": 30}
    assert normalized["is_pass_source"].tolist() == [True, True]
    assert normalized["cpoe"].tolist() == [4.5, 4.5]


def test_canonical_snapshot_prevents_mixing_old_weekly_files(tmp_path, monkeypatch):
    season_dir = tmp_path / "pbp" / "2025"
    season_dir.mkdir(parents=True)
    pd.DataFrame({"id": [1]}).to_parquet(
        season_dir / "regular_01.parquet", index=False
    )
    pd.DataFrame({"id": [2]}).to_parquet(
        season_dir / "canonical.parquet", index=False
    )
    monkeypatch.setattr(store, "RAW_DIR", tmp_path)

    loaded = store.read_season_pbp(2025)

    assert loaded["id"].tolist() == [2]
