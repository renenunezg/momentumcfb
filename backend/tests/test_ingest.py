import pandas as pd

from backend.etl.ingest import to_snake, write_parquet


def test_to_snake_converts_camel_case():
    df = pd.DataFrame({"gameId": [1], "homeTeam": ["A"], "ppa": [0.1]})
    out = to_snake(df)
    assert list(out.columns) == ["game_id", "home_team", "ppa"]


def test_write_parquet_overwrites_partition(tmp_path):
    path = tmp_path / "pbp" / "2024" / "regular_01.parquet"
    write_parquet(pd.DataFrame({"a": [1, 2]}), path)
    write_parquet(pd.DataFrame({"a": [3]}), path)
    assert pd.read_parquet(path)["a"].tolist() == [3]
