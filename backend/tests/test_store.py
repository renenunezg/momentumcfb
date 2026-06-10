import pandas as pd

import backend.etl.store as store
from backend.etl.ingest import write_parquet


def test_read_season_pbp_concats_weeks(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RAW_DIR", tmp_path)
    base = tmp_path / "pbp" / "2024"
    write_parquet(pd.DataFrame({"ppa": [0.1, 0.2]}), base / "regular_01.parquet")
    write_parquet(pd.DataFrame({"ppa": [0.3]}), base / "regular_02.parquet")
    out = store.read_season_pbp(2024)
    assert len(out) == 3
