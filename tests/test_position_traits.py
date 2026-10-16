import pandas as pd
import pytest

from backend.players import ingest
from backend.players.traits import TraitSpec, context, register


def test_ingest_portal_writes_one_call_per_season(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)

    class Client:
        calls = []

        def get(self, path, params=None, **_):
            self.calls.append((path, params))
            return [
                {
                    "season": 2021,
                    "firstName": "A",
                    "lastName": "B",
                    "position": "QB",
                    "origin": "Troy",
                    "destination": "Auburn",
                    "transferDate": None,
                    "rating": None,
                    "stars": 3,
                    "eligibility": "Immediate",
                }
            ]

        def ensure_budget(self, calls):
            pass

    frame = ingest.ingest_portal(Client(), 2021)
    assert Client.calls == [("/player/portal", {"year": 2021})]
    assert list(frame.columns)[:4] == ["season", "first_name", "last_name", "position"]
    assert ingest.read_season_source(2021, "portal").shape[0] == 1


def test_player_presence_counts_zero_stat_box_rows_as_present():
    box = pd.DataFrame([
        {"game_id": 1, "team": "Away", "athlete_id": "de1", "athlete_name": "de1", "category": "defensive", "stat_name": "TOT", "stat": "0"},
        {"game_id": 2, "team": "Away", "athlete_id": "de1", "athlete_name": "de1", "category": "defensive", "stat_name": "TOT", "stat": "3"},
    ])
    games = pd.DataFrame({"game_id": [1, 2, 3], "model_week": [1, 2, 3], "home_team": "Home", "away_team": "Away", "completed": True})
    credit = pd.DataFrame(columns=["athlete_id", "team", "game_id"])
    out = context.player_presence(box, credit, games)
    de1 = out[out["athlete_id"].eq("de1")].set_index("game_id")["present"]
    assert de1.to_dict() == {1: True, 2: True, 3: False}


def test_register_keys_by_name_and_rejects_duplicates_and_bad_aggregates(monkeypatch):
    monkeypatch.setattr("backend.players.traits.TRAITS", {})
    from backend.players import traits

    spec = TraitSpec("def.havoc", ("DL", "LB", "DB"), "counted")
    assert register(spec) is spec
    assert traits.TRAITS == {"def.havoc": spec}
    with pytest.raises(ValueError, match="already registered"):
        register(TraitSpec("def.havoc", ("DL",), "counted"))
    with pytest.raises(ValueError, match="unknown aggregate"):
        TraitSpec("x", ("QB",), "counted", aggregate="rates")
