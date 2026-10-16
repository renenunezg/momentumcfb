import pandas as pd
import pytest

from backend.players import ingest
from backend.players.traits import TraitSpec, context, register, targets


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


def test_recover_targets_parses_incomplete_pass_text_and_prefers_feed_rows():
    plays = pd.DataFrame([
        {"id": "p1", "season": 2024, "game_id": 1, "offense": "Home", "play_type": "Pass Incompletion", "play_text": "A Quarterback pass incomplete to Jon Smith Jr."},
        {"id": "p2", "season": 2024, "game_id": 1, "offense": "Home", "play_type": "Pass Incompletion", "play_text": "A Quarterback pass incomplete to Jon Smith Jr."},
        {"id": "p3", "season": 2024, "game_id": 1, "offense": "Home", "play_type": "Pass Incompletion", "play_text": "A Quarterback pass incomplete"},
    ])
    stats = pd.DataFrame([{"play_id": "p2", "athlete_id": "wr9", "stat_type": "Target", "team": "Home"}])
    roster = pd.DataFrame([{"id": "wr1", "first_name": "Jon", "last_name": "Smith Jr.", "team": "Home"}])
    out = targets.recover_targets(plays, stats, roster).set_index("play_id")
    assert out.loc["p1", "athlete_id"] == "wr1" and out.loc["p1", "source"] == "text"
    assert out.loc["p2", "athlete_id"] == "wr9" and out.loc["p2", "source"] == "feed"
    assert "p3" not in out.index
    coverage = targets.target_coverage(plays, out.reset_index())
    assert coverage.iloc[0][["incompletions", "feed", "text", "uncovered"]].tolist() == [3, 1, 1, 1]


def test_recover_targets_skips_ambiguous_names_and_trims_penalty_text():
    plays = pd.DataFrame([
        {"id": "a", "season": 2024, "game_id": 1, "offense": "Home", "play_type": "Pass Incompletion", "play_text": "Q One pass incomplete to José Núñez, broken up by D Two"},
        {"id": "b", "season": 2024, "game_id": 1, "offense": "Home", "play_type": "Pass Incompletion", "play_text": "Q One pass incomplete to Sam Twin for no gain"},
        {"id": "c", "season": 2024, "game_id": 1, "offense": "Home", "play_type": "Pass Incompletion", "play_text": "Q One pass incomplete. PENALTY Away holding. Penalty: pass incomplete to Ghost Name"},
        {"id": "d", "season": 2024, "game_id": 1, "offense": "Away", "play_type": "Pass Incompletion", "play_text": "Q Two pass incomplete to Sam Twin (D One)"},
        {"id": "e", "season": 2024, "game_id": 1, "offense": "Home", "play_type": "Pass Reception", "play_text": "Q One pass complete to Sam Twin for 5 yds"},
    ])
    stats = pd.DataFrame(columns=["play_id", "athlete_id", "stat_type", "team"])
    roster = pd.DataFrame([
        {"id": 7, "first_name": "Jose", "last_name": "Nunez", "team": "Home"},
        {"id": 8, "first_name": "Sam", "last_name": "Twin", "team": "Home"},
        {"id": 9, "first_name": "Sam", "last_name": "Twin Jr.", "team": "Home"},
        {"id": 10, "first_name": "Sam", "last_name": "Twin", "team": "Away"},
    ])
    out = targets.recover_targets(plays, stats, roster).set_index("play_id")
    assert out["athlete_id"].to_dict() == {"a": "7", "d": "10"}
    assert targets.normalize_name("D.J. Moore III") == "dj moore"
