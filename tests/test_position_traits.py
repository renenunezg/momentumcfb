import pandas as pd
import pytest

from backend.players import ingest
from backend.players.traits import TraitSpec, context, expectations, register, targets


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


def test_recover_targets_ignores_null_feed_ids_and_normalizes_float_ids():
    plays = pd.DataFrame([
        {"id": "n", "season": 2024, "game_id": 1, "offense": "Home", "play_type": "Pass Incompletion", "play_text": "Q One pass incomplete to Jon Smith"},
        {"id": "f", "season": 2024, "game_id": 1, "offense": "Home", "play_type": "Pass Incompletion", "play_text": "Q One pass incomplete"},
    ])
    stats = pd.DataFrame([
        {"play_id": "n", "athlete_id": None, "stat_type": "Target", "team": "Home"},
        {"play_id": "f", "athlete_id": 4.0, "stat_type": "Target", "team": "Home"},
    ])
    roster = pd.DataFrame([{"id": 11.0, "first_name": "Jon", "last_name": "Smith", "team": "Home"}])
    out = targets.recover_targets(plays, stats, roster).set_index("play_id")
    assert out.loc["n", "athlete_id"] == "11" and out.loc["n", "source"] == "text"
    assert out.loc["f", "athlete_id"] == "4" and out.loc["f", "source"] == "feed"


def test_expectation_tables_use_only_earlier_seasons_and_fall_back_to_the_mean():
    prior = pd.DataFrame({"season": 2021, "down": [1, 1, 3], "distance": [10, 10, 2], "yards_to_goal": [75, 75, 25],
                          "is_rush": True, "is_competitive": True, "yards_gained": [2.0, 6.0, 1.0], "is_success": [False, True, True],
                          "play_type": "Rush", "play_text": ""})
    current = prior.assign(season=2022, yards_gained=99.0)
    tables = expectations.fit_expectation_tables({2021: prior, 2022: current}, before_season=2022)
    assert tables.source_seasons == [2021]
    probe = pd.DataFrame({"down": [1, 4], "distance": [10, 1], "yards_to_goal": [75, 3]})
    expected = expectations.expected_rush_yards(tables, probe)
    assert expected.iloc[0] == pytest.approx(4.0)
    assert expected.iloc[1] == pytest.approx(3.0)  # unseen bucket: global mean of 2, 6, 1
    with pytest.raises(ValueError):
        expectations.fit_expectation_tables({2022: current}, before_season=2022)


def test_buckets_field_goal_labels_and_conversion_table():
    edges = pd.DataFrame({
        "down": [1, 2, 3, 4, 1, 1, 1, 1, 1, 1, 1],
        "distance": [0, 2, 3, 6, 7, 10, 11, 5, 5, 5, 5],
        "yards_to_goal": [80, 79, 40, 39, 20, 19, 99, 1, 50, 60, 70],
    })
    got = expectations.bucket(edges)
    assert got["distance_bucket"].tolist()[:7] == ["1-2", "1-2", "3-6", "3-6", "7-10", "7-10", "11+"]
    assert got["field_bucket"].tolist()[:7] == ["own", "mid", "mid", "plus", "plus", "red", "own"]
    assert got["down_bucket"].tolist()[:4] == [1, 2, 3, 4]
    assert "fg_bucket" not in got  # no play_text, no field goal bucket

    kicks = [
        ("Field Goal Good", "A B 49 yd FG GOOD"),
        ("Field Goal Missed", "A B 46 yard field goal MISSED"),
        ("Blocked Field Goal", "A B 47 Yd Field Goal BLOCKED"),
        ("Field Goal Good", "A B 50 yd FG GOOD"),
        ("Blocked Field Goal Touchdown", "C D 70 Yd Return of Blocked Field Goal (E F Kick)"),
        ("Field Goal Good", "no distance in this text"),
    ]
    # first-and-10 in the middle: two successes in three competitive plays, then one failed second down;
    # the last play is a success that is not competitive and must stay out
    scrimmage = [True, True, False, False, True]
    prior = pd.DataFrame({
        "season": 2021,
        "down": [4] * len(kicks) + [1, 1, 1, 2, 1],
        "distance": [5] * len(kicks) + [10] * 5,
        "yards_to_goal": [30] * len(kicks) + [60] * 5,
        "is_rush": False,
        "is_competitive": [False] * len(kicks) + [True, True, True, True, False],
        "yards_gained": 0.0,
        "is_success": [False] * len(kicks) + scrimmage,
        "play_type": [kind for kind, _ in kicks] + ["Pass Reception"] * 5,
        "play_text": [text for _, text in kicks] + [""] * 5,
    })
    tables = expectations.fit_expectation_tables({2021: prior}, before_season=2022)

    fg = tables.field_goal.set_index("fg_bucket")
    assert fg["expected"].to_dict() == {45: pytest.approx(1 / 3), 50: 1.0}
    assert fg["plays"].to_dict() == {45: 3, 50: 1}  # the return TD and the unparsed kick are out

    probe = pd.DataFrame({"play_text": ["Z 49 yd FG", "Z 52 yard field goal", "Z 61 yd FG", "no distance"]}, index=[10, 11, 12, 13])
    expected = expectations.expected_field_goal(tables, probe)  # bucket() is called here: no bucket columns on probe
    assert expected.index.tolist() == [10, 11, 12, 13]
    assert expected.iloc[0] == pytest.approx(1 / 3)
    assert expected.iloc[1] == pytest.approx(1.0)  # 52 falls in the 50 bin
    assert expected.iloc[2] == pytest.approx(0.5)  # unseen 60 bin: mean of the four fitted kicks
    assert expected.iloc[3] == pytest.approx(0.5)  # unparsed distance

    conversion = tables.conversion
    assert conversion["plays"].sum() == 4  # the non-competitive play and the kicks are excluded
    situation = pd.DataFrame({"down": [1, 3], "distance": [10, 10], "yards_to_goal": [60, 60]}, index=[5, 6])
    got = expectations.expected_conversion(tables, situation)
    assert got.loc[5] == pytest.approx(2 / 3)
    assert got.loc[6] == pytest.approx(0.5)  # unseen third down: global mean of the four plays
