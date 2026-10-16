import numpy as np
import pandas as pd
import pytest

from backend.players import ingest
from backend.players.traits import (
    TRAITS,
    TraitSpec,
    absence,
    adjust,
    context,
    expectations,
    offense,
    register,
    targets,
    teammate,
)


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


def test_adjust_rate_subtracts_only_prior_week_opponent_deviation():
    games = pd.DataFrame({"game_id": [1, 2, 3], "model_week": [1, 2, 3]})
    events = pd.DataFrame([
        {"game_id": 1, "athlete_id": "a", "opponent": "D1", "numerator": 2.0, "denominator": 10.0},
        {"game_id": 1, "athlete_id": "b", "opponent": "D2", "numerator": 0.0, "denominator": 10.0},
        {"game_id": 2, "athlete_id": "c", "opponent": "D1", "numerator": 1.0, "denominator": 10.0},
    ])
    out = adjust.adjust_rate(events, games).set_index(["game_id", "athlete_id"])["numerator"]
    assert out.loc[(1, "a")] == 2.0  # week 1: no prior evidence, unchanged
    # week 2 vs D1: D1 allowed 0.2 through week 1, league 0.1, so subtract 10 * 0.1
    assert out.loc[(2, "c")] == pytest.approx(0.0)


def test_relative_rate_compares_to_teammates_in_the_same_games():
    events = pd.DataFrame([
        {"game_id": 1, "team": "T", "athlete_id": "rb1", "numerator": 4.0, "denominator": 10.0},
        {"game_id": 1, "team": "T", "athlete_id": "rb2", "numerator": 1.0, "denominator": 10.0},
        {"game_id": 2, "team": "T", "athlete_id": "rb1", "numerator": 3.0, "denominator": 10.0},
    ])
    out = teammate.relative_rate(events).set_index(["athlete_id", "game_id"])
    assert out.loc[("rb1", 1), "numerator"] == pytest.approx(3.0)   # (0.4 - 0.1) * 10
    assert np.isnan(out.loc[("rb1", 2), "numerator"])                # no teammate carries


def test_with_without_needs_an_absence_and_shrinks_by_games():
    presence = pd.DataFrame({"athlete_id": "de1", "team": "T", "game_id": [1, 2, 3, 4], "present": [True, True, True, False]})
    unit = pd.DataFrame({"team": "T", "game_id": [1, 2, 3, 4], "epa_allowed": [-1.0, -1.0, -1.0, 2.0]})
    out = absence.with_without(presence, unit, "epa_allowed", shrink_games=3.0).iloc[0]
    assert out["raw_difference"] == pytest.approx(3.0)
    assert out["impact"] == pytest.approx(3.0 * 3 / (3 + 3))
    never_absent = presence.assign(present=True)
    assert np.isnan(absence.with_without(never_absent, unit, "epa_allowed", 3.0).iloc[0]["impact"])


# Passer-row share of the play's EPA by outcome, as in credit.OFFENSE_SHARES.
PASSER_SHARE = {"Pass Reception": 0.6, "Pass Incompletion": 0.7, "Sack": 0.5, "Pass Interception Return": 0.8}
OUTCOME_EPA = {"Pass Reception": 0.5, "Pass Incompletion": -0.4, "Sack": -1.0, "Pass Interception Return": -3.0}


def _starter_dropbacks():
    """Twelve dropbacks: one 25-yard completion, five short ones, three incompletions,
    a sack, an interception, and a two-minute completion in the fourth quarter."""
    kinds = (["Pass Reception"] * 6 + ["Pass Incompletion"] * 3
             + ["Sack", "Pass Interception Return", "Pass Reception"])
    yards = [25, 8, 8, 8, 8, 8, 0, 0, 0, -7, 0, 5]
    out = []
    for i, (kind, gained) in enumerate(zip(kinds, yards)):
        last = i == len(kinds) - 1
        out.append({
            "play_type": kind, "yards_gained": float(gained), "down": i % 4 + 1, "distance": 10,
            "period": 4 if last else 1 + i // 6, "clock": 90 if last else 600,
            "epa": 2.0 if gained >= 20 else OUTCOME_EPA[kind],
            "wpa": -0.1 if kind == "Pass Interception Return" else 0.01,
        })
    return out


def qb_context():
    """Home hosts Away in week 1 and visits in week 2. Each starter has twelve
    dropbacks and two third-down scrambles per game, all at his team's receiver
    except the sack and the interception; Home's backup throws two completions
    to the same receiver in week 2."""
    games = pd.DataFrame({"game_id": [1, 2], "model_week": [1, 2], "home_team": ["Home", "Away"],
                          "away_team": ["Away", "Home"], "completed": True})
    starters = {"Home": ("qb_home", "wr_home"), "Away": ("qb_away", "wr_away")}
    plays, credit = [], []

    def add(game_id, team, passer, receiver, play):
        play_id = len(plays) + 1
        opponent = "Away" if team == "Home" else "Home"
        kind = play["play_type"]
        success = play["yards_gained"] >= (play["distance"] if play["down"] >= 3 else 0.5 * play["distance"])
        plays.append({"id": play_id, "game_id": game_id, "offense": team, "defense": opponent,
                      "down": play["down"], "distance": play["distance"], "yards_to_goal": 60,
                      "yards_gained": play["yards_gained"], "period": play["period"],
                      "clock_seconds_remaining": play["clock"], "is_competitive": True,
                      "is_rush": kind == "Rush", "is_pass": kind != "Rush", "is_success": bool(success),
                      "play_type": kind, "play_text": "", "epa": play["epa"], "wpa": play["wpa"]})
        base = {"play_id": str(play_id), "team": team, "opponent": opponent, "side": "offense",
                "game_id": game_id, "model_week": game_id, "weight": 1.0, "plays": 1.0, "is_competitive": True}
        if kind == "Rush":
            roles = [(passer, "rusher", 0.7)]
        else:
            roles = [(passer, "passer", PASSER_SHARE[kind])]
            if kind == "Pass Reception":
                roles.append((receiver, "receiver", 0.4))
            elif kind == "Pass Incompletion":
                roles.append((receiver, "target", 0.3))
        for athlete_id, role, share in roles:
            credit.append({**base, "athlete_id": athlete_id, "role": role, "share": share,
                           "channel": "rush" if kind == "Rush" else "pass",
                           "raw_epa": share * play["epa"], "adjusted_epa": share * play["epa"],
                           "wpa": share * play["wpa"]})

    for game_id in (1, 2):
        for team, (passer, receiver) in starters.items():
            for play in _starter_dropbacks():
                add(game_id, team, passer, receiver, play)
            for gained in (6.0, 12.0):
                add(game_id, team, passer, receiver, {"play_type": "Rush", "yards_gained": gained, "down": 3,
                                                      "distance": 5, "period": 3, "clock": 600, "epa": 0.3, "wpa": 0.01})
    for _ in range(2):
        add(2, "Home", "qb_backup", "wr_home", {"play_type": "Pass Reception", "yards_gained": 9.0, "down": 1,
                                               "distance": 10, "period": 3, "clock": 600, "epa": 1.0, "wpa": 0.02})

    positions = pd.DataFrame({
        "athlete_id": ["qb_home", "qb_away", "qb_backup", "wr_home", "wr_away"],
        "position": ["QB", "QB", "QB", "WR", "WR"],
        "position_group": ["QB", "QB", "QB", "WR", "WR"],
    })
    # No bucket matches the probe plays, so every lookup falls back to the table mean of 0.4.
    conversion = pd.DataFrame({"down_bucket": pd.array([9], dtype="Int64"), "distance_bucket": ["none"],
                               "field_bucket": ["none"], "expected": [0.4], "plays": [1]})
    tables = expectations.ExpectationTables(rush_yards=pd.DataFrame(), conversion=conversion,
                                            field_goal=pd.DataFrame(), source_seasons=[2025])
    empty = pd.DataFrame()
    ctx = context.SeasonContext(season=2026, plays=pd.DataFrame(plays), games=games, effects=empty,
                                credit=pd.DataFrame(credit), box=empty, roster=empty, teams=empty,
                                positions=positions, presence=empty)
    return ctx, tables


NO_RECOVERED_TARGETS = pd.DataFrame(columns=["play_id", "athlete_id", "source"])


def test_qb_traits_cover_every_registered_qb_trait_once_per_game():
    ctx, tables = qb_context()
    out = offense.qb_traits(ctx, tables, recovered_targets=NO_RECOVERED_TARGETS)
    expected = {name for name, spec in TRAITS.items() if "QB" in spec.position_groups}
    assert set(out["trait"]) == expected
    assert not out.duplicated(["athlete_id", "game_id", "trait"]).any()
    assert set(out["athlete_id"]) == {"qb_home", "qb_away", "qb_backup"}


def test_qb_pressure_handling_falls_back_to_the_league_rate_with_one_passer():
    ctx, tables = qb_context()
    out = offense.qb_traits(ctx, tables, NO_RECOVERED_TARGETS)
    away = out[out["trait"].eq("qb.pressure_handling") & out["athlete_id"].eq("qb_away")]
    assert not away.empty
    assert away["numerator"].notna().all()
    assert (away["kind"] == "inferred").all()


def test_qb_floor_is_nan_under_ten_dropbacks():
    ctx, tables = qb_context()
    out = offense.qb_traits(ctx, tables, NO_RECOVERED_TARGETS)
    backup = out[out["trait"].eq("qb.floor") & out["athlete_id"].eq("qb_backup")]
    assert not backup.empty
    assert backup["numerator"].isna().all()


def test_qb_ball_security_charges_the_interception_win_probability():
    ctx, tables = qb_context()
    out = offense.qb_traits(ctx, tables, NO_RECOVERED_TARGETS).set_index(["trait", "athlete_id", "game_id"])
    # week 1 has no prior opponent evidence, so the numerator is the raw offense-view WPA
    row = out.loc[("qb.ball_security", "qb_away", 1)]
    assert row["numerator"] == pytest.approx(-0.1)
    assert row["denominator"] == pytest.approx(12.0)


def test_qb_receiver_independence_compares_to_the_receivers_rate_with_other_passers():
    ctx, tables = qb_context()
    out = offense.qb_traits(ctx, tables, NO_RECOVERED_TARGETS).set_index(["trait", "athlete_id", "game_id"])
    # wr_home's ten week-1 targets from qb_home carry 2.16 passer EPA, 0.216 per target
    backup = out.loc[("qb.receiver_independence", "qb_backup", 2)]
    assert backup["numerator"] == pytest.approx(2 * 0.6 - 2 * 0.216)
    assert backup["denominator"] == pytest.approx(2.0)
    assert np.isnan(out.loc[("qb.receiver_independence", "qb_home", 1), "numerator"])

    # Under poor coverage only receptions count: seven from qb_home is below the ten needed.
    poor = offense.qb_traits(ctx, tables, NO_RECOVERED_TARGETS, poor_coverage=True)
    poor = poor.set_index(["trait", "athlete_id", "game_id"])
    assert np.isnan(poor.loc[("qb.receiver_independence", "qb_backup", 2), "numerator"])
