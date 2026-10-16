import dataclasses
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.draft import research
from backend.players import ingest, pipeline
from backend.players.traits import (
    TRAITS,
    TraitSpec,
    absence,
    adjust,
    backtest,
    context,
    defense,
    expectations,
    headline,
    line,
    offense,
    register,
    snapshots,
    targets,
    teammate,
    transfers,
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


def test_box_positions_group_box_only_players_from_the_roster(caplog):
    positions = pd.DataFrame({"athlete_id": ["lb1"], "position": ["LB"], "position_group": ["LB"]})
    box = pd.DataFrame({"athlete_id": ["lb1", "cb1", "ghost"], "team": "Home", "game_id": 1})
    roster = pd.DataFrame({"id": ["lb1", "cb1"], "position": ["LB", "CB"]})
    with caplog.at_level("INFO", logger="backend.players.traits.context"):
        out = context.box_positions(positions, box, roster).set_index("athlete_id")
    assert out.loc["cb1", ["position", "position_group"]].tolist() == ["CB", "DB"]  # tackles only, no credit
    assert "ghost" not in out.index  # neither rostered nor credited
    assert len(out) == 2
    assert "1 box-score players have no roster row and no credit; skipped" in caplog.text


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


# Home hosts Away in week 1 and visits in week 2.
GAMES = pd.DataFrame({"game_id": [1, 2], "model_week": [1, 2], "home_team": ["Home", "Away"],
                      "away_team": ["Away", "Home"], "completed": True})


def _flat_tables(rush_yards=4.0):
    """No bucket matches a probe play, so every lookup falls back to the table mean:
    0.4 conversions and ``rush_yards`` yards per carry."""
    def flat(expected):
        return pd.DataFrame({"down_bucket": pd.array([9], dtype="Int64"), "distance_bucket": ["none"],
                             "field_bucket": ["none"], "expected": [expected], "plays": [1]})
    return expectations.ExpectationTables(rush_yards=flat(rush_yards), conversion=flat(0.4),
                                          field_goal=pd.DataFrame(), source_seasons=[2025])


def _season_context(plays, credit, positions):
    empty = pd.DataFrame()
    return context.SeasonContext(season=2026, plays=pd.DataFrame(plays), games=GAMES, effects=empty,
                                 credit=pd.DataFrame(credit), box=empty, roster=empty, teams=empty,
                                 positions=positions, presence=empty)


def qb_context():
    """Home hosts Away in week 1 and visits in week 2. Each starter has twelve
    dropbacks and two third-down scrambles per game, all at his team's receiver
    except the sack and the interception; Home's backup throws two completions
    to the same receiver in week 2."""
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
    return _season_context(plays, credit, positions), _flat_tables()


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


def _add_play(plays, credit, game_id, team, play_type, credited, *, yards=0.0, down=1, distance=10,
              yards_to_goal=60, epa=0.0, wpa=0.0):
    """Append one competitive play and a credit row for each (athlete, role, share); return its id."""
    play_id = len(plays) + 1
    opponent = "Away" if team == "Home" else "Home"
    success = yards >= (distance if down >= 3 else 0.5 * distance)
    plays.append({"id": play_id, "game_id": game_id, "offense": team, "defense": opponent, "down": down,
                  "distance": distance, "yards_to_goal": yards_to_goal, "yards_gained": yards, "period": 1,
                  "clock_seconds_remaining": 600, "is_competitive": True, "is_rush": play_type == "Rush",
                  "is_pass": play_type.startswith("Pass"), "is_success": bool(success), "play_type": play_type,
                  "play_text": "", "epa": epa, "wpa": wpa})
    for athlete_id, role, share in credited:
        credit.append({"play_id": str(play_id), "team": team, "opponent": opponent, "side": "offense",
                       "game_id": game_id, "model_week": game_id, "weight": 1.0, "plays": 1.0,
                       "is_competitive": True, "athlete_id": athlete_id, "role": role, "share": share,
                       "channel": "rush" if play_type == "Rush" else "pass", "raw_epa": share * epa,
                       "adjusted_epa": share * epa, "wpa": share * wpa})
    return str(play_id)


def rb_context():
    """Home's two backs share week 1: rb1 runs six times for +0.5 credited EPA a carry (one
    stuffed, one 12-yard breakaway, one third-and-1 conversion), rb2 six times for -0.5 (three
    stuffed). In week 2 rb1 is Home's only back: four carries, a fumble, a catch, and an
    incompletion with no target row. Expected rushing yards are 4 a carry everywhere."""
    plays, credit = [], []
    rb1 = [("rb1", "rusher", 0.7)]
    for yards in (4.0, 4.0, 4.0, 12.0, 0.0):
        _add_play(plays, credit, 1, "Home", "Rush", rb1, yards=yards, epa=0.5 / 0.7)
    _add_play(plays, credit, 1, "Home", "Rush", rb1, yards=4.0, down=3, distance=1, epa=0.5 / 0.7)
    for yards in (0.0, 0.0, -1.0, 2.0, 2.0, 2.0):
        _add_play(plays, credit, 1, "Home", "Rush", [("rb2", "rusher", 0.7)], yards=yards, epa=-0.5 / 0.7)
    for _ in range(4):
        _add_play(plays, credit, 2, "Home", "Rush", rb1, yards=3.0, epa=0.1)
    _add_play(plays, credit, 2, "Home", "Fumble Recovery (Opponent)", [("rb1", "fumbler", 1.0)], epa=-2.0, wpa=-0.2)
    _add_play(plays, credit, 2, "Home", "Pass Reception", [("qb1", "passer", 0.6), ("rb1", "receiver", 0.4)],
              yards=6.0, epa=1.0)
    _add_play(plays, credit, 2, "Home", "Pass Incompletion", [("qb1", "passer", 0.7)], epa=-0.4)
    positions = pd.DataFrame({"athlete_id": ["rb1", "rb2", "qb1"], "position_group": ["RB", "RB", "QB"]})
    return _season_context(plays, credit, positions), _flat_tables()


def _pair(out, trait, athlete_id, game_id):
    return out.loc[(trait, athlete_id, game_id), ["numerator", "denominator"]].tolist()


def test_rb_line_independence_is_relative_to_backs_behind_the_same_line():
    ctx, tables = rb_context()  # two backs on Home in game 1: rb1 +0.5 EPA/carry, rb2 -0.5; one back in game 2
    out = offense.rb_traits(ctx, tables, NO_RECOVERED_TARGETS)
    li = out[out["trait"].eq("rb.line_independence")].set_index(["athlete_id", "game_id"])
    assert li.loc[("rb1", 1), "numerator"] / li.loc[("rb1", 1), "denominator"] == pytest.approx(1.0)
    assert np.isnan(li.loc[("rb1", 2), "numerator"])


def test_rb_traits_cover_every_registered_rb_trait_once_per_game():
    ctx, tables = rb_context()
    out = offense.rb_traits(ctx, tables, NO_RECOVERED_TARGETS)
    specs = {name: spec for name, spec in TRAITS.items() if "RB" in spec.position_groups}
    assert set(out["trait"]) == set(specs)
    assert out["kind"].eq(out["trait"].map(lambda name: specs[name].kind)).all()
    assert not out.duplicated(["athlete_id", "game_id", "trait"]).any()
    assert set(out["athlete_id"]) == {"rb1", "rb2"}


def test_rb_carry_traits_measure_against_expectation_and_teammates():
    ctx, tables = rb_context()
    out = offense.rb_traits(ctx, tables, NO_RECOVERED_TARGETS).set_index(["trait", "athlete_id", "game_id"])
    assert _pair(out, "rb.yards_over_expected", "rb1", 1) == pytest.approx([28.0 - 6 * 4.0, 6.0])
    assert _pair(out, "rb.breakaway", "rb1", 1) == pytest.approx([1.0, 6.0])
    assert _pair(out, "rb.short_yardage", "rb1", 1) == pytest.approx([1.0 - 0.4, 1.0])
    # rb1 was stuffed once in six carries against rb2's three in six: 3 expected, 1 allowed
    assert _pair(out, "rb.stuff_avoidance", "rb1", 1) == pytest.approx([2.0, 6.0])
    assert _pair(out, "rb.stuff_avoidance", "rb2", 1) == pytest.approx([-2.0, 6.0])


def test_rb_receiving_and_ball_security_count_recovered_targets():
    ctx, tables = rb_context()
    incompletion = ctx.plays.loc[ctx.plays["play_type"].eq("Pass Incompletion"), "id"].astype(str)
    recovered = pd.DataFrame({"play_id": incompletion, "athlete_id": "rb1", "source": "text"})
    out = offense.rb_traits(ctx, tables, recovered).set_index(["trait", "athlete_id", "game_id"])
    # the catch keeps its 0.4 credit row; the recovered target takes the 0.3 target share of -0.4
    assert _pair(out, "rb.receiving", "rb1", 2) == pytest.approx([0.4 - 0.12, 2.0])
    # four carries, two targets, and the fumbled play are touches
    assert _pair(out, "rb.ball_security", "rb1", 2) == pytest.approx([-0.2, 7.0])
    assert _pair(out, "rb.ball_security", "rb1", 1) == pytest.approx([0.0, 6.0])

    poor = offense.rb_traits(ctx, tables, recovered, poor_coverage=True)
    poor = poor.set_index(["trait", "athlete_id", "game_id"])
    assert _pair(poor, "rb.receiving", "rb1", 2) == pytest.approx([0.4, 1.0])
    assert _pair(poor, "rb.ball_security", "rb1", 2) == pytest.approx([-0.2, 6.0])


def receiver_context():
    """Week 1: wr1 has two feed targets (a 25-yard third-down catch and a first-down
    incompletion), wr2 one third-down incompletion that only the play text names, and te1 only
    a carry. Week 2: qb1 completes five passes to wr1 and throws five incompletions at te1."""
    plays, credit = [], []
    _add_play(plays, credit, 1, "Home", "Pass Reception", [("qb1", "passer", 0.6), ("wr1", "receiver", 0.4)],
              yards=25.0, down=3, distance=5, epa=2.0)
    feed = _add_play(plays, credit, 1, "Home", "Pass Incompletion",
                     [("qb1", "passer", 0.7), ("wr1", "target", 0.3)], epa=-0.4)
    text = _add_play(plays, credit, 1, "Home", "Pass Incompletion", [("qb1", "passer", 0.7)],
                     down=3, distance=5, epa=-0.4)
    _add_play(plays, credit, 1, "Home", "Rush", [("te1", "rusher", 0.7)], yards=5.0, epa=0.2)
    for _ in range(5):
        _add_play(plays, credit, 2, "Home", "Pass Reception", [("qb1", "passer", 0.6), ("wr1", "receiver", 0.4)],
                  yards=9.0, epa=1.0)
        _add_play(plays, credit, 2, "Home", "Pass Incompletion", [("qb1", "passer", 0.7), ("te1", "target", 0.3)],
                  epa=-0.4)
    positions = pd.DataFrame({"athlete_id": ["wr1", "wr2", "te1", "qb1"], "position_group": ["WR", "WR", "TE", "QB"]})
    # recover_targets returns the feed's Target rows as well as the text matches
    recovered = pd.DataFrame([{"play_id": feed, "athlete_id": "wr1", "source": "feed"},
                              {"play_id": text, "athlete_id": "wr2", "source": "text"}])
    return _season_context(plays, credit, positions), _flat_tables(), recovered


def test_receiver_target_share_uses_recovered_targets():
    ctx, tables, recovered = receiver_context()  # wr1: 2 feed targets; wr2: 1 recovered target; team total 3
    out = offense.receiver_traits(ctx, tables, recovered)
    share = out[out["trait"].eq("rec.target_share")].groupby("athlete_id")[["numerator", "denominator"]].sum()
    assert share.loc["wr2"].tolist() == [1.0, 3.0]


def test_receiver_traits_cover_wr_and_te_once_per_game():
    ctx, tables, recovered = receiver_context()
    out = offense.receiver_traits(ctx, tables, recovered)
    specs = {name: spec for name, spec in TRAITS.items() if "WR" in spec.position_groups}
    assert set(out["trait"]) == set(specs)
    assert all(spec.position_groups == ("WR", "TE") for spec in specs.values())
    assert out["kind"].eq(out["trait"].map(lambda name: specs[name].kind)).all()
    assert not out.duplicated(["athlete_id", "game_id", "trait"]).any()
    assert set(out["athlete_id"]) == {"wr1", "wr2", "te1"}
    # te1 appeared on a carry in week 1 and drew none of the three targets
    out = out.set_index(["trait", "athlete_id", "game_id"])
    assert _pair(out, "rec.target_share", "te1", 1) == pytest.approx([0.0, 3.0])


def test_receiver_rates_credit_the_recovered_target_at_the_target_share():
    ctx, tables, recovered = receiver_context()
    out = offense.receiver_traits(ctx, tables, recovered).set_index(["trait", "athlete_id", "game_id"])
    assert _pair(out, "rec.value_per_target", "wr2", 1) == pytest.approx([0.3 * -0.4, 1.0])
    assert _pair(out, "rec.value_per_target", "wr1", 1) == pytest.approx([0.4 * 2.0 + 0.3 * -0.4, 2.0])
    assert _pair(out, "rec.explosive", "wr1", 1) == pytest.approx([1.0, 2.0])
    assert _pair(out, "rec.late_down", "wr1", 1) == pytest.approx([1.0 - 0.4, 1.0])
    assert _pair(out, "rec.late_down", "wr2", 1) == pytest.approx([0.0 - 0.4, 1.0])


def test_receiver_qb_independence_is_relative_to_the_passers_other_targets():
    ctx, tables, recovered = receiver_context()
    out = offense.receiver_traits(ctx, tables, recovered).set_index(["trait", "athlete_id", "game_id"])
    # week 2: wr1's 0.4 a catch against te1's -0.12 a target from the same passer
    assert _pair(out, "rec.qb_independence", "wr1", 2) == pytest.approx([5 * (0.4 + 0.12), 5.0])
    assert np.isnan(out.loc[("rec.qb_independence", "wr1", 1), "numerator"])  # one other target


def test_receiver_traits_count_receptions_under_poor_coverage():
    ctx, tables, recovered = receiver_context()
    out = offense.receiver_traits(ctx, tables, recovered, poor_coverage=True)
    assert "wr2" not in set(out["athlete_id"])
    out = out.set_index(["trait", "athlete_id", "game_id"])
    assert _pair(out, "rec.target_share", "wr1", 1) == pytest.approx([1.0, 1.0])
    assert _pair(out, "rec.target_share", "te1", 1) == pytest.approx([0.0, 1.0])
    assert _pair(out, "rec.value_per_target", "wr1", 1) == pytest.approx([0.8, 1.0])
    assert _pair(out, "rec.explosive", "wr1", 1) == pytest.approx([1.0, 1.0])


# Home hosts Away in odd weeks and visits in even weeks.
DEFENSE_GAMES = pd.DataFrame({"game_id": [1, 2, 3, 4], "model_week": [1, 2, 3, 4],
                              "home_team": ["Home", "Away", "Home", "Away"],
                              "away_team": ["Away", "Home", "Away", "Home"], "completed": True,
                              "home_classification": "fbs", "away_classification": "fbs"})
DEFENSIVE_STATS = ("TOT", "SOLO", "TFL", "SACKS", "QB HUR", "PD")


def defense_context():
    """Home's de1 plays weeks 1-3 (week 3 with every stat zero) and misses week 4; lb1 and db1
    play every week; Away's ad1 plays weeks 1-2. Away's offense runs one rush a game for -1 EPA
    in weeks 1-3 and +2 in week 4. Sacks of Away's passer: de1 alone in week 1 (WPA -0.04),
    lb1 alone in week 1 (-0.10), and de1 and lb1 split in week 2 (-0.06). de1 forces a fumble
    in week 1."""
    lines = {
        ("de1", "Home", 1): {"TOT": 4, "SOLO": 3, "TFL": 2, "SACKS": 1, "QB HUR": 2},
        ("de1", "Home", 2): {"TOT": 3, "SOLO": 2, "TFL": 1, "SACKS": 0.5, "QB HUR": 1, "PD": 1},
        ("de1", "Home", 3): {},
        ("lb1", "Home", 1): {"TOT": 6, "SOLO": 4, "TFL": 1, "SACKS": 1, "PD": 1},
        ("lb1", "Home", 2): {"TOT": 5, "SOLO": 3, "TFL": 0.5, "SACKS": 0.5},
        ("lb1", "Home", 3): {"TOT": 4, "SOLO": 2},
        ("lb1", "Home", 4): {"TOT": 5, "SOLO": 3, "INT": 1},
        ("db1", "Home", 1): {"TOT": 3, "SOLO": 3, "PD": 1},
        ("db1", "Home", 2): {"TOT": 2, "SOLO": 2, "INT": 1},
        ("db1", "Home", 3): {"TOT": 4, "SOLO": 4},
        ("db1", "Home", 4): {"TOT": 2, "SOLO": 1, "PD": 2},
        ("ad1", "Away", 1): {"TOT": 2},
        ("ad1", "Away", 2): {"TOT": 3, "TFL": 1},
    }
    box = []
    for (athlete_id, team, game_id), stats in lines.items():
        for stat_name in DEFENSIVE_STATS:
            box.append({"game_id": game_id, "team": team, "athlete_id": athlete_id, "category": "defensive",
                        "stat_name": stat_name, "stat": float(stats.get(stat_name, 0))})
        if "INT" in stats:
            box.append({"game_id": game_id, "team": team, "athlete_id": athlete_id, "category": "interceptions",
                        "stat_name": "INT", "stat": float(stats["INT"])})
    box = pd.DataFrame(box).assign(athlete_name="", stat_text="")

    plays, credit = [], []

    def add(game_id, play_type, epa, wpa=0.0, credited=()):
        play_id = len(plays) + 1
        home = DEFENSE_GAMES.loc[game_id - 1, "home_team"]
        plays.append({"id": play_id, "game_id": game_id, "season": 2026, "week": game_id,
                      "season_type": "regular", "home": home, "away": "Home" if home == "Away" else "Away",
                      "offense": "Away", "defense": "Home", "offense_score": 0, "defense_score": 0,
                      "drive_id": f"{game_id}-1", "drive_number": 1, "play_number": play_id, "period": 1,
                      "clock": {"minutes": 10, "seconds": 0}, "clock_seconds_remaining": 600, "down": 2,
                      "distance": 8, "yards_to_goal": 60, "yards_gained": -6.0 if play_type == "Sack" else 3.0,
                      "scoring": False, "play_type": play_type, "play_text": "", "is_success": False,
                      "epa": epa, "wpa": wpa})
        for athlete_id, role, share in credited:
            credit.append({"play_id": str(play_id), "game_id": game_id, "model_week": game_id, "team": "Home",
                           "opponent": "Away", "side": "defense", "athlete_id": athlete_id, "role": role,
                           "share": share, "weight": 1.0, "is_competitive": True,
                           "wpa": -wpa * share})

    for game_id, epa in zip((1, 2, 3, 4), (-1.0, -1.0, -1.0, 2.0)):
        add(game_id, "Rush", epa)
    add(1, "Sack", -0.5, -0.04, [("de1", "sacker", 1.0)])
    add(1, "Sack", -0.5, -0.10, [("lb1", "sacker", 1.0)])
    add(2, "Sack", -0.5, -0.06, [("de1", "sacker", 0.5), ("lb1", "sacker", 0.5)])
    add(1, "Fumble Recovery (Opponent)", -2.0, -0.08, [("de1", "fumble_forcer", 1.0)])

    plays, credit = pd.DataFrame(plays), pd.DataFrame(credit)
    positions = pd.DataFrame({"athlete_id": ["de1", "lb1", "db1", "ad1"], "position_group": ["DL", "LB", "DB", "DL"]})
    empty = pd.DataFrame()
    return context.SeasonContext(season=2026, plays=plays, games=DEFENSE_GAMES, effects=empty, credit=credit,
                                 box=box, roster=empty, teams=empty, positions=positions,
                                 presence=context.player_presence(box, credit, DEFENSE_GAMES))


def test_defense_traits_treat_a_zero_stat_box_row_as_present_and_need_an_absence():
    ctx = defense_context()  # de1 present weeks 1-3 (week 3 all zeros), absent week 4; lb1 present every week
    out = defense.defense_traits(ctx)
    # one row per (player, week); the latest week carries the estimate through every game
    latest = out[out["trait"].eq("def.impact") & out["model_week"].eq(4)]
    impact = latest.set_index("athlete_id")["numerator"]
    assert np.isfinite(impact.loc["de1"])
    assert np.isnan(impact.loc["lb1"])


def test_defense_every_week_counts_games_with_any_disruption():
    ctx = defense_context()
    out = defense.defense_traits(ctx)
    ew = out[out["trait"].eq("def.every_week") & out["athlete_id"].eq("de1")].sort_values("game_id")
    assert ew["numerator"].tolist() == [1.0, 1.0, 0.0]


def test_defense_rates_follow_the_box_score_and_the_sack_win_probability():
    ctx = defense_context()
    out = defense.defense_traits(ctx)
    per_game = out[out["game_id"].notna()].set_index(["trait", "athlete_id", "game_id"])
    # week 1 has no prior sacks to price de1's two hurries; in week 2 the median sack is
    # worth 0.07 (0.04 and 0.10), on top of de1's half of the 0.06 sack
    assert np.isnan(per_game.loc[("def.pass_rush", "de1", 1), "numerator"])
    assert _pair(per_game, "def.pass_rush", "lb1", 1) == pytest.approx([0.10, 1.0])
    assert _pair(per_game, "def.pass_rush", "de1", 2) == pytest.approx([0.03 + 0.07, 1.0])
    assert _pair(per_game, "def.pass_rush", "de1", 3) == pytest.approx([0.0, 1.0])
    assert _pair(per_game, "def.finishing", "de1", 1) == pytest.approx([1.0, 3.0])
    assert _pair(per_game, "def.finishing", "de1", 2) == pytest.approx([0.5, 1.5])
    assert ("def.finishing", "de1", 3) not in per_game.index  # no sack chance
    # havoc in week 1 is raw: two TFLs and the forced fumble. Through week 1 Away allowed
    # 6 havoc plays to three Home defenders (2 a game) against a league 6 in 4 (1.5), so
    # week 2's two plays (a TFL and a PD) lose 0.5. Through week 2 Away allowed 9.5 in 6
    # against a league 10.5 in 8.
    assert _pair(per_game, "def.havoc", "de1", 1) == pytest.approx([3.0, 1.0])
    assert _pair(per_game, "def.havoc", "de1", 2) == pytest.approx([1.5, 1.0])
    assert _pair(per_game, "def.havoc", "de1", 3) == pytest.approx([-(9.5 / 6 - 10.5 / 8), 1.0])

    groups = out["athlete_id"].map(ctx.positions.set_index("athlete_id")["position_group"])
    allowed = out["trait"].map(lambda name: TRAITS[name].position_groups)
    assert all(group in names for group, names in zip(groups, allowed))
    assert set(out.loc[out["athlete_id"].eq("db1"), "trait"]) == {
        name for name, spec in TRAITS.items() if "DB" in spec.position_groups
    }
    assert not out.duplicated(["athlete_id", "game_id", "model_week", "trait"]).any()
    assert out["game_id"].dtype == "Int64"
    assert out.loc[out["trait"].eq("def.impact"), "game_id"].isna().all()


def test_line_traits_emit_one_unit_row_per_team_game():
    ctx, tables = rb_context()
    out = line.line_traits(ctx, tables)
    assert out["athlete_id"].str.startswith("unit:").all()
    assert (out["kind"] == "inferred").all()
    assert not out.duplicated(["athlete_id", "game_id", "trait"]).any()


def test_line_traits_count_the_whole_team_and_have_no_protection_baseline_yet():
    ctx, tables = rb_context()  # Home runs 12 carries in week 1 (4 stuffed), has 2 dropbacks in week 2
    out = line.line_traits(ctx, tables).set_index(["trait", "athlete_id", "game_id"])
    assert _pair(out, "ol.stuff_rate", "unit:Home", 1) == pytest.approx([-4.0, 12.0])
    assert _pair(out, "ol.short_yardage", "unit:Home", 1) == pytest.approx([1.0 - 0.4, 1.0])
    # 4 + 4 + 4 + 7 (the 12-yard run is capped) + 0 + 4, then 0 + 0 - 1.2 + 2 + 2 + 2
    assert _pair(out, "ol.line_yards", "unit:Home", 1) == pytest.approx([27.8, 12.0])
    # no dropbacks in week 1, so week 2 has no league sack rate: the numerator is NaN, the dropbacks stay
    protection = out.loc[("ol.protection", "unit:Home", 2)]
    assert np.isnan(protection["numerator"]) and protection["denominator"] == pytest.approx(2.0)
    assert ("ol.protection", "unit:Home", 1) not in out.index
    assert not any(name.startswith("unit:Away") for name in out.index.get_level_values("athlete_id"))


def test_line_protection_is_expected_sacks_minus_sacks_allowed():
    ctx, tables = qb_context()  # each offense: 12 dropbacks and 1 sack per game, in both weeks
    out = line.line_traits(ctx, tables).set_index(["trait", "athlete_id", "game_id"])
    # week 1 has no prior league rate; week 2's is 2 sacks in 24 dropbacks, so Home's 14 dropbacks
    # (the backup's two included) expect 14 / 12 sacks and allowed one. The opponent's week 1 had
    # no valid expectation, so there is nothing to adjust by.
    assert np.isnan(out.loc[("ol.protection", "unit:Home", 1), "numerator"])
    week_two = out.loc[("ol.protection", "unit:Home", 2)]
    assert week_two["denominator"] == pytest.approx(14.0)
    assert week_two["numerator"] == pytest.approx(14 / 12 - 1.0)


KICK_GAMES = pd.DataFrame({
    "game_id": [1, 2], "model_week": [1, 2], "home_team": ["Home", "Away"], "away_team": ["Away", "Home"],
    "completed": True, "home_classification": ["fbs", "fbs"], "away_classification": ["fcs", "fbs"],
})


def kicking_context():
    """Game 1 (Away is FCS, so Home's rows weigh 0.5): Home's k1 kicks field goals from 30 (good),
    45 (missed) and 25 (good); Away's k2 makes one from 40. Game 2: Home
    has two kickers with attempts, so its field goals are skipped; k2 makes one from 35. The table
    expects 0.9 from 25, 0.8 from 30, 0.7 from 35, 0.6 from 40, 0.5 from 45. Punters: p1 and p2
    kick in both games."""
    def play(game_id, offense, text, play_type):
        return {"id": len(plays) + 1, "game_id": game_id, "offense": offense,
                "defense": "Away" if offense == "Home" else "Home", "play_type": play_type,
                "play_text": text, "is_competitive": False}

    plays = []
    for args in (
        (1, "Home", "J Kicker 30 yd FG GOOD", "Field Goal Good"),
        (1, "Home", "J Kicker 45 yd FG MISSED", "Field Goal Missed"),
        (1, "Home", "J Kicker 25 yd FG GOOD", "Field Goal Good"),
        (1, "Away", "A Kicker 40 yd FG GOOD", "Field Goal Good"),
        (2, "Home", "J Kicker 30 yd FG GOOD", "Field Goal Good"),
        (2, "Home", "B Kicker 30 yd FG MISSED", "Field Goal Missed"),
        (2, "Away", "A Kicker 35 yd FG GOOD", "Field Goal Good"),
    ):
        plays.append(play(*args))

    box = []

    def add(game_id, team, athlete_id, category, stat_name, stat=np.nan, text=""):
        box.append({"game_id": game_id, "team": team, "athlete_id": athlete_id, "athlete_name": "",
                    "category": category, "stat_name": stat_name, "stat": stat, "stat_text": text})

    for game_id, team, athlete_id, made in ((1, "Home", "k1", "2/3"), (1, "Away", "k2", "1/1"),
                                             (2, "Home", "k1", "1/1"), (2, "Home", "k3", "0/1"),
                                             (2, "Away", "k2", "1/1")):
        add(game_id, team, athlete_id, "kicking", "FG", text=made)
        add(game_id, team, athlete_id, "kicking", "XP", text="1/1")
    for game_id, team, athlete_id, punts, yards in ((1, "Home", "p1", 4, 160), (1, "Away", "p2", 2, 70),
                                                    (2, "Home", "p1", 3, 120), (2, "Away", "p2", 3, 105)):
        add(game_id, team, athlete_id, "punting", "NO", punts)
        add(game_id, team, athlete_id, "punting", "YDS", yards)
        add(game_id, team, athlete_id, "punting", "AVG", yards / punts)
    positions = pd.DataFrame({"athlete_id": ["k1", "k2", "k3", "p1", "p2"],
                              "position_group": ["K", "K", "K", "P", "P"]})
    field_goal = pd.DataFrame({"fg_bucket": pd.array([25, 30, 35, 40, 45], dtype="Int64"),
                               "expected": [0.9, 0.8, 0.7, 0.6, 0.5], "plays": 10})
    tables = expectations.ExpectationTables(rush_yards=pd.DataFrame(), conversion=pd.DataFrame(),
                                            field_goal=field_goal, source_seasons=[2025])
    empty = pd.DataFrame()
    ctx = context.SeasonContext(season=2026, plays=pd.DataFrame(plays), games=KICK_GAMES, effects=empty,
                                credit=empty, box=pd.DataFrame(box), roster=empty, teams=empty,
                                positions=positions, presence=empty)
    return ctx, tables


def test_kicking_traits_score_field_goals_and_punts_against_expectation(caplog):
    ctx, tables = kicking_context()
    with caplog.at_level("INFO", logger=line.__name__):
        out = line.kicking_traits(ctx, tables)
    assert (out["kind"] == "counted").all()
    assert not out.duplicated(["athlete_id", "game_id", "trait"]).any()
    out = out.set_index(["trait", "athlete_id", "game_id"])
    # k1 in game 1: (1 - .8) + (0 - .5) + (1 - .9) = -0.2 over 3 kicks, all at the 0.5 FCS weight
    assert _pair(out, "k.field_goal", "k1", 1) == pytest.approx([-0.1, 1.5])
    assert _pair(out, "k.field_goal", "k2", 1) == pytest.approx([0.4, 1.0])
    assert _pair(out, "k.field_goal", "k2", 2) == pytest.approx([0.3, 1.0])
    # Home had two kickers with attempts in game 2, so neither is credited
    assert {athlete for trait, athlete, _ in out.index if trait == "k.field_goal"} == {"k1", "k2"}
    assert ("k.field_goal", "k1", 2) not in out.index and ("k.field_goal", "k3", 2) not in out.index
    assert "1 team-games skipped: ambiguous kicker" in caplog.text
    # no league average before week 2; then 230 yards on 6 punts through week 1
    assert not any(trait == "p.punt_average" and game == 1 for trait, _, game in out.index)
    assert _pair(out, "p.punt_average", "p1", 2) == pytest.approx([120 - 3 * 230 / 6, 3.0])
    assert _pair(out, "p.punt_average", "p2", 2) == pytest.approx([105 - 3 * 230 / 6, 3.0])


AS_OF = datetime(2024, 10, 1, tzinfo=timezone.utc)


def _trait_row(athlete_id, model_week, game_id, trait, numerator, denominator=10.0):
    return {"athlete_id": athlete_id, "model_week": model_week, "game_id": game_id, "trait": trait,
            "numerator": numerator, "denominator": denominator, "kind": TRAITS[trait].kind}


def _trait_frame(rows):
    return pd.DataFrame(rows).astype({"game_id": "Int64"})


def snapshot_context():
    """Weeks 1, 2 and 4 have completed games; week 3's game was not played and
    week 5 is still ahead. qb1 (A) throws 2/10 explosive in week 1, 4/10 in week 2
    and 1/10 in week 4; qb2 (B) 1/10 in weeks 1 and 2; qb3 (C) appears only in
    week 4; qb4 plays for FCS D. Nobody is on the roster but qb1."""
    games = pd.DataFrame({
        "game_id": [1, 2, 3, 4, 5, 6, 7],
        "model_week": [1, 1, 2, 2, 3, 4, 5],
        "completed": [True, True, True, True, False, True, False],
        "home_team": ["A", "C", "A", "C", "A", "A", "B"],
        "away_team": ["B", "D", "B", "D", "C", "C", "D"],
        "home_classification": ["fbs", "fbs", "fbs", "fbs", "fbs", "fbs", "fbs"],
        "away_classification": ["fbs", "fcs", "fbs", "fcs", "fbs", "fbs", "fcs"],
        "start_date": pd.date_range("2024-08-31", periods=7, freq="7D"),
    })
    frames = _trait_frame([
        _trait_row("qb1", 1, 1, "qb.explosive", 2.0),
        _trait_row("qb1", 2, 3, "qb.explosive", 4.0),
        _trait_row("qb1", 4, 6, "qb.explosive", 1.0),
        _trait_row("qb2", 1, 1, "qb.explosive", 1.0),
        _trait_row("qb2", 2, 3, "qb.explosive", 1.0),
        _trait_row("qb3", 4, 6, "qb.explosive", 5.0),
        _trait_row("qb4", 1, 2, "qb.explosive", 3.0),
        _trait_row("qb4", 2, 4, "qb.explosive", 3.0),
    ])
    positions = pd.DataFrame({"athlete_id": ["qb1"], "position": ["QB"], "position_group": ["QB"]})
    teams = pd.DataFrame({"school": ["A", "B", "C", "D"], "id": [11, 12, 13, 14],
                          "classification": ["fbs", "fbs", "fbs", "fcs"]})
    identity = pd.DataFrame({"athlete_id": ["qb1", "qb2", "qb3", "qb4"],
                             "athlete_name": ["One", "Two", "Three", "Four"], "team": ["A", "B", "C", "D"]})
    return frames, positions, games, teams, identity


def test_snapshots_are_cumulative_and_carry_a_dead_week_forward():
    frames, positions, games, teams, identity = snapshot_context()
    out = snapshots.weekly_trait_snapshots(frames, positions, games, teams, identity, 2024, AS_OF)
    assert list(out.columns) == snapshots.TRAIT_SNAPSHOT_COLUMNS
    assert sorted(out["week"].unique()) == [1, 2, 3, 4]  # week 5 has no completed game yet
    qb = out[out["athlete_id"].eq("qb1") & out["trait"].eq("qb.explosive")].set_index("week")
    assert qb.loc[1, "value"] == pytest.approx(0.2)
    assert qb.loc[2, "value"] == pytest.approx(0.3)
    assert qb.loc[3, "value"] == pytest.approx(0.3)
    assert qb.loc[4, "value"] == pytest.approx(7 / 30)
    assert qb["games"].tolist() == [1, 2, 2, 3]
    assert (qb["team_id"] == 11).all() and (qb["athlete_name"] == "One").all()
    assert (out["model_version"] == snapshots.MODEL_VERSION).all() and (out["as_of"] == AS_OF.isoformat()).all()


def test_snapshot_percentiles_only_rank_qualified_players_and_keep_unrostered_players():
    frames, positions, games, teams, identity = snapshot_context()
    out = snapshots.weekly_trait_snapshots(frames, positions.iloc[:0], games, teams, identity, 2024, AS_OF)
    assert set(out["position_group"]) == {"QB"}  # group from the trait registry
    week2 = out[out["week"].eq(2) & out["trait"].eq("qb.explosive")]
    assert week2.loc[~week2["qualified"], "percentile"].isna().all()
    # week 4 needs two games: qb3 has one and is not ranked
    week4 = out[out["week"].eq(4) & out["trait"].eq("qb.explosive")].set_index("athlete_id")
    assert not week4.loc["qb3", "qualified"] and np.isnan(week4.loc["qb3", "percentile"])
    assert week4.loc["qb1", "percentile"] == pytest.approx(100.0)
    assert week4.loc["qb2", "percentile"] == pytest.approx(0.0)


def test_split_half_reliability_reports_each_game_count():
    frames, positions, games, teams, identity = snapshot_context()
    out = snapshots.split_half_reliability(frames, positions, games_at=(2,))
    assert set(out.columns) == {"position_group", "trait", "games", "players", "correlation"}
    assert out.empty  # three players with two games is under the minimum

    # Twelve QBs with four games each. Games 1, 2 and 4 carry a rate that differs
    # by player only; game 3 is noise. At 2 total games each half is one game
    # (games 1 and 2) and the halves agree exactly; at 4 the noise game enters
    # the odd half; 6 games is more than anyone has.
    noise = [0.0, 2.0, 1.0, 4.0, 0.0, 3.0, 1.0, 1.0, 4.0, 2.0, 0.0, 3.0]
    rows = [
        _trait_row(f"q{i}", week, 100 * i + week, "qb.explosive", noise[i] if week == 3 else float(i % 3))
        for i in range(12) for week in range(1, 5)
    ]
    out = snapshots.split_half_reliability(_trait_frame(rows), positions.iloc[:0], games_at=(2, 4, 6))
    assert out[["position_group", "trait", "games", "players"]].values.tolist() == [
        ["QB", "qb.explosive", 2, 12],
        ["QB", "qb.explosive", 4, 12],
    ]
    odd = (np.arange(12) % 3 + np.array(noise)) / 20
    even = np.arange(12) % 3 / 10
    assert out["correlation"].tolist() == pytest.approx([1.0, np.corrcoef(odd, even)[0, 1]])
    assert out["correlation"].iloc[1] < 0.99


def test_snapshot_latest_and_quantile_aggregates_and_fbs_only_percentiles(monkeypatch):
    frames, positions, games, teams, identity = snapshot_context()
    extra = _trait_frame([
        # qb5 (A): floor games 0.4, -0.1, NaN, 0.2, 0.0
        _trait_row("qb5", 1, 1, "qb.floor", 0.4, 1.0),
        _trait_row("qb5", 1, 2, "qb.floor", -0.1, 1.0),
        _trait_row("qb5", 2, 3, "qb.floor", np.nan, 1.0),
        _trait_row("qb5", 2, 4, "qb.floor", 0.2, 1.0),
        _trait_row("qb5", 4, 6, "qb.floor", 0.0, 1.0),
        # dl1 (B): one game of havoc, weekly impact in weeks 1 and 2 only
        _trait_row("dl1", 1, 1, "def.havoc", 1.0),
        _trait_row("dl1", 1, pd.NA, "def.impact", 0.5, 1.0),
        _trait_row("dl1", 2, pd.NA, "def.impact", 0.8, 1.0),
        # the D (FCS) and A lines
        _trait_row("unit:D", 1, 2, "ol.stuff_rate", 1.0),
        _trait_row("unit:A", 1, 1, "ol.stuff_rate", 2.0),
    ])
    identity = pd.concat([identity, pd.DataFrame({"athlete_id": ["qb5", "dl1"], "athlete_name": ["Five", "Dee"],
                                                  "team": ["A", "B"]})], ignore_index=True)
    positions = pd.concat([positions, pd.DataFrame({"athlete_id": ["dl1"], "position": ["DE"],
                                                    "position_group": ["DL"]})], ignore_index=True)
    out = snapshots.weekly_trait_snapshots(pd.concat([frames, extra], ignore_index=True), positions, games,
                                           teams, identity, 2024, AS_OF)
    by = out.set_index(["athlete_id", "trait", "week"])
    # quantile20 needs four games with a value: three through week 2, four through week 4
    assert np.isnan(by.loc[("qb5", "qb.floor", 2), "value"])
    assert by.loc[("qb5", "qb.floor", 4), "value"] == pytest.approx(np.quantile([0.4, -0.1, 0.2, 0.0], 0.2))
    assert by.loc[("qb5", "qb.floor", 2), "games"] == 4  # the NaN game still counts as played
    # latest keeps the last weekly value and adds no games
    assert [by.loc[("dl1", "def.impact", week), "value"] for week in (1, 2, 3, 4)] == pytest.approx([0.5, 0.8, 0.8, 0.8])
    assert by.loc[("dl1", "def.impact", 4), "games"] == 1
    assert by.loc[("dl1", "def.impact", 1), "percentile"] == pytest.approx(100.0)  # alone in the group
    assert by.loc[("dl1", "def.impact", 1), "position_group"] == "DL"
    # FCS players and units are qualified but never ranked
    assert by.loc[("qb4", "qb.explosive", 2), "qualified"]
    assert np.isnan(by.loc[("qb4", "qb.explosive", 2), "percentile"])
    assert by.loc[("unit:D", "ol.stuff_rate", 1), "position_group"] == "OL"
    assert by.loc[("unit:D", "ol.stuff_rate", 1), "team"] == "D"
    assert np.isnan(by.loc[("unit:D", "ol.stuff_rate", 1), "percentile"])
    assert by.loc[("unit:A", "ol.stuff_rate", 1), "percentile"] == pytest.approx(100.0)
    week2 = out[out["week"].eq(2) & out["trait"].eq("qb.explosive")].set_index("athlete_id")["percentile"]
    assert week2.loc["qb1"] == pytest.approx(100.0) and week2.loc["qb2"] == pytest.approx(0.0)

    spec = TRAITS["qb.explosive"]
    for higher_is_better, expected in ((False, [0.0, 100.0]), (None, [np.nan, np.nan])):
        monkeypatch.setitem(TRAITS, "qb.explosive", dataclasses.replace(spec, higher_is_better=higher_is_better))
        out = snapshots.weekly_trait_snapshots(frames, positions, games, teams, identity, 2024, AS_OF)
        week2 = out[out["week"].eq(2) & out["trait"].eq("qb.explosive")].set_index("athlete_id")["percentile"]
        assert week2.loc[["qb1", "qb2"]].tolist() == pytest.approx(expected, nan_ok=True)


def test_match_transfers_requires_both_rosters_to_agree_on_one_player():
    portal = pd.DataFrame([{"season": 2022, "first_name": "Jon", "last_name": "Smith", "origin": "Troy", "destination": "Auburn"},
                           {"season": 2022, "first_name": "Al", "last_name": "Dupe", "origin": "Troy", "destination": "Auburn"}])
    roster_to = pd.DataFrame([{"id": "7", "first_name": "Jon", "last_name": "Smith", "team": "Auburn"},
                              {"id": "8", "first_name": "Al", "last_name": "Dupe", "team": "Auburn"},
                              {"id": "9", "first_name": "Al", "last_name": "Dupe", "team": "Auburn"}])
    roster_from = pd.DataFrame([{"id": "7", "first_name": "Jon", "last_name": "Smith", "team": "Troy"},
                                {"id": "8", "first_name": "Al", "last_name": "Dupe", "team": "Troy"}])
    out = transfers.match_transfers(portal, roster_to, roster_from)
    assert out["athlete_id"].tolist() == ["7"]


def test_match_transfers_counts_each_dropped_reason_and_handles_an_empty_portal():
    def person(first, last, team, id_):
        return {"id": id_, "first_name": first, "last_name": last, "team": team}

    def move(first, last, origin, destination):
        return {"season": 2023, "first_name": first, "last_name": last, "origin": origin, "destination": destination}

    portal = pd.DataFrame([
        move("Jon", "Smith Jr.", "Troy", "Auburn"),  # matches despite suffix and integer ids
        move("Ann", "Gone", "Troy", "Auburn"),       # absent from the destination roster
        move("Bo", "Twin", "Troy", "Auburn"),        # two destination players share the name
        move("Cy", "Lost", "Troy", "Auburn"),        # absent from the origin roster
        move("Di", "Clone", "Troy", "Auburn"),       # two origin players share the name
        move("Ed", "Swap", "Troy", "Auburn"),        # different ids on the two rosters
    ])
    roster_to = pd.DataFrame([
        person("Jon", "Smith", "Auburn", 7), person("Bo", "Twin", "Auburn", 11), person("Bo", "Twin", "Auburn", 12),
        person("Cy", "Lost", "Auburn", 13), person("Di", "Clone", "Auburn", 14), person("Ed", "Swap", "Auburn", 15),
    ])
    roster_from = pd.DataFrame([
        person("Jon", "Smith", "Troy", 7.0), person("Di", "Clone", "Troy", 14), person("Di", "Clone", "Troy", 99),
        person("Ed", "Swap", "Troy", 16),
    ])
    out = transfers.match_transfers(portal, roster_to, roster_from)
    assert out.to_dict("records") == [{"athlete_id": "7", "season": 2023, "origin": "Troy", "destination": "Auburn"}]
    assert out.attrs["dropped"] == {
        "no_destination_match": 1, "ambiguous_destination": 1,
        "no_origin_match": 1, "ambiguous_origin": 1, "id_mismatch": 1,
    }

    for empty in (pd.DataFrame({"season": [2020]}), portal.iloc[0:0]):
        out = transfers.match_transfers(empty, roster_to, roster_from)
        assert out.empty and list(out.columns) == ["athlete_id", "season", "origin", "destination"]
        assert sum(out.attrs["dropped"].values()) == 0


def _snapshot(season, rows, group="WR"):
    """Snapshot rows from (athlete, week, value[, qualified[, group]]) tuples."""
    return pd.DataFrame([
        {"season": season, "week": week, "athlete_id": athlete, "trait": "wr.test",
         "value": value, "qualified": rest[0] if rest else True,
         "position_group": rest[1] if len(rest) > 1 else group}
        for athlete, week, value, *rest in rows
    ])


def _portability_case():
    """Two season pairs; final-week values at week 15, week 3 holds poison.

    Pair 2021 -> 2022: stayers a0..a3 on y = x (x = 0..3); movers m0..m9 on y = 0.5x + 1 + e.
    Pair 2022 -> 2023: stayers b0..b3 on y = x + 2 (x = 4..7); movers n0..n9 on y = 0.5x + 3 - e.
    e = (1, -1, -1, 1, 1, -1, -1, 1, 1, -1) over x = 0..9. Player a3 is TE in 2021 and WR in 2022.
    """
    noise = [1, -1, -1, 1, 1, -1, -1, 1, 1, -1]
    y21, y22, y23 = [], [], []
    for x in range(4):
        y21 += [(f"a{x}", 15, float(x), True, "TE" if x == 3 else "WR"), (f"a{x}", 3, -99.0)]
        y22 += [(f"a{x}", 15, float(x)), (f"a{x}", 3, -99.0)]
        y22 += [(f"b{x}", 15, float(x + 4)), (f"b{x}", 3, -99.0)]
        y23 += [(f"b{x}", 15, float(x + 6)), (f"b{x}", 3, -99.0)]
    for x in range(10):
        y21 += [(f"m{x}", 15, float(x)), (f"m{x}", 3, -99.0)]
        y22 += [(f"m{x}", 15, 0.5 * x + 1 + noise[x]), (f"m{x}", 3, -99.0)]
        y22 += [(f"n{x}", 15, float(x)), (f"n{x}", 3, -99.0)]
        y23 += [(f"n{x}", 15, 0.5 * x + 3 - noise[x]), (f"n{x}", 3, -99.0)]
    moved = pd.DataFrame({"athlete_id": [f"m{x}" for x in range(10)] + [f"n{x}" for x in range(10)] + ["a0"],
                          "season": [2022] * 10 + [2023] * 10 + [2021], "origin": "X", "destination": "Y"})
    return {2021: _snapshot(2021, y21), 2022: _snapshot(2022, y22), 2023: _snapshot(2023, y23)}, moved


def test_portability_fits_an_intercept_pools_season_pairs_and_uses_the_later_group():
    snaps, moved = _portability_case()
    out = transfers.portability(snaps, moved)
    assert list(out.columns) == ["position_group", "trait", "stayers", "movers", "stay_slope", "move_slope", "move_slope_se", "portable_share"]
    assert out[["position_group", "trait", "stayers", "movers"]].values.tolist() == [["WR", "wr.test", 8, 20]]  # a3 counts as WR, not TE
    row = out.iloc[0]
    # Stayers pooled, x = 0..3 and 4..7: mean x 3.5, Sxx = 42, Sxy = 58, so slope 58/42
    # (pair 1 alone gives 1, so pooling and the intercept both matter; a through-origin fit differs).
    assert row["stay_slope"] == pytest.approx(58 / 42)
    # Movers pooled, x = 0..9 twice: Sxx = 2 * 82.5 = 165. The noise is mirrored between pairs and
    # sums to zero against centred x, so the slope is exactly 0.5. Residuals about the pooled fit
    # are e - 1 (pair 1) and -(e - 1) (pair 2): SSE = 2 * sum((e - 1)^2) = 2 * (10 + 10) = 40.
    assert row["move_slope"] == pytest.approx(0.5)
    assert row["move_slope_se"] == pytest.approx(np.sqrt(40 / 18 / 165))
    assert row["portable_share"] == pytest.approx(0.5 / (58 / 42))


def test_portability_needs_twenty_movers_and_a_positive_stayer_slope():
    snaps, moved = _portability_case()
    few = transfers.portability(snaps, moved[~moved["athlete_id"].isin(["m0", "n0"])])
    assert few.iloc[0]["movers"] == 18 and np.isnan(few.iloc[0]["portable_share"])
    assert not np.isnan(few.iloc[0]["move_slope"])  # the slope is still reported
    # negate the stayers' later values (and, in 2023, the movers'): the pooled stayer slope is -58/42
    snaps[2022] = snaps[2022].assign(value=np.where(snaps[2022]["athlete_id"].str.startswith("a"), -snaps[2022]["value"], snaps[2022]["value"]))
    snaps[2023] = snaps[2023].assign(value=-snaps[2023]["value"])
    falling = transfers.portability(snaps, moved).iloc[0]
    assert falling["stay_slope"] == pytest.approx(-58 / 42) and np.isnan(falling["portable_share"])


def test_portability_drops_nan_values_instead_of_poisoning_the_cell():
    snaps, moved = _portability_case()
    snaps[2022] = pd.concat([snaps[2022], _snapshot(2022, [("nan1", 15, np.nan), ("nan2", 15, 5.0)])])
    snaps[2021] = pd.concat([snaps[2021], _snapshot(2021, [("nan1", 15, 1.0), ("nan2", 15, np.nan)])])
    out = transfers.portability(snaps, moved)
    assert out.iloc[0]["stayers"] == 8
    assert out.iloc[0]["stay_slope"] == pytest.approx(58 / 42)
    assert out.iloc[0]["move_slope_se"] == pytest.approx(np.sqrt(40 / 18 / 165))


def test_portability_without_final_week_only_pairs_the_same_week_across_seasons():
    def two_weeks(season, offset):
        rows = []
        for x in range(4):
            rows += [(f"s{x}", 1, x + offset), (f"s{x}", 2, (x if season == 1 else 2 * x) + offset)]
        for x in range(10):
            rows += [(f"m{x}", 1, x + offset), (f"m{x}", 2, x + offset)]
        return rows

    before = _snapshot(2021, [(a, w, float(x)) for a, w, x in two_weeks(1, 0)])
    # stayers: week 1 y = x, week 2 y = 2x; movers: y = 0.5x + 1 in both weeks
    after_rows = []
    for x in range(4):
        after_rows += [(f"s{x}", 1, float(x)), (f"s{x}", 2, 2.0 * x)]
    for x in range(10):
        after_rows += [(f"m{x}", 1, 0.5 * x + 1), (f"m{x}", 2, 0.5 * x + 1)]
    after = _snapshot(2022, after_rows)
    moved = pd.DataFrame({"athlete_id": [f"m{x}" for x in range(10)], "season": 2022, "origin": "X", "destination": "Y"})

    every_week = transfers.portability({2021: before, 2022: after}, moved, final_week_only=False).iloc[0]
    # Same-week pairs only: 4 stayers x 2 weeks = 8 rows (a cross join would give 16), 20 mover rows.
    # Stayers pooled, x = 0..3 twice: Sxx = 10, Sxy = 5 (week 1) + 10 (week 2), so slope 1.5.
    assert (every_week["stayers"], every_week["movers"]) == (8, 20)
    assert every_week["stay_slope"] == pytest.approx(1.5)
    assert every_week["move_slope"] == pytest.approx(0.5)
    assert every_week["portable_share"] == pytest.approx(0.5 / 1.5)

    last_week = transfers.portability({2021: before, 2022: after}, moved).iloc[0]
    assert (last_week["stayers"], last_week["movers"]) == (4, 10)
    assert last_week["stay_slope"] == pytest.approx(2.0)


# Week 1 qualified QBs on qb.explosive: qa1 0.2 (A), qb1 0.1 (B), qc1 0.1 (C), qd1 0.0 (D);
# qa2 (A, 0.0) is unqualified, so it is standardized but sets no scale. Mean 0.1,
# sd (ddof 1) sqrt(0.02 / 3), so z(qa1) = sqrt(1.5) and z(qa2) = -sqrt(1.5). A's shares are
# 30 / 40 and 10 / 40: aggregate 0.75 sqrt(1.5) - 0.25 sqrt(1.5) = sqrt(1.5) / 2. C's lone QB
# has z 0, so the week-2 A-versus-C difference is sqrt(1.5) / 2 (0.6124). Week 2's own
# snapshot (qa1 at 0.3) would give a different number.
EXPECTED_WEEK1_Z_DIFFERENCE = np.sqrt(1.5) / 2


def matrix_context():
    """Weeks 1 to 3; A hosts B, then C, then D, and qa1's explosive rate rises each week."""
    games = pd.DataFrame({
        "game_id": [1, 2, 3, 4, 5, 6],
        "season": 2024,
        "model_week": [1, 1, 2, 2, 3, 3],
        "completed": True,
        "neutral_site": [False, False, False, True, False, False],
        "home_team": ["A", "C", "A", "B", "A", "B"],
        "away_team": ["B", "D", "C", "D", "D", "E"],
        "home_classification": "fbs",
        "away_classification": ["fbs", "fbs", "fbs", "fbs", "fbs", "fcs"],
        "home_points": [21, 10, 28, 17, 35, 40],
        "away_points": [14, 13, 20, 24, 7, 3],
    })
    rows = []
    for week, qa1 in ((1, 0.2), (2, 0.3), (3, 0.4)):
        for athlete, team, value, exposure, qualified in (
            ("qa1", "A", qa1, 30.0 * week, True), ("qa2", "A", 0.0, 10.0 * week, False),
            ("qb1", "B", 0.1, 40.0 * week, True), ("qc1", "C", 0.1, 40.0 * week, True),
            ("qd1", "D", 0.0, 40.0 * week, True),
        ):
            rows.append({"season": 2024, "week": week, "athlete_id": athlete, "team": team, "position_group": "QB",
                         "games": week, "trait": "qb.explosive", "value": value, "qualified": qualified,
                         "exposure": exposure})
    snaps = pd.DataFrame(rows)
    values = pd.DataFrame({"season": 2024, "week": 1, "athlete_id": ["qa1", "qb1"], "team": ["A", "B"],
                           "position_group": "QB", "games": 1, "shrunk_per_game": [1.0, -1.0]})
    return snaps, values, games


def synthetic_matrix(seed):
    """Four dev seasons of 300 games; margin = 3 QB:qb.explosive + 2 home_field + N(0, 1).

    RB:rb.breakaway and the other features have no effect, and one game in ten is neutral.
    """
    rng = np.random.default_rng(seed)
    null_features = ["RB:rb.breakaway", "WR:rec.explosive", "DL:def.havoc", "QB:epa"]
    parts = []
    for season in (2020, 2021, 2022, 2023):
        n = 300
        frame = pd.DataFrame({
            "game_id": season * 1000 + np.arange(n),
            "season": season,
            "model_week": rng.integers(2, 15, n),
            "home_field": (rng.random(n) > 0.1).astype(float),
            "QB:qb.explosive": rng.normal(size=n),
        })
        for name in null_features:
            frame[name] = rng.normal(size=n)
        frame["margin"] = 3 * frame["QB:qb.explosive"] + 2 * frame["home_field"] + rng.normal(size=n)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def test_design_matrix_uses_the_prior_week_snapshot_and_skips_week_one():
    snaps, values, games = matrix_context()  # 3 weeks, home QB explosive rises each week
    m = headline.design_matrix(snaps, values, games, 2024)
    assert m["model_week"].min() == 2
    row = m[m["game_id"].eq(games.loc[games["model_week"].eq(2), "game_id"].iloc[0])].iloc[0]
    assert row["QB:qb.explosive"] == pytest.approx(EXPECTED_WEEK1_Z_DIFFERENCE)
    assert (row["margin"], row["home_field"]) == (8, 1)
    # the FCS opponent's game is not a row; B-D was neutral; every registered trait is a column
    assert sorted(m["game_id"]) == [3, 4, 5]
    assert m.set_index("game_id").loc[4, "home_field"] == 0
    assert {"QB:qb.floor", "WR:rec.target_share", "TE:rec.target_share", "OL:ol.protection", "QB:epa"} <= set(m.columns)
    # v4 epa week 1: qa1 +1, qb1 -1 (mean 0, sd sqrt(2)); A's share of qa1 is 0.75, C has no v4 row
    assert row["QB:epa"] == pytest.approx(0.75 / np.sqrt(2))
    with pytest.raises(ValueError, match="exposure"):
        headline.design_matrix(snaps.drop(columns="exposure"), values, games, 2024)


def test_fit_headline_recovers_a_planted_weight_and_drops_noise():
    m = synthetic_matrix(seed=0)  # margin = 3*QB:qb.explosive + 2*home_field + noise, plus a pure-noise column RB:rb.breakaway
    fit = headline.fit_headline(m, dev_seasons=[2020, 2021, 2022, 2023])
    assert fit.weights["QB:qb.explosive"] == pytest.approx(3.0, abs=0.3)
    assert "RB:rb.breakaway" in fit.dropped
    assert fit.dropped_reason["RB:rb.breakaway"] == "insignificant"
    assert fit.penalty > 0


def test_headline_fit_refuses_to_score_its_own_seasons(tmp_path):
    fit = headline.fit_headline(synthetic_matrix(seed=1), dev_seasons=[2020, 2021, 2022, 2023])
    headline.save_headline_fit(fit, tmp_path / "fit.json")
    loaded = headline.load_headline_fit(tmp_path / "fit.json")
    headline.validate_headline_fit(loaded, 2025)
    pd.testing.assert_series_equal(loaded.analytic_se, fit.analytic_se)
    assert loaded.dropped_reason == fit.dropped_reason
    with pytest.raises(ValueError):
        headline.validate_headline_fit(loaded, 2023)


def test_score_players_uses_the_fit_scale_replacement_and_reliability():
    snapshot = pd.DataFrame({
        "week": 6, "athlete_id": ["q1", "q2", "q3", "q4", "q5"], "team": "A", "position_group": "QB",
        "games": [2, 4, 6, 12, 1], "trait": "qb.explosive", "value": [0.0, 0.1, 0.2, 0.3, 0.9],
        "qualified": [True, True, True, True, False],
    })
    legs = snapshot.assign(trait="qb.legs", value=[5.0, 0.0, 0.0, 0.0, 0.0])  # dropped: weight 0
    snapshot = pd.concat([snapshot, legs], ignore_index=True)
    values = pd.DataFrame({"week": 6, "athlete_id": ["q1", "q4"], "position_group": "QB", "games": 4,
                           "shrunk_per_game": [-1.0, 1.0]})
    features = ["QB:qb.explosive", "QB:epa"]
    fit = headline.HeadlineFit(
        weights=pd.Series({"QB:qb.explosive": 2.0, "QB:epa": 1.0, "home_field": 3.0, "intercept": 0.0}),
        penalty=1.0, loso_sd=pd.Series(0.1, index=features), jackknife_sd=pd.Series(0.15, index=features),
        analytic_se=pd.Series(0.12, index=features), dropped=["QB:qb.legs"],
        dropped_reason={"QB:qb.legs": "insignificant"}, dev_seasons=[2022, 2023],
        model_version=headline.MODEL_VERSION,
        feature_scale=pd.DataFrame({"feature": features, "mean": 0.0, "sd": [0.5, 1.0]}),
        reliability=pd.DataFrame({"position_group": "QB", "games": [4, 8], "correlation": [0.5, 0.7]}),
        replaces_value={"QB": True},
    )
    out = headline.score_players(snapshot, values, fit).set_index("athlete_id")
    # qualified mean 0.15, sd sqrt(0.05 / 3); q5 is standardized on that scale but sets none of it
    z = (np.array([0.0, 0.1, 0.2, 0.3, 0.9]) - 0.15) / np.sqrt(0.05 / 3)
    epa = np.array([-1, 0, 0, 1, 0]) / np.sqrt(2)
    raw = 2.0 * z / 0.5 + 1.0 * epa / 1.0
    assert out["raw_score"].to_numpy() == pytest.approx(raw)
    assert out["headline"].to_numpy() == pytest.approx(raw - np.quantile(raw[:4], 0.3))
    assert out["headline_percentile"].tolist()[:4] == pytest.approx([0.0, 100 / 3, 200 / 3, 100.0])
    assert np.isnan(out.loc["q5", "headline_percentile"])
    # r(2) = 0.25 from r(0) = 0, r(6) = 0.6 between 4 and 8, flat at 0.7 beyond 8
    r = np.array([0.25, 0.5, 0.6, 0.7, 0.125])
    assert out["uncertainty"].to_numpy() == pytest.approx(np.std(raw[:4], ddof=1) * np.sqrt(1 - r))
    assert out["replaces_value"].all()


def test_replacement_rule_needs_both_a_better_mae_and_equal_reliability():
    swap = pd.DataFrame([
        {"position_group": "QB", "mae_v4_only": 13.0, "mae_with_traits": 12.8, "reliability_6_v4": 0.5, "reliability_6_headline": 0.55},
        {"position_group": "RB", "mae_v4_only": 13.0, "mae_with_traits": 12.8, "reliability_6_v4": 0.5, "reliability_6_headline": 0.4},
        {"position_group": "DL", "mae_v4_only": 13.0, "mae_with_traits": 13.1, "reliability_6_v4": 0.5, "reliability_6_headline": 0.9},
    ])
    out = backtest.replacement_flags(swap).set_index("position_group")["replaces_value"]
    assert out.to_dict() == {"QB": True, "RB": False, "DL": False}


def exposure_frames():
    """wr1 is targeted in weeks 1 and 3 (week 2 is dead), wr2 only in week 1 with no
    numerator, ghost has no position group, and unit:A is A's offensive line."""
    return _trait_frame([
        _trait_row("wr1", 1, 1, "rec.value_per_target", 1.0, denominator=4.0),
        _trait_row("wr1", 1, 1, "rec.explosive", 1.0, denominator=99.0),
        _trait_row("wr1", 3, 3, "rec.value_per_target", 2.0, denominator=6.0),
        _trait_row("wr2", 1, 1, "rec.value_per_target", np.nan, denominator=2.0),
        _trait_row("ghost", 1, 1, "rec.value_per_target", 1.0, denominator=5.0),
        _trait_row("unit:A", 1, 1, "ol.stuff_rate", 3.0),
        _trait_row("unit:A", 3, 3, "ol.stuff_rate", 2.0),
    ])


def test_exposure_shares_carry_dead_weeks_count_units_and_skip_unmapped_players():
    groups = pd.Series({"wr1": "WR", "wr2": "WR", "unit:A": "OL"})
    out = headline.exposure_shares(exposure_frames(), groups)
    wide = out.pivot(index="athlete_id", columns="week", values="exposure")
    assert sorted(wide.index) == ["unit:A", "wr1", "wr2"]
    assert wide.loc["wr1"].tolist() == [4.0, 4.0, 10.0]
    assert wide.loc["wr2"].tolist() == [2.0, 2.0, 2.0]
    assert wide.loc["unit:A"].tolist() == [1.0, 1.0, 1.0]


def test_with_exposure_carries_the_last_week_and_zeroes_players_without_usage():
    groups = pd.Series({"wr1": "WR", "wr2": "WR", "wr3": "WR", "unit:A": "OL"})
    snaps = pd.DataFrame({
        "athlete_id": ["wr1"] * 4 + ["wr3", "unit:A"],
        "week": [1, 2, 3, 4, 2, 4],
        "trait": "rec.value_per_target",
    })
    out = backtest.with_exposure(snaps, exposure_frames(), groups)
    assert len(out) == len(snaps)
    assert out["exposure"].tolist() == [4.0, 4.0, 10.0, 10.0, 0.0, 1.0]


def test_season_identity_takes_each_players_latest_row_and_names_line_units():
    games = pd.DataFrame({"game_id": [1, 3], "model_week": [1, 3]})
    credit = pd.DataFrame({"athlete_id": [7], "athlete_name": ["Seven"], "team": ["A"], "model_week": [1]})
    box = pd.DataFrame({"game_id": [3, 1], "athlete_id": ["7", "8"], "athlete_name": ["Seven", "Eight"],
                        "team": ["B", "A"]})
    out = backtest.season_identity(credit, box, games, exposure_frames()).set_index("athlete_id")
    assert sorted(out.index) == ["7", "8", "unit:A"]
    assert out.loc["7", "team"] == "B"
    assert out.loc["8", ["athlete_name", "team"]].tolist() == ["Eight", "A"]
    assert out.loc["unit:A", ["athlete_name", "team"]].tolist() == ["A offensive line", "A"]


def test_swap_features_keep_the_full_set_for_one_group_and_epa_elsewhere():
    features = headline.feature_names()
    out = backtest.swap_features(features, "QB")
    qb = [name for name in features if name.startswith("QB:")]
    assert set(qb) <= set(out) and "QB:qb.explosive" in out and "QB:epa" in out
    assert sorted(set(out) - set(qb)) == sorted(f"{group}:epa" for group in headline.GROUPS if group != "QB")
    assert len(out) == len(set(out))


def _paired_fit():
    features = ["QB:qb.explosive", "QB:epa"]
    index = ["intercept", "home_field", *features]
    zeros = pd.Series(0.0, index=index)
    return headline.HeadlineFit(
        weights=pd.Series([0.0, 0.0, 1.0, 1.0], index=index), penalty=1.0, loso_sd=zeros,
        jackknife_sd=zeros, analytic_se=zeros, dropped=[], dropped_reason={}, dev_seasons=[2023, 2024],
        model_version=headline.MODEL_VERSION,
        feature_scale=pd.DataFrame({"feature": features, "mean": 0.0, "sd": 1.0}),
    )


def test_paired_split_half_scores_both_sides_on_one_athlete_set():
    """12 credited QBs with six games each, plus ghost: six frame games and no credit."""
    rng = np.random.default_rng(0)
    rows, credit = [], []
    for player in [f"q{i}" for i in range(12)] + ["ghost"]:
        for game in range(1, 7):
            game_id = 1000 * len(rows) + game
            rows.append(_trait_row(player, game, game_id, "qb.explosive", rng.uniform(0, 5)))
            if player != "ghost":
                credit.append({"athlete_id": player, "game_id": game_id, "model_week": game,
                               "adjusted_epa": rng.normal()})
    frames = _trait_frame(rows)
    snaps = pd.DataFrame({"athlete_id": frames["athlete_id"].unique(), "position_group": "QB"})
    data = {"frames": frames, "credited": pd.DataFrame(credit), "snapshots": snaps}

    chosen = backtest.paired_games(frames, data["credited"], 6)
    assert "ghost" not in set(chosen["athlete_id"]) and chosen["athlete_id"].nunique() == 12

    head, v4 = backtest.paired_split_half({2024: data}, _paired_fit(), (6,))
    assert head[["position_group", "games", "players"]].to_dict("records") == [
        {"position_group": "QB", "games": 6, "players": 12}
    ]
    assert v4[["position_group", "games", "players"]].equals(head[["position_group", "games", "players"]])
    assert head["correlation"].notna().all() and v4["correlation"].notna().all()
    assert head["correlation"].iloc[0] != pytest.approx(v4["correlation"].iloc[0])


def _frozen_fit(dev_seasons, replaces_qb=True):
    features = ["QB:qb.explosive", "QB:epa", "RB:rb.yards_over_expected"]
    return headline.HeadlineFit(
        weights=pd.Series({"intercept": 0.0, "home_field": 3.0, "QB:qb.explosive": 2.0, "QB:epa": 1.0,
                           "RB:rb.yards_over_expected": 0.5}),
        penalty=1.0, loso_sd=pd.Series(0.1, index=features), jackknife_sd=pd.Series(0.15, index=features),
        analytic_se=pd.Series(0.12, index=features), dropped=["QB:qb.legs"],
        dropped_reason={"QB:qb.legs": "insignificant"}, dev_seasons=dev_seasons,
        model_version=headline.MODEL_VERSION,
        feature_scale=pd.DataFrame({"feature": features, "mean": 0.0, "sd": [0.5, 1.0, 0.2]}),
        reliability=pd.DataFrame({"position_group": "QB", "games": [4, 8], "correlation": [0.5, 0.7]}),
        replaces_value={"QB": replaces_qb, "RB": False},
    )


FROZEN_FIT_2024 = _frozen_fit([2022, 2023, 2024])


def _fake_season_2026():
    rows = []
    for week in (1, 2):
        for athlete, group, trait, value, qualified in (
            ("q1", "QB", "qb.explosive", 0.1 * week, True), ("q2", "QB", "qb.explosive", 0.3, True),
            ("q1", "QB", "qb.legs", 4.0, True), ("q2", "QB", "qb.legs", 1.0, True),
            ("q3", "QB", "qb.explosive", 0.2, False),
            ("r1", "RB", "rb.yards_over_expected", 0.2, True), ("r2", "RB", "rb.yards_over_expected", 0.4, True),
        ):
            rows.append({"season": 2026, "week": week, "as_of": "old", "model_version": "old",
                         "athlete_id": athlete, "athlete_name": athlete.upper(), "team": "A", "team_id": 1,
                         "position_group": group, "games": week, "trait": trait, "value": value,
                         "percentile": 50.0, "kind": "raw", "qualified": qualified, "exposure": 10.0 * week})
    return {"snapshots": pd.DataFrame(rows)}


FAKE_SEASON_2026 = _fake_season_2026()
FAKE_V4_2026 = pd.DataFrame({
    "week": [1, 1, 2, 2], "athlete_id": ["q1", "q2", "q1", "q2"], "position_group": "QB", "games": [1, 1, 2, 2],
    "shrunk_per_game": [-1.0, 1.0, -0.5, 0.5], "adjusted_per_game": 0.0, "value_above_replacement": 0.0,
    "model_version": "cfb_player_value_v4",
})


def test_run_position_traits_scores_with_a_frozen_fit_and_never_refits(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.store, "PROCESSED_DIR", tmp_path)
    headline.save_headline_fit(FROZEN_FIT_2024, tmp_path / "players" / "models" / "position_traits_2024.json")
    monkeypatch.setattr(pipeline.backtest, "build_season_traits", lambda season: FAKE_SEASON_2026)
    monkeypatch.setattr(pipeline.headline, "fit_headline", lambda *a, **k: pytest.fail("refit"))
    monkeypatch.setattr(pipeline.store, "read_processed", lambda *parts, **k: FAKE_V4_2026)
    out = pipeline.run_position_traits(2026, AS_OF)
    assert (out["model_version"] == "cfb_position_traits_v1").all()
    assert set(out.loc[out["trait"].eq("headline"), "kind"]) == {"fitted"}
    assert out["uncertainty"].notna().where(out["trait"].eq("headline"), True).all()
    assert list(out.columns) == [
        *snapshots.TRAIT_SNAPSHOT_COLUMNS, "exposure", "uncertainty", "replaces_value", "in_headline"
    ]
    on_traits = out["trait"].ne("headline")
    assert out.loc[on_traits, "exposure"].tolist() == FAKE_SEASON_2026["snapshots"]["exposure"].tolist()
    assert out.loc[~on_traits, "exposure"].isna().all()
    assert (out["as_of"] == AS_OF.isoformat()).all()
    assert pd.read_parquet(tmp_path / "players" / "position_traits" / "2026.parquet").shape == out.shape

    headlines = out[out["trait"].eq("headline")]
    assert len(headlines) == 10  # three QBs and two RBs in each of two weeks
    q3 = headlines[headlines["athlete_id"].eq("q3")]
    assert not q3["qualified"].any() and q3["percentile"].isna().all()
    assert set(q3["team"]) == {"A"} and set(q3["athlete_name"]) == {"Q3"}
    assert (headlines.groupby("week").size() == 5).all()
    # the fitted feature is in the headline, the dropped one is not, headlines always are
    flags = out.groupby("trait")["in_headline"].all()
    assert flags["qb.explosive"] and flags["headline"] and not out.loc[out["trait"].eq("qb.legs"), "in_headline"].any()
    assert out.groupby("position_group")["replaces_value"].all().to_dict() == {"QB": True, "RB": False}
    q1 = headlines[headlines["athlete_id"].eq("q1") & headlines["week"].eq(2)].iloc[0]
    scored = headline.score_players(
        FAKE_SEASON_2026["snapshots"].query("week == 2"), FAKE_V4_2026.query("week == 2"), FROZEN_FIT_2024
    ).set_index("athlete_id")
    assert q1["value"] == pytest.approx(scored.loc["q1", "headline"])
    assert q1["games"] == 2 and q1["uncertainty"] == pytest.approx(scored.loc["q1", "uncertainty"])


def test_run_position_traits_uses_the_latest_fit_before_the_season(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.store, "PROCESSED_DIR", tmp_path)
    models = tmp_path / "players" / "models"
    headline.save_headline_fit(_frozen_fit([2022, 2023, 2024]), models / "position_traits_2024.json")
    headline.save_headline_fit(_frozen_fit([2022, 2023, 2024, 2025]), models / "position_traits_2025.json")
    headline.save_headline_fit(_frozen_fit([2022, 2023, 2026]), models / "position_traits_2026.json")
    assert pipeline.latest_headline_fit_path(2026).name == "position_traits_2025.json"
    with pytest.raises(FileNotFoundError, match="position-traits-fit"):
        pipeline.latest_headline_fit_path(2024)


def test_run_position_traits_refuses_a_fit_that_saw_the_season(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.store, "PROCESSED_DIR", tmp_path)
    headline.save_headline_fit(
        _frozen_fit([2022, 2026]), tmp_path / "players" / "models" / "position_traits_2025.json"
    )
    monkeypatch.setattr(pipeline.backtest, "build_season_traits", lambda season: pytest.fail("built"))
    with pytest.raises(ValueError, match="cannot score 2026"):
        pipeline.run_position_traits(2026, AS_OF)


def _two_week_traits() -> pd.DataFrame:
    rows = [
        {"season": 2026, "week": week, "as_of": "2026-10-01T00:00:00+00:00", "model_version": "cfb_position_traits_v1",
         "athlete_id": athlete, "athlete_name": athlete.upper(), "team": "A", "team_id": 1, "position_group": "QB",
         "games": week, "trait": trait, "value": 1.0, "percentile": 50.0, "kind": "fitted", "qualified": True,
         "exposure": 10.0, "uncertainty": np.nan, "replaces_value": True, "in_headline": True}
        for week in (7, 8) for athlete in ("q1", "q2") for trait in ("headline", "qb.explosive")
    ]
    return pd.DataFrame(rows)[pipeline.POSITION_TRAIT_COLUMNS]


def write_fake_player_artifacts(root: Path, season: int) -> None:
    """The v4 values, one Heisman board, the history and the meta that publishing reads."""
    values = FAKE_V4_2026.assign(overall_rank=[1, 2, 1, 2])
    pipeline.store.write_processed(values, *pipeline.player_values_artifact(season))
    pipeline.store.write_processed(
        pd.DataFrame({"season": [season], "week": [8], "athlete_id": ["q1"]}),
        *pipeline.heisman_board_artifact(season, 8),
    )
    for name in (pipeline.HISTORY_ARTIFACT, pipeline.META_ARTIFACT):
        pipeline.store.write_processed(pd.DataFrame({"season": [season]}), *name)


def test_player_artifacts_publish_only_the_latest_trait_week(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.store, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "PROCESSED_DIR", tmp_path)
    write_fake_player_artifacts(tmp_path, season=2026)
    traits = _two_week_traits()
    pipeline.store.write_processed(traits, *pipeline.position_traits_artifact(2026))
    out = pipeline.read_player_artifacts(2026)["position_traits"]
    assert out["week"].unique().tolist() == [traits["week"].max()]
    assert len(out) == len(traits) // 2
    assert list(out.columns) == pipeline.POSITION_TRAIT_COLUMNS


def test_player_artifacts_tolerate_a_missing_trait_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.store, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "PROCESSED_DIR", tmp_path)
    write_fake_player_artifacts(tmp_path, season=2026)
    artifacts = pipeline.read_player_artifacts(2026)
    assert artifacts["position_traits"].empty
    assert list(artifacts["position_traits"].columns) == pipeline.POSITION_TRAIT_COLUMNS
    assert set(artifacts) >= {"player_values", "heisman_board", "heisman_history", "player_model_meta"}


def test_migration_declares_the_trait_table_contract():
    sql = next(Path("sql").glob("*_position_traits.sql")).read_text()
    for column in ("replaces_value boolean", "in_headline boolean", "uncertainty double precision"):
        assert column in sql
    assert "primary key (season, athlete_id, trait)" in sql.lower()
    assert "site_revalidate" not in sql
    published = set(pipeline.POSITION_TRAIT_COLUMNS) - {"exposure", "qualified"}
    declared = {line.split()[0] for line in sql.splitlines() if line.startswith("    ")}
    assert published <= declared and "exposure" not in declared


def draft_fixture():
    """Two FBS players with v4 values at week 8, plus a third QB the roster does not list."""
    roster = pd.DataFrame({
        "id": [1, 2], "first_name": ["Q", "R"], "last_name": ["One", "One"], "team": ["A", "A"],
        "position": ["QB", "RB"], "year": [3, 3], "height": [75, 70], "weight": [210, 200],
    })
    catalog = pd.DataFrame({"school": ["A"], "classification": ["fbs"]})
    values = pd.DataFrame({
        "athlete_id": ["1", "2"], "team": ["A", "A"], "position": ["QB", "RB"], "position_group": ["QB", "RB"],
        "week": [8, 8], "as_of": ["2026-10-01T00:00:00+00:00"] * 2, "model_version": ["cfb_player_value_v4"] * 2,
        "games": [6, 6], "value_above_replacement": [5.0, 4.0], "position_rank": [2, 1],
    })
    return roster, catalog, values


def _headline_row(athlete_id, week, group, value, replaces, as_of="2026-10-01T00:00:00+00:00", trait="headline"):
    return {
        "athlete_id": athlete_id, "week": week, "as_of": as_of, "trait": trait, "value": value,
        "position_group": group, "replaces_value": replaces, "model_version": "cfb_position_traits_v1",
    }


DRAFT_AS_OF = pd.Timestamp("2026-10-02", tz="UTC")


def test_watchlist_uses_the_headline_where_it_replaces_v4():
    roster, catalog, values = draft_fixture()
    traits = pd.DataFrame([
        _headline_row("1", 8, "QB", 9.0, True),
        _headline_row("2", 8, "RB", 1.0, False),
        _headline_row("3", 8, "QB", 12.0, True),  # off the roster, still sets the group rank
        _headline_row("1", 8, "QB", -3.0, True, trait="qb.explosive"),
        _headline_row("1", 7, "QB", 99.0, True),  # an earlier week never wins
        _headline_row("1", 9, "QB", 50.0, True, as_of="2026-10-09T00:00:00+00:00"),  # after the cutoff
    ])
    board, audit = research.watchlist(roster, catalog, values, DRAFT_AS_OF, traits=traits)
    by_id = board.set_index("athlete_id")
    assert by_id.loc["1", "value_above_replacement"] == 9.0
    assert by_id.loc["1", "model_version"] == "cfb_position_traits_v1"
    assert by_id.loc["1", "position_rank"] == 2
    assert by_id.loc["2", "value_above_replacement"] == 4.0
    assert by_id.loc["2", "model_version"] == "cfb_player_value_v4" and by_id.loc["2", "position_rank"] == 1
    assert audit["headline_replaced_groups"] == ["QB"]

    plain, plain_audit = research.watchlist(roster, catalog, values, DRAFT_AS_OF)
    assert plain.set_index("athlete_id").loc["1", "value_above_replacement"] == 5.0
    assert plain_audit["headline_replaced_groups"] == []
