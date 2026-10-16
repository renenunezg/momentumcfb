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
    defense,
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
    for (athlete_id, team, game_id), line in lines.items():
        for stat_name in DEFENSIVE_STATS:
            box.append({"game_id": game_id, "team": team, "athlete_id": athlete_id, "category": "defensive",
                        "stat_name": stat_name, "stat": float(line.get(stat_name, 0))})
        if "INT" in line:
            box.append({"game_id": game_id, "team": team, "athlete_id": athlete_id, "category": "interceptions",
                        "stat_name": "INT", "stat": float(line["INT"])})
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
