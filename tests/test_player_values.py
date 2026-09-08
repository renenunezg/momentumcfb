"""Player value keeps the invariants the credit design promises."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backend.players import value
from backend.players.credit import assign_play_credit


def _play(play_id, epa, offense="Home", defense="Away", competitive=True, rush=False):
    return {
        "id": play_id,
        "season": 2025,
        "week": 3,
        "season_type": "regular",
        "game_id": 1,
        "offense": offense,
        "defense": defense,
        "epa": epa,
        "is_competitive": competitive,
        "is_rush": rush,
        "is_pass": not rush,
    }


def _stat(play_id, athlete, team, stat_type):
    return {
        "play_id": play_id,
        "athlete_id": athlete,
        "athlete_name": athlete,
        "team": team,
        "stat_type": stat_type,
        "stat": 1,
    }


PLAYS = pd.DataFrame(
    [
        _play("p1", 1.2),  # completion: passer and receiver
        _play("p2", -0.6),  # incompletion with a tagged target
        _play("p3", -0.5),  # incompletion, no target row
        _play("p4", 0.8, rush=True),  # rush
        _play("p5", -2.1),  # sack taken, two sackers
        _play("p6", -4.0),  # interception thrown and returned
        _play("p7", -3.5),  # fumbled reception: fumbler only
        _play("p8", -0.4),  # pass breakup
        _play("p9", 2.5, competitive=False),  # garbage-time completion
    ]
)
STATS = pd.DataFrame(
    [
        _stat("p1", "qb", "Home", "Completion"),
        _stat("p1", "wr", "Home", "Reception"),
        _stat("p2", "qb", "Home", "Incompletion"),
        _stat("p2", "wr", "Home", "Target"),
        _stat("p3", "qb", "Home", "Incompletion"),
        _stat("p4", "rb", "Home", "Rush"),
        _stat("p5", "qb", "Home", "Sack Taken"),
        _stat("p5", "de1", "Away", "Sack"),
        _stat("p5", "de2", "Away", "Sack"),
        _stat("p6", "qb", "Home", "Interception Thrown"),
        _stat("p6", "cb", "Away", "Interception"),
        _stat("p7", "qb", "Home", "Completion"),
        _stat("p7", "wr", "Home", "Reception"),
        _stat("p7", "wr", "Home", "Fumble"),
        _stat("p7", "lb", "Away", "Fumble Recovered"),
        _stat("p8", "qb", "Home", "Incompletion"),
        _stat("p8", "cb", "Away", "Pass Breakup"),
        _stat("p9", "qb", "Home", "Completion"),
        _stat("p9", "wr", "Home", "Reception"),
    ]
)


def test_credit_and_unit_remainder_reconstruct_every_play():
    credit = assign_play_credit(STATS, PLAYS)
    paid = credit.groupby(["play_id", "side"])["credit_epa"].sum().unstack(fill_value=0)
    for play in PLAYS.itertuples(index=False):
        # Whatever the players do not carry is the unit's, so the paid
        # credit can never exceed the play on either side of the ball.
        offense_paid = paid["offense"].get(play.id, 0.0)
        defense_paid = paid["defense"].get(play.id, 0.0)
        assert abs(offense_paid) <= abs(play.epa) + 1e-12
        assert abs(defense_paid) <= abs(play.epa) + 1e-12
        assert offense_paid * play.epa >= 0
        assert defense_paid * play.epa <= 0
    shares = credit.groupby(["play_id", "side"])["share"].sum()
    assert shares.le(1.0 + 1e-12).all()
    # The fumbled reception charges the fumbler alone: no passer credit.
    p7 = credit[credit["play_id"].eq("p7")]
    assert set(p7["role"]) == {"fumbler", "fumble_recoverer"}
    assert np.isclose(p7.loc[p7["role"].eq("fumbler"), "credit_epa"].iloc[0], -3.5)
    # Two sackers split the whole defensive EPA.
    sackers = credit[credit["role"].eq("sacker")]
    assert np.allclose(sackers["credit_epa"], 1.05)


def test_garbage_time_counts_for_wpa_but_not_epa_value(monkeypatch, tmp_path):
    credit = assign_play_credit(STATS, PLAYS)
    credit["weight"] = 1.0
    credit["raw_epa"] = credit["credit_epa"]
    credit["adjusted_epa"] = credit["credit_epa"]
    wpa = pd.DataFrame({"play_id": PLAYS["id"], "wpa": 0.05})
    out = value.finalize_credit(credit, wpa)
    garbage = out[out["play_id"].eq("p9")]
    assert (garbage["plays"] == 0).all()
    assert (garbage["raw_epa"] == 0).all()
    assert (garbage["adjusted_epa"] == 0).all()
    assert (garbage["wpa"] != 0).all()
    live = out[out["play_id"].eq("p1")]
    assert (live["plays"] == 1).all()

    # A fresh weekly runner uses the frozen records grading fetched, without
    # needing past local forecast artifacts or accepting post-kickoff rows.
    monkeypatch.setattr(value.store, "PROCESSED_DIR", tmp_path)
    value.store.write_processed(
        pd.DataFrame(
            {
                "game_id": [1, 2],
                "home_margin": [3.0, 99.0],
                "margin_sd": 14.0,
                "as_of": ["2026-08-28T00:00:00Z", "2026-08-30T00:00:00Z"],
                "start_date": "2026-08-29T16:00:00Z",
            }
        ),
        "players",
        "published_projections",
        "2026.parquet",
    )
    anchors = value._projection_anchors(
        2026, pd.DataFrame({"game_id": [1, 2], "model_week": 0})
    )
    assert anchors["game_id"].tolist() == [1]
    assert anchors["home_margin"].tolist() == [3.0]


def _games(weeks_completed: set[int]) -> pd.DataFrame:
    teams = ["A", "B", "C", "D", "E", "F"]
    rows = []
    game_id = 100
    for week in range(1, 5):
        pairs = [(0, 1), (2, 3), (4, 5)] if week % 2 else [(0, 2), (1, 4), (3, 5)]
        for home, away in pairs:
            rows.append(
                {
                    "game_id": game_id,
                    "season": 2025,
                    "model_week": week,
                    "week": week,
                    "season_type": "regular",
                    "completed": week in weeks_completed,
                    "start_date": f"2025-09-{week * 6:02d}T00:00:00+00:00",
                    "home_team_id": home + 1,
                    "home_team": teams[home],
                    "home_classification": "fbs",
                    "away_team_id": away + 1,
                    "away_team": teams[away],
                    "away_classification": "fbs",
                }
            )
            game_id += 1
    return pd.DataFrame(rows)


def _unit_games(games: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for game in games.itertuples(index=False):
        for team, opponent in (
            (game.home_team, game.away_team),
            (game.away_team, game.home_team),
        ):
            rush_plays = int(rng.integers(25, 35))
            pass_plays = int(rng.integers(25, 35))
            rows.append(
                {
                    "game_id": game.game_id,
                    "team": team,
                    "opponent": opponent,
                    "rush_ppa": float(rng.normal(3.0, 4.0)),
                    "pass_ppa": float(rng.normal(5.0, 5.0)),
                    "protection_ppa_allowed": float(rng.normal(-3.0, 1.5)),
                    "adjusted_line_yards": float(rng.normal(90.0, 15.0)),
                    "rush_plays": rush_plays,
                    "pass_plays": pass_plays,
                    "sacks_allowed": int(rng.integers(0, 4)),
                    "line_yard_carries": rush_plays,
                }
            )
    return pd.DataFrame(rows)


def test_opponent_effect_for_a_week_never_sees_that_weeks_games():
    with_week_four = _games({1, 2, 3, 4})
    without_week_four = _games({1, 2, 3})
    units = _unit_games(with_week_four)
    effects_seen = value.season_unit_effects(with_week_four, units)
    effects_unseen = value.season_unit_effects(without_week_four, units)
    week_four_seen = (
        effects_seen[effects_seen["model_week"].eq(4)]
        .sort_values("team")
        .reset_index(drop=True)
    )
    week_four_unseen = (
        effects_unseen[effects_unseen["model_week"].eq(4)]
        .sort_values("team")
        .reset_index(drop=True)
    )
    assert not np.allclose(week_four_seen["pass_defense"], 0.0)
    pd.testing.assert_frame_equal(week_four_seen, week_four_unseen)


def test_shrinkage_and_replacement_baseline():
    per_game = np.arange(11, dtype=float) / 10.0
    games = 10
    frame = pd.DataFrame(
        {
            "athlete_id": [f"qb{index}" for index in range(11)] + ["rookie"],
            "athlete_name": "x",
            "team": "T",
            "side": "offense",
            "position": "QB",
            "position_group": "QB",
            "plays": [300.0] * 11 + [30.0],
            "adjusted_epa": list(per_game * games) + [2.0],
            "raw_epa": 0.0,
            "wpa": 0.0,
            "fcs_plays": 0.0,
            "games": [games] * 11 + [1],
        }
    )
    snapshot = value._rank_snapshot(frame, week=10).set_index("athlete_id")
    group_mean = np.average(
        frame["adjusted_epa"] / frame["games"], weights=frame["games"]
    )
    rookie = snapshot.loc["rookie"]
    assert group_mean < rookie["shrunk_per_game"] < 2.0
    # A one-game cameo does not qualify to set the replacement level.
    qualified = snapshot[snapshot["games"].ge(5)]
    assert "rookie" not in qualified.index
    expected = qualified["shrunk_per_game"].quantile(value.REPLACEMENT_PERCENTILE)
    assert np.isclose(snapshot["replacement_per_game"].iloc[0], expected)
    # Equal game counts make shrinkage affine, so the 30th percentile raw
    # rate is exactly the replacement rate and that player is worth zero.
    assert np.isclose(snapshot.loc["qb3", "value_above_replacement"], 0.0)
    assert (snapshot["overall_rank"].sort_values().to_numpy() == np.arange(1, 13)).all()


def test_weekly_snapshots_are_cumulative():
    credited = pd.DataFrame(
        {
            "game_id": [1, 2, 3],
            "athlete_id": ["a", "a", "a"],
            "athlete_name": "A",
            "team": "T",
            "model_week": [1, 2, 3],
            "side": "offense",
            "role": "passer",
            "plays": [40.0, 40.0, 40.0],
            "raw_epa": [4.0, 4.0, 4.0],
            "adjusted_epa": [5.0, 5.0, 5.0],
            "wpa": [0.1, 0.1, 0.1],
            "weight": 1.0,
            "fcs_plays": [0.0, 0.0, 40.0],
        }
    )
    positions = pd.DataFrame(
        {"athlete_id": ["a"], "position": ["QB"], "position_group": ["QB"]}
    )
    teams = pd.DataFrame({"school": ["T"], "id": [1], "classification": ["fbs"]})
    out = value.weekly_snapshots(
        credited, positions, teams, 2025, datetime(2025, 10, 1, tzinfo=timezone.utc)
    )
    assert out["week"].tolist() == [1, 2, 3]
    assert out["plays"].tolist() == [40.0, 80.0, 120.0]
    assert np.allclose(out["adjusted_epa"], [5.0, 10.0, 15.0])
    assert np.isclose(out["fcs_play_share"].iloc[-1], 1 / 3)
    assert out["games"].tolist() == [1, 2, 3]
