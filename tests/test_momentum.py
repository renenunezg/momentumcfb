import pandas as pd
import pytest

from backend.features import ingame

HOME = "Home U"
AWAY = "Away St"


def _play(play_id, minutes, offense, play_type, drive, **overrides):
    row = {
        "game_id": 9101,
        "id": play_id,
        "drive_id": f"d{drive}",
        "drive_number": drive,
        "play_number": play_id,
        "offense": offense,
        "defense": AWAY if offense == HOME else HOME,
        "home": HOME,
        "away": AWAY,
        "offense_score": 0,
        "defense_score": 0,
        "offense_timeouts": 3,
        "defense_timeouts": 3,
        "period": 1,
        "clock": {"minutes": minutes, "seconds": 0},
        "down": 1,
        "distance": 10,
        "yards_to_goal": 75,
        "yards_gained": 0,
        "scoring": False,
        "play_type": play_type,
        "play_text": play_type,
        "ppa": None,
        "season": 2024,
        "week": 5,
        "season_type": "regular",
    }
    row.update(overrides)
    return row


def _game_plays():
    return pd.DataFrame(
        [
            _play(1, 15, AWAY, "Kickoff", 1),
            _play(2, 14, HOME, "Rush", 2, yards_gained=6, ppa=0.3),
            _play(
                3, 13, HOME, "Pass Incompletion", 2,
                down=2, distance=4, yards_to_goal=69, ppa=-0.5,
            ),
            _play(4, 12, HOME, "Punt", 2, down=4, yards_to_goal=65),
            _play(
                5, 11, AWAY, "Rush", 3,
                yards_to_goal=55, yards_gained=5, ppa=0.2,
            ),
            _play(
                6, 10, AWAY, "Interception", 3,
                down=2, distance=5, yards_to_goal=50, ppa=-2.0,
            ),
            _play(
                7, 9, HOME, "Rush", 4,
                yards_to_goal=45, yards_gained=2, ppa=-0.2,
            ),
            _play(
                8, 8, HOME, "Rush", 4,
                down=4, distance=1, yards_to_goal=43, yards_gained=3, ppa=0.8,
            ),
            _play(9, 7, HOME, "Field Goal Missed", 4, down=4, yards_to_goal=20),
            _play(
                10, 6, AWAY, "Rush", 5,
                down=4, distance=2, yards_to_goal=80, ppa=-0.3,
            ),
            _play(11, 0, AWAY, "End of Game", 6, period=4),
        ]
    )


def test_process_evidence_is_presnap_and_attributes_every_family():
    evidence = ingame.build_process_evidence(_game_plays())
    assert list(evidence["play_index"]) == list(range(1, 12))

    # Pre-snap shift: the punt row itself must not yet see its own stop.
    punt_state = evidence[evidence["play_index"].eq(4)].iloc[0]
    assert punt_state["away_stops_forced"] == 0
    after_punt = evidence[evidence["play_index"].eq(5)].iloc[0]
    assert after_punt["away_stops_forced"] == 1
    assert after_punt["evidence_plays"] == 2
    assert after_punt["home_field_position_total"] == 0

    final = evidence[evidence["play_index"].eq(11)].iloc[0]
    assert final["evidence_plays"] == 7
    assert final["scrimmage_plays_before"] == 7
    assert final["home_epa_total"] == pytest.approx(0.4)
    assert final["away_epa_total"] == pytest.approx(-2.1)
    assert final["home_successes"] == 2
    assert final["away_successes"] == 1
    # Punt and missed kick stop the home offense; the failed fourth down
    # stops the away offense. The interception counts only as a turnover.
    assert final["away_stops_forced"] == 2
    assert final["home_stops_forced"] == 1
    assert final["home_turnovers_forced"] == 1
    assert final["away_turnovers_forced"] == 0
    assert final["home_field_position_total"] == 30
    assert final["away_field_position_total"] == 15
    assert final["home_fourth_down_attempts"] == 1
    assert final["home_fourth_down_conversions"] == 1
    assert final["away_fourth_down_attempts"] == 1
    assert final["away_fourth_down_conversions"] == 0
    assert final["home_missed_kicks"] == 1
    assert final["away_missed_kicks"] == 0

    # Decayed counterparts tick on scrimmage plays: contributions at scrimmage
    # counts 1, 2, 5, 6 decay against the 7 scrimmage plays run before play 11.
    lam = 0.5 ** (1.0 / 30.0)
    assert final["home_epa_total_hl30"] == pytest.approx(
        0.3 * lam**6 - 0.5 * lam**5 - 0.2 * lam**2 + 0.8 * lam
    )
    # The punt is the newest evidence at play 5, so it carries full weight.
    assert after_punt["away_stops_forced_hl30"] == pytest.approx(1.0)


def test_extended_leakage_check_covers_process_evidence(monkeypatch):
    plays = _game_plays()
    assert (
        ingame.leakage_problems(
            plays, [9101], builder=ingame.build_momentum_states
        )
        == []
    )

    real_build = ingame.build_process_evidence

    def contaminated_build(frame):
        evidence = real_build(frame)
        evidence["home_epa_total"] = evidence.groupby("game_id")[
            "home_epa_total"
        ].transform("max")
        return evidence

    monkeypatch.setattr(ingame, "build_process_evidence", contaminated_build)
    problems = ingame.leakage_problems(
        plays, [9101], builder=ingame.build_momentum_states
    )
    assert problems and "change when later plays are removed" in problems[0]
