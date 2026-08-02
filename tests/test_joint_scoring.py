from datetime import datetime, timezone

import pandas as pd

from backend.model.joint_scoring import fit_joint_scoring


def _mini_season() -> pd.DataFrame:
    teams = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
    }
    matchups = [
        (1, "A", "B", 31, 17, 3.2, -1.0, False),
        (1, "C", "D", 20, 21, 0.5, 0.7, True),
        (2, "A", "C", 28, 14, 2.7, -0.6, False),
        (2, "D", "B", 24, 20, 1.1, 0.2, False),
        (3, "B", "C", 23, 20, 0.8, 0.4, False),
        (3, "D", "A", 17, 27, -0.3, 2.1, True),
    ]
    rows = []
    for game_id, matchup in enumerate(matchups, start=100):
        week, home, away, home_points, away_points, home_epa, away_epa, neutral = matchup
        rows.append(
            {
                "game_id": game_id,
                "season": 2026,
                "week": week,
                "model_week": week,
                "home_team_id": teams[home],
                "home_team": home,
                "home_classification": "fbs",
                "away_team_id": teams[away],
                "away_team": away,
                "away_classification": "fbs",
                "neutral_site": neutral,
                "home_points": home_points,
                "away_points": away_points,
                "game_possessions": 12.0,
                "home_epa_per_possession": home_epa,
                "away_epa_per_possession": away_epa,
            }
        )
    return pd.DataFrame(rows)


def test_joint_model_is_leak_free_and_reconciles_outputs():
    games = _mini_season()
    as_of = datetime(2026, 9, 15, tzinfo=timezone.utc)
    fitted = fit_joint_scoring(games, forecast_week=3, as_of=as_of)
    target = games[games["model_week"].eq(3)]
    ratings = fitted.ratings()
    projections = fitted.project(target)

    changed_future = games.copy()
    changed_future.loc[changed_future["model_week"].eq(3), "home_points"] = 100
    refitted = fit_joint_scoring(changed_future, forecast_week=3, as_of=as_of)

    assert [rating.to_record() for rating in ratings] == [
        rating.to_record() for rating in refitted.ratings()
    ]
    rating_by_id = {rating.team_id: rating for rating in ratings}
    for rating in ratings:
        assert rating.power_rating == rating.offense_points + rating.defense_points
    for projection in projections:
        home = rating_by_id[projection.home_team_id]
        away = rating_by_id[projection.away_team_id]
        expected_margin = (
            home.power_rating - away.power_rating + projection.home_field_points
        )
        assert abs(projection.home_margin - expected_margin) < 1e-10
        assert projection.model_total == (
            projection.expected_home_points + projection.expected_away_points
        )
        assert projection.margin_sd > 0
        assert projection.total_sd > 0
        assert -1 < projection.margin_total_correlation < 1

    fcs_games = games.copy()
    fcs_games.loc[fcs_games["home_team"].eq("D"), "home_classification"] = "fcs"
    fcs_games.loc[fcs_games["away_team"].eq("D"), "away_classification"] = "fcs"
    fcs_fit = fit_joint_scoring(fcs_games, forecast_week=3, as_of=as_of)
    fcs_ratings = pd.DataFrame(rating.to_record() for rating in fcs_fit.ratings())
    classifications = fcs_fit.teams.set_index("team_id")["classification"]
    fcs_ratings["classification"] = fcs_ratings["team_id"].map(classifications)
    assert set(fcs_ratings["classification"]) == {"fbs", "fcs"}
    assert abs(
        fcs_ratings.loc[
            fcs_ratings["classification"].eq("fbs"), "power_rating"
        ].mean()
    ) < 1e-10
