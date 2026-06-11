import pandas as pd

from backend.backtest.harness import walk_forward_season


def toy_games():
    rows = []
    gid = 0
    for w in range(1, 5):
        rows.append(
            {
                "game_id": gid,
                "season": 2024,
                "week": w,
                "week_index": w - 1,
                "season_type": "regular",
                "home_team": "A",
                "away_team": "B",
                "neutral_site": False,
                "home_points": 28,
                "away_points": 21,
                "margin": 7,
                "closing_spread": -3.0,
                "home_off_epa_total": 9.0,
                "away_off_epa_total": 4.0,
            }
        )
        gid += 1
    return pd.DataFrame(rows)


def test_walk_forward_never_leaks_and_predicts_each_week():
    seen = []

    def stub_fitter(train, prior, n_states, test):
        assert (train["week_index"] < test["week_index"].min()).all()
        seen.append(int(test["week_index"].min()))
        return pd.Series([5.0] * len(test), index=test.index)

    prior = pd.DataFrame(
        {"off_prior": [1.0, -1.0], "def_prior": [1.0, -1.0], "sd": [6.0, 6.0]},
        index=pd.Index(["A", "B"], name="team"),
    )
    out = walk_forward_season(toy_games(), prior, fitter=stub_fitter)
    assert seen == [1, 2, 3]
    assert len(out) == 3
    assert (out["model_margin"] == 5.0).all()
    assert "actual_margin" in out.columns
