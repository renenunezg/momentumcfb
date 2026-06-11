import numpy as np
import pandas as pd
import pytest

from backend.model.priors import preseason_prior
from backend.model.state_space import fit_ratings, predict_margin


def synthetic_season():
    rng = np.random.default_rng(7)
    true = {"A": 12.0, "B": 4.0, "C": -4.0, "D": -12.0}
    teams = list(true)
    rows = []
    gid = 0
    for w in range(1, 7):
        order = rng.permutation(teams)
        for h, a in (order[:2], order[2:]):
            margin = true[h] - true[a] + 2.5 + rng.normal(0, 6)
            total = 50 + rng.normal(0, 8)
            rows.append(
                {
                    "game_id": gid,
                    "week_index": w - 1,
                    "home_team": h,
                    "away_team": a,
                    "neutral_site": False,
                    "home_points": (total + margin) / 2,
                    "away_points": (total - margin) / 2,
                    "home_off_epa_total": (true[h] - true[a]) / 2 + 10 + rng.normal(0, 4),
                    "away_off_epa_total": (true[a] - true[h]) / 2 + 10 + rng.normal(0, 4),
                }
            )
            gid += 1
    return pd.DataFrame(rows), teams


@pytest.mark.slow
def test_model_orders_teams_and_predicts():
    games, teams = synthetic_season()
    talent = pd.Series({t: 850.0 for t in teams})
    returning = pd.Series({t: 0.6 for t in teams})
    prior = preseason_prior(teams, talent, returning)

    ratings, idata = fit_ratings(games, prior, n_states=6, method="advi", seed=1)

    nets = ratings.set_index("team")["net"]
    assert nets["A"] > nets["B"] > nets["C"] > nets["D"]

    # synthetic teams have symmetric off/def, so the fitted split should too
    assert (ratings["off"] - ratings["defense"]).abs().max() < 3.0

    test = pd.DataFrame(
        {
            "home_team": ["A"],
            "away_team": ["D"],
            "neutral_site": [True],
        }
    )
    pred = predict_margin(idata, test, list(prior.index))
    assert 10 < pred[0] < 40
