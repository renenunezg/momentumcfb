import pandas as pd

from backend.model.priors import fit_prior_weights, preseason_prior


def test_prior_is_centered_and_split():
    talent = pd.Series({"A": 980.0, "B": 850.0, "C": 700.0})
    returning = pd.Series({"A": 0.8, "B": 0.6, "C": 0.4})
    prior = preseason_prior(["A", "B", "C"], talent, returning)
    assert abs(prior["net_prior"].mean()) < 1e-9
    assert prior.loc["A", "net_prior"] > prior.loc["C", "net_prior"]
    assert abs(prior.loc["A", "off_prior"] * 2 - prior.loc["A", "net_prior"]) < 1e-9


def test_prev_rating_blends_in():
    talent = pd.Series({"A": 900.0, "B": 900.0})
    returning = pd.Series({"A": 0.6, "B": 0.6})
    prev = pd.Series({"A": 10.0, "B": -10.0})
    prior = preseason_prior(["A", "B"], talent, returning, prev_final=prev)
    assert prior.loc["A", "net_prior"] > prior.loc["B", "net_prior"]


def test_fcs_row_added():
    talent = pd.Series({"A": 900.0, "B": 800.0})
    returning = pd.Series({"A": 0.6, "B": 0.6})
    prior = preseason_prior(["A", "B", "FCS"], talent, returning)
    assert prior.loc["FCS", "net_prior"] < -15
    assert prior.loc["FCS", "sd"] < 4


def test_fit_prior_weights_recovers_signal():
    rows = pd.DataFrame(
        {
            "prev_final": [10.0, 0.0, -10.0, 5.0, -5.0, 8.0],
            "talent_z": [1.0, 0.0, -1.0, 0.5, -0.5, 0.9],
            "ret_c": [0.1, 0.0, -0.1, 0.05, -0.05, 0.1],
            "final": [9.0, 0.0, -9.0, 4.5, -4.5, 7.5],
        }
    )
    w = fit_prior_weights(rows)
    assert w["prev"] > 0
