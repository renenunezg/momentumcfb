import numpy as np
import pandas as pd
import pymc as pm


def build_model(games: pd.DataFrame, prior: pd.DataFrame, n_states: int) -> pm.Model:
    teams = list(prior.index)
    team_index = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    h = games["home_team"].map(team_index).to_numpy()
    a = games["away_team"].map(team_index).to_numpy()
    w = games["week_index"].to_numpy().astype(int)
    home_ind = (~games["neutral_site"].astype(bool)).to_numpy().astype(float)
    home_pts = games["home_points"].to_numpy().astype(float)
    away_pts = games["away_points"].to_numpy().astype(float)

    epa_ok = (games["home_off_epa_total"].notna() & games["away_off_epa_total"].notna()).to_numpy()
    he, ae, we = h[epa_ok], a[epa_ok], w[epa_ok]
    home_ind_e = home_ind[epa_ok]
    home_epa = games["home_off_epa_total"].to_numpy().astype(float)[epa_ok]
    away_epa = games["away_off_epa_total"].to_numpy().astype(float)[epa_ok]

    off_mu = prior["off_prior"].to_numpy()
    def_mu = prior["def_prior"].to_numpy()
    sd0 = prior["sd"].to_numpy() / 2

    with pm.Model() as model:
        tau = pm.HalfNormal("tau", sigma=0.7)
        # sum-to-zero deviations pin the off/def level, which the likelihood
        # cannot identify; base_pts and hfa carry the global levels instead
        off0_raw = pm.Normal("off0_raw", mu=0.0, sigma=sd0, shape=n_teams)
        def0_raw = pm.Normal("def0_raw", mu=0.0, sigma=sd0, shape=n_teams)
        off0 = off_mu + off0_raw - off0_raw.mean()
        def0 = def_mu + def0_raw - def0_raw.mean()
        off_steps_raw = pm.Normal("off_steps_raw", 0.0, tau, shape=(n_states - 1, n_teams))
        def_steps_raw = pm.Normal("def_steps_raw", 0.0, tau, shape=(n_states - 1, n_teams))
        off_steps = off_steps_raw - off_steps_raw.mean(axis=1, keepdims=True)
        def_steps = def_steps_raw - def_steps_raw.mean(axis=1, keepdims=True)
        off = pm.Deterministic(
            "off",
            pm.math.concatenate(
                [off0[None, :], off0[None, :] + off_steps.cumsum(axis=0)], axis=0
            ),
        )
        defense = pm.Deterministic(
            "defense",
            pm.math.concatenate(
                [def0[None, :], def0[None, :] + def_steps.cumsum(axis=0)], axis=0
            ),
        )

        hfa = pm.Normal("hfa", mu=2.5, sigma=1.0)
        base_pts = pm.Normal("base_pts", mu=28.0, sigma=3.0)
        sigma_pts = pm.HalfNormal("sigma_pts", sigma=12.0)

        mu_home = base_pts + off[w, h] - defense[w, a] + 0.5 * hfa * home_ind
        mu_away = base_pts + off[w, a] - defense[w, h] - 0.5 * hfa * home_ind
        pm.Normal("obs_home_pts", mu=mu_home, sigma=sigma_pts, observed=home_pts)
        pm.Normal("obs_away_pts", mu=mu_away, sigma=sigma_pts, observed=away_pts)

        if len(home_epa):
            base_epa = pm.Normal("base_epa", mu=10.0, sigma=5.0)
            k = pm.HalfNormal("epa_scale", sigma=1.0)
            sigma_epa = pm.HalfNormal("sigma_epa", sigma=12.0)
            mu_hepa = base_epa + k * (off[we, he] - defense[we, ae] + 0.5 * hfa * home_ind_e)
            mu_aepa = base_epa + k * (off[we, ae] - defense[we, he] - 0.5 * hfa * home_ind_e)
            pm.Normal("obs_home_epa", mu=mu_hepa, sigma=sigma_epa, observed=home_epa)
            pm.Normal("obs_away_epa", mu=mu_aepa, sigma=sigma_epa, observed=away_epa)

    return model


def fit(model: pm.Model, method: str = "advi", seed: int = 42):
    with model:
        if method == "nuts":
            return pm.sample(
                draws=1000,
                tune=1000,
                chains=2,
                target_accept=0.9,
                random_seed=seed,
                progressbar=False,
            )
        approx = pm.fit(40000, method="advi", random_seed=seed, progressbar=False)
        return approx.sample(1000, random_seed=seed)


def extract_ratings(idata, teams: list, state: int = -1) -> pd.DataFrame:
    off = idata.posterior["off"].mean(("chain", "draw")).values[state]
    deff = idata.posterior["defense"].mean(("chain", "draw")).values[state]
    off_sd = idata.posterior["off"].std(("chain", "draw")).values[state]
    def_sd = idata.posterior["defense"].std(("chain", "draw")).values[state]
    return pd.DataFrame(
        {
            "team": teams,
            "off": off,
            "defense": deff,
            "net": off + deff,
            "net_sd": np.sqrt(off_sd**2 + def_sd**2),
        }
    ).sort_values("net", ascending=False, ignore_index=True)


def fit_ratings(games, prior, n_states, method="advi", seed=42):
    model = build_model(games, prior, n_states)
    idata = fit(model, method=method, seed=seed)
    return extract_ratings(idata, list(prior.index)), idata


def predict_margin(idata, games: pd.DataFrame, teams: list) -> np.ndarray:
    team_index = {t: i for i, t in enumerate(teams)}
    off = idata.posterior["off"].mean(("chain", "draw")).values[-1]
    deff = idata.posterior["defense"].mean(("chain", "draw")).values[-1]
    hfa = float(idata.posterior["hfa"].mean(("chain", "draw")))
    net = off + deff
    h = games["home_team"].map(team_index).to_numpy()
    a = games["away_team"].map(team_index).to_numpy()
    home_ind = (~games["neutral_site"].astype(bool)).to_numpy().astype(float)
    return net[h] - net[a] + hfa * home_ind
