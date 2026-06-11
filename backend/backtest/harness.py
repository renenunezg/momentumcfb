import pandas as pd

from backend.config import EVAL_SEASONS, FCS_LABEL
from backend.etl import store
from backend.features.games import build_games
from backend.model.priors import fit_prior_weights, preseason_prior
from backend.model.state_space import build_model, extract_ratings, fit, predict_margin


def pymc_fitter(method: str = "nuts", seed: int = 42):
    def _fitter(train, prior, n_states, test):
        model = build_model(train, prior, n_states)
        idata = fit(model, method=method, seed=seed)
        return pd.Series(
            predict_margin(idata, test, list(prior.index)), index=test.index
        )

    return _fitter


def walk_forward_season(games: pd.DataFrame, prior: pd.DataFrame, fitter) -> pd.DataFrame:
    reg = games[games["season_type"] == "regular"]
    out = []
    for w in sorted(reg["week_index"].unique()):
        if w == 0:
            continue
        train = games[games["week_index"] < w]
        test = reg[reg["week_index"] == w].copy()
        if train.empty or test.empty:
            continue
        # n_states = w + 1 adds one unobserved state, the one step ahead forecast
        test["model_margin"] = fitter(train, prior, int(w) + 1, test)
        test["actual_margin"] = test["margin"]
        out.append(test)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def season_teams(games: pd.DataFrame) -> list:
    return sorted(set(games["home_team"]) | set(games["away_team"]))


def load_season_inputs(season: int, variant: str):
    games_raw = store.read_games(season)
    lines = store.read_lines(season)
    epa = store.read_processed("team_game_epa", f"{season}.parquet")
    games = build_games(games_raw, lines, epa, variant=variant)

    talent_df = store.read_talent(season)
    team_col = "school" if "school" in talent_df.columns else "team"
    # CFBD ships duplicate rows in some seasons (2023 talent)
    talent = talent_df.drop_duplicates(team_col).set_index(team_col)["talent"].astype(float)

    ret_df = store.read_returning(season)
    pct_col = "percent_ppa" if "percent_ppa" in ret_df.columns else "percent_p_p_a"
    returning = ret_df.drop_duplicates("team").set_index("team")[pct_col].astype(float)
    return games, talent, returning


def build_season_prior(season, teams, talent, returning, history):
    prev_final = None
    weights = None
    if history:
        last = max(history)
        prev_final = history[last]["final"].set_index("team")["net"]
        rows = [h["prior_rows"] for h in history.values() if h["prior_rows"] is not None]
        if len(rows) >= 2:
            weights = fit_prior_weights(pd.concat(rows, ignore_index=True))
    return preseason_prior(teams, talent, returning, prev_final=prev_final, weights=weights)


def run_backtest(seasons, variant="filtered", method="nuts", seed=42) -> pd.DataFrame:
    fitter = pymc_fitter(method=method, seed=seed)
    history: dict[int, dict] = {}
    predictions = []

    for season in sorted(seasons):
        games, talent, returning = load_season_inputs(season, variant)
        teams = season_teams(games)
        prior = build_season_prior(season, teams, talent, returning, history)

        if season in EVAL_SEASONS:
            preds = walk_forward_season(games, prior, fitter)
            preds["season"] = season
            predictions.append(preds)
            print(f"{season}: predicted {len(preds)} games")

        n_states = int(games["week_index"].max()) + 1
        model = build_model(games, prior, n_states)
        idata = fit(model, method=method, seed=seed)
        final = extract_ratings(idata, list(prior.index))

        fbs = prior.index != FCS_LABEL
        if history:
            prev_net = history[max(history)]["final"].set_index("team")["net"]
            prev_col = prev_net.reindex(prior.index[fbs])
        else:
            prev_col = pd.Series(index=prior.index[fbs], dtype=float)
        prior_rows = pd.DataFrame(
            {
                "prev_final": prev_col,
                "talent_z": prior.loc[fbs, "talent_z"],
                "ret_c": prior.loc[fbs, "ret_c"],
                "final": final.set_index("team")["net"].reindex(prior.index[fbs]),
            }
        ).dropna()
        history[season] = {"final": final, "prior_rows": prior_rows}
        print(f"{season}: season fit done, top team {final.iloc[0]['team']}")

    return pd.concat(predictions, ignore_index=True)
