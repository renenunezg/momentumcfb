"""Market-history rating for the market-informed margin.

A ridge rating fitted to closing spreads of games that kicked off before the
forecast week, never the target game's own line. The prior is the previous
season's final market rating, chained through every cached season the way the
points-only prior is. It feeds only ``market_informed_home_margin``; the pure
``home_margin`` never sees a line.

Settings were selected on the 2020 through 2022 walk-forward and scored once
on 2023 through 2025 (4,254 Division I games with a close). Replayed through
this module on joint_scoring_v13, the model side of the blend improved by
0.184 MAE (t -5.4) and the market-informed margin, which also holds the game's
own line, by 0.073 (12.070 to 11.997, t -4.3), negative in every season. About
70 percent of the gain is the current season's earlier lines.
"""

import numpy as np
import pandas as pd

from backend.etl import store
from backend.features.scoring import build_weekly_scoring_games, load_scoring_team_games
from backend.model.joint_scoring import _fit_points_rating, _team_catalog
from backend.serving.market import flatten_closing_lines

MARKET_PRIOR_MODEL_VERSION = "market_carryover_v1"
MARKET_PRIOR_SD_POINTS = 9.0
LINE_HALF_LIFE_WEEKS = 3.0
EARLY_WEEKS = 3
EARLY_HISTORY_WEIGHT = 0.55
LATE_HISTORY_WEIGHT = 0.35
MARKET_PRIOR_COLUMNS = [
    "season",
    "team",
    "classification",
    "market_rating",
    "home_field_points",
    "model_version",
    "chain_start_season",
]


def history_weight(forecast_week: int) -> float:
    return EARLY_HISTORY_WEIGHT if forecast_week <= EARLY_WEEKS else LATE_HISTORY_WEIGHT


def lined_games(games: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Division I games with a closing spread, the spread standing in as the score."""
    out = games.merge(
        flatten_closing_lines(lines)[["game_id", "closing_spread"]],
        on="game_id",
        how="inner",
        validate="one_to_one",
    )
    out = out[out["closing_spread"].notna()]
    return out.assign(home_points=-out["closing_spread"], away_points=0.0)


def _fit_market_rating(
    lined: pd.DataFrame,
    catalog: pd.DataFrame,
    prior_by_team: dict[str, float],
    recency: np.ndarray,
) -> tuple[np.ndarray, float]:
    prior = catalog["team"].map(prior_by_team)
    return _fit_points_rating(
        lined,
        {int(team_id): index for index, team_id in enumerate(catalog["team_id"])},
        catalog["classification"].fillna("").str.lower().to_numpy(),
        recency,
        prior.fillna(0.0).to_numpy(float),
        prior.notna().to_numpy(),
        MARKET_PRIOR_SD_POINTS,
    )


def build_market_prior(season: int) -> pd.DataFrame:
    """Chain the closing-line rating through every cached season before ``season``.

    Teams without a line in a season keep their older rating; teams never
    lined get no entry and shrink toward their classification inside the fit.
    """
    seasons = sorted(
        int(name) for name in store.processed_names("team_games") if name.isdigit()
    )
    seasons = [year for year in seasons if year < season]
    if not seasons or seasons[-1] != season - 1:
        raise FileNotFoundError(
            f"{season}: the previous season's games and lines are required "
            "to build the market carryover prior"
        )
    carried: dict[str, float] = {}
    classes: dict[str, str] = {}
    home_field = 0.0
    for year in seasons:
        games = build_weekly_scoring_games(
            store.read_games(year), load_scoring_team_games(year)
        )
        lined = lined_games(games, store.read_lines(year))
        catalog = _team_catalog(lined)
        rating, home_field = _fit_market_rating(
            lined, catalog, carried, np.ones(len(lined))
        )
        carried.update(zip(catalog["team"], rating.astype(float)))
        classes.update(zip(catalog["team"], catalog["classification"]))
    frame = pd.DataFrame(
        {
            "season": seasons[-1],
            "team": list(carried),
            "classification": [classes[team] for team in carried],
            "market_rating": list(carried.values()),
            "home_field_points": home_field,
            "model_version": MARKET_PRIOR_MODEL_VERSION,
            "chain_start_season": seasons[0],
        }
    )
    return frame[MARKET_PRIOR_COLUMNS]


def load_market_prior(season: int) -> pd.DataFrame:
    try:
        frame = store.read_preseason_forecast_artifact(season, 1, "market_prior")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{season}: frozen preseason market_prior is required for the "
            "market-informed margin; build it with "
            f"`python -m backend market-prior --season {season}` from cached "
            "lines and install it with the preseason runtime bundle"
        ) from exc
    missing = sorted(set(MARKET_PRIOR_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError("market prior is missing columns: " + ", ".join(missing))
    if frame.empty or not frame["season"].eq(season - 1).all():
        raise ValueError("market prior must describe the previous season")
    if (
        frame["team"].duplicated().any()
        or not np.isfinite(frame["market_rating"].to_numpy(float)).all()
    ):
        raise ValueError("market prior must hold one finite rating per team")
    return frame


def market_history_margins(
    games: pd.DataFrame,
    lines: pd.DataFrame,
    forecast_week: int,
    as_of: pd.Timestamp,
    target: pd.DataFrame,
    market_prior: pd.DataFrame,
) -> pd.Series:
    """Home margin implied by lines of games that kicked off before the week.

    Returns an empty series when no earlier game of the season has a line, so
    the market-informed margin falls back to its previous construction.
    """
    if not any(
        offer.get("spread") is not None for offers in lines["lines"] for offer in offers
    ):
        return pd.Series(dtype=float)
    lined = lined_games(games, lines)
    lined = lined[
        lined["model_week"].lt(forecast_week)
        & pd.to_datetime(lined["start_date"], utc=True).lt(as_of)
    ]
    if lined.empty:
        return pd.Series(dtype=float)
    catalog = _team_catalog(games)
    recency = 0.5 ** (
        (forecast_week - 1 - lined["model_week"].to_numpy(float)) / LINE_HALF_LIFE_WEEKS
    )
    rating, home_field = _fit_market_rating(
        lined,
        catalog,
        dict(zip(market_prior["team"], market_prior["market_rating"].astype(float))),
        recency,
    )
    by_team = dict(zip(catalog["team_id"].astype(int), rating))
    margin = (
        target["home_team_id"].astype(int).map(by_team)
        - target["away_team_id"].astype(int).map(by_team)
        + home_field * (~target["neutral_site"].astype(bool)).astype(float)
    )
    return pd.Series(margin.to_numpy(float), index=target["game_id"].to_numpy())
