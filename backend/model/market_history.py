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

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.etl import store
from backend.features.scoring import build_weekly_scoring_games, load_scoring_team_games
from backend.model.joint_scoring import (
    PointsRatingFit,
    _fit_points_rating,
    _team_catalog,
)
from backend.serving.market import flatten_closing_lines

MARKET_PRIOR_MODEL_VERSION = "market_carryover_v1"
MARKET_PRIOR_SD_POINTS = 9.0
LINE_HALF_LIFE_WEEKS = 3.0
EARLY_WEEKS = 3
EARLY_HISTORY_WEIGHT = 0.55
LATE_HISTORY_WEIGHT = 0.35
MARKET_UNCERTAINTY_VERSION = "market_uncertainty_v1"
# Gaussian NLL optimum on 2,775 chronological 2020-2022 forecasts.
# On 4,376 reused 2023-2025 validation games, 80% future-line coverage
# improves from 91.75% to 77.81%, with NLL 3.12434 -> 3.06137.
# This scales future-line prediction intervals, not latent rating SDs.
MARKET_LINE_SD_SCALE = 0.6866868661624285
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
) -> PointsRatingFit:
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
        fitted = _fit_market_rating(lined, catalog, carried, np.ones(len(lined)))
        home_field = fitted.home_field
        carried.update(zip(catalog["team"], fitted.rating.astype(float)))
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


@dataclass(frozen=True, slots=True)
class MarketHistoryFit:
    teams: pd.DataFrame
    fitted: PointsRatingFit
    season: int
    week: int
    as_of: pd.Timestamp
    training: pd.DataFrame

    def ratings(self) -> pd.DataFrame:
        """Conditional strength SD, distinct from future-line prediction SD.

        This covariance conditions on the estimated noise, prior means and
        fallback pools. Future-line coverage does not validate latent team
        strength coverage, so no predictive calibration scalar is applied here.
        """
        out = self.teams[["team_id", "team", "classification"]].copy()
        games = pd.concat(
            [self.training["home_team_id"], self.training["away_team_id"]]
        ).value_counts()
        out["market_rating"] = self.fitted.rating
        out["market_rating_sd"] = np.sqrt(
            self.fitted.rating_variance(out["classification"].to_numpy())
        )
        out["games_with_lines"] = out["team_id"].map(games).fillna(0).astype(int)
        out["season"], out["week"], out["as_of"] = self.season, self.week, self.as_of
        out["home_field_points"] = self.fitted.home_field
        out["last_training_kickoff"] = self.training["start_date"].max()
        out["model_version"] = MARKET_UNCERTAINTY_VERSION
        out["sd_method"] = "conditional_ridge_posterior"
        out["source"] = "prior_week_closing_spread_median"
        return out

    def project(self, target: pd.DataFrame) -> pd.DataFrame:
        """Forecast a later closing line, not a realized game margin."""
        index = dict(zip(self.teams["team_id"].astype(int), range(len(self.teams))))
        home = target["home_team_id"].map(index).to_numpy(int)
        away = target["away_team_id"].map(index).to_numpy(int)
        venue = (~target["neutral_site"].astype(bool)).to_numpy(float)
        covariance = self.fitted.parameter_covariance
        parameter_variance = (
            covariance[home, home]
            + covariance[away, away]
            - 2 * covariance[home, away]
            + venue**2 * covariance[-1, -1]
            + 2 * venue * (covariance[home, -1] - covariance[away, -1])
        )
        return pd.DataFrame(
            {
                "market_history_home_margin": (
                    self.fitted.rating[home]
                    - self.fitted.rating[away]
                    + self.fitted.home_field * venue
                ),
                "market_parameter_variance": np.maximum(parameter_variance, 0.0),
                "market_observation_variance": self.fitted.observation_variance,
                "market_line_sd": MARKET_LINE_SD_SCALE
                * np.sqrt(
                    np.maximum(parameter_variance, 0.0)
                    + self.fitted.observation_variance
                ),
                "season": self.season,
                "week": self.week,
                "as_of": self.as_of,
                "model_version": MARKET_UNCERTAINTY_VERSION,
                "sd_method": "dev_scaled_future_closing_line_normal",
                "sd_scale": MARKET_LINE_SD_SCALE,
                "last_training_kickoff": self.training["start_date"].max(),
            },
            index=pd.Index(target["game_id"].to_numpy(), name="game_id"),
        )


def fit_market_history(
    games: pd.DataFrame,
    lines: pd.DataFrame,
    forecast_week: int,
    as_of: pd.Timestamp,
    market_prior: pd.DataFrame,
) -> MarketHistoryFit | None:
    """Home margin implied by lines of games that kicked off before the week.

    Returns None when no earlier game of the season has a line, so
    the market-informed margin falls back to its previous construction.
    """
    if not any(
        offer.get("spread") is not None for offers in lines["lines"] for offer in offers
    ):
        return None
    lined = lined_games(games, lines)
    lined = lined[
        lined["model_week"].lt(forecast_week)
        & pd.to_datetime(lined["start_date"], utc=True).lt(as_of)
    ]
    if lined.empty:
        return None
    catalog = _team_catalog(games)
    recency = 0.5 ** (
        (forecast_week - 1 - lined["model_week"].to_numpy(float)) / LINE_HALF_LIFE_WEEKS
    )
    fitted = _fit_market_rating(
        lined,
        catalog,
        dict(zip(market_prior["team"], market_prior["market_rating"].astype(float))),
        recency,
    )
    return MarketHistoryFit(
        catalog,
        fitted,
        int(games["season"].iloc[0]),
        forecast_week,
        as_of,
        lined,
    )
