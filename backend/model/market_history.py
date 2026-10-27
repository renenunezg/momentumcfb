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

The market-history total is the same construction for closing totals: a
symmetric ridge rating (closing_total = base + home + away) with the previous
season's final total rating as the carried prior, feeding only the model side
of ``market_informed_total``. Settings were selected on the 2020 through 2022
walk-forward (production-like replay, FBS vs FBS with a close) and scored once
on 2023 through 2025: the model side of the total improved by 0.134 MAE
(12.871 to 12.736, t -2.0; 0.182 and t -3.7 on all Division I games with a
close) and the market-informed total at the 0.5 market weight by 0.042
(12.649 to 12.608, t -1.2; 0.059 and t -2.3 on all Division I). The rating
alone forecasts the closing total within 3.4 points on the development
seasons, and about a fifth of the gain comes before week 4.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.etl import store
from backend.features.scoring import build_weekly_scoring_games, load_scoring_team_games
from backend.model.joint_scoring import (
    SRS_FALLBACK_TOLERANCE_POINTS,
    SRS_NOISE_FLOOR,
    PointsRatingFit,
    _fit_points_rating,
    _solve_ridge,
    _team_catalog,
)
from backend.serving.market import flatten_closing_lines, flatten_closing_totals

MARKET_PRIOR_MODEL_VERSION = "market_carryover_v1"
MARKET_PRIOR_SD_POINTS = 9.0
LINE_HALF_LIFE_WEEKS = 3.0
EARLY_WEEKS = 3
EARLY_HISTORY_WEIGHT = 0.55
LATE_HISTORY_WEIGHT = 0.35
MARKET_UNCERTAINTY_VERSION = "market_uncertainty_v1"
# Market-history total: a symmetric ridge rating fitted to earlier games'
# closing totals (closing_total = base + home + away). Settings were selected
# on the 2020-2022 walk-forward by model-side total MAE; the half-life ties
# with 1 to 4 weeks there (paired t below 1), so the margin version's value is
# kept. The model side of the market-informed total is
# (1 - weight) * model_total + weight * market_history_total.
MARKET_TOTAL_PRIOR_SD_POINTS = 6.0
TOTAL_LINE_HALF_LIFE_WEEKS = 3.0
TOTAL_EARLY_HISTORY_WEIGHT = 0.68
TOTAL_LATE_HISTORY_WEIGHT = 0.71
MARKET_TOTAL_PRIOR_COLUMN = "market_total_rating"
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


def total_history_weight(forecast_week: int) -> float:
    if forecast_week <= EARLY_WEEKS:
        return TOTAL_EARLY_HISTORY_WEIGHT
    return TOTAL_LATE_HISTORY_WEIGHT


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


def lined_total_games(games: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Division I games with a closing total; empty when no offer posts one."""
    try:
        totals = flatten_closing_totals(lines)
    except ValueError:
        return games.iloc[0:0].assign(closing_total=pd.Series(dtype=float))
    out = games.merge(
        totals[["game_id", "closing_total"]],
        on="game_id",
        how="inner",
        validate="one_to_one",
    )
    return out[out["closing_total"].notna()]


@dataclass(frozen=True, slots=True)
class TotalRatingFit:
    rating: np.ndarray
    base_total: float
    parameter_covariance: np.ndarray
    observation_variance: float


def _fit_total_rating(
    lined: pd.DataFrame,
    catalog: pd.DataFrame,
    prior_by_team: dict[str, float],
    recency: np.ndarray,
) -> TotalRatingFit:
    """Symmetric ridge: closing_total = base + rating_home + rating_away.

    Ratings shrink toward the carried prior in points; teams without one
    shrink toward their classification's fitted mean, iterated to a fixed
    point. The base is a free intercept. Rows are weighted by recency over the
    residual variance of a first pass, as the margin rating does. Ratings are
    centered on the FBS mean, with the base absorbing the shift.
    """
    team_index = {
        int(team_id): index for index, team_id in enumerate(catalog["team_id"])
    }
    classes = pd.Series(catalog["classification"].fillna("").str.lower().to_numpy())
    n_teams, n_games = len(catalog), len(lined)
    home = lined["home_team_id"].map(team_index).to_numpy(int)
    away = lined["away_team_id"].map(team_index).to_numpy(int)
    total = lined["closing_total"].to_numpy(float)
    design = np.zeros((n_games, n_teams + 1))
    design[np.arange(n_games), home] = 1.0
    design[np.arange(n_games), away] += 1.0
    design[:, -1] = 1.0
    prior = catalog["team"].map(prior_by_team)
    missing = prior.isna().to_numpy()
    base_guess = float(np.average(total, weights=recency))
    prior_mean: np.ndarray = np.append(prior.fillna(0.0).to_numpy(float), base_guess)
    prior_sd = np.full(n_teams + 1, MARKET_TOTAL_PRIOR_SD_POINTS)
    prior_sd[-1] = 1e3

    def solve(noise: float) -> tuple[np.ndarray, np.ndarray]:
        weights = recency / noise
        mean = prior_mean.copy()
        parameters, covariance = _solve_ridge(design, total, weights, mean, prior_sd)
        for _ in range(30 if missing.any() else 0):
            updated = mean.copy()
            for _, members in classes.groupby(classes).groups.items():
                selection = np.asarray(members)
                gap = missing[selection]
                if gap.any():
                    updated[selection[gap]] = parameters[selection].mean()
            shift = float(np.abs(updated - mean).max())
            mean = updated
            parameters, covariance = _solve_ridge(
                design, total, weights, mean, prior_sd
            )
            if shift < SRS_FALLBACK_TOLERANCE_POINTS:
                break
        return parameters, covariance

    noise = max(
        float(np.average(np.square(total - base_guess), weights=recency)),
        SRS_NOISE_FLOOR,
    )
    residual = total - design @ solve(noise)[0]
    noise = max(
        float(np.average(np.square(residual), weights=recency)), SRS_NOISE_FLOOR
    )
    parameters, covariance = solve(noise)
    rating = parameters[:-1]
    center = classes.eq("fbs").to_numpy()
    shift = float(rating[center].mean() if center.any() else rating.mean())
    return TotalRatingFit(
        rating - shift, float(parameters[-1]) + 2 * shift, covariance, noise
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
    carried_total: dict[str, float] = {}
    classes: dict[str, str] = {}
    home_field = 0.0
    for year in seasons:
        games = build_weekly_scoring_games(
            store.read_games(year), load_scoring_team_games(year)
        )
        lines = store.read_lines(year)
        lined = lined_games(games, lines)
        catalog = _team_catalog(lined)
        fitted = _fit_market_rating(lined, catalog, carried, np.ones(len(lined)))
        home_field = fitted.home_field
        carried.update(zip(catalog["team"], fitted.rating.astype(float)))
        classes.update(zip(catalog["team"], catalog["classification"]))
        lined_totals = lined_total_games(games, lines)
        if not lined_totals.empty:
            total_catalog = _team_catalog(lined_totals)
            total_fit = _fit_total_rating(
                lined_totals, total_catalog, carried_total, np.ones(len(lined_totals))
            )
            carried_total.update(
                zip(total_catalog["team"], total_fit.rating.astype(float))
            )
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
    # Teams lined for a total but never for a spread are not carried; the
    # total fit shrinks them toward their classification like any new team.
    frame[MARKET_TOTAL_PRIOR_COLUMN] = frame["team"].map(carried_total)
    return frame[[*MARKET_PRIOR_COLUMNS, MARKET_TOTAL_PRIOR_COLUMN]]


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


@dataclass(frozen=True, slots=True)
class MarketTotalHistoryFit:
    teams: pd.DataFrame
    fitted: TotalRatingFit
    season: int
    week: int
    as_of: pd.Timestamp
    training: pd.DataFrame

    def ratings(self) -> pd.DataFrame:
        out = self.teams[["team_id", "team", "classification"]].copy()
        games = pd.concat(
            [self.training["home_team_id"], self.training["away_team_id"]]
        ).value_counts()
        out["market_total_rating"] = self.fitted.rating
        out["base_total"] = self.fitted.base_total
        out["games_with_totals"] = out["team_id"].map(games).fillna(0).astype(int)
        out["season"], out["week"], out["as_of"] = self.season, self.week, self.as_of
        out["last_training_kickoff"] = self.training["start_date"].max()
        out["source"] = "prior_week_closing_total_median"
        return out

    def project(self, target: pd.DataFrame) -> pd.DataFrame:
        """Forecast a later closing total, not a realized game total."""
        index = dict(zip(self.teams["team_id"].astype(int), range(len(self.teams))))
        home = target["home_team_id"].map(index).to_numpy(int)
        away = target["away_team_id"].map(index).to_numpy(int)
        return pd.DataFrame(
            {
                "market_history_total": (
                    self.fitted.base_total
                    + self.fitted.rating[home]
                    + self.fitted.rating[away]
                ),
                "season": self.season,
                "week": self.week,
                "as_of": self.as_of,
                "last_training_kickoff": self.training["start_date"].max(),
            },
            index=pd.Index(target["game_id"].to_numpy(), name="game_id"),
        )


def fit_market_total_history(
    games: pd.DataFrame,
    lines: pd.DataFrame,
    forecast_week: int,
    as_of: pd.Timestamp,
    market_prior: pd.DataFrame,
) -> MarketTotalHistoryFit | None:
    """Total implied by closing totals of games that kicked off before the week.

    Returns None when no earlier game of the season has a total, so the
    market-informed total falls back to its previous construction. A market
    prior frozen before this rating carries no total column; every team then
    shrinks toward its classification's level.
    """
    lined = lined_total_games(games, lines)
    lined = lined[
        lined["model_week"].lt(forecast_week)
        & pd.to_datetime(lined["start_date"], utc=True).lt(as_of)
    ]
    if lined.empty:
        return None
    catalog = _team_catalog(games)
    recency = 0.5 ** (
        (forecast_week - 1 - lined["model_week"].to_numpy(float))
        / TOTAL_LINE_HALF_LIFE_WEEKS
    )
    prior: dict[str, float] = {}
    if MARKET_TOTAL_PRIOR_COLUMN in market_prior:
        carried = market_prior[MARKET_TOTAL_PRIOR_COLUMN].astype(float)
        prior = dict(zip(market_prior["team"][carried.notna()], carried.dropna()))
    fitted = _fit_total_rating(lined, catalog, prior, recency)
    return MarketTotalHistoryFit(
        catalog, fitted, int(games["season"].iloc[0]), forecast_week, as_of, lined
    )
