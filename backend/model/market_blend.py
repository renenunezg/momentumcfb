"""Output-only market context for pregame projections.

The pure model remains the source of team ratings and of the pure score,
margin, and total columns that picks, grading, and evaluation read. These
helpers add separately named market-informed margin, total, and score columns,
which are the forecast the site publishes, so consumers cannot mistake a
blended number for the model's independent opinion.
"""

import numpy as np
import pandas as pd

# The 0.50 cap improved identical-cohort holdout MAE from 13.284 to 12.314.
# The full closing line remained better at 11.991, so this is explicitly a
# product blend that preserves model opinion, not independent model skill.
DEFAULT_MARKET_WEIGHT = 0.50
MARKET_WEIGHT_CAP = 0.50
# Independent totals policy. Closing-only research does not establish a
# better executable blend; retain the existing model share and dispersion.
TOTAL_MARKET_WEIGHT = 0.50


def consensus_total(game_offers: pd.DataFrame) -> float:
    """Median posted total across a game's offers, or NaN without one."""
    if game_offers.empty or "market" not in game_offers:
        return float("nan")
    points = pd.to_numeric(
        game_offers.loc[game_offers["market"].eq("totals"), "point"], errors="coerce"
    ).dropna()
    return float(points.median()) if len(points) else float("nan")


def _consensus_home_spread(offers: pd.DataFrame) -> pd.Series:
    if {"market", "selection", "point"}.issubset(offers.columns):
        home = offers[
            offers["market"].eq("spreads") & offers["selection"].eq("home")
        ].copy()
        home["home_spread"] = pd.to_numeric(home["point"], errors="coerce")
    elif {"game_id", "home_spread"}.issubset(offers.columns):
        home = offers[["game_id", "home_spread"]].copy()
        home["home_spread"] = pd.to_numeric(home["home_spread"], errors="coerce")
    else:
        return pd.Series(dtype=float)
    return (
        home.dropna(subset=["home_spread"]).groupby("game_id")["home_spread"].median()
    )


def add_market_informed_margins(
    projections: pd.DataFrame,
    offers: pd.DataFrame,
    weight: float = DEFAULT_MARKET_WEIGHT,
    history: pd.Series | None = None,
    history_weight: float = 0.0,
) -> pd.DataFrame:
    """Add pure and market-informed margin, total, and score fields.

    ``history`` maps game ID to the margin implied by the market-history
    rating (earlier games' closing lines, never this game's). It replaces
    ``history_weight`` of the pure margin on the model side of the blend and
    also informs games without a current line. The market-informed total
    moves the model total toward the median posted total, and the
    market-informed scores are the pair with that total and the
    market-informed margin.
    """
    if not 0 <= weight <= MARKET_WEIGHT_CAP:
        raise ValueError(f"market weight must be between 0 and {MARKET_WEIGHT_CAP:g}")
    if not 0 <= history_weight < 1:
        raise ValueError("market history weight must be in [0, 1)")
    required = {"game_id", "home_margin", "home_spread", "model_total"}
    missing = sorted(required - set(projections.columns))
    if missing:
        raise ValueError("projections are missing blend columns: " + ", ".join(missing))

    out = projections.copy()
    consensus = _consensus_home_spread(offers)
    out["pure_home_margin"] = pd.to_numeric(out["home_margin"], errors="raise")
    out["pure_home_spread"] = pd.to_numeric(out["home_spread"], errors="raise")
    out["market_home_spread"] = out["game_id"].map(consensus)
    has_market = out["market_home_spread"].notna()
    out["market_weight"] = np.where(has_market, weight, 0.0)
    market_margin = -out["market_home_spread"]
    out["market_history_home_margin"] = (
        np.nan if history is None else out["game_id"].map(history)
    )
    has_history = out["market_history_home_margin"].notna()
    out["market_history_weight"] = np.where(has_history, history_weight, 0.0)
    model_side = np.where(
        has_history,
        (1.0 - history_weight) * out["pure_home_margin"]
        + history_weight * out["market_history_home_margin"],
        out["pure_home_margin"],
    )
    out["market_informed_home_margin"] = np.where(
        has_market,
        (1.0 - weight) * model_side + weight * market_margin,
        model_side,
    )
    out["market_informed_home_spread"] = -out["market_informed_home_margin"]

    model_total = pd.to_numeric(out["model_total"], errors="raise")
    market_total = out["game_id"].map(
        {
            game_id: consensus_total(group)
            for game_id, group in offers.groupby("game_id")
        }
        if "game_id" in offers
        else {}
    )
    blended_total = np.where(
        market_total.notna(),
        (1.0 - TOTAL_MARKET_WEIGHT) * model_total + TOTAL_MARKET_WEIGHT * market_total,
        model_total,
    )
    # A total below the margin would publish a negative score.
    margin = out["market_informed_home_margin"]
    out["market_informed_total"] = np.maximum(blended_total, margin.abs())
    out["market_informed_home_points"] = (out["market_informed_total"] + margin) / 2.0
    out["market_informed_away_points"] = (out["market_informed_total"] - margin) / 2.0
    return out


def align_ratings_to_forecast(
    ratings: pd.DataFrame, projections: pd.DataFrame
) -> pd.DataFrame:
    """Shift team ratings so they reproduce the week's published lines.

    The fitted ratings carry no market information, quarterback report, or
    pace term, so their difference plus home field can sit several points from
    the published market-informed line. Each game's gap is split evenly
    between its two teams (the minimum-norm solution, which also handles a
    team with two games), leaving teams without a game unchanged. Offense and
    defense each take half of a team's shift, so they still sum to the power
    rating. ``forecast_alignment_points`` records the shift, so the fitted
    rating stays recoverable.
    """
    out = ratings.copy()
    index = {int(team_id): row for row, team_id in enumerate(out["team_id"])}
    if len(index) != len(out):
        raise ValueError("rating alignment requires unique team IDs")
    home = projections["home_team_id"].map(index)
    away = projections["away_team_id"].map(index)
    if home.isna().any() or away.isna().any():
        raise ValueError("every projected team needs a rating to align")
    home, away = home.to_numpy(dtype=int), away.to_numpy(dtype=int)
    power = out["power_rating"].to_numpy(dtype=float)
    gap = (
        projections["market_informed_home_margin"].to_numpy(dtype=float)
        - power[home]
        + power[away]
        - projections["home_field_points"].to_numpy(dtype=float)
    )
    if not np.isfinite(gap).all():
        raise ValueError("rating alignment requires finite margins and ratings")
    games = np.zeros((len(projections), len(out)))
    rows = np.arange(len(projections))
    games[rows, home] = 1.0
    games[rows, away] = -1.0
    shift = np.linalg.lstsq(games, gap, rcond=None)[0]
    out["power_rating"] = power + shift
    out["offense_points"] = out["offense_points"] + shift / 2.0
    out["defense_points"] = out["defense_points"] + shift / 2.0
    out["forecast_alignment_points"] = shift
    return out
