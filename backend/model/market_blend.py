"""Output-only market context for pregame projections.

The pure model remains the source of team ratings, expected scores, totals,
and model-versus-market comparisons. These helpers add a separately named
market-informed margin so consumers cannot mistake a blended line for the
model's independent opinion.
"""

import numpy as np
import pandas as pd

# The 0.50 cap improved identical-cohort holdout MAE from 13.284 to 12.314.
# The full closing line remained better at 11.991, so this is explicitly a
# product blend that preserves model opinion, not independent model skill.
DEFAULT_MARKET_WEIGHT = 0.50
MARKET_WEIGHT_CAP = 0.50


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
) -> pd.DataFrame:
    """Add pure and market-informed margin fields to projection records."""
    if not 0 <= weight <= MARKET_WEIGHT_CAP:
        raise ValueError(f"market weight must be between 0 and {MARKET_WEIGHT_CAP:g}")
    required = {"game_id", "home_margin", "home_spread"}
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
    out["market_informed_home_margin"] = np.where(
        has_market,
        (1.0 - weight) * out["pure_home_margin"] + weight * market_margin,
        out["pure_home_margin"],
    )
    out["market_informed_home_spread"] = -out["market_informed_home_margin"]
    return out
