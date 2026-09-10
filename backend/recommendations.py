"""Pregame decisions and settlement at the exact recommended line and price."""

from datetime import datetime, timezone
from math import ceil, floor

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from backend.model.distributions import marginal_cdf
from backend.model.market_blend import DEFAULT_MARKET_WEIGHT
from backend.odds.markets import _american_profit, priced_candidates

POLICY_VERSION = "cfb-picks-v6"
# Minimum points the priced line must sit beyond the offered price's
# break-even line. Measured in margin or total points, the same yardstick for
# favourites and underdogs, so a mispriced tail cannot clear the gate on one
# side only. A versioned starting policy, not a fit to live-season outcomes.
MIN_EDGE_POINTS = 2.0
# Dispersion of actual results around the priced (market-informed) line, from
# the chronological calibration replay of 2021 through 2025 (3,208 games with
# a closing spread and total). The pure model's predictive spread includes
# rating uncertainty the market blend has already removed, and its 17 to 18
# point width overstated every underdog's win probability. The market
# residual is the honest width for a line that is half market.
PRICING_MARGIN_SD = 15.35
PRICING_TOTAL_SD = 15.93
# Weight of the pure model in the margin that prices a moneyline. The
# half-market blend that prices spreads is a product line that keeps model
# opinion; for winning outright it is not calibrated. On the 2021 through
# 2025 chronological replay (3,481 games with a closing spread) the probit fit
# of the outcome on close + w * (model - close) gives w = 0.04 with se 0.05,
# and the held-out 2024 to 2025 log loss is lowest near 0.2 (0.5317 against
# 0.5338 at 0.5). No curve shape beat the fixed-sd normal out of sample, so
# only the location changes. 0.2 is the most model opinion the held-out
# seasons support; 0.5 priced moneyline picks 8 to 9 points above their
# realised win rate.
H2H_MODEL_WEIGHT = 0.2
# Minimum recommended picks per decision batch (one model week). When the
# edge gate leaves fewer, the highest-edge positive-EV offers are promoted
# and recorded with FLOOR_REASON so the ledger separates gate picks from
# floor picks. A product volume choice, not a calibration result.
VOLUME_FLOOR = 15
FLOOR_REASON = "volume_floor"
EDGE_SEARCH_POINTS = 200.0
MAX_OFFER_AGE = pd.Timedelta(hours=1)
MAX_FORECAST_AGE = pd.Timedelta(days=7)
MARKETS = ("h2h", "spreads", "totals")
RECOMMENDATION_COLUMNS = [
    "game_id",
    "market",
    "season",
    "week",
    "start_date",
    "home_team",
    "away_team",
    "model_version",
    "forecast_as_of",
    "home_missing_input_count",
    "away_missing_input_count",
    "policy_version",
    "decision_at",
    "status",
    "reason",
    "selection",
    "side",
    "point",
    "price",
    "provider",
    "provider_key",
    "market_fetched_at",
    "odds_api_event_id",
    "provider_start_date",
    "provider_last_update",
    "match_score",
    "win_probability",
    "push_probability",
    "probability_edge",
    "edge_points",
    "expected_value_per_unit",
    "stake_units",
    "model_home_margin",
    "model_total",
    "market_total",
    "margin_sd",
    "total_sd",
    "degrees_of_freedom",
]
SETTLEMENT_COLUMNS = [
    "game_id",
    "market",
    "decision_at",
    "outcome",
    "home_points",
    "away_points",
    "profit_units",
    "graded_at",
]


def _timestamp(value):
    return pd.to_datetime(value, utc=True, errors="coerce")


def _win_loss(mean, sd, df, offer):
    """Win and loss probabilities for the bettor's side of an offer.

    ``mean`` is the priced home margin for sides or the priced total for
    totals. Half-point lines keep the original CDF. Integer lines reserve the
    mass between the adjacent half points for a returned stake.
    """
    home_side = offer["side"] in ("home", "over")
    if offer["market"] == "h2h":
        win = float(marginal_cdf(mean if home_side else -mean, sd, df))
        return win, 1.0 - win
    point = offer["point"]
    if offer["market"] == "spreads":
        mean = mean if home_side else -mean
        threshold = -point
    else:
        threshold = point
    win = float(marginal_cdf(mean - floor(threshold) - 0.5, sd, df))
    loss = float(marginal_cdf(ceil(threshold) - 0.5 - mean, sd, df))
    return (loss, win) if offer["side"] == "under" else (win, loss)


def _priced_mean_sd(projection, offer):
    if offer["market"] == "totals":
        return projection.model_total, projection.total_sd
    if offer["market"] == "h2h":
        return projection.h2h_home_margin, projection.margin_sd
    return projection.market_informed_home_margin, projection.margin_sd


def _h2h_home_margins(projections):
    """Consensus market margin moved H2H_MODEL_WEIGHT of the way to the pure
    model, or NaN without a posted spread so a moneyline is never priced from
    the pure model alone."""
    if "market_home_spread" not in projections:
        return np.full(len(projections), np.nan)
    spread = pd.to_numeric(projections["market_home_spread"], errors="coerce")
    pure = pd.to_numeric(
        projections["pure_home_margin"]
        if "pure_home_margin" in projections
        else projections["home_margin"],
        errors="coerce",
    )
    market = -spread
    return np.where(
        spread.notna() & pure.notna(),
        market + H2H_MODEL_WEIGHT * (pure - market),
        np.nan,
    )


def _probabilities(projection, offer):
    """Round the frozen continuous marginal to integer scores, retaining pushes.

    Sides use the market-informed margin (pure model shrunk toward the
    pre-decision consensus spread). Totals use the model total shrunk toward
    the median posted total across the decision-time offers, supplied on the
    projection. Both are priced with the dispersion stored on the projection.
    """
    mean, sd = _priced_mean_sd(projection, offer)
    win, loss = _win_loss(mean, sd, projection.degrees_of_freedom, offer)
    if offer["market"] == "h2h":
        return win, 0.0, loss
    return win, max(0.0, 1.0 - win - loss), loss


def _edge_points(projection, offer, implied):
    """Points the priced line must move against the pick before its no-push
    win probability falls to the price's break-even probability.

    Positive favours the pick. The same shift is required of a favourite and
    an underdog, unlike a probability gap, which is largest where the density
    is highest and is then multiplied by the payout in expected value.
    """
    mean, sd = _priced_mean_sd(projection, offer)
    direction = 1.0 if offer["side"] in ("home", "over") else -1.0

    def gap(delta):
        win, loss = _win_loss(
            mean - direction * delta, sd, projection.degrees_of_freedom, offer
        )
        return win / (win + loss) - implied

    try:
        return float(brentq(gap, -EDGE_SEARCH_POINTS, EDGE_SEARCH_POINTS))
    except ValueError:
        return float("nan")


def _offer_reason(offer, paired, now, start):
    fetched = _timestamp(offer["market_fetched_at"])
    updated = _timestamp(offer["provider_last_update"])
    # An explicit bookmaker list is optional: the configured regional feed
    # also supplies real quotes. CFBD lines have no Odds API event provenance.
    event_id = offer["odds_api_event_id"]
    if pd.isna(event_id) or not event_id:
        return "unverified_source"
    provider_start = _timestamp(offer["provider_start_date"])
    if pd.isna(provider_start) or provider_start != start:
        return "kickoff_mismatch"
    if any(
        pd.isna(offer[key]) or not offer[key] for key in ("provider_key", "provider")
    ):
        return "missing_provider"
    if (
        pd.isna(offer["match_score"])
        or not np.isfinite(offer["match_score"])
        or offer["match_score"] < 0.95
    ):
        return "uncertain_game_match"
    if pd.isna(fetched) or pd.isna(updated):
        return "missing_price_timestamp"
    if fetched >= start or updated >= start:
        return "in_play_offer"
    if not (updated <= fetched <= now) or now - updated > MAX_OFFER_AGE:
        return "stale_price"
    opposite = {"home": "away", "away": "home", "over": "under", "under": "over"}[
        offer["side"]
    ]
    other_point = -offer["point"] if offer["market"] == "spreads" else offer["point"]
    if not any(
        other["provider_key"] == offer["provider_key"]
        and other["odds_api_event_id"] == offer["odds_api_event_id"]
        and other["provider_start_date"] == offer["provider_start_date"]
        and other["side"] == opposite
        and other["point"] == other_point
        and other["provider_last_update"] == offer["provider_last_update"]
        and other["market_fetched_at"] == offer["market_fetched_at"]
        for other in paired
    ):
        return "unpaired_market"
    return None


def _consensus_total(game_offers):
    """Median posted total across a game's offers, or NaN without one."""
    if game_offers.empty or "market" not in game_offers:
        return float("nan")
    points = pd.to_numeric(
        game_offers.loc[game_offers["market"].eq("totals"), "point"], errors="coerce"
    ).dropna()
    return float(points.median()) if len(points) else float("nan")


def build_recommendations(projections, offers, *, decision_at=None):
    """One best eligible side per game and market, or an explicit No Play.

    Sides use the market-informed margin and totals the model total blended
    toward the median posted total, so every edge is measured after shrinking
    toward the market being bet into. Each pick records that market total.
    Moneylines are priced from the consensus market margin moved
    H2H_MODEL_WEIGHT toward the pure model. Everything is priced with the
    empirical dispersion around the priced line, not the pure model's wider
    predictive spread. An offer qualifies when the priced line sits at least
    MIN_EDGE_POINTS beyond the price's break-even line and expected value is
    positive; the best offer per market is the one with the most points of
    edge, never the largest payout. If fewer than VOLUME_FLOOR picks qualify
    in the batch, the highest-edge positive-EV offers are promoted with
    FLOOR_REASON. Historical calibration is diagnostic and never gates
    forward recommendations or replaces their probabilities. Stakes are
    always one unit, with no compounding.
    """
    now = _timestamp(decision_at or datetime.now(timezone.utc))
    projections = projections.assign(h2h_home_margin=_h2h_home_margins(projections))
    groups = {game_id: group for game_id, group in offers.groupby("game_id")}
    rows = []
    for projection in projections.itertuples():
        start, forecast = (
            _timestamp(projection.start_date),
            _timestamp(projection.as_of),
        )
        reason = None
        if pd.isna(start) or pd.isna(forecast) or not forecast <= now < start:
            reason = "not_pregame"
        elif now - forecast > MAX_FORECAST_AGE:
            reason = "stale_forecast"
        elif any(
            pd.isna(getattr(projection, f"{side}_missing_input_count", None))
            # Preseason and weekly counts always include unavailable injury
            # data. Preserve that flag without making every game a No Play.
            # Any additional missing model input still blocks a pick.
            or not 0 <= getattr(projection, f"{side}_missing_input_count", -1) <= 1
            for side in ("home", "away")
        ):
            reason = "missing_model_inputs"
        game_offers = groups.get(projection.game_id, offers.iloc[:0]).dropna(
            subset=["price"]
        )
        candidates = priced_candidates(projection, game_offers)
        market_total = _consensus_total(game_offers)
        priced_projection = projection._replace(
            margin_sd=PRICING_MARGIN_SD, total_sd=PRICING_TOTAL_SD
        )
        if np.isfinite(market_total) and np.isfinite(projection.model_total):
            priced_projection = priced_projection._replace(
                model_total=(1.0 - DEFAULT_MARKET_WEIGHT) * projection.model_total
                + DEFAULT_MARKET_WEIGHT * market_total
            )
        for market in MARKETS:
            row = {
                key: getattr(projection, key, None)
                for key in (
                    "game_id",
                    "season",
                    "week",
                    "start_date",
                    "home_team",
                    "away_team",
                    "model_version",
                    "home_missing_input_count",
                    "away_missing_input_count",
                    "model_total",
                    "degrees_of_freedom",
                )
            }
            row.update(
                market=market,
                forecast_as_of=forecast,
                decision_at=now,
                policy_version=POLICY_VERSION,
                status="no_play",
                reason=reason,
                stake_units=0.0,
                # The margin this market was actually priced from.
                model_home_margin=(
                    priced_projection.h2h_home_margin
                    if market == "h2h"
                    else projection.market_informed_home_margin
                ),
                model_total=priced_projection.model_total,
                market_total=market_total if np.isfinite(market_total) else None,
                margin_sd=priced_projection.margin_sd,
                total_sd=priced_projection.total_sd,
            )
            priced = [c for c in candidates if c["market"] == market]
            empty_reason = reason or "no_valid_price"
            if market == "h2h" and not np.isfinite(priced_projection.h2h_home_margin):
                priced = []
                empty_reason = reason or "missing_market_spread"
                row["model_home_margin"] = projection.market_informed_home_margin
            evaluated = []
            for candidate in priced:
                if candidate["market"] != "h2h" and candidate["point"] * 2 != round(
                    candidate["point"] * 2
                ):
                    continue
                win, push, loss = _probabilities(priced_projection, candidate)
                profit = _american_profit(candidate["price"])
                implied = 1 / (profit + 1)
                edge = win / (win + loss) - implied
                ev = win * profit - loss
                points = _edge_points(priced_projection, candidate, implied)
                block = reason or _offer_reason(candidate, priced, now, start)
                if (
                    not np.isfinite([win, push, loss, edge, ev, points]).all()
                    or not 0 < win < 1
                    or loss <= 0
                ):
                    block = "invalid_probability"
                if block is None and (points < MIN_EDGE_POINTS or ev <= 0):
                    block = "below_edge_threshold"
                evaluated.append((block, points, candidate, win, push, edge, ev))
            # A blocked offer must never hide a qualifying offer with less edge.
            if evaluated:
                block, points, best, win, push, edge, ev = max(
                    evaluated,
                    key=lambda item: (
                        item[0] is None,
                        item[1],
                        str(item[2]["provider_key"]),
                    ),
                )
                for key in (
                    "selection",
                    "side",
                    "point",
                    "price",
                    "provider",
                    "provider_key",
                    "market_fetched_at",
                    "odds_api_event_id",
                    "provider_start_date",
                    "provider_last_update",
                    "match_score",
                ):
                    row[key] = best[key]
                row.update(
                    win_probability=win,
                    push_probability=push,
                    probability_edge=edge,
                    edge_points=points,
                    expected_value_per_unit=ev,
                    status="recommended" if block is None else "no_play",
                    reason=block or "qualifying_edge",
                    stake_units=1.0 if block is None else 0.0,
                )
            else:
                row["reason"] = empty_reason
            rows.append(row)
    decisions = pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)
    shortfall = VOLUME_FLOOR - int(decisions["status"].eq("recommended").sum())
    if shortfall > 0:
        eligible = decisions[
            decisions["reason"].eq("below_edge_threshold")
            & decisions["edge_points"].gt(0)
            & decisions["expected_value_per_unit"].gt(0)
        ]
        promote = eligible.sort_values("edge_points", ascending=False).index[:shortfall]
        decisions.loc[promote, ["status", "reason", "stake_units"]] = [
            "recommended",
            FLOOR_REASON,
            1.0,
        ]
    return decisions


def grade_recommendations(recommendations, games, *, graded_at=None):
    """Settle recorded picks only; never infer a past recommendation from EV."""
    now = _timestamp(graded_at or datetime.now(timezone.utc))
    schedule = games.set_index("id")
    rows = []
    for pick in recommendations.itertuples():
        if pick.outcome != "pending" or pick.game_id not in schedule.index:
            continue
        game = schedule.loc[pick.game_id]
        if game["home_team"] != pick.home_team or game["away_team"] != pick.away_team:
            raise ValueError(f"game {pick.game_id}: schedule identity changed")
        current_start = _timestamp(game["start_date"])
        if pd.isna(current_start):
            continue
        result = None
        # A changed kickoff voids the original price contract. Do not reuse a
        # recommendation on a postponed or rescheduled fixture.
        if current_start != _timestamp(pick.start_date):
            result = "void"
        elif (
            now < _timestamp(pick.start_date)
            or pd.isna(game["completed"])
            or not bool(game["completed"])
            or pd.isna(game["home_points"])
            or pd.isna(game["away_points"])
        ):
            continue
        if pick.status != "recommended":
            if now < _timestamp(pick.start_date):
                continue
            result = "no_play"
        if result is None:
            scores = [game["home_points"], game["away_points"]]
            if any(not np.isfinite(s) or s < 0 or s != int(s) for s in scores):
                raise ValueError(f"game {pick.game_id}: invalid final score")
            margin = float(game["home_points"] - game["away_points"])
            total = float(game["home_points"] + game["away_points"])
            balance = (
                (margin if pick.side == "home" else -margin)
                if pick.market == "h2h"
                else (margin if pick.side == "home" else -margin) + pick.point
                if pick.market == "spreads"
                else (total - pick.point) * (1 if pick.side == "over" else -1)
            )
            result = "win" if balance > 0 else "loss" if balance < 0 else "push"
            if pick.market == "h2h" and balance == 0:
                result = "void"
        profit = (
            pick.stake_units * _american_profit(pick.price)
            if result == "win"
            else -pick.stake_units
            if result == "loss"
            else 0.0
        )
        rows.append(
            dict(
                game_id=pick.game_id,
                market=pick.market,
                decision_at=pick.decision_at,
                outcome=result,
                home_points=game["home_points"],
                away_points=game["away_points"],
                profit_units=profit,
                graded_at=now,
            )
        )
    return pd.DataFrame(rows, columns=SETTLEMENT_COLUMNS)
