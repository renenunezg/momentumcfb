"""Pregame decisions and settlement at the exact recommended line and price."""

from datetime import datetime, timezone
from math import ceil, floor

import numpy as np
import pandas as pd

from backend.model.distributions import marginal_cdf
from backend.odds.markets import _american_profit, priced_candidates

POLICY_VERSION = "cfb-picks-v4"
MIN_PROBABILITY_EDGE = 0.045
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
    "expected_value_per_unit",
    "stake_units",
    "model_home_margin",
    "model_total",
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


def _probabilities(projection, offer):
    """Round the frozen continuous marginal to integer scores, retaining pushes.

    Sides use the market-informed margin (pure model shrunk toward the
    pre-decision consensus spread); totals have no market blend and stay pure.
    Half-point lines keep the original CDF. Integer lines reserve the mass
    between the adjacent half points for a returned stake.
    """
    if offer["market"] == "h2h":
        mean = projection.market_informed_home_margin * (
            1 if offer["side"] == "home" else -1
        )
        win = float(
            marginal_cdf(mean, projection.margin_sd, projection.degrees_of_freedom)
        )
        return win, 0.0, 1.0 - win
    point = offer["point"]
    if offer["market"] == "spreads":
        mean = projection.market_informed_home_margin * (
            1 if offer["side"] == "home" else -1
        )
        threshold = -point
        sd = projection.margin_sd
    else:
        mean = projection.model_total
        threshold = point
        sd = projection.total_sd
    df = projection.degrees_of_freedom
    win = float(marginal_cdf(mean - floor(threshold) - 0.5, sd, df))
    loss = float(marginal_cdf(ceil(threshold) - 0.5 - mean, sd, df))
    if offer["side"] == "under":
        win, loss = loss, win
    return win, max(0.0, 1.0 - win - loss), loss


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


def build_recommendations(projections, offers, *, decision_at=None):
    """One best eligible side per game and market, or an explicit No Play.

    Sides use the market-informed margin so an edge is measured after shrinking
    toward the market being bet into; totals stay pure. Historical calibration
    is diagnostic and never gates forward recommendations or replaces their
    probabilities. The 4.5 percentage-point gate is a versioned starting
    policy, not a fit to live-season outcomes. Stakes are always one unit,
    with no compounding.
    """
    now = _timestamp(decision_at or datetime.now(timezone.utc))
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
                    "margin_sd",
                    "total_sd",
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
                model_home_margin=projection.market_informed_home_margin,
            )
            priced = [c for c in candidates if c["market"] == market]
            evaluated = []
            for candidate in priced:
                if candidate["market"] != "h2h" and candidate["point"] * 2 != round(
                    candidate["point"] * 2
                ):
                    continue
                win, push, loss = _probabilities(projection, candidate)
                profit = _american_profit(candidate["price"])
                edge = win / (win + loss) - 1 / (profit + 1)
                ev = win * profit - loss
                block = reason or _offer_reason(candidate, priced, now, start)
                if (
                    not np.isfinite([win, push, loss, edge, ev]).all()
                    or not 0 < win < 1
                    or loss <= 0
                ):
                    block = "invalid_probability"
                if block is None and (edge < MIN_PROBABILITY_EDGE or ev <= 0):
                    block = "below_edge_threshold"
                evaluated.append((block, ev, candidate, win, push, edge))
            # A blocked high EV offer must never hide a qualifying lower EV offer.
            if evaluated:
                block, ev, best, win, push, edge = max(
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
                    expected_value_per_unit=ev,
                    status="recommended" if block is None else "no_play",
                    reason=block or "qualifying_edge",
                    stake_units=1.0 if block is None else 0.0,
                )
            else:
                row["reason"] = reason or "no_valid_price"
            rows.append(row)
    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)


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
