"""Pregame decisions and settlement at the exact recommended line and price."""

from datetime import datetime, timezone
from math import ceil, floor

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import t

from backend.model.distributions import marginal_cdf, marginal_scale
from backend.model.market_blend import consensus_total
from backend.model.spread_pricing import (
    SPREAD_CALIBRATION_VERSION,
    SPREAD_METHOD,
    SPREAD_MODEL_WEIGHT,
    spread_margin,
    spread_win_loss,
)
from backend.odds.markets import _american_profit, priced_candidates

POLICY_VERSION = "cfb-picks-v11"
# Minimum points the priced line must sit beyond the offered price's
# break-even line. Measured in margin or total points, the same yardstick for
# favourites and underdogs, so a mispriced tail cannot clear the gate on one
# side only. A versioned starting policy, not a fit to live-season outcomes.
MIN_EDGE_POINTS = 2.0
# Retain the pricing scales from the 2021-2025 chronological replay (3,208
# games), separately from pure-model predictive uncertainty. The frozen
# spread key calibration uses this fixed margin scale and 500 degrees of
# freedom; neither scale is retuned on current-season outcomes.
PRICING_MARGIN_SD = 15.35
PRICING_TOTAL_SD = 15.93
# Preserve the chosen research policy's model contribution in margin space.
# Paired moneylines replace the spread as the market reference; the weight
# remains a research choice, not a claim of independently validated profit.
H2H_MODEL_WEIGHT = 0.20
MONEYLINE_METHOD = "paired_moneyline_margin_blend_v1"
# Research target per decision batch, not a mandatory minimum. Shortlist the
# highest-edge eligible spreads/totals first, then cap each market at half
# the shortfall (rounded up). Never replace capped picks with other offers.
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
    "decision_forecast",
    "pricing_context",
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
    "closing_point",
    "closing_price",
    "closing_source",
    "clv_points",
]
CLOSING_LINE_COLUMNS = [
    "closing_point",
    "closing_price",
    "closing_source",
    "clv_points",
]
CLOSING_SOURCE = "cfbd_lines_median"


def _timestamp(value):
    return pd.to_datetime(value, utc=True, errors="coerce")


def _win_loss(mean, sd, df, offer):
    """Win and loss probabilities for the bettor's side of an offer.

    ``mean`` is the priced home margin for sides or the priced total for
    totals. Integer lines reserve push mass; spread masses include the frozen
    football key-number correction.
    """
    home_side = offer["side"] in ("home", "over")
    point = offer["point"]
    if offer["market"] == "spreads":
        mean = mean if home_side else -mean
        threshold = -point
        return spread_win_loss(mean, sd, df, threshold)
    else:
        threshold = point
    win = float(marginal_cdf(mean - floor(threshold) - 0.5, sd, df))
    loss = float(marginal_cdf(ceil(threshold) - 0.5 - mean, sd, df))
    return (loss, win) if offer["side"] == "under" else (win, loss)


def _priced_mean_sd(projection, offer):
    if offer["market"] == "totals":
        return projection.model_total, projection.total_sd
    return projection.market_informed_home_margin, projection.margin_sd


def _probabilities(projection, offer):
    """Round the frozen continuous marginal to integer scores, retaining pushes.

    Spreads use the 20% model/history pricing margin with frozen key weights;
    totals retain the published total. Moneylines blend the model
    margin with the paired market's equivalent margin and have no push mass.
    """
    if offer["market"] == "h2h":
        home = projection.h2h_home_probability
        win = home if offer["side"] == "home" else 1.0 - home
        return win, 0.0, 1.0 - win
    mean, sd = _priced_mean_sd(projection, offer)
    win, loss = _win_loss(mean, sd, projection.degrees_of_freedom, offer)
    return win, max(0.0, 1.0 - win - loss), loss


def _edge_points(projection, offer, implied):
    """Points the priced line must move against the pick before its no-push
    win probability falls to the price's break-even probability.

    Positive favours the pick. The same shift is required of a favourite and
    an underdog, unlike a probability gap, which is largest where the density
    is highest and is then multiplied by the payout in expected value.
    """
    if offer["market"] == "h2h":
        # Preserve the two-point selection yardstick without deriving the
        # probability from a spread. These are probability-equivalent points.
        win, _, _ = _probabilities(projection, offer)
        df = projection.degrees_of_freedom
        scale = marginal_scale(projection.margin_sd, df)
        df = np.inf if pd.isna(df) else df
        return float(scale * (t.ppf(win, df) - t.ppf(implied, df)))
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
        or offer["match_score"] > 1
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


def _moneyline_reference(candidates, latest, now, start):
    """Pair prices before removing vig; give each bookmaker one vote.

    Only the latest capture is eligible for either reference or execution.
    Duplicate conflicting prices, mixed events, stale and unpaired offers
    cannot manufacture a reference or resurrect an older executable price.
    """
    offers = [c for c in candidates if c["market"] == "h2h"]
    offers = [c for c in offers if _timestamp(c["market_fetched_at"]) == latest]
    groups = {}
    for offer in offers:
        key = (offer["provider_key"], offer["odds_api_event_id"])
        groups.setdefault(key, []).append(offer)
    pairs, eligible = [], []
    for (provider_key, event_id), group in sorted(groups.items(), key=str):
        updated = max(
            (_timestamp(c["provider_last_update"]) for c in group), default=pd.NaT
        )
        group = [c for c in group if _timestamp(c["provider_last_update"]) == updated]
        sides = {
            side: [c for c in group if c["side"] == side] for side in ("home", "away")
        }
        if any(len({c["price"] for c in side}) != 1 for side in sides.values()):
            continue
        if any(_offer_reason(c, group, now, start) is not None for c in group):
            continue
        home, away = sides["home"][0], sides["away"][0]
        h = 1.0 / (1.0 + _american_profit(home["price"]))
        a = 1.0 / (1.0 + _american_profit(away["price"]))
        if not 1.0 <= h + a <= 1.20:
            continue
        pairs.append(
            dict(
                provider_key=provider_key,
                provider=home["provider"],
                odds_api_event_id=event_id,
                provider_start_date=start.isoformat(),
                provider_last_update=updated.isoformat(),
                market_fetched_at=latest.isoformat(),
                home_price=home["price"],
                away_price=away["price"],
                home_match_score=float(home["match_score"]),
                away_match_score=float(away["match_score"]),
                overround=h + a,
                home_probability=h / (h + a),
            )
        )
        eligible.extend(group)
    if len({p["odds_api_event_id"] for p in pairs}) > 1:
        pairs, eligible = [], []
    probability = (
        float(np.median([p["home_probability"] for p in pairs])) if pairs else None
    )
    return probability, pairs, eligible


def build_recommendations(projections, offers, *, decision_at=None, games=None):
    """One best eligible side per game and market, or an explicit No Play.

    Spreads use a separate 20% model/history pricing margin; totals retain the
    published market-informed total. A late QB report creates a separately recorded
    decision forecast without rewriting the published original. Each
    pick records the median posted total at decision time.
    Moneylines move the paired market's equivalent margin H2H_MODEL_WEIGHT
    toward the pure model margin before converting back to win probability.
    Spreads and totals are priced with the
    empirical dispersion around the priced line, not the pure model's wider
    predictive spread. An offer qualifies when the priced line sits at least
    MIN_EDGE_POINTS beyond the price's break-even line and expected value is
    positive; the best offer per market is the one with the most points of
    edge, never the largest payout. If fewer than VOLUME_FLOOR picks qualify
    in the batch, shortlist the highest-edge positive-EV spreads/totals, then
    cap each market at ceil(shortfall / 2), with no backfilling. Promotions use
    FLOOR_REASON. Frozen development calibration prices spreads but never
    gates forward recommendations on realized results. Stakes are
    always one unit, with no compounding.
    """
    now = _timestamp(decision_at or datetime.now(timezone.utc))
    game_types = (
        games.set_index("id", verify_integrity=True)["season_type"].to_dict()
        if games is not None and "season_type" in games
        else {}
    )
    projections = projections.assign(h2h_home_probability=np.nan)
    groups = {game_id: group for game_id, group in offers.groupby("game_id")}
    rows = []
    for projection in projections.itertuples():
        season_type = getattr(projection, "season_type", None)
        season_type_source = "forecast"
        if not isinstance(season_type, str) or season_type not in (
            "regular",
            "postseason",
        ):
            season_type = game_types.get(projection.game_id)
            season_type_source = "schedule"
        if not isinstance(season_type, str) or season_type not in (
            "regular",
            "postseason",
        ):
            season_type, season_type_source = None, "unavailable"
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
        game_offers = groups.get(projection.game_id, offers.iloc[:0])
        # Read the latest capture before dropping invalid prices: a broken new
        # snapshot must not resurrect a previously available moneyline.
        latest_moneyline_capture = pd.to_datetime(
            game_offers.loc[game_offers.market.eq("h2h"), "market_fetched_at"],
            utc=True,
            errors="coerce",
        ).max()
        game_offers = game_offers.dropna(subset=["price"])
        candidates = priced_candidates(projection, game_offers)
        market_probability, reference_pairs, moneyline_offers = _moneyline_reference(
            candidates, latest_moneyline_capture, now, start
        )
        pure_margin = getattr(projection, "pure_home_margin", projection.home_margin)
        pure_margin = (
            float(pure_margin)
            if pd.notna(pure_margin) and np.isfinite(pure_margin)
            else None
        )
        market_margin = pricing_margin = home_probability = None
        if (
            market_probability is not None
            and 0 < market_probability < 1
            and pure_margin is not None
        ):
            df = projection.degrees_of_freedom
            scale = marginal_scale(PRICING_MARGIN_SD, df)
            market_margin = float(
                scale * t.ppf(market_probability, np.inf if pd.isna(df) else df)
            )
            pricing_margin = market_margin + H2H_MODEL_WEIGHT * (
                pure_margin - market_margin
            )
            home_probability = float(
                marginal_cdf(pricing_margin, PRICING_MARGIN_SD, df)
            )
        # The supplied forecast is either the published original or the
        # recorded availability-adjusted decision forecast. The
        # median posted total at decision time is recorded for reference.
        market_total = consensus_total(game_offers)
        priced_projection = projection._replace(
            margin_sd=PRICING_MARGIN_SD,
            total_sd=PRICING_TOTAL_SD,
            h2h_home_probability=home_probability,
        )
        published_total = getattr(projection, "market_informed_total", np.nan)
        if pd.notna(published_total) and np.isfinite(published_total):
            priced_projection = priced_projection._replace(model_total=published_total)
        spread_inputs = spread_margin(projection)
        for market in MARKETS:
            market_projection = priced_projection
            if market == "spreads" and spread_inputs is not None:
                market_projection = priced_projection._replace(
                    market_informed_home_margin=spread_inputs[0],
                    degrees_of_freedom=500.0,
                )
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
                decision_forecast=getattr(projection, "decision_forecast", None),
                pricing_context=dict(
                    version=1,
                    purpose="research",
                    validation_status="unvalidated",
                    season_type=season_type,
                    season_type_source=season_type_source,
                    is_postseason=season_type == "postseason" if season_type else None,
                    research_flags=(
                        ["postseason"]
                        if season_type == "postseason"
                        else ["season_type_unknown"]
                        if season_type is None
                        else []
                    ),
                    probability_source=(
                        MONEYLINE_METHOD
                        if market == "h2h"
                        else SPREAD_METHOD
                        if market == "spreads"
                        else "market_informed_forecast"
                    ),
                    **(
                        dict(
                            model_weight=H2H_MODEL_WEIGHT,
                            blend_space="margin",
                            model_home_margin=pure_margin,
                            market_home_probability=market_probability,
                            market_equivalent_home_margin=market_margin,
                            pricing_home_margin=pricing_margin,
                            home_probability=home_probability,
                            reference_pairs=reference_pairs,
                            edge_units="probability_equivalent_points",
                        )
                        if market == "h2h"
                        else {}
                    ),
                ),
                decision_at=now,
                policy_version=POLICY_VERSION,
                status="no_play",
                reason=reason,
                stake_units=0.0,
                # Forecast context only for H2H; its pricing source is explicit above.
                model_home_margin=projection.market_informed_home_margin,
                model_total=priced_projection.model_total,
                market_total=market_total if np.isfinite(market_total) else None,
                margin_sd=priced_projection.margin_sd,
                total_sd=priced_projection.total_sd,
            )
            if market == "spreads" and spread_inputs is not None:
                row["model_home_margin"] = spread_inputs[0]
                row["degrees_of_freedom"] = 500.0
                row["pricing_context"].update(
                    model_weight=SPREAD_MODEL_WEIGHT,
                    model_side_home_margin=spread_inputs[1],
                    market_home_margin=spread_inputs[2],
                    pricing_home_margin=spread_inputs[0],
                    published_home_margin=projection.market_informed_home_margin,
                    reference_as_of=forecast.isoformat(),
                    calibration_version=SPREAD_CALIBRATION_VERSION,
                    blend_space="margin",
                )
            priced = [c for c in candidates if c["market"] == market]
            empty_reason = reason or "no_valid_price"
            if market == "spreads" and spread_inputs is None:
                priced = []
                empty_reason = reason or "missing_spread_reference"
            if market == "h2h" and home_probability is None:
                priced = []
                empty_reason = reason or (
                    "missing_moneyline_reference"
                    if market_probability is None
                    else "invalid_probability"
                )
            evaluated = []
            for candidate in priced:
                if candidate["market"] != "h2h" and candidate["point"] * 2 != round(
                    candidate["point"] * 2
                ):
                    continue
                win, push, loss = _probabilities(market_projection, candidate)
                profit = _american_profit(candidate["price"])
                implied = 1 / (profit + 1)
                edge = win / (win + loss) - implied
                ev = win * profit - loss
                points = _edge_points(market_projection, candidate, implied)
                block = reason or _offer_reason(candidate, priced, now, start)
                if market == "h2h" and not any(
                    candidate is c for c in moneyline_offers
                ):
                    block = reason or "invalid_moneyline_pair"
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
                if market == "spreads" and spread_inputs is not None:
                    direction = 1.0 if best["side"] == "home" else -1.0
                    model_shift = direction * (spread_inputs[0] - spread_inputs[2])
                    quote_gap = direction * spread_inputs[2] + best["point"]
                    raw_edge = model_shift + quote_gap
                    row["pricing_context"]["spread_edge_breakdown"] = dict(
                        perspective="selected_side",
                        model_disagreement_points=direction
                        * (spread_inputs[1] - spread_inputs[2]),
                        model_contribution_points=model_shift,
                        quote_difference_points=quote_gap,
                        raw_line_edge_points=raw_edge,
                        price_distribution_adjustment_points=points - raw_edge,
                        edge_points=points,
                    )
                    if abs(quote_gap) > 1e-9:
                        row["pricing_context"]["research_flags"].append(
                            "spread_quote_differs_from_reference"
                        )
            else:
                row["reason"] = empty_reason
            rows.append(row)
    decisions = pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)
    shortfall = max(0, VOLUME_FLOOR - int(decisions["status"].eq("recommended").sum()))
    market_cap = (shortfall + 1) // 2
    shortlisted, capped = set(), set()
    if shortfall > 0:
        eligible = decisions[
            decisions["reason"].eq("below_edge_threshold")
            & decisions["market"].ne("h2h")
            & decisions["edge_points"].gt(0)
            & decisions["expected_value_per_unit"].gt(0)
        ]
        shortlist = eligible.sort_values(
            ["edge_points", "game_id", "market"], ascending=[False, True, True]
        ).head(shortfall)
        promote = shortlist.groupby("market", sort=False).head(market_cap).index
        shortlisted = set(shortlist.index)
        capped = shortlisted - set(promote)
        decisions.loc[promote, ["status", "reason", "stake_units"]] = [
            "recommended",
            FLOOR_REASON,
            1.0,
        ]
    exposure = decisions.groupby("game_id").stake_units.sum()
    for row in decisions.itertuples():
        row.pricing_context.update(
            floor_policy="shortlist_market_soft_cap_v1",
            floor_target=VOLUME_FLOOR,
            floor_shortfall=shortfall,
            floor_market_cap=market_cap,
            floor_shortlisted=row.Index in shortlisted,
            floor_capped=row.Index in capped,
            floor_backfill=False,
            recommendation_tier=(
                "exploratory"
                if row.reason == FLOOR_REASON
                else "edge_qualified"
                if row.status == "recommended"
                else "no_play"
            ),
            exposure_group=str(row.game_id),
            decision_batch_game_stake_units=float(exposure.loc[row.game_id]),
            stake_method="flat_one_unit_per_pick",
        )
    return decisions


def closing_line_value(pick, closing):
    """The closing number for the pick's side and the points it beat the close.

    ``closing`` is indexed by game_id with the CFBD median closing_spread,
    closing_total and closing_home_moneyline/closing_away_moneyline. Positive
    clv_points means the market moved toward the pick after the decision.
    Spreads and totals compare lines; a moneyline records only the closing
    price for its side because its value lives in the price.
    """
    empty = dict.fromkeys(CLOSING_LINE_COLUMNS)
    if closing is None or pick.game_id not in closing.index:
        return empty
    row = closing.loc[pick.game_id]
    if pick.market == "h2h":
        price = row.get(f"closing_{pick.side}_moneyline")
        if price is None or pd.isna(price):
            return empty
        return dict(empty, closing_price=float(price), closing_source=CLOSING_SOURCE)
    if pick.market == "spreads":
        spread = row.get("closing_spread")
        if spread is None or pd.isna(spread):
            return empty
        point = float(spread) if pick.side == "home" else -float(spread)
        return dict(
            empty,
            closing_point=point,
            closing_source=CLOSING_SOURCE,
            clv_points=float(pick.point) - point,
        )
    total = row.get("closing_total")
    if total is None or pd.isna(total):
        return empty
    direction = 1.0 if pick.side == "over" else -1.0
    return dict(
        empty,
        closing_point=float(total),
        closing_source=CLOSING_SOURCE,
        clv_points=direction * (float(total) - float(pick.point)),
    )


def closing_line_backfill(recommendations, closing):
    """Closing numbers for settled picks that were graded before closing
    lines were recorded. Rows carry only the closing fields."""
    settled = recommendations[
        recommendations["status"].eq("recommended")
        & recommendations["outcome"].isin(["win", "loss", "push"])
        & recommendations["closing_source"].isna()
    ]
    rows = [
        dict(
            game_id=pick.game_id,
            market=pick.market,
            decision_at=pick.decision_at,
            **closing_line_value(pick, closing),
        )
        for pick in settled.itertuples()
    ]
    frame = pd.DataFrame(
        rows, columns=["game_id", "market", "decision_at", *CLOSING_LINE_COLUMNS]
    )
    return frame[frame["closing_source"].notna()].reset_index(drop=True)


def grade_recommendations(recommendations, games, closing=None, *, graded_at=None):
    """Settle recorded picks only; never infer a past recommendation from EV.

    ``closing`` optionally supplies the CFBD median closing lines so every
    settled pick records its closing line value alongside the outcome.
    """
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
                **(
                    closing_line_value(pick, closing)
                    if result in ("win", "loss", "push")
                    else dict.fromkeys(CLOSING_LINE_COLUMNS)
                ),
            )
        )
    return pd.DataFrame(rows, columns=SETTLEMENT_COLUMNS)
