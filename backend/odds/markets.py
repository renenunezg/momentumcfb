import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def _normalized_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore")
    return re.sub(r"[^a-z0-9]", "", ascii_value.decode().lower())


def _name_score(left: str, right: str) -> float:
    normalized_left = _normalized_name(left)
    normalized_right = _normalized_name(right)
    if normalized_left.startswith(normalized_right) or normalized_right.startswith(
        normalized_left
    ):
        return 1.0
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def match_event(
    commence_time, home_team: str, away_team: str, schedule: pd.DataFrame
) -> tuple[int | None, float]:
    commence = pd.to_datetime(commence_time, utc=True)
    time_difference = (
        pd.to_datetime(schedule["start_date"], utc=True) - commence
    ).abs()
    candidates = schedule[time_difference.le(pd.Timedelta(hours=2))].copy()
    if candidates.empty:
        return None, 0.0
    candidates["match_score"] = candidates.apply(
        lambda game: 0.5
        * (
            _name_score(home_team, game.home_team)
            + _name_score(away_team, game.away_team)
        ),
        axis=1,
    )
    best = candidates.sort_values("match_score", ascending=False).iloc[0]
    if float(best.match_score) < 0.65:
        return None, float(best.match_score)
    return int(best.game_id), float(best.match_score)


def _stored_sequence(value) -> list | np.ndarray:
    """Nested payloads read back from parquet arrive as numpy arrays, whose
    truthiness raises once they hold more than one element."""
    return value if isinstance(value, (list, np.ndarray)) else []


def flatten_odds_api_offers(
    events: pd.DataFrame, schedule: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    offer_columns = [
        "game_id",
        "odds_api_event_id",
        "provider_key",
        "provider",
        "market",
        "selection",
        "point",
        "price",
        "provider_last_update",
        "event_link",
        "market_link",
        "bet_link",
        "execution_eligibility_verified",
        "market_fetched_at",
        "match_score",
    ]
    offers = []
    matches = []
    if events.empty:
        return pd.DataFrame(columns=offer_columns), pd.DataFrame()

    for event in events.itertuples():
        game_id, match_score = match_event(
            event.commence_time, event.home_team, event.away_team, schedule
        )
        matches.append(
            {
                "odds_api_event_id": event.id,
                "commence_time": event.commence_time,
                "odds_home_team": event.home_team,
                "odds_away_team": event.away_team,
                "game_id": game_id,
                "match_score": match_score,
                "matched": game_id is not None,
            }
        )
        if game_id is None:
            continue
        for bookmaker in _stored_sequence(event.bookmakers):
            for market in _stored_sequence(bookmaker.get("markets")):
                market_key = market.get("key")
                if market_key not in {"spreads", "totals"}:
                    continue
                for outcome in _stored_sequence(market.get("outcomes")):
                    if market_key == "spreads":
                        if outcome.get("name") == event.home_team:
                            selection = "home"
                        elif outcome.get("name") == event.away_team:
                            selection = "away"
                        else:
                            continue
                    else:
                        selection = str(outcome.get("name", "")).lower()
                        if selection not in {"over", "under"}:
                            continue
                    offers.append(
                        {
                            "game_id": game_id,
                            "odds_api_event_id": event.id,
                            "provider_key": bookmaker.get("key"),
                            "provider": bookmaker.get("title"),
                            "market": market_key,
                            "selection": selection,
                            "point": outcome.get("point"),
                            "price": outcome.get("price"),
                            "provider_last_update": market.get("last_update")
                            or bookmaker.get("last_update"),
                            "event_link": bookmaker.get("link"),
                            "market_link": market.get("link"),
                            "bet_link": outcome.get("link"),
                            "execution_eligibility_verified": bool(
                                event.execution_eligibility_verified
                            ),
                            "market_fetched_at": event.source_fetched_at,
                            "match_score": match_score,
                        }
                    )
    frame = pd.DataFrame(offers, columns=offer_columns)
    for column in ("point", "price"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, pd.DataFrame(matches)


def _american_profit(price: float) -> float:
    return price / 100.0 if price > 0 else 100.0 / abs(price)


def _fair_american(probability: float) -> float:
    probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
    if probability >= 0.5:
        return -100.0 * probability / (1.0 - probability)
    return 100.0 * (1.0 - probability) / probability


def compare_priced_offers(
    projections: pd.DataFrame, offers: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for projection in projections.itertuples():
        game_offers = offers[offers["game_id"].eq(projection.game_id)].dropna(
            subset=["point", "price"]
        )
        eligible = game_offers[
            game_offers["execution_eligibility_verified"].astype(bool)
        ]
        executable = not eligible.empty
        candidate_offers = eligible if executable else game_offers
        scale_by_market = {
            "spreads": projection.margin_sd
            * np.sqrt(
                (projection.degrees_of_freedom - 2.0)
                / projection.degrees_of_freedom
            ),
            "totals": projection.total_sd
            * np.sqrt(
                (projection.degrees_of_freedom - 2.0)
                / projection.degrees_of_freedom
            ),
        }
        candidates = []
        for offer in candidate_offers.itertuples():
            if offer.price == 0:
                continue
            if offer.market == "spreads" and offer.selection == "home":
                edge = projection.home_margin + offer.point
                selection = projection.home_team
            elif offer.market == "spreads" and offer.selection == "away":
                edge = -projection.home_margin + offer.point
                selection = projection.away_team
            elif offer.market == "totals" and offer.selection == "over":
                edge = projection.model_total - offer.point
                selection = "Over"
            elif offer.market == "totals" and offer.selection == "under":
                edge = offer.point - projection.model_total
                selection = "Under"
            else:
                continue
            uncertainty = (
                projection.margin_sd
                if offer.market == "spreads"
                else projection.total_sd
            )
            probability = float(
                student_t.cdf(
                    edge / scale_by_market[offer.market],
                    projection.degrees_of_freedom,
                )
            )
            expected_value = (
                probability * _american_profit(float(offer.price))
                - (1.0 - probability)
            )
            candidates.append(
                {
                    "market": offer.market,
                    "selection": selection,
                    "point": float(offer.point),
                    "price": float(offer.price),
                    "provider": offer.provider,
                    "provider_key": offer.provider_key,
                    "provider_last_update": offer.provider_last_update,
                    "event_link": offer.event_link,
                    "market_link": offer.market_link,
                    "bet_link": offer.bet_link,
                    "edge_points": edge,
                    "edge_standardized": edge / uncertainty,
                    "model_cover_probability": probability,
                    "model_fair_price": _fair_american(probability),
                    "expected_value_per_unit": expected_value,
                }
            )
        row = {
            "game_id": projection.game_id,
            "start_date": projection.start_date,
            "away_team": projection.away_team,
            "home_team": projection.home_team,
            "model_home_spread": projection.home_spread,
            "model_total": projection.model_total,
            "margin_sd": projection.margin_sd,
            "total_sd": projection.total_sd,
            "model_as_of": projection.as_of,
            "market_available": not game_offers.empty,
            "priced_offer_available": bool(candidates),
            "executable_offer_available": executable,
        }
        if candidates:
            best = max(candidates, key=lambda item: item["expected_value_per_unit"])
            row.update({f"best_offer_{key}": value for key, value in best.items()})
            row["review_status"] = (
                "requires_current_source_review"
                if best["edge_points"] >= 4.0
                and best["expected_value_per_unit"] > 0
                else "below_material_review_threshold"
            )
        else:
            row["review_status"] = "no_priced_offer"
        row["recommendation_status"] = "not_recommended"
        rows.append(row)
    comparisons = pd.DataFrame(rows)
    if "best_offer_expected_value_per_unit" not in comparisons:
        comparisons["best_offer_expected_value_per_unit"] = np.nan
    return comparisons.sort_values(
        "best_offer_expected_value_per_unit",
        ascending=False,
        na_position="last",
        ignore_index=True,
    )
