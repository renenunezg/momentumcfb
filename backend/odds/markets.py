import json
import re
import unicodedata
from difflib import SequenceMatcher
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from backend.model.distributions import marginal_cdf

TEAM_NAME_ALIASES = {
    "liusharks": "longislanduniversity",
    "umassminutemen": "massachusetts",
    "albany": "ualbany",
    "youngstownstpenguins": "youngstownstate",
    "appalachianstatemountaineers": "appstate",
    "citadelbulldogs": "thecitadel",
    "houstonbaptisthuskies": "houstonchristian",
    "southeasternlouisianalions": "selouisiana",
    "southernmississippigoldeneagles": "southernmiss",
    "nichollsstatecolonels": "nicholls",
    "samhoustonstatebearkats": "samhouston",
}


@cache
def _team_labels():
    # Exact FBS/FCS school-plus-mascot labels from the CFBD team catalogue.
    # Unknown provider names cannot earn confidence from a shared prefix.
    return json.loads(Path(__file__).with_name("team_labels.json").read_text())


def _normalized_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore")
    normalized = re.sub(r"[^a-z0-9]", "", ascii_value.decode().lower())
    return TEAM_NAME_ALIASES.get(normalized, _team_labels().get(normalized, normalized))


def _name_score(left: str, right: str) -> float:
    normalized_left = _normalized_name(left)
    normalized_right = _normalized_name(right)
    if normalized_left and normalized_left == normalized_right:
        return 1.0
    # Approximate matches remain diagnostic evidence, not recommendation proof.
    return min(0.94, SequenceMatcher(None, normalized_left, normalized_right).ratio())


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
        lambda game: min(
            _name_score(home_team, game.home_team),
            _name_score(away_team, game.away_team),
        ),
        axis=1,
    )
    best = candidates.sort_values("match_score", ascending=False).iloc[0]
    if float(best.match_score) < 0.95:
        return None, float(best.match_score)
    if candidates["match_score"].eq(best.match_score).sum() != 1:
        return None, float(best.match_score)
    return int(best.game_id), float(best.match_score)


def _stored_sequence(value) -> list | np.ndarray:
    """Nested payloads read back from parquet arrive as numpy arrays, whose
    truthiness raises once they hold more than one element."""
    return value if isinstance(value, (list, np.ndarray)) else []


OFFER_COLUMNS = [
    "game_id",
    "odds_api_event_id",
    "commence_time",
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


def offer_selection(
    market_key: str, outcome_name: str | None, home_team: str, away_team: str
) -> str | None:
    """Map an Odds API outcome onto home/away or over/under, or None to skip."""
    if market_key == "spreads":
        if outcome_name == home_team:
            return "home"
        if outcome_name == away_team:
            return "away"
        return None
    if market_key == "totals":
        selection = str(outcome_name or "").lower()
        return selection if selection in {"over", "under"} else None
    return None


def flatten_odds_api_offers(
    events: pd.DataFrame, schedule: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    offers = []
    matches = []
    if events.empty:
        return pd.DataFrame(columns=OFFER_COLUMNS), pd.DataFrame()

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
                for outcome in _stored_sequence(market.get("outcomes")):
                    selection = offer_selection(
                        market_key,
                        outcome.get("name"),
                        event.home_team,
                        event.away_team,
                    )
                    if selection is None:
                        continue
                    offers.append(
                        {
                            "game_id": game_id,
                            "odds_api_event_id": event.id,
                            "commence_time": event.commence_time,
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
    frame = pd.DataFrame(offers, columns=OFFER_COLUMNS)
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


def priced_candidates(projection, game_offers: pd.DataFrame) -> list[dict]:
    candidates = []
    for offer in game_offers.itertuples():
        if not np.isfinite(offer.price) or abs(offer.price) < 100:
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
            projection.margin_sd if offer.market == "spreads" else projection.total_sd
        )
        if not np.isfinite(uncertainty) or uncertainty <= 0 or not np.isfinite(edge):
            continue
        probability = float(
            marginal_cdf(
                edge,
                uncertainty,
                projection.degrees_of_freedom,
            )
        )
        expected_value = probability * _american_profit(float(offer.price)) - (
            1.0 - probability
        )
        candidates.append(
            {
                "market": offer.market,
                "side": offer.selection,
                "market_fetched_at": offer.market_fetched_at,
                "odds_api_event_id": getattr(offer, "odds_api_event_id", None),
                "provider_start_date": getattr(offer, "commence_time", None),
                "execution_eligibility_verified": offer.execution_eligibility_verified,
                "match_score": offer.match_score,
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
    return candidates


def compare_priced_offers(
    projections: pd.DataFrame, offers: pd.DataFrame
) -> pd.DataFrame:
    from backend.recommendations import build_recommendations

    decisions = build_recommendations(projections, offers)
    recommended = {
        game_id: group.sort_values("expected_value_per_unit", ascending=False).iloc[0]
        for game_id, group in decisions[decisions["status"].eq("recommended")].groupby(
            "game_id"
        )
    }
    rows = []
    offer_groups = {game_id: group for game_id, group in offers.groupby("game_id")}
    for projection in projections.itertuples():
        game_offers = offer_groups.get(projection.game_id, offers.iloc[:0]).dropna(
            subset=["point"]
        )
        priced = game_offers.dropna(subset=["price"])
        eligible = priced[priced["execution_eligibility_verified"].eq(True)]
        executable = not eligible.empty
        candidates = priced_candidates(projection, eligible if executable else priced)
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
        decision = recommended.get(projection.game_id)
        if candidates:
            best = max(candidates, key=lambda item: item["expected_value_per_unit"])
            if decision is not None:
                matching = [
                    c
                    for c in candidates
                    if c["market"] == decision.market
                    and c["side"] == decision.side
                    and c["point"] == decision.point
                    and c["price"] == decision.price
                    and c["provider_key"] == decision.provider_key
                ]
                if matching:
                    best = matching[0].copy()
                    best["expected_value_per_unit"] = decision.expected_value_per_unit
                    best["model_cover_probability"] = decision.win_probability
                    best["model_fair_price"] = _fair_american(
                        decision.win_probability / (1 - decision.push_probability)
                    )
                else:
                    decision = None
            row.update(
                {
                    f"best_offer_{key}": value
                    for key, value in best.items()
                    if key
                    not in {
                        "side",
                        "market_fetched_at",
                        "execution_eligibility_verified",
                        "match_score",
                        "odds_api_event_id",
                        "provider_start_date",
                    }
                }
            )
            row["review_status"] = (
                "requires_current_source_review"
                if best["edge_points"] >= 4.0 and best["expected_value_per_unit"] > 0
                else "below_material_review_threshold"
            )
        else:
            row["review_status"] = "no_priced_offer"
        row["recommendation_status"] = (
            "recommended" if decision is not None else "not_recommended"
        )
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
