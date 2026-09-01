import pandas as pd

from backend.model.market_blend import add_market_informed_margins
from backend.model.preseason import _flatten_cfbd_offers
from backend.odds.markets import compare_priced_offers
from backend.publish import MARKET_COMPARISONS_COLUMNS, _serving_frame


def test_cfbd_line_comparisons_fill_the_published_contract():
    lines = pd.DataFrame(
        [
            {
                "id": 401,
                "home_team": "Home",
                "away_team": "Away",
                "source_fetched_at": "2026-08-27T12:00:00+00:00",
                "lines": [
                    {"provider": "Book A", "spread": -3.5, "overUnder": 51.5},
                    {"provider": "Book B", "spread": -4.0, "overUnder": None},
                ],
            }
        ]
    )
    projections = pd.DataFrame(
        [
            {
                "game_id": 401,
                "start_date": "2026-08-29T16:00:00+00:00",
                "home_team": "Home",
                "away_team": "Away",
                "home_margin": 6.0,
                "home_spread": -6.0,
                "model_total": 55.0,
                "margin_sd": 13.0,
                "total_sd": 13.0,
                "as_of": "2026-08-27T13:00:00+00:00",
                "degrees_of_freedom": 8.0,
            }
        ]
    )

    offers = _flatten_cfbd_offers(lines)
    blended = add_market_informed_margins(projections, offers)
    published = _serving_frame(
        compare_priced_offers(blended, offers), MARKET_COMPARISONS_COLUMNS
    )
    row = published.iloc[0]

    assert blended["market_home_spread"].iloc[0] == -3.75
    assert row["market_available"] is True
    assert row["priced_offer_available"] is False
    assert row["review_status"] == "no_priced_offer"
    assert row["model_home_spread"] == -6.0
    assert row["best_offer_point"] is None
    assert row["best_offer_expected_value_per_unit"] is None
