"""The cfb.closing_lines contract: one captured row per game, preferred by
grading and pick CLV over the labelled CFBD median fallback."""

from types import SimpleNamespace

import pandas as pd
import pytest

from backend import grading
from backend.recommendations import CLOSING_SOURCE, closing_line_value
from backend.serving.closing_lines import (
    CLOSING_LINES_COLUMNS,
    ESPN_SOURCE,
    ODDS_API_SOURCE,
    build_closing_lines,
    closing_source,
)

KICKOFF = pd.Timestamp("2026-08-29T16:00:00Z")
CLOSE_AT = KICKOFF - pd.Timedelta(minutes=1)


def _offer(provider, market, selection, point, price, *, seconds_before, **extra):
    return {
        "game_id": 101,
        "snapshot_id": "close",
        "fetched_at": CLOSE_AT,
        "phase": "pregame",
        "market": market,
        "selection": selection,
        "provider_key": provider,
        "provider_last_update": KICKOFF - pd.Timedelta(seconds=seconds_before),
        "point": point,
        "price": price,
        **extra,
    }


def _anchor(spread):
    return pd.DataFrame(
        {
            "game_id": [101],
            "closing_spread": [spread],
            "closing_snapshot_id": ["close"],
            "closing_fetched_at": [CLOSE_AT],
        }
    )


def test_median_row_takes_draftkings_prices_at_the_median_point():
    offers = pd.DataFrame(
        [
            _offer("a", "spreads", "home", -3.5, -115, seconds_before=20),
            _offer("a", "spreads", "away", 3.5, -105, seconds_before=20),
            _offer("draftkings", "spreads", "home", -3.5, -108, seconds_before=50),
            _offer("draftkings", "spreads", "away", 3.5, -112, seconds_before=50),
            _offer("b", "spreads", "home", -4.0, -110, seconds_before=30),
            _offer("b", "spreads", "away", 4.0, -110, seconds_before=30),
            _offer("a", "totals", "over", 51.5, -105, seconds_before=25),
            _offer("a", "totals", "under", 51.5, -115, seconds_before=25),
            _offer("draftkings", "totals", "over", 51.5, -102, seconds_before=45),
            _offer("draftkings", "totals", "under", 51.5, -118, seconds_before=40),
            # One side only: not a paired total.
            _offer("b", "totals", "over", 52.0, -110, seconds_before=30),
            # A book without a spread never prices the total consensus.
            _offer("c", "totals", "over", 49.0, -110, seconds_before=30),
            _offer("c", "totals", "under", 49.0, -110, seconds_before=30),
            # Live quotes are never part of the close.
            _offer(
                "a",
                "spreads",
                "home",
                -10.0,
                -110,
                seconds_before=-60,
                snapshot_id="live",
                phase="live",
                fetched_at=KICKOFF + pd.Timedelta(minutes=1),
            ),
        ]
    )

    rows = build_closing_lines(
        _anchor(-3.5), offers, {101: KICKOFF}, season=2026, source=ODDS_API_SOURCE
    )

    assert list(rows.columns) == CLOSING_LINES_COLUMNS
    row = rows.iloc[0]
    assert row["home_spread"] == -3.5 and row["n_spread_offers"] == 3
    assert row["total"] == 51.5 and row["n_total_offers"] == 2
    assert row["price_book"] == "draftkings"
    assert (row["spread_home_price"], row["spread_away_price"]) == (-108, -112)
    assert (row["over_price"], row["under_price"]) == (-102, -118)
    # observed_at is that book's latest update among the quotes stored.
    assert row["observed_at"] == KICKOFF - pd.Timedelta(seconds=40)
    assert row["fetched_at"] == CLOSE_AT and row["start_date"] == KICKOFF
    assert row["source"] == ODDS_API_SOURCE
    assert pd.isna(row["home_moneyline"]) and pd.isna(row["away_moneyline"])


def test_median_row_prices_stay_null_when_no_book_stands_at_the_median():
    offers = pd.DataFrame(
        [
            _offer("a", "spreads", "home", -3.0, -110, seconds_before=20),
            _offer("b", "spreads", "home", -4.0, -110, seconds_before=10),
            _offer("a", "totals", "over", 50.0, -110, seconds_before=20),
            _offer("a", "totals", "under", 50.0, -110, seconds_before=20),
            _offer("b", "totals", "over", 51.0, -110, seconds_before=10),
            _offer("b", "totals", "under", 51.0, -110, seconds_before=15),
        ]
    )
    row = build_closing_lines(
        _anchor(-3.5), offers, {101: KICKOFF}, season=2026, source=ODDS_API_SOURCE
    ).iloc[0]

    assert row["home_spread"] == -3.5 and row["total"] == 50.5
    assert row["price_book"] is None
    assert (
        row[["spread_home_price", "spread_away_price", "over_price", "under_price"]]
        .isna()
        .all()
    )
    # Without a price book the consensus' latest update is the observation.
    assert row["observed_at"] == KICKOFF - pd.Timedelta(seconds=10)

    # Without DraftKings at the median the freshest book names the prices.
    offers = pd.DataFrame(
        [
            _offer("a", "spreads", "home", -3.5, -115, seconds_before=20),
            _offer("b", "spreads", "home", -3.5, -105, seconds_before=10),
            _offer("b", "spreads", "away", 3.5, -115, seconds_before=10),
            _offer("c", "spreads", "home", -4.0, -110, seconds_before=5),
        ]
    )
    row = build_closing_lines(
        _anchor(-3.5), offers, {101: KICKOFF}, season=2026, source=ODDS_API_SOURCE
    ).iloc[0]
    assert row["price_book"] == "b"
    assert (row["spread_home_price"], row["spread_away_price"]) == (-105, -115)
    assert pd.isna(row["total"]) and row["n_total_offers"] == 0


def test_espn_fallback_row_carries_the_free_moneyline_and_the_feed_label():
    offers = pd.DataFrame(
        [
            _offer("draftkings", "spreads", "home", -7.5, -110, seconds_before=30),
            _offer("draftkings", "spreads", "away", 7.5, -110, seconds_before=30),
            _offer("draftkings", "totals", "over", 48.5, -110, seconds_before=30),
            _offer("draftkings", "totals", "under", 48.5, -110, seconds_before=30),
            _offer("draftkings", "h2h", "home", None, -290, seconds_before=30),
            _offer("draftkings", "h2h", "away", None, 235, seconds_before=30),
        ]
    )
    row = build_closing_lines(
        _anchor(-7.5),
        offers,
        {101: KICKOFF},
        season=2026,
        source=closing_source("espn"),
    ).iloc[0]
    assert row["source"] == ESPN_SOURCE
    assert (row["home_moneyline"], row["away_moneyline"]) == (-290, 235)
    assert row["n_spread_offers"] == 1 and row["n_total_offers"] == 1
    with pytest.raises(ValueError, match="no closing line source"):
        closing_source("cfbd")


def test_median_row_refuses_an_anchor_mismatch_or_a_post_kickoff_snapshot():
    offers = pd.DataFrame(
        [_offer("a", "spreads", "home", -3.5, -110, seconds_before=20)]
    )
    with pytest.raises(ValueError, match="disagrees with the validated anchor"):
        build_closing_lines(
            _anchor(-3.0), offers, {101: KICKOFF}, season=2026, source=ODDS_API_SOURCE
        )
    with pytest.raises(ValueError, match="no validated kickoff"):
        build_closing_lines(_anchor(-3.5), offers, {}, season=2026, source="x")
    with pytest.raises(ValueError, match="not pregame"):
        build_closing_lines(
            _anchor(-3.5),
            offers,
            {101: CLOSE_AT - pd.Timedelta(minutes=5)},
            season=2026,
            source=ODDS_API_SOURCE,
        )


def _captured(**overrides):
    row = {
        "game_id": 1,
        "season": 2026,
        "start_date": KICKOFF,
        "source": ODDS_API_SOURCE,
        "observed_at": CLOSE_AT,
        "fetched_at": CLOSE_AT,
        "home_spread": -6.5,
        "spread_home_price": -105.0,
        "spread_away_price": -115.0,
        "total": 47.5,
        "over_price": -108.0,
        "under_price": -112.0,
        "home_moneyline": None,
        "away_moneyline": None,
        "n_spread_offers": 4,
        "n_total_offers": 3,
        "price_book": "draftkings",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_grading_prefers_the_captured_row_and_labels_the_cfbd_fallback():
    lines = pd.DataFrame(
        {
            "game_id": [1, 2],
            "lines": [
                [
                    {"spread": -3.0, "overUnder": 44.0},
                    {"spread": -4.0, "overUnder": 46.0},
                ],
                [{"spread": 6.5, "overUnder": 50.0}],
            ],
        }
    )
    resolved = grading.resolve_closing_lines(lines, _captured()).set_index("game_id")
    assert resolved.loc[1, "closing_spread"] == -6.5
    assert resolved.loc[1, "closing_total"] == 47.5
    assert resolved.loc[1, "closing_source"] == ODDS_API_SOURCE
    assert resolved.loc[1, "n_spread_offers"] == 4
    assert resolved.loc[2, "closing_spread"] == 6.5
    assert resolved.loc[2, "closing_source"] == CLOSING_SOURCE

    # A captured row is taken whole: its missing total is not filled from CFBD.
    partial = grading.resolve_closing_lines(lines, _captured(total=None))
    assert pd.isna(partial.set_index("game_id").loc[1, "closing_total"])

    # Without captures every game is the CFBD median, and no table is needed.
    cfbd_only = grading.resolve_closing_lines(lines, None).set_index("game_id")
    assert cfbd_only.loc[1, "closing_spread"] == -3.5
    assert cfbd_only["closing_source"].eq(CLOSING_SOURCE).all()
    with pytest.raises(ValueError, match="more than one row"):
        grading.resolve_closing_lines(lines, pd.concat([_captured(), _captured()]))


def test_performance_metrics_report_graded_games_per_closing_source():
    graded = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": 2026,
                "week": 1,
                "start_date": KICKOFF,
                "model_version": "test",
                "home_classification": "fbs",
                "away_classification": "fbs",
                "home_missing_input_count": 0,
                "away_missing_input_count": 0,
                "pure_home_margin": 4.0,
                "market_informed_home_margin": 3.0,
                "model_total": 50.0,
                "market_informed_total": 49.0,
                "margin_sd": 14.0,
                "degrees_of_freedom": 500.0,
                "home_win_probability": 0.6,
                "probability_method": grading.PROBABILITY_METHOD,
                "closing_spread": spread,
                "closing_total": 48.0,
                "closing_source": source,
                "actual_margin": 3,
                "actual_total": 45,
            }
            for game_id, spread, source in (
                (1, -3.0, ODDS_API_SOURCE),
                (2, -2.0, ODDS_API_SOURCE),
                (3, 1.0, CLOSING_SOURCE),
                (4, None, None),
            )
        ]
    )
    metrics = grading.compute_performance_metrics(graded)
    overall = metrics[metrics["segment_kind"].eq("overall")].set_index(
        "prediction_source"
    )
    assert "closing_source_games" in grading.PERFORMANCE_METRIC_COLUMNS
    assert overall.loc["pure_model", "games"] == 4
    assert overall.loc["pure_model", "games_with_market"] == 3
    assert overall.loc["pure_model", "closing_source_games"] == (
        f'{{"{CLOSING_SOURCE}": 1, "{ODDS_API_SOURCE}": 2}}'
    )
    assert overall.loc["closing_market", "closing_source_games"] == (
        f'{{"{CLOSING_SOURCE}": 1, "{ODDS_API_SOURCE}": 2}}'
    )


def test_clv_prefers_the_captured_point_and_price_and_keeps_cfbd_moneylines():
    closing = pd.DataFrame(
        {
            "game_id": [1, 2],
            "closing_spread": [-6.0, -1.0],
            "closing_total": [46.5, 40.0],
            "closing_home_moneyline": [-240.0, -105.0],
            "closing_away_moneyline": [200.0, -115.0],
        }
    ).set_index("game_id")
    captured = _captured().set_index("game_id")

    def pick(game_id, market, side, point=None, **extra):
        return SimpleNamespace(
            game_id=game_id, market=market, side=side, point=point, **extra
        )

    home = closing_line_value(
        pick(1, "spreads", "home", -3.0, decision_market_point=-4.0), closing, captured
    )
    assert home["closing_point"] == -6.5 and home["closing_price"] == -105.0
    assert home["closing_source"] == ODDS_API_SOURCE
    assert home["clv_points"] == 3.5 and home["clv_market_points"] == 2.5

    away = closing_line_value(pick(1, "spreads", "away", 3.0), closing, captured)
    assert away["closing_point"] == 6.5 and away["closing_price"] == -115.0
    assert away["clv_points"] == -3.5

    under = closing_line_value(
        pick(1, "totals", "under", 49.0, market_total=48.0), closing, captured
    )
    assert under["closing_point"] == 47.5 and under["closing_price"] == -112.0
    assert under["clv_points"] == 1.5 and under["clv_market_points"] == 0.5

    # The moneyline keeps the CFBD median; the capture buys no h2h market.
    moneyline = closing_line_value(pick(1, "h2h", "home"), closing, captured)
    assert moneyline["closing_price"] == -240.0
    assert moneyline["closing_source"] == CLOSING_SOURCE

    # A game the capture missed falls back to the CFBD median, labelled.
    fallback = closing_line_value(pick(2, "spreads", "home", -2.0), closing, captured)
    assert fallback["closing_point"] == -1.0 and fallback["closing_price"] is None
    assert fallback["closing_source"] == CLOSING_SOURCE

    # A captured row without a total leaves that market to CFBD.
    no_total = _captured(total=None, over_price=None, under_price=None).set_index(
        "game_id"
    )
    over = closing_line_value(pick(1, "totals", "over", 44.0), closing, no_total)
    assert over["closing_point"] == 46.5 and over["closing_source"] == CLOSING_SOURCE
