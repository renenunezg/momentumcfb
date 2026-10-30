"""One captured closing line per game for ``cfb.closing_lines``.

The kickoff capture freezes, per target game, the last pregame snapshot that
its validation accepted (``cfb.serving_anchors`` week 0 carries the snapshot
id and fetch time). This module turns that snapshot into the shared
closing-line row: the provider median home spread and total, the prices of
one named bookmaker standing at those medians, and the provenance the
freeze trigger checks. It reads stored snapshots only; nothing here polls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ODDS_API_SOURCE = "odds_api_median"
ESPN_SOURCE = "espn:draftkings"
# DraftKings prices are stored when it stands at the median point; otherwise
# the most recently updated bookmaker at that point names the prices.
PREFERRED_PRICE_BOOK = "draftkings"

CLOSING_LINES_COLUMNS = [
    "game_id",
    "season",
    "start_date",
    "source",
    "observed_at",
    "fetched_at",
    "home_spread",
    "spread_home_price",
    "spread_away_price",
    "total",
    "over_price",
    "under_price",
    "home_moneyline",
    "away_moneyline",
    "n_spread_offers",
    "n_total_offers",
    "price_book",
]


def closing_source(odds_source: str) -> str:
    """The closing_lines source label for the configured odds feed."""
    if odds_source == "odds_api":
        return ODDS_API_SOURCE
    if odds_source == "espn":
        return ESPN_SOURCE
    raise ValueError(f"no closing line source for odds feed {odds_source!r}")


def _clean_quotes(offers: pd.DataFrame) -> pd.DataFrame:
    """Finite, timestamped quotes with one row per provider and selection."""
    quotes = offers.copy()
    quotes["point"] = pd.to_numeric(quotes["point"], errors="coerce")
    quotes["price"] = pd.to_numeric(quotes["price"], errors="coerce")
    quotes["provider_last_update"] = pd.to_datetime(
        quotes["provider_last_update"], utc=True, errors="coerce"
    )
    quotes = quotes.dropna(subset=["provider_key", "provider_last_update"])
    quotes = quotes[np.isfinite(quotes["price"].to_numpy(float))]
    # A provider contributes one quote per selection: its latest if a payload
    # repeats one, which mirrors the anchor's spread resolution.
    quotes = quotes.sort_values("provider_last_update", kind="stable")
    return quotes.drop_duplicates(["provider_key", "selection"], keep="last")


def paired_closing_totals(
    totals: pd.DataFrame, spread_providers: set[str]
) -> pd.DataFrame:
    """One paired total per provider: over and under at one point, from a
    provider that also priced the spread. Columns: provider_key, point,
    over_price, under_price, oldest_update, latest_update."""
    columns = [
        "provider_key",
        "point",
        "over_price",
        "under_price",
        "oldest_update",
        "latest_update",
    ]
    required = {"provider_key", "provider_last_update", "selection", "point", "price"}
    if totals.empty or not required.issubset(totals.columns):
        return pd.DataFrame(columns=columns)
    quotes = _clean_quotes(totals[totals["selection"].isin(["over", "under"])])
    quotes = quotes[
        np.isfinite(quotes["point"].to_numpy(float))
        & quotes["provider_key"].astype(str).isin(spread_providers).to_numpy(bool)
    ]
    rows = []
    for provider, group in quotes.groupby("provider_key", sort=True):
        if set(group["selection"]) != {"over", "under"}:
            continue
        if group["point"].nunique() != 1:
            continue
        priced = group.set_index("selection")
        rows.append(
            {
                "provider_key": str(provider),
                "point": float(group["point"].iloc[0]),
                "over_price": float(priced.loc["over", "price"]),
                "under_price": float(priced.loc["under", "price"]),
                "oldest_update": group["provider_last_update"].min(),
                "latest_update": group["provider_last_update"].max(),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _price_book(candidates: pd.DataFrame) -> str | None:
    """DraftKings when it stands at the median, else the freshest bookmaker."""
    if candidates.empty:
        return None
    keys = candidates["provider_key"].astype(str)
    if keys.eq(PREFERRED_PRICE_BOOK).any():
        return PREFERRED_PRICE_BOOK
    freshest = candidates.sort_values("provider_last_update", kind="stable").iloc[-1]
    return str(freshest["provider_key"])


def _side_price(quotes: pd.DataFrame, provider: str | None, selection: str):
    if provider is None:
        return None
    row = quotes[
        quotes["provider_key"].astype(str).eq(provider)
        & quotes["selection"].eq(selection)
    ]
    return float(row["price"].iloc[0]) if len(row) == 1 else None


def _closing_row(
    anchor,
    game_offers: pd.DataFrame,
    kickoff: pd.Timestamp,
    *,
    season: int,
    source: str,
) -> dict:
    game_id = int(anchor["game_id"])
    label = f"game {game_id}"
    closing_fetched_at = pd.Timestamp(anchor["closing_fetched_at"])
    if closing_fetched_at.tzinfo is None:
        closing_fetched_at = closing_fetched_at.tz_localize("UTC")
    pregame = game_offers[game_offers["phase"].eq("pregame")]
    snapshot = pregame[pregame["snapshot_id"].eq(anchor["closing_snapshot_id"])]

    spreads = _clean_quotes(snapshot[snapshot["market"].eq("spreads")])
    home = spreads[spreads["selection"].eq("home")].dropna(subset=["point"])
    if home.empty:
        raise ValueError(f"{label}: closing snapshot carries no home spread")
    home_spread = float(home["point"].median())
    if home_spread != float(anchor["closing_spread"]):
        raise ValueError(
            f"{label}: snapshot median spread {home_spread} disagrees with the "
            f"validated anchor {anchor['closing_spread']}"
        )
    spread_providers = set(home["provider_key"].astype(str))

    # Totals close in the last pregame poll that requested them, which is the
    # spread snapshot unless kickoff moved after that pull.
    pregame_totals = pregame[
        pregame["market"].eq("totals")
        & pd.to_datetime(pregame["fetched_at"], utc=True).le(closing_fetched_at)
    ]
    if not pregame_totals.empty:
        totals_snapshot = pregame_totals.sort_values("fetched_at", kind="stable")[
            "snapshot_id"
        ].iloc[-1]
        pregame_totals = pregame_totals[
            pregame_totals["snapshot_id"].eq(totals_snapshot)
        ]
    paired = paired_closing_totals(pregame_totals, spread_providers)
    total = float(paired["point"].median()) if not paired.empty else None

    at_median = home[home["point"].eq(home_spread)]
    book = _price_book(at_median)
    spread_home_price = _side_price(spreads, book, "home")
    spread_away_price = _side_price(spreads, book, "away")
    book_total = (
        paired[paired["provider_key"].eq(book) & paired["point"].eq(total)]
        if book is not None and total is not None
        else paired.iloc[0:0]
    )
    over_price = float(book_total["over_price"].iloc[0]) if len(book_total) else None
    under_price = float(book_total["under_price"].iloc[0]) if len(book_total) else None
    moneylines = _clean_quotes(snapshot[snapshot["market"].eq("h2h")])
    home_moneyline = _side_price(moneylines, book, "home")
    away_moneyline = _side_price(moneylines, book, "away")

    if book is not None:
        used = [home[home["provider_key"].astype(str).eq(book)]["provider_last_update"]]
        if len(book_total):
            used.append(book_total["latest_update"])
        observed_at = pd.concat(used).max()
    else:
        updates = [home["provider_last_update"]]
        if not paired.empty:
            updates.append(paired["latest_update"])
        observed_at = pd.concat(updates).max()

    return {
        "game_id": game_id,
        "season": season,
        "start_date": kickoff,
        "source": source,
        "observed_at": observed_at,
        "fetched_at": closing_fetched_at,
        "home_spread": home_spread,
        "spread_home_price": spread_home_price,
        "spread_away_price": spread_away_price,
        "total": total,
        "over_price": over_price,
        "under_price": under_price,
        "home_moneyline": home_moneyline,
        "away_moneyline": away_moneyline,
        "n_spread_offers": len(spread_providers),
        "n_total_offers": int(len(paired)),
        "price_book": book,
    }


def build_closing_lines(
    anchors: pd.DataFrame,
    offers: pd.DataFrame,
    kickoffs: dict[int, pd.Timestamp],
    *,
    season: int,
    source: str,
) -> pd.DataFrame:
    """One ``cfb.closing_lines`` row per validated market anchor.

    ``anchors`` is the frozen week-0 market anchor artifact (game_id,
    closing_spread, closing_snapshot_id, closing_fetched_at); ``offers`` the
    stored live odds offers of the season; ``kickoffs`` the kickoff each
    anchor was validated against, which is the row's start_date.
    """
    required = {
        "game_id",
        "closing_spread",
        "closing_snapshot_id",
        "closing_fetched_at",
    }
    missing = sorted(required - set(anchors.columns))
    if missing:
        raise ValueError("market anchors lack columns: " + ", ".join(missing))
    if anchors["game_id"].duplicated().any():
        raise ValueError("market anchors carry more than one row for a game")
    rows = []
    for anchor in anchors.sort_values("game_id", kind="stable").itertuples(index=False):
        record = anchor._asdict()
        game_id = int(record["game_id"])
        kickoff = kickoffs.get(game_id)
        if kickoff is None or pd.isna(kickoff):
            raise ValueError(f"game {game_id} has no validated kickoff")
        kickoff = pd.Timestamp(kickoff)
        if kickoff.tzinfo is None:
            kickoff = kickoff.tz_localize("UTC")
        row = _closing_row(
            record,
            offers[offers["game_id"].eq(game_id)],
            kickoff,
            season=season,
            source=source,
        )
        if row["fetched_at"] >= kickoff:
            raise ValueError(f"game {game_id}: closing snapshot is not pregame")
        rows.append(row)
    return pd.DataFrame(rows, columns=CLOSING_LINES_COLUMNS)
