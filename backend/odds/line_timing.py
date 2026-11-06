"""DraftKings lines at fixed checkpoints, for closing-line value and timing tests.

``cfb.line_timing`` keeps one row per game with three checkpoints, each a
smallint array of LINE_FIELDS in order:

- ``opened``: DraftKings' opening line, which ESPN's scoreboard keeps beside
  the current one, so it is read on Monday rather than captured when posted.
- ``decided``: the current line when the Monday forecast and picks publish.
- ``closed``: the captured closing line, written with ``cfb.closing_lines``.

Points are stored in half-points so every value fits a smallint, which keeps a
season near 0.2 MB. Each checkpoint is written once; a rerun never overwrites
it. ESPN is unmetered, so no Odds API credit is spent.
"""

import logging
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from backend import db
from backend.odds.espn import ESPN_BOOKMAKERS, EspnOddsClient, _point, _price

log = logging.getLogger(__name__)

LINE_FIELDS = (
    "home_spread_half_points",
    "spread_home_price",
    "spread_away_price",
    "total_half_points",
    "over_price",
    "under_price",
    "home_moneyline",
    "away_moneyline",
)
_QUOTES = (
    ("pointSpread", "home", True),
    ("pointSpread", "home", False),
    ("pointSpread", "away", False),
    ("total", "over", True),
    ("total", "over", False),
    ("total", "under", False),
    ("moneyline", "home", False),
    ("moneyline", "away", False),
)


def _half(point: float | None) -> int | None:
    return None if point is None else int(round(point * 2))


def _checkpoint(odds: dict, when: str) -> list[int | None] | None:
    """One checkpoint's values from an ESPN odds object, or None if unpriced."""
    values: list[int | None] = []
    for market, side, is_point in _QUOTES:
        quote = ((odds.get(market) or {}).get(side) or {}).get(when) or {}
        values.append(_half(_point(quote)) if is_point else _price(quote))
    return values if any(value is not None for value in values) else None


def espn_checkpoints(events: list[dict]) -> dict[int, tuple[list | None, list | None]]:
    """DraftKings (opening, current) values by game id. CFB game ids are ESPN's."""
    lines = {}
    for event in events:
        for odds in event["competitions"][0].get("odds") or []:
            if str((odds.get("provider") or {}).get("id")) in ESPN_BOOKMAKERS:
                lines[int(event["id"])] = (
                    _checkpoint(odds, "open"),
                    _checkpoint(odds, "close"),
                )
    return lines


def record_decision_lines(
    season: int, projections: pd.DataFrame, *, client=None, now=None
) -> int:
    """Store the opening and current DraftKings lines of the week's future games."""
    if not db.writes_allowed():
        raise RuntimeError("Line timing requires MOMENTUMCFB_DB_WRITES=1")
    now = now or datetime.now(timezone.utc)
    starts = pd.to_datetime(projections["start_date"], utc=True)
    future = projections[starts > now]
    if future.empty:
        return 0
    future_starts = pd.to_datetime(future["start_date"], utc=True)
    read_at, events = (client or EspnOddsClient()).scoreboard_events(
        future_starts.min().to_pydatetime(), future_starts.max().to_pydatetime()
    )
    lines = espn_checkpoints(events)
    rows = [
        {"g": int(game_id), "s": season, "o": opened, "d": current, "t": read_at}
        for game_id in future["game_id"].astype(int)
        for opened, current in [lines.get(game_id, (None, None))]
        if current is not None
    ]
    if not rows:
        return 0
    with db.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO cfb.line_timing (game_id, season, opened, decided, decided_at) "
                "VALUES (:g, :s, CAST(:o AS smallint[]), CAST(:d AS smallint[]), :t) "
                "ON CONFLICT (game_id) DO UPDATE SET "
                "opened = coalesce(cfb.line_timing.opened, excluded.opened), "
                "decided = coalesce(cfb.line_timing.decided, excluded.decided), "
                "decided_at = coalesce(cfb.line_timing.decided_at, excluded.decided_at)"
            ),
            rows,
        )
    return len(rows)


def _closing_values(row) -> list[int | None]:
    def price(value):
        return None if value is None or pd.isna(value) else int(round(value))

    def point(value):
        return None if value is None or pd.isna(value) else _half(float(value))

    return [
        point(row.home_spread),
        price(row.spread_home_price),
        price(row.spread_away_price),
        point(row.total),
        price(row.over_price),
        price(row.under_price),
        price(row.home_moneyline),
        price(row.away_moneyline),
    ]


def record_closed_lines(conn, closing: pd.DataFrame) -> int:
    """Store each captured closing line as the game's closed checkpoint, once."""
    rows = [
        {
            "g": int(row.game_id),
            "s": int(row.season),
            "c": _closing_values(row),
            "t": row.fetched_at,
        }
        for row in closing.itertuples()
    ]
    rows = [row for row in rows if any(v is not None for v in row["c"])]
    if not rows:
        return 0
    conn.execute(
        text(
            "INSERT INTO cfb.line_timing (game_id, season, closed, closed_at) "
            "VALUES (:g, :s, CAST(:c AS smallint[]), :t) "
            "ON CONFLICT (game_id) DO UPDATE SET "
            "closed = coalesce(cfb.line_timing.closed, excluded.closed), "
            "closed_at = coalesce(cfb.line_timing.closed_at, excluded.closed_at)"
        ),
        rows,
    )
    return len(rows)
