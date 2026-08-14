"""Publish serving tables to the cfb schema of the momentum Supabase project.

Supabase is the only interface between this model and the website
(momentumweb); the five tables written here are that API surface:

- teams                 team identity dimension (logos, colors), full refresh
- team_ratings          one row per team per published (season, week)
- game_projections      current forecast, one row per game
- market_comparisons    best priced offer vs model, one row per game
- backtest_predictions  frozen walk-forward history, full refresh

Writes use per-key DELETE + append (or TRUNCATE + append for the full
backtest refresh) so RLS, policies, and indexes on the tables survive;
to_sql(if_exists="replace") would drop them.
"""

from __future__ import annotations

import re

import pandas as pd
from sqlalchemy import text

from backend.config import PROCESSED_DIR, RAW_DIR

TEAMS_COLUMNS = [
    "team_id",
    "team",
    "mascot",
    "abbreviation",
    "conference",
    "classification",
    "color",
    "alternate_color",
    "logo_light",
    "logo_dark",
]

TEAM_RATINGS_COLUMNS = [
    "season",
    "week",
    "as_of",
    "model_version",
    "team_id",
    "team",
    "conference",
    "classification",
    "offense_points",
    "defense_points",
    "power_rating",
    "scoring_environment",
    "expected_possessions",
    "power_rating_sd",
    "missing_input_count",
]

GAME_PROJECTIONS_COLUMNS = [
    "game_id",
    "season",
    "week",
    "as_of",
    "model_version",
    "start_date",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "neutral_site",
    "home_field_points",
    "expected_home_points",
    "expected_away_points",
    "home_margin",
    "home_spread",
    "model_total",
    "margin_sd",
    "total_sd",
    "margin_total_correlation",
    "distribution",
    "degrees_of_freedom",
    "home_classification",
    "away_classification",
    "home_missing_input_count",
    "away_missing_input_count",
]

MARKET_COMPARISONS_COLUMNS = [
    "game_id",
    "start_date",
    "home_team",
    "away_team",
    "model_home_spread",
    "model_total",
    "margin_sd",
    "total_sd",
    "model_as_of",
    "market_available",
    "priced_offer_available",
    "executable_offer_available",
    "review_status",
    "recommendation_status",
    "best_offer_market",
    "best_offer_selection",
    "best_offer_point",
    "best_offer_price",
    "best_offer_provider",
    "best_offer_provider_key",
    "best_offer_provider_last_update",
    "best_offer_event_link",
    "best_offer_market_link",
    "best_offer_bet_link",
    "best_offer_edge_points",
    "best_offer_edge_standardized",
    "best_offer_model_cover_probability",
    "best_offer_model_fair_price",
    "best_offer_expected_value_per_unit",
]

BACKTEST_PREDICTIONS_COLUMNS = [
    "game_id",
    "season",
    "week",
    "week_index",
    "season_type",
    "home_team",
    "away_team",
    "neutral_site",
    "home_points",
    "away_points",
    "margin",
    "closing_spread",
    "model_margin",
    "actual_margin",
]

_TIMESTAMP_COLUMNS = {
    "as_of",
    "start_date",
    "model_as_of",
    "best_offer_provider_last_update",
}


def _artifact_dir(source: str, kind: str):
    if source == "preseason":
        return PROCESSED_DIR / "preseason" / kind
    return PROCESSED_DIR / kind


def _serving_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Project onto the serving columns and convert values for Postgres.

    Timestamps become tz-aware datetimes and every NaN becomes None:
    Postgres double precision would accept a literal NaN, but PostgREST
    cannot serialize it to JSON, so NULL is the only safe missing value.
    """
    out = df.reindex(columns=columns)
    for column in columns:
        if column in _TIMESTAMP_COLUMNS:
            out[column] = pd.to_datetime(out[column], utc=True, format="ISO8601")
        out[column] = out[column].astype(object).where(out[column].notna(), None)
    return out


_HEX_COLOR = re.compile(r"^#[0-9a-f]{6}$")


def _hex_color(value) -> str | None:
    """CFBD stores a missing color as the literal string '#null', which would
    reach the browser as an invalid CSS color; anything not a six-digit hex
    becomes NULL so the site can fall back deliberately."""
    if not isinstance(value, str):
        return None
    color = value.strip().lower()
    return color if _HEX_COLOR.match(color) else None


# The site renders logos at roughly 20px, and the schedule page loads two per
# game; at CFBD's 500px variant that is ~9MB of PNG for a full week, so serve
# the 128px variant, which still has ample headroom over the display size.
LOGO_TARGET_WIDTH = 128


def _logo(logos, dark: bool) -> str | None:
    """CFBD ships each logo at eight widths in a light and a dark variant
    (.../logos/128/333.png, .../logos-dark/128/333.png). Pick the narrowest
    variant at or above the target width, or the widest available below it."""
    marker = "/logos-dark/" if dark else "/logos/"
    candidates = []
    for url in logos if logos is not None else ():
        if not isinstance(url, str) or marker not in url:
            continue
        segment = url.rsplit(marker, 1)[1].split("/", 1)[0]
        if segment.isdigit():
            candidates.append((int(segment), url))
    if not candidates:
        return None
    at_or_above = [c for c in candidates if c[0] >= LOGO_TARGET_WIDTH]
    return min(at_or_above)[1] if at_or_above else max(candidates)[1]


def teams_path(season: int):
    return RAW_DIR / "preseason" / str(season) / "teams.parquet"


def load_teams(season: int) -> pd.DataFrame:
    """Team identity from the season's preseason /teams snapshot.

    Every division is kept, not just D1: game_projections can name a
    non-D1 opponent, and the site resolves logos by team_id alone.
    """
    raw = pd.read_parquet(teams_path(season))
    out = pd.DataFrame(
        {
            "team_id": pd.to_numeric(raw["id"], errors="coerce").astype("Int64"),
            "team": raw["school"],
            "mascot": raw["mascot"],
            "abbreviation": raw["abbreviation"],
            "conference": raw["conference"],
            "classification": raw["classification"],
            "color": raw["color"].map(_hex_color),
            "alternate_color": raw["alternate_color"].map(_hex_color),
            "logo_light": raw["logos"].map(lambda v: _logo(v, dark=False)),
            "logo_dark": raw["logos"].map(lambda v: _logo(v, dark=True)),
        }
    )
    out = out.dropna(subset=["team_id"]).drop_duplicates(subset=["team_id"])
    return _serving_frame(out, TEAMS_COLUMNS)


def load_team_ratings(source: str, season: int, week: int) -> pd.DataFrame:
    path = _artifact_dir(source, "ratings") / f"{season}_{week:02d}.parquet"
    return _serving_frame(pd.read_parquet(path), TEAM_RATINGS_COLUMNS)


def load_game_projections(source: str, season: int, week: int) -> pd.DataFrame:
    path = _artifact_dir(source, "projections") / f"{season}_{week:02d}.parquet"
    return _serving_frame(pd.read_parquet(path), GAME_PROJECTIONS_COLUMNS)


def load_market_comparisons(season: int, week: int) -> pd.DataFrame:
    path = (
        _artifact_dir("preseason", "market_comparisons")
        / f"{season}_{week:02d}.parquet"
    )
    return _serving_frame(pd.read_parquet(path), MARKET_COMPARISONS_COLUMNS)


def load_backtest_predictions() -> pd.DataFrame:
    path = PROCESSED_DIR / "backtest" / "predictions_filtered.parquet"
    return _serving_frame(pd.read_parquet(path), BACKTEST_PREDICTIONS_COLUMNS)


def publish(
    season: int,
    week: int,
    source: str = "preseason",
    include_backtest: bool = True,
) -> dict[str, int]:
    """Publish the serving tables for one (season, week) in one transaction."""
    from backend.db import engine

    # Team identity comes from the preseason /teams snapshot, which an
    # in-season fit publish does not produce; skip it rather than fail.
    teams = load_teams(season) if teams_path(season).exists() else None
    ratings = load_team_ratings(source, season, week)
    projections = load_game_projections(source, season, week)
    # Market comparisons are produced only by the preseason forecast; fit
    # artifacts project games without pricing the market.
    market = load_market_comparisons(season, week) if source == "preseason" else None
    backtest = load_backtest_predictions() if include_backtest else None

    # Delete projections by game id, not by week, so a game that moved weeks
    # between publishes cannot survive as a duplicate row.
    game_ids = [int(g) for g in projections["game_id"]]

    with engine.begin() as conn:
        if teams is not None:
            # A dimension with no natural version: replace it wholesale so a
            # rebrand or a reclassification cannot leave a stale row behind.
            conn.execute(text("DELETE FROM teams"))
            teams.to_sql("teams", con=conn, if_exists="append", index=False)

        conn.execute(
            text("DELETE FROM team_ratings WHERE season = :s AND week = :w"),
            {"s": season, "w": week},
        )
        ratings.to_sql("team_ratings", con=conn, if_exists="append", index=False)

        conn.execute(
            text("DELETE FROM game_projections WHERE game_id = ANY(:ids)"),
            {"ids": game_ids},
        )
        projections.to_sql(
            "game_projections", con=conn, if_exists="append", index=False
        )

        if market is not None:
            conn.execute(
                text("DELETE FROM market_comparisons WHERE game_id = ANY(:ids)"),
                {"ids": [int(g) for g in market["game_id"]]},
            )
            market.to_sql(
                "market_comparisons", con=conn, if_exists="append", index=False
            )

        if backtest is not None:
            # Full refresh: TRUNCATE + append preserves RLS and indexes.
            conn.execute(text("TRUNCATE TABLE backtest_predictions"))
            backtest.to_sql(
                "backtest_predictions", con=conn, if_exists="append", index=False
            )

    # Read back stored totals so the caller reports what the website will see.
    tables = [
        "teams",
        "team_ratings",
        "game_projections",
        "market_comparisons",
        "backtest_predictions",
    ]
    with engine.connect() as conn:
        return {
            table: int(
                conn.execute(text(f"SELECT count(*) FROM {table}")).scalar()
            )
            for table in tables
        }
