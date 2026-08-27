"""Publish serving tables to the cfb schema of the momentum Supabase project.

Supabase is the only interface between this model and the website
(momentumweb); the tables written here are that API surface:

- teams                 team identity dimension (logos, colors), full refresh
- team_ratings          one row per team per published (season, week)
- team_unit_ratings     descriptive opponent-adjusted unit companions
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

# Identity only. Conference and classification are deliberately absent: every
# consumer already loads team_ratings, which carries both.
TEAMS_COLUMNS = [
    "team_id",
    "team",
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

TEAM_UNIT_RATINGS_COLUMNS = [
    "season",
    "week",
    "as_of",
    "model_version",
    "source_season",
    "team_id",
    "team",
    "classification",
    "unit_history_missing",
    "rush_offense",
    "pass_offense",
    "rush_defense",
    "pass_defense",
    "pass_block",
    "run_block",
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
    "conference_game",
]

GAME_PROJECTIONS_OPTIONAL_COLUMNS = [
    "pure_home_margin",
    "pure_home_spread",
    "market_home_spread",
    "market_weight",
    "market_informed_home_margin",
    "market_informed_home_spread",
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

# Serving anchor artifacts are keyed by artifact week: anchor_week 0 is the
# frozen market closing-spread capture, anchor_week >= 1 is a projection
# artifact for that week. The market columns are NULL on projection rows.
SERVING_ANCHORS_COLUMNS = [
    "season",
    "anchor_week",
    "game_id",
    "model_week",
    "home_margin",
    "margin_sd",
    "closing_spread",
    "n_spread_offers",
    "margin_sd_method",
    "market_anchor_source",
    "closing_snapshot_id",
    "closing_fetched_at",
    "latest_provider_update",
]

_SERVING_ANCHOR_CONTRACT = ["game_id", "model_week", "home_margin", "margin_sd"]

_TIMESTAMP_COLUMNS = {
    "as_of",
    "start_date",
    "model_as_of",
    "best_offer_provider_last_update",
    "closing_fetched_at",
    "latest_provider_update",
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


def load_team_unit_ratings(source: str, season: int, week: int) -> pd.DataFrame:
    path = _artifact_dir(source, "unit_ratings") / f"{season}_{week:02d}.parquet"
    return _serving_frame(pd.read_parquet(path), TEAM_UNIT_RATINGS_COLUMNS)


def load_game_projections(source: str, season: int, week: int) -> pd.DataFrame:
    path = _artifact_dir(source, "projections") / f"{season}_{week:02d}.parquet"
    projections = pd.read_parquet(path)
    # conference_game is schedule metadata the model never consumes, so it is
    # absent from the forecast artifact; take it from the same snapshot the
    # forecast ran against rather than inferring it from matching conference
    # names, which would call two independents a conference game.
    schedule_path = (
        RAW_DIR / "preseason" / str(season) / "games.parquet"
        if source == "preseason"
        else RAW_DIR / "games" / f"{season}.parquet"
    )
    if schedule_path.exists():
        schedule = pd.read_parquet(
            schedule_path, columns=["id", "conference_game"]
        )
        projections["conference_game"] = projections["game_id"].map(
            schedule.set_index("id")["conference_game"]
        )
    return _serving_frame(
        projections,
        [*GAME_PROJECTIONS_COLUMNS, *GAME_PROJECTIONS_OPTIONAL_COLUMNS],
    )


def _table_exists(conn, table: str) -> bool:
    return conn.execute(
        text("SELECT to_regclass(:table_name)"),
        {"table_name": table},
    ).scalar() is not None


def _table_columns(conn, table: str) -> set[str]:
    return {
        str(row.column_name)
        for row in conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = :table_name"
            ),
            {"table_name": table},
        )
    }


def load_market_comparisons(
    source: str, season: int, week: int
) -> pd.DataFrame:
    directory = _artifact_dir(source, "market_comparisons")
    path = directory / f"{season}_{week:02d}.parquet"
    return _serving_frame(pd.read_parquet(path), MARKET_COMPARISONS_COLUMNS)


def load_backtest_predictions() -> pd.DataFrame:
    path = PROCESSED_DIR / "backtest" / "predictions_filtered.parquet"
    return _serving_frame(pd.read_parquet(path), BACKTEST_PREDICTIONS_COLUMNS)


def publish_serving_anchors(season: int, anchor_week: int) -> int:
    """Publish one stored serving anchor artifact to cfb.serving_anchors.

    A CI capture runner is ephemeral, so the anchor artifact must reach the
    database to survive the job; fetch_serving_anchors hydrates it back into
    any local store.
    """
    from backend.db import engine
    from backend.serving.anchors import (
        load_serving_anchors,
        serving_anchor_artifact,
    )

    # The loader round-trip is the artifact's validity proof; a frame that
    # fails it must never reach serving consumers.
    load_serving_anchors("serving", season=season, week=anchor_week)
    parts = serving_anchor_artifact(season, anchor_week)
    frame = pd.read_parquet(PROCESSED_DIR.joinpath(*parts))
    frame = frame.assign(season=season, anchor_week=anchor_week)
    rows = _serving_frame(frame, SERVING_ANCHORS_COLUMNS)

    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM serving_anchors "
                "WHERE season = :s AND anchor_week = :w"
            ),
            {"s": season, "w": anchor_week},
        )
        rows.to_sql("serving_anchors", con=conn, if_exists="append", index=False)
    with engine.connect() as conn:
        return int(
            conn.execute(
                text(
                    "SELECT count(*) FROM serving_anchors "
                    "WHERE season = :s AND anchor_week = :w"
                ),
                {"s": season, "w": anchor_week},
            ).scalar()
        )


def fetch_serving_anchors(season: int, anchor_week: int) -> int:
    """Hydrate the local serving anchor artifact from cfb.serving_anchors."""
    from backend.db import engine
    from backend.etl import store
    from backend.serving.anchors import (
        load_serving_anchors,
        serving_anchor_artifact,
    )

    with engine.connect() as conn:
        frame = pd.read_sql_query(
            text(
                "SELECT "
                + ", ".join(c for c in SERVING_ANCHORS_COLUMNS if c != "anchor_week")
                + " FROM serving_anchors "
                "WHERE season = :s AND anchor_week = :w ORDER BY game_id"
            ),
            conn,
            params={"s": season, "w": anchor_week},
        )
    if frame.empty:
        raise ValueError(
            f"cfb.serving_anchors holds no rows for season {season} "
            f"anchor week {anchor_week}"
        )
    # Projection artifacts carry no market columns; drop the all-NULL ones so
    # the hydrated parquet matches the shape the builders write locally.
    optional = [c for c in frame.columns if c not in _SERVING_ANCHOR_CONTRACT]
    frame = frame.drop(columns=[c for c in optional if frame[c].isna().all()])
    store.write_processed(frame, *serving_anchor_artifact(season, anchor_week))
    stored = load_serving_anchors("serving", season=season, week=anchor_week)
    return len(stored)


def weekly_forecast_is_published(
    season: int,
    week: int,
    model_version: str,
) -> bool:
    """Return whether this weekly model version already reached serving."""
    from backend.db import engine

    with engine.connect() as conn:
        if not _table_exists(conn, "team_ratings"):
            return False
        return bool(
            conn.execute(
                text(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM team_ratings "
                    "WHERE season = :season AND week = :week "
                    "AND model_version = :model_version"
                    ")"
                ),
                {
                    "season": season,
                    "week": week,
                    "model_version": model_version,
                },
            ).scalar()
        )


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
    try:
        unit_ratings = load_team_unit_ratings(source, season, week)
    except FileNotFoundError:
        unit_ratings = None
    projections = load_game_projections(source, season, week)
    market = load_market_comparisons(source, season, week)
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

        if unit_ratings is not None and _table_exists(conn, "team_unit_ratings"):
            conn.execute(
                text(
                    "DELETE FROM team_unit_ratings "
                    "WHERE season = :s AND week = :w"
                ),
                {"s": season, "w": week},
            )
            unit_ratings.to_sql(
                "team_unit_ratings", con=conn, if_exists="append", index=False
            )

        conn.execute(
            text("DELETE FROM game_projections WHERE game_id = ANY(:ids)"),
            {"ids": game_ids},
        )
        projection_columns = _table_columns(conn, "game_projections")
        projections[[
            column for column in projections.columns if column in projection_columns
        ]].to_sql(
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
        if _table_exists(conn, "team_unit_ratings"):
            tables.insert(2, "team_unit_ratings")
        return {
            table: int(
                conn.execute(text(f"SELECT count(*) FROM {table}")).scalar()
            )
            for table in tables
        }
