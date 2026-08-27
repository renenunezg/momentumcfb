"""Market closing lines flattened into outcome-free serving anchors.

The in-game model is a pregame anchor updated by game state, so it inherits
whatever error its anchor carries. On the 2021-2025 backtest games the model's
projected margin runs MAE 13.40 against the closing line's 12.14, worse in
every season and week bucket, so the live model anchors on the market closing
spread: a live serve starts after kickoff, when the closing line is already
fixed and known, so the anchor is not forward-looking.

One game's closing spread is the median of the provider closing spreads in
the raw ``lines`` feed, and ``home_margin = -closing_spread``. The market
prices a spread but not a margin standard deviation, so ``margin_sd`` is a
single frozen constant: the residual standard deviation of the final margin
around the market margin over the development seasons that have stored lines.
Fitting that constant reads final scores, like every other fit; the stored
artifact carries no outcome column, and every row records the method that
produced ``margin_sd`` so a later reader cannot mistake the constant for a
per-game fitted spread.
"""

from math import isfinite

import numpy as np
import pandas as pd

from backend.etl import store
from backend.model.calibration import DEVELOPMENT_SEASONS

# Market anchors cover a whole season, not one projection week; the artifact
# is stored under week 00 so the untouched serving loader and serve-game
# (--anchor-source serving --week 0) find it by the ordinary naming rule.
MARKET_ANCHOR_WEEK = 0

CROSS_CHECK_ARTIFACT = ("backtest", "predictions_filtered.parquet")

MARKET_ANCHOR_COLUMNS = [
    "game_id",
    "season",
    "model_week",
    "home_margin",
    "margin_sd",
    "closing_spread",
    "n_spread_offers",
    "margin_sd_method",
    "market_anchor_source",
]

LIVE_MARKET_PROVENANCE_COLUMNS = [
    "closing_snapshot_id",
    "closing_fetched_at",
    "latest_provider_update",
]


def flatten_closing_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """Resolve one closing spread per game from the nested provider offers.

    The closing spread is the median across every priced provider spread,
    which is exactly the resolution the backtest market comparison used.
    """
    records = []
    for game_id, offers in zip(lines["game_id"], lines["lines"]):
        for offer in offers:
            spread = offer.get("spread")
            if spread is not None and isfinite(float(spread)):
                records.append((game_id, float(spread)))
    if not records:
        raise ValueError("lines carry no priced spreads to flatten")
    flat = pd.DataFrame(records, columns=["game_id", "spread"])
    grouped = flat.groupby("game_id")["spread"].agg(
        closing_spread="median", n_spread_offers="size"
    )
    return grouped.reset_index()


def flatten_live_closing_lines(offers: pd.DataFrame) -> pd.DataFrame:
    """Resolve one pregame Odds API spread per game without live leakage.

    Each game uses the latest stored snapshot that the capture classified as
    pregame. The closing spread is the median home-team spread across the
    providers in that snapshot. Quotes first observed after kickoff are never
    eligible, even if no pregame quote was captured for the game.
    """
    required = {
        "game_id",
        "snapshot_id",
        "fetched_at",
        "commence_time",
        "phase",
        "provider_key",
        "provider_last_update",
        "market",
        "selection",
        "point",
    }
    missing = sorted(required - set(offers.columns))
    if missing:
        raise ValueError("live odds offers are missing columns: " + ", ".join(missing))

    spreads = offers[
        offers["market"].eq("spreads")
        & offers["selection"].eq("home")
        & offers["phase"].eq("pregame")
    ].copy()
    spreads["point"] = pd.to_numeric(spreads["point"], errors="coerce")
    spreads["fetched_at"] = pd.to_datetime(
        spreads["fetched_at"], utc=True, errors="coerce"
    )
    spreads["commence_time"] = pd.to_datetime(
        spreads["commence_time"], utc=True, errors="coerce"
    )
    spreads["provider_last_update"] = pd.to_datetime(
        spreads["provider_last_update"], utc=True, errors="coerce"
    )
    spreads = spreads.dropna(
        subset=[
            "game_id",
            "snapshot_id",
            "fetched_at",
            "commence_time",
            "provider_key",
            "provider_last_update",
            "point",
        ]
    )
    finite = np.isfinite(spreads["point"].to_numpy(float))
    spreads = spreads[finite & spreads["fetched_at"].le(spreads["commence_time"])]
    if spreads.empty:
        raise ValueError("live odds captures carry no eligible pregame home spreads")

    # A provider should contribute at most one home spread to the consensus.
    # Keep its most recently updated quote if a payload contains duplicates.
    spreads = spreads.sort_values("provider_last_update", kind="stable")
    spreads = spreads.drop_duplicates(
        ["game_id", "snapshot_id", "provider_key"], keep="last"
    )
    snapshots = (
        spreads.groupby(["game_id", "snapshot_id"], as_index=False)
        .agg(closing_fetched_at=("fetched_at", "max"))
        .sort_values(["game_id", "closing_fetched_at"], kind="stable")
    )
    latest = snapshots.drop_duplicates("game_id", keep="last")
    selected = spreads.merge(
        latest[["game_id", "snapshot_id"]],
        on=["game_id", "snapshot_id"],
        how="inner",
        validate="many_to_one",
    )
    return (
        selected.groupby(["game_id", "snapshot_id"], as_index=False)
        .agg(
            closing_spread=("point", "median"),
            n_spread_offers=("provider_key", "nunique"),
            closing_fetched_at=("fetched_at", "max"),
            latest_provider_update=("provider_last_update", "max"),
        )
        .rename(columns={"snapshot_id": "closing_snapshot_id"})
    )


def _schedule_model_weeks(season: int) -> pd.DataFrame:
    """Chronological model week per game from the schedule alone."""
    schedule = store.read_games(season).rename(columns={"id": "game_id"})
    schedule = schedule.dropna(subset=["week"])
    regular = schedule["season_type"].eq("regular")
    if not regular.any():
        raise ValueError(f"season {season} schedule has no regular-season games")
    max_regular_week = int(schedule.loc[regular, "week"].max())
    model_week = np.where(
        regular, schedule["week"], max_regular_week + schedule["week"]
    )
    return pd.DataFrame(
        {"game_id": schedule["game_id"], "model_week": model_week.astype(int)}
    )


def fit_market_margin_sd(
    seasons: tuple[int, ...] = DEVELOPMENT_SEASONS,
) -> tuple[float, str]:
    """Fit the frozen market ``margin_sd`` constant on development seasons.

    The constant is the residual standard deviation of the final home margin
    around ``-closing_spread`` pooled over the development seasons with
    stored lines. Holdout sensitivity is flat near this value, so nothing
    sharper than a constant is warranted yet.
    """
    residuals = []
    fitted_seasons = []
    for season in seasons:
        try:
            closing = flatten_closing_lines(store.read_lines(season))
        except FileNotFoundError:
            continue
        schedule = store.read_games(season).rename(columns={"id": "game_id"})
        completed = schedule[schedule["completed"].fillna(False)]
        completed = completed[["game_id", "home_points", "away_points"]].dropna()
        merged = completed.merge(closing, on="game_id", how="inner")
        if merged.empty:
            continue
        residuals.append(
            merged["home_points"] - merged["away_points"] + merged["closing_spread"]
        )
        fitted_seasons.append(season)
    if not residuals:
        raise ValueError(
            "no development season has both stored lines and final scores; "
            "cannot fit the market margin_sd constant"
        )
    pooled = pd.concat(residuals, ignore_index=True)
    margin_sd = float(pooled.std(ddof=1))
    method = (
        "constant residual sd of final margin around -closing_spread, "
        f"development seasons {min(fitted_seasons)}-{max(fitted_seasons)} "
        f"({len(pooled)} games)"
    )
    return margin_sd, method


def _build_market_anchors(
    closing: pd.DataFrame, season: int, source: str
) -> pd.DataFrame:
    weeks = _schedule_model_weeks(season)
    anchors = closing.merge(weeks, on="game_id", how="inner", validate="one_to_one")
    if anchors.empty:
        raise ValueError(
            f"season {season}: no lined game appears in the stored schedule"
        )
    margin_sd, method = fit_market_margin_sd()
    anchors["season"] = season
    anchors["home_margin"] = -anchors["closing_spread"]
    anchors["margin_sd"] = margin_sd
    anchors["margin_sd_method"] = method
    anchors["market_anchor_source"] = source
    columns = MARKET_ANCHOR_COLUMNS + [
        column for column in LIVE_MARKET_PROVENANCE_COLUMNS if column in anchors.columns
    ]
    return (
        anchors[columns]
        .sort_values(["model_week", "game_id"], kind="stable")
        .reset_index(drop=True)
    )


def build_market_anchors(season: int) -> pd.DataFrame:
    """Build market anchors from CFBD's nested provider line payloads."""
    closing = flatten_closing_lines(store.read_lines(season))
    return _build_market_anchors(closing, season, "cfbd_lines")


def build_live_market_anchors(season: int) -> pd.DataFrame:
    """Build market anchors from append-only pregame Odds API captures."""
    from backend.odds.live import verify_live_snapshots

    problems, frames = verify_live_snapshots(season)
    if problems:
        raise ValueError("invalid live odds captures: " + "; ".join(problems))
    closing = flatten_live_closing_lines(frames["offers"])
    return _build_market_anchors(closing, season, "odds_api_live_capture")


def cross_check_closing_spreads(
    anchors: pd.DataFrame, season: int
) -> tuple[int, list[str]]:
    """Compare flattened closing spreads with the resolved backtest column.

    ``backtest/predictions_filtered.parquet`` carries the closing spread each
    backtest game was compared against; every one of them must reproduce
    exactly from the raw lines. Seasons absent from the backtest (a live
    season) have nothing to check and pass with zero comparisons.
    """
    location = "/".join(CROSS_CHECK_ARTIFACT)
    try:
        reference = store.read_processed(
            *CROSS_CHECK_ARTIFACT, columns=["game_id", "season", "closing_spread"]
        )
    except FileNotFoundError:
        return 0, []
    reference = reference[
        reference["season"].eq(season) & reference["closing_spread"].notna()
    ]
    if reference.empty:
        return 0, []
    merged = reference.merge(
        anchors[["game_id", "closing_spread"]].rename(
            columns={"closing_spread": "flattened_spread"}
        ),
        on="game_id",
        how="left",
    )
    problems = []
    missing = merged["flattened_spread"].isna()
    if missing.any():
        problems.append(
            f"{location}: {int(missing.sum())} of {len(merged)} backtest "
            f"closing spreads for {season} have no flattened market anchor"
        )
    mismatched = merged[
        ~missing & merged["flattened_spread"].ne(merged["closing_spread"])
    ]
    if not mismatched.empty:
        examples = ", ".join(
            f"game {int(row.game_id)} flattened {row.flattened_spread} "
            f"vs backtest {row.closing_spread}"
            for row in mismatched.head(3).itertuples()
        )
        problems.append(
            f"{location}: {len(mismatched)} flattened closing spreads "
            f"disagree with the backtest resolution ({examples})"
        )
    return len(merged), problems
