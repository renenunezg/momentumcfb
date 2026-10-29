"""Public ledgers behind the CFB performance page, and their rebuild.

``export_ledgers`` writes two CSV files from the published tables: one row per
graded game (``cfb.graded_games``) and one row per pick decision
(``cfb.recommendations``, plus every archived decision that was replaced before
kickoff and every later pregame decision that did not replace a frozen pick).
``rebuild_from_ledgers`` reads only those files and recomputes every number the
performance page shows, so an outsider can check the page without database
access: the forecast metrics with ``backend.grading.compute_performance_metrics``
and the pick summary with the same segments as ``cfb.recommendation_performance``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from backend.grading import (
    GRADED_GAME_COLUMNS,
    THIN_SAMPLE_GAMES,
    compute_performance_metrics,
)
from backend.recommendations import CLOSING_LINE_COLUMNS, RECOMMENDATION_COLUMNS

FORECAST_RULE = (
    "absolute error of each source's home margin against the final margin; the "
    "CFBD median closing line is a benchmark graded beside the model, never an "
    "input; win probability is scored with the t marginal named in "
    "probability_method"
)
PICK_RULE = (
    "flat one unit at the recorded American price; a push returns the stake; a "
    "changed kickoff or tied moneyline voids; only rows with ledger_status "
    "'graded' count"
)
GRADED = "graded"
SUPERSEDED = "superseded_before_kickoff"
LATER_DECISION = "later_pregame_decision"
SETTLED = ("win", "loss", "push")
TIMESTAMP_COLUMNS = ("start_date", "decision_at", "graded_at", "published_at")


def forecast_ledger_name(season: int) -> str:
    return f"cfb_forecast_ledger_{season}.csv"


def pick_ledger_name(season: int) -> str:
    return f"cfb_pick_ledger_{season}.csv"


def build_pick_ledger(
    picks: pd.DataFrame, superseded: pd.DataFrame, decisions: pd.DataFrame
) -> pd.DataFrame:
    """The graded picks, then each replaced or later decision on its game.

    ``superseded`` rows were published and replaced before kickoff; only the
    replacement is graded. ``decisions`` are the last pregame decisions the
    site displayed; one that differs from the frozen pick is listed so the
    reader can see the pick stayed graded after a later price-driven change.
    """
    key = ["game_id", "market", "decision_at"]
    graded = picks.assign(ledger_status=GRADED)
    replaced = superseded.merge(picks[key], on=key, how="left", indicator=True)
    replaced = replaced[replaced["_merge"].eq("left_only")].drop(columns="_merge")
    # A later decision matters only where it would read differently from the
    # graded row: another status, side, line or price.
    shown = ["status", "side", "point", "price"]
    later = decisions.merge(
        picks[["game_id", "market", *shown]],
        on=["game_id", "market"],
        suffixes=("", "_graded"),
    )
    differs = np.zeros(len(later), dtype=bool)
    for column in shown:
        a, b = later[column], later[f"{column}_graded"]
        differs |= ~((a == b) | (a.isna() & b.isna())).to_numpy()
    later = later[differs].drop(columns=[f"{c}_graded" for c in shown])
    # The ledger is written as text, so the parts share one object dtype
    # rather than inferring numeric types from partly empty columns.
    ledger = pd.concat(
        [
            frame.astype(object)
            for frame in (
                graded,
                replaced.assign(ledger_status=SUPERSEDED),
                later.assign(ledger_status=LATER_DECISION),
            )
        ],
        ignore_index=True,
    )
    return ledger.assign(grading_rule=PICK_RULE).sort_values(
        ["start_date", "game_id", "market", "decision_at"], ignore_index=True
    )


def export_ledgers(season: int, directory: Path) -> dict[str, int]:
    """Write both ledgers for one season from the published tables."""
    from sqlalchemy import text

    from backend.db import engine
    from backend.publish import CFB_SCHEMA, fetch_graded_games

    graded = fetch_graded_games(season).assign(grading_rule=FORECAST_RULE)
    with engine.connect() as conn:

        def read(table: str) -> pd.DataFrame:
            return pd.read_sql_query(
                text(f"SELECT * FROM {CFB_SCHEMA}.{table} WHERE season = :s"),
                conn,
                params={"s": season},
            )

        archives = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = :schema "
                "AND table_name LIKE 'recommendations\\_archive\\_%' "
                "ORDER BY table_name"
            ),
            {"schema": CFB_SCHEMA},
        ).scalars()
        picks = read("recommendations")
        decisions = read("recommendation_decisions")
        archived = [read(table).assign(archive_table=table) for table in archives]
    required = [*RECOMMENDATION_COLUMNS, *CLOSING_LINE_COLUMNS]
    for name, frame in (
        ("recommendations", picks),
        ("recommendation_decisions", decisions),
    ):
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"cfb.{name} lacks {missing}; apply migrations first")
    # Archive tables are copies of cfb.recommendations made by hand when a
    # published slate was replaced before kickoff (cfb.recommendations_archive_
    # 20261012_week7_v8 holds the policy v8 week-7 slate replaced by v11 on
    # 2026-10-12). No migration creates them, so the export reads whichever
    # exist. They keep the columns of their day; later columns are empty, and a
    # replaced decision never settled, so its stored outcome is cleared.
    columns = [*picks.columns, "archive_table"]
    superseded = (
        pd.concat(archived, ignore_index=True).reindex(columns=columns)
        if archived
        else pd.DataFrame(columns=columns)
    ).assign(outcome=None, profit_units=None, graded_at=None)
    started = pd.to_datetime(decisions["start_date"], utc=True).le(
        pd.Timestamp.now(tz="UTC")
    )
    ledger = build_pick_ledger(picks, superseded, decisions[started])
    directory.mkdir(parents=True, exist_ok=True)
    graded.to_csv(directory / forecast_ledger_name(season), index=False)
    ledger.to_csv(directory / pick_ledger_name(season), index=False)
    return {
        "graded_games": len(graded),
        **ledger["ledger_status"].value_counts().to_dict(),
    }


def _edge_bucket(picks: pd.DataFrame) -> pd.Series:
    edge, probability = picks["edge_points"], picks["probability_edge"]
    by_probability = np.select(
        [probability < 0.075, probability < 0.10], ["4.5-7.5 pp", "7.5-10 pp"], "10+ pp"
    )
    by_points = np.select(
        [edge < 2, edge < 4, edge < 7],
        ["under 2 pts (floor)", "2-4 pts", "4-7 pts"],
        "7+ pts",
    )
    return pd.Series(
        np.where(edge.isna(), by_probability, by_points), index=picks.index
    )


def _side_label(picks: pd.DataFrame) -> pd.Series:
    spread = np.select(
        [picks["point"] < 0, picks["point"] > 0], ["favorite", "underdog"], "pickem"
    )
    price = np.select(
        [picks["price"] < -100, picks["price"] > 100], ["favorite", "underdog"], "even"
    )
    side = np.where(
        picks["market"].eq("totals"),
        picks["side"],
        np.where(picks["market"].eq("spreads"), spread, price),
    )
    return picks["market"] + ":" + pd.Series(side, index=picks.index).astype(str)


def _segments(picks: pd.DataFrame) -> pd.DataFrame:
    """One row per pick and segment, as ``cfb.recommendation_performance`` builds them."""
    recommended = picks["status"].eq("recommended")
    qualification = np.select(
        [picks["reason"].eq("volume_floor"), picks["reason"].eq("qualifying_edge")],
        ["exploratory", "edge_qualified"],
        "legacy",
    )
    every = pd.Series(True, index=picks.index)
    labelled = [
        ("overall", pd.Series("All picks", index=picks.index), every),
        ("market", picks["market"], every),
        (
            "week",
            picks["season"].astype(str) + " Week " + picks["week"].astype(str),
            every,
        ),
        ("policy", picks["policy_version"], every),
        ("edge", _edge_bucket(picks), every),
        ("qualification", pd.Series(qualification, index=picks.index), recommended),
        ("side", _side_label(picks), recommended),
    ]
    return pd.concat(
        [
            picks[mask].assign(segment_kind=kind, segment=label[mask])
            for kind, label, mask in labelled
        ],
        ignore_index=True,
    )


def _share_positive(values: pd.Series) -> float:
    present = values.dropna()
    return float(present.gt(0).mean()) if len(present) else np.nan


def pick_summary(picks: pd.DataFrame) -> pd.DataFrame:
    """The performance page's pick numbers from graded ledger rows."""
    segments = _segments(picks)
    rows = []
    for (season, kind, label), group in segments.groupby(
        ["season", "segment_kind", "segment"], sort=True
    ):
        recommended = group["status"].eq("recommended")
        settled = group["outcome"].isin(SETTLED)
        wins = int(group["outcome"].eq("win").sum())
        losses = int(group["outcome"].eq("loss").sum())
        pushes = int(group["outcome"].eq("push").sum())
        staked = float(group.loc[settled, "stake_units"].sum())
        profit = float(group.loc[settled, "profit_units"].sum())
        rows.append(
            {
                "season": season,
                "segment_kind": kind,
                "segment": label,
                "picks": int(recommended.sum()),
                "no_plays": int(group["status"].eq("no_play").sum()),
                "pending": int((recommended & group["outcome"].eq("pending")).sum()),
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "voids": int((recommended & group["outcome"].eq("void")).sum()),
                "staked_units": staked,
                "profit_units": profit,
                "average_ev": group.loc[recommended, "expected_value_per_unit"].mean(),
                "first_decision_at": group["decision_at"].min(),
                "last_decision_at": group["decision_at"].max(),
                "last_graded_at": group["graded_at"].max(),
                "average_clv_points": group["clv_points"].mean(),
                "clv_sample": int(group["clv_points"].notna().sum()),
                "clv_positive_share": _share_positive(group["clv_points"]),
                "average_clv_market_points": group["clv_market_points"].mean(),
                "clv_market_points_sample": int(
                    group["clv_market_points"].notna().sum()
                ),
                "clv_market_points_positive_share": _share_positive(
                    group["clv_market_points"]
                ),
                "average_clv_market_probability": group[
                    "clv_market_probability"
                ].mean(),
                "clv_market_probability_sample": int(
                    group["clv_market_probability"].notna().sum()
                ),
                "clv_market_probability_positive_share": _share_positive(
                    group["clv_market_probability"]
                ),
                "roi": profit / staked if staked else np.nan,
                "win_rate": wins / (wins + losses) if wins + losses else np.nan,
                "thin_sample": wins + losses + pushes < THIN_SAMPLE_GAMES,
            }
        )
    return pd.DataFrame(rows)


def read_ledgers(season: int, directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both ledgers with their timestamps parsed; picks keep every status."""
    graded = pd.read_csv(directory / forecast_ledger_name(season))
    for column in (
        "start_date",
        "forecast_as_of",
        "forecast_published_at",
        "graded_at",
    ):
        graded[column] = pd.to_datetime(graded[column], utc=True, format="ISO8601")
    picks = pd.read_csv(directory / pick_ledger_name(season))
    for column in TIMESTAMP_COLUMNS:
        picks[column] = pd.to_datetime(picks[column], utc=True, format="ISO8601")
    missing = sorted(set(GRADED_GAME_COLUMNS) - set(graded.columns))
    if missing:
        raise ValueError(f"forecast ledger lacks {missing}")
    return graded, picks


def rebuild_from_ledgers(
    season: int, directory: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute the performance page from the two ledger files alone."""
    graded, picks = read_ledgers(season, directory)
    metrics = compute_performance_metrics(graded).drop(columns="computed_at")
    summary = pick_summary(picks[picks["ledger_status"].eq(GRADED)])
    metrics.to_csv(directory / f"performance_metrics_rebuilt_{season}.csv", index=False)
    summary.to_csv(directory / f"pick_summary_rebuilt_{season}.csv", index=False)
    return metrics, summary
