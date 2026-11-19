"""Explicit pregame quarterback availability as a documented adjustment.

CFBD carries no availability data and play-by-play text must never be read
as an injury signal, so the only sanctioned source is the ``cfb.qb_availability``
table: one row per season, week and team naming the starting quarterback's
status, its source and the time it was reported. Only rows reported before the
forecast or decision cutoff count, so a frozen forecast can always be replayed.

A ruled-out starter costs his team ``qb_out_points`` of his raw value, ``a + b
x max(raw, 0)``, where raw is his points above replacement per game times his
share of the team's games (``backend.players.traits.spread``). The players
pipeline publishes the regular-season amount as his PSR (``spread_value`` in
``cfb.position_traits``), and the forecast reads the PSR published before its
cutoff back as points, so the site's number is the line's move. In bowls the
move adds the bowl intercept's excess, ``QB_OUT_POSTSEASON_POINTS -
QB_OUT_POINTS``.

Fit (2026-11-19, ``research/spread_weight_20261119``): on the joint_scoring_v13
walk-forward, a team whose QB1 by plays took no snaps, against the closing line
minus the model, 2020-2023, OLS with HC1 errors. The starter is priced from
the snapshot one week staler than the game's, as the Monday forecast reads the
previous publish: ``a`` = 0.36 (SE 0.23) in the regular season, 2.60 (0.72)
in bowls, where opt-outs come with a starter's, and ``b`` = 0.72 (0.10)
shared. 736 regular-season and 68 bowl absences. A separate bowl slope was not
determined (1.08, SE 0.30, on 68 events). Flooring PSR at zero won on
leave-one-season-out within 2020-2023: the market charges a below-replacement
starter's absence about one point, not a gain. The realized margin agrees on
the slope, 1.81 (0.42).

Walk-forward 2024-2025 against the flat 1.5 / 4.0 (370 games with a QB1 out):
margin MAE 11.915 vs 11.956, RMSE 14.919 vs 14.954, distance to the close
3.473 vs 3.524, win log loss 0.5057 vs 0.5037 (paired difference +0.0020,
95% CI -0.0029 to +0.0068); all 1,491 games MAE 12.380 vs 12.390. Bowls with
a QB1 out (35) MAE 9.827 vs 10.222.

A starter with no published PSR (not his team's expected starter, no qualified
sample this season, or no publish yet) is priced at raw ``UNPRICED_STARTER_RAW``,
the 2020-2023 mean of max(raw, 0) over ruled-out regular-season starters. He
then costs the average change, 1.16, the same as a flat fit on those events.
"""

import numpy as np
import pandas as pd

from backend.etl.store import RAW_DIR

QB_OUT_POINTS = 0.36
QB_OUT_POSTSEASON_POINTS = 2.60
QB_OUT_POINTS_PER_PSR = 0.72
UNPRICED_STARTER_RAW = 1.11
ADJUSTING_STATUSES = ("out", "doubtful")
AVAILABILITY_COLUMNS = ["home_qb_out", "away_qb_out", "qb_availability_points"]


def _pregame_out_rows(
    availability: pd.DataFrame | None, season: int, week: int, as_of
) -> pd.DataFrame:
    if availability is None or availability.empty:
        return pd.DataFrame(columns=["team", "player"])
    reported = pd.to_datetime(availability["reported_at"], utc=True)
    return availability[
        availability["season"].eq(season)
        & availability["week"].eq(week)
        & availability["status"].str.lower().isin(ADJUSTING_STATUSES)
        & reported.lt(pd.Timestamp(as_of))
    ]


def pregame_qb_outs(
    availability: pd.DataFrame | None, season: int, week: int, as_of
) -> set[str]:
    """Teams whose starting quarterback was reported out or doubtful for the
    week before ``as_of``."""
    return set(_pregame_out_rows(availability, season, week, as_of)["team"])


def _folded(name) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalpha())


def qb_out_points(raw: float, postseason: bool = False) -> float:
    """Points a ruled-out starter with raw value ``raw`` costs, ``a + b x
    max(raw, 0)``; the regular-season amount is his published PSR."""
    base = QB_OUT_POSTSEASON_POINTS if postseason else QB_OUT_POINTS
    return base + QB_OUT_POINTS_PER_PSR * max(raw, 0.0)


UNPRICED_STARTER_POINTS = qb_out_points(UNPRICED_STARTER_RAW)


def starter_spread_values(
    availability: pd.DataFrame | None,
    spread_values: pd.DataFrame | None,
    season: int,
    week: int,
    as_of,
) -> dict[str, float]:
    """Each team in ``pregame_qb_outs`` whose ruled-out starter has a PSR
    published before ``as_of``: team -> PSR, regular-season points.

    ``spread_values`` holds the published QB ``spread_value`` rows (``team,
    athlete_name, value, as_of``); the report's player is matched by team and
    name, letters only. A team left out is priced at ``UNPRICED_STARTER_POINTS``.
    """
    outs = _pregame_out_rows(availability, season, week, as_of)
    if outs.empty or spread_values is None or spread_values.empty:
        return {}
    published = spread_values[
        pd.to_datetime(spread_values["as_of"], utc=True).lt(pd.Timestamp(as_of))
    ]
    lookup = {
        (row.team, _folded(row.athlete_name)): float(row.value)
        for row in published.itertuples()
    }
    priced = {}
    for row in outs.itertuples():
        value = lookup.get((row.team, _folded(row.player)))
        if value is not None:
            priced[row.team] = value
    return priced


def apply_qb_availability(
    projections: pd.DataFrame,
    outs: set[str],
    postseason_game_ids: set[int] = frozenset(),
    starter_psr: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Lower each affected team's expected points and every derived margin.

    ``starter_psr`` maps a team to its ruled-out starter's published PSR, the
    regular-season move in points (``starter_spread_values``); a team without
    one moves ``UNPRICED_STARTER_POINTS``. A bowl adds the bowl intercept's
    excess.

    Pure margins move by the full net adjustment. A market-informed margin
    moves only by its model share because the market already priced the
    absence. A team already flagged on the frame (a published forecast being
    re-decided) is not adjusted twice. The columns in AVAILABILITY_COLUMNS
    are always present so the published record shows when nothing applied.
    """
    out = projections.copy()
    flagged = {}
    fresh = {}
    for side in ("home", "away"):
        column = f"{side}_qb_out"
        flagged[side] = (
            out[column].fillna(False).astype(bool)
            if column in out
            else pd.Series(False, index=out.index)
        )
        fresh[side] = out[f"{side}_team"].isin(outs) & ~flagged[side]
        out[column] = flagged[side] | fresh[side]
    starter_psr = starter_psr or {}
    postseason = out["game_id"].isin(postseason_game_ids)
    bowl_excess = np.where(postseason, QB_OUT_POSTSEASON_POINTS - QB_OUT_POINTS, 0.0)
    points = {
        side: out[f"{side}_team"]
        .map(starter_psr)
        .fillna(UNPRICED_STARTER_POINTS)
        .to_numpy(dtype=float)
        + bowl_excess
        for side in ("home", "away")
    }
    # A near-zero score cannot lose the full availability penalty. Use the
    # applied loss for every derived margin, total and market blend as well.
    home_loss = np.minimum(
        np.where(fresh["home"], points["home"], 0.0), out["expected_home_points"]
    )
    away_loss = np.minimum(
        np.where(fresh["away"], points["away"], 0.0), out["expected_away_points"]
    )
    net = away_loss - home_loss
    previous = (
        pd.to_numeric(out["qb_availability_points"], errors="coerce").fillna(0.0)
        if "qb_availability_points" in out
        else 0.0
    )
    out["qb_availability_points"] = previous + net
    if not (fresh["home"].any() or fresh["away"].any()):
        return out
    out["expected_home_points"] = (out["expected_home_points"] - home_loss).clip(
        lower=0.0
    )
    out["expected_away_points"] = (out["expected_away_points"] - away_loss).clip(
        lower=0.0
    )
    for margin, spread in (
        ("home_margin", "home_spread"),
        ("pure_home_margin", "pure_home_spread"),
    ):
        if margin in out:
            out[margin] = out[margin] + net
            out[spread] = -out[margin]
    if "market_informed_home_margin" in out:
        share = 1.0 - pd.to_numeric(out["market_weight"], errors="coerce").fillna(0.0)
        if "market_history_weight" in out:
            # Earlier weeks' lines cannot know this week's report either.
            share = share * (1.0 - out["market_history_weight"].fillna(0.0))
        out["market_informed_home_margin"] = (
            out["market_informed_home_margin"] + share * net
        )
        out["market_informed_home_spread"] = -out["market_informed_home_margin"]
    out["model_total"] = out["model_total"] - home_loss - away_loss
    if "market_informed_total" in out:
        if "market_total_weight" not in out or out["market_total_weight"].isna().any():
            raise ValueError(
                "Late availability requires a published market_total_weight; republish the forecast first"
            )
        weights = pd.to_numeric(out["market_total_weight"], errors="raise")
        if not weights.between(0, 1).all():
            raise ValueError("Invalid published total blend weight")
        total_share = 1 - weights
        if "market_history_total_weight" in out:
            # Earlier weeks' totals cannot know this week's report either.
            total_share = total_share * (
                1.0 - out["market_history_total_weight"].fillna(0.0)
            )
        margin = out["market_informed_home_margin"]
        out["market_informed_total"] = np.maximum(
            out["market_informed_total"] - total_share * (home_loss + away_loss),
            margin.abs(),
        )
        out["market_informed_home_points"] = (out["market_informed_total"] + margin) / 2
        out["market_informed_away_points"] = (out["market_informed_total"] - margin) / 2
    return out


def observed_starters(season: int) -> pd.DataFrame:
    """Each team's most recent starting quarterback from completed box scores.

    The starter is the passer with the most attempts in the game. This is an
    observed fact about played games, never an availability inference: it
    exists to point at teams whose last starter differs from their usual one
    so a report can be checked and entered with its source.
    """
    files = sorted((RAW_DIR / "players" / str(season)).glob("box_*.parquet"))
    if not files:
        return pd.DataFrame(
            columns=[
                "team",
                "last_game_id",
                "last_starter",
                "usual_starter",
                "starts",
                "changed",
            ]
        )
    box = pd.concat(pd.read_parquet(path) for path in files)
    passing = box[box["category"].eq("passing") & box["stat_name"].eq("C/ATT")].copy()
    passing["attempts"] = pd.to_numeric(
        passing["stat"].astype(str).str.split("/").str[-1], errors="coerce"
    ).fillna(0)
    starters = (
        passing.sort_values("attempts", ascending=False)
        .drop_duplicates(["game_id", "team"])
        .sort_values(["season_type", "week", "game_id"])
    )
    usual = (
        starters.groupby(["team", "athlete_name"]).size().rename("starts").reset_index()
    )
    usual = usual.sort_values("starts", ascending=False).drop_duplicates("team")
    last = starters.drop_duplicates("team", keep="last")[
        ["team", "game_id", "athlete_name"]
    ]
    out = last.rename(
        columns={"game_id": "last_game_id", "athlete_name": "last_starter"}
    ).merge(usual.rename(columns={"athlete_name": "usual_starter"}), on="team")
    out["changed"] = out["last_starter"].ne(out["usual_starter"])
    return out.sort_values(["changed", "team"], ascending=[False, True]).reset_index(
        drop=True
    )
