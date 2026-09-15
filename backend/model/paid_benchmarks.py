"""Descriptive comparisons with external adjusted metrics, never model inputs."""

import pandas as pd

from backend.cfbd.snapshots import receipts

ADJUSTED_ENDPOINTS = (
    "/wepa/team/season",
    "/wepa/players/passing",
    "/wepa/players/rushing",
    "/wepa/players/kicking",
)


def latest_adjusted(endpoint, season, as_of, *, root=None):
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("benchmark cutoff must be timezone-aware")
    available = [
        (p, r)
        for p, r in receipts(endpoint, root=root)
        if r["params"].get("year") == season and pd.Timestamp(r["fetched_at"]) <= cutoff
    ]
    if not available:
        return pd.DataFrame()
    path, receipt = max(available, key=lambda x: pd.Timestamp(x[1]["fetched_at"]))
    frame = pd.json_normalize(receipt["payload"])
    if frame.empty:
        return frame
    if not frame.year.eq(season).all():
        raise ValueError("adjusted metric season differs from requested season")
    frame["provider_fetched_at"] = receipt["fetched_at"]
    frame["provider_snapshot"] = str(path)
    frame["evidence_kind"] = "retrospective_external_benchmark"
    return frame


def team_benchmark(ratings, provider):
    """Compare ranks because PPP strengths and provider EPA use different scales."""
    if provider.empty:
        return pd.DataFrame(), pd.DataFrame()
    if ratings.team_id.duplicated().any() or provider.teamId.duplicated().any():
        raise ValueError("team benchmarks require one row per team")
    joined = ratings.merge(
        provider,
        left_on="team_id",
        right_on="teamId",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_provider"),
    )
    if not joined.season.eq(joined.year).all():
        raise ValueError("team benchmark seasons differ")
    rows = []
    for model, external, direction in (
        ("offense_points", "epa.total", 1),
        ("defense_points", "epaAllowed.total", -1),
    ):
        a, b = joined[model], joined[external] * direction
        valid = a.notna() & b.notna()
        rows.append(
            dict(
                metric=model,
                games_or_teams=int(valid.sum()),
                spearman=a[valid].rank().corr(b[valid].rank()),
                interpretation="rank_agreement_not_forecast_accuracy",
            )
        )
    return joined, pd.DataFrame(rows)


def player_benchmark(values, provider):
    if provider.empty or values.empty:
        return pd.DataFrame(), pd.DataFrame()
    latest = values[values.week.eq(values.week.max())].copy()
    latest["athlete_id"] = latest.athlete_id.astype(str)
    source = provider.copy()
    source["athleteId"] = source.athleteId.astype(str)
    joined = latest.merge(
        source,
        left_on=["athlete_id", "team"],
        right_on=["athleteId", "team"],
        validate="one_to_one",
        suffixes=("", "_provider"),
    )
    if not joined.season.eq(joined.year).all():
        raise ValueError("player benchmark seasons differ")
    # Kicking exposes total points above average replacement (PAAR), not WEPA.
    model_column, provider_column = (
        ("adjusted_rate", "wepa")
        if "wepa" in joined
        else ("value_above_replacement", "paar")
    )
    valid = joined[model_column].notna() & joined[provider_column].notna()
    result = pd.DataFrame(
        [
            dict(
                players=int(valid.sum()),
                model_metric=model_column,
                provider_metric=provider_column,
                spearman=joined.loc[valid, model_column]
                .rank()
                .corr(joined.loc[valid, provider_column].rank()),
                interpretation="rank_agreement_not_Heisman_forecast_accuracy",
            )
        ]
    )
    return joined, result
