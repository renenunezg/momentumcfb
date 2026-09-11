"""Fail-closed preflight and orchestration for one kickoff market window."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil, isfinite

import pandas as pd

from backend.etl import store
from backend.model.ingame import SERVING_ANCHOR_COLUMNS
from backend.odds.client import OddsAPIClient, OddsAPIError
from backend.odds.forecast import (
    check_weekly_capture_readiness,
    load_weekly_capture_forecast,
)
from backend.odds.live import (
    CLOSING_MARKETS,
    LIVE_MARKETS,
    live_poll_quota_cost,
    load_division_one_schedule,
    run_live_polling,
    verify_live_snapshots,
)
from backend.serving.anchors import (
    load_serving_anchors,
    serving_anchor_artifact,
)
from backend.serving.market import MARKET_ANCHOR_WEEK, build_live_market_anchors

log = logging.getLogger(__name__)

REQUIRED_PRESEASON_SOURCES = frozenset(
    {
        "teams",
        "games",
        "talent",
        "returning",
        "portal",
        "coaches",
        "recruiting",
        "lines",
        "prior_coaches",
        "prior_talent",
    }
)
ALLOWED_EMPTY_SOURCES = frozenset({"talent"})


@dataclass(frozen=True, slots=True)
class KickoffTarget:
    game_ids: tuple[int, ...]
    first_kickoff: pd.Timestamp
    last_kickoff: pd.Timestamp
    labels: tuple[str, ...]
    kickoffs: tuple[pd.Timestamp, ...] = ()


@dataclass(frozen=True, slots=True)
class KickoffWindowPlan:
    starts_at: pd.Timestamp
    ends_at: pd.Timestamp
    polls: int


@dataclass(frozen=True, slots=True)
class KickoffReadiness:
    target: KickoffTarget | None
    problems: tuple[str, ...]
    warnings: tuple[str, ...]
    details: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.problems


@dataclass(frozen=True, slots=True)
class KickoffRunResult:
    plan: KickoffWindowPlan
    anchor_count: int
    target_details: tuple[str, ...]
    polls_completed: int = 0


def _utc(value=None) -> pd.Timestamp:
    if value is None:
        value = datetime.now(timezone.utc)
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _artifact_frame(season: int, week: int, name: str) -> pd.DataFrame:
    return store.read_preseason_forecast_artifact(season, week, name)


def check_forecast_readiness(
    season: int,
    week: int,
    *,
    as_of=None,
    forecast_directory: str | None = None,
    max_forecast_age_hours: float = 48.0,
    max_source_age_hours: float = 48.0,
) -> tuple[list[str], list[str], list[str]]:
    """Validate the final local forecast and its outcome-free anchors."""
    now = _utc(as_of)
    if forecast_directory is not None:
        return check_weekly_capture_readiness(
            forecast_directory, season, as_of=now, max_age_hours=max_forecast_age_hours
        )
    problems: list[str] = []
    warnings: list[str] = []
    details: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    for name in (
        "source_manifest",
        "ratings",
        "unit_ratings",
        "projections",
        "market_comparisons",
    ):
        try:
            frames[name] = _artifact_frame(season, week, name)
        except (FileNotFoundError, ValueError) as exc:
            problems.append(f"missing or unreadable preseason {name}: {exc}")

    manifest = frames.get("source_manifest")
    if manifest is not None:
        required_columns = {
            "source",
            "source_fetched_at",
            "row_count",
            "is_empty",
        }
        missing_columns = sorted(required_columns - set(manifest.columns))
        if missing_columns:
            problems.append(
                "source manifest is missing columns: " + ", ".join(missing_columns)
            )
        else:
            duplicated = manifest["source"].duplicated()
            if duplicated.any():
                problems.append(
                    f"source manifest has {int(duplicated.sum())} duplicate sources"
                )
            sources = set(manifest["source"].dropna().astype(str))
            missing_sources = sorted(REQUIRED_PRESEASON_SOURCES - sources)
            if missing_sources:
                problems.append(
                    "source manifest is missing: " + ", ".join(missing_sources)
                )
            for row in manifest.itertuples():
                source = str(row.source)
                row_count = int(row.row_count)
                is_empty = bool(row.is_empty)
                if is_empty != (row_count == 0):
                    problems.append(f"source {source} row_count and is_empty disagree")
                if is_empty:
                    if source in ALLOWED_EMPTY_SOURCES:
                        warnings.append(
                            f"source {source} is empty; the documented neutral "
                            "fallback remains active"
                        )
                    elif source not in REQUIRED_PRESEASON_SOURCES:
                        warnings.append(
                            f"optional source {source} is empty; required CFBD "
                            "sources remain available"
                        )
                    else:
                        problems.append(f"required source {source} is empty")
            source_times = pd.to_datetime(
                manifest["source_fetched_at"], utc=True, errors="coerce"
            )
            if source_times.isna().any():
                problems.append("source manifest has invalid fetched timestamps")
            elif not source_times.empty:
                oldest = source_times.min()
                age_hours = (now - oldest).total_seconds() / 3600.0
                if age_hours < -5 / 60:
                    problems.append("source manifest timestamps are in the future")
                elif age_hours > max_source_age_hours:
                    problems.append(
                        f"oldest preseason source is {age_hours:.1f}h old; "
                        f"maximum is {max_source_age_hours:.1f}h"
                    )
                details.append(
                    f"preseason sources: {len(manifest)} rows, oldest {age_hours:.1f}h"
                )

    projections = frames.get("projections")
    forecast_time = None
    if projections is not None:
        if projections.empty:
            problems.append("preseason projections are empty")
        elif "forecast_created_at" not in projections:
            problems.append("preseason projections lack forecast_created_at")
        else:
            created = pd.to_datetime(
                projections["forecast_created_at"], utc=True, errors="coerce"
            )
            if created.isna().any() or created.nunique() != 1:
                problems.append(
                    "preseason projections do not share one valid forecast timestamp"
                )
            else:
                forecast_time = created.iloc[0]
                age_hours = (now - forecast_time).total_seconds() / 3600.0
                if age_hours < -5 / 60:
                    problems.append("forecast timestamp is in the future")
                elif age_hours > max_forecast_age_hours:
                    problems.append(
                        f"preseason forecast is {age_hours:.1f}h old; maximum is "
                        f"{max_forecast_age_hours:.1f}h"
                    )
                details.append(
                    f"preseason forecast: {len(projections)} games, "
                    f"created {forecast_time.isoformat()}"
                )

    ratings = frames.get("ratings")
    units = frames.get("unit_ratings")
    if ratings is not None and units is not None:
        if ratings.empty:
            problems.append("preseason ratings are empty")
        if set(ratings.get("team_id", [])) != set(units.get("team_id", [])):
            problems.append("team ratings and unit ratings cover different teams")
        else:
            details.append(f"team ratings: {len(ratings)} teams")

    comparisons = frames.get("market_comparisons")
    if projections is not None and comparisons is not None:
        if set(projections.get("game_id", [])) != set(comparisons.get("game_id", [])):
            problems.append("projections and market comparisons cover different games")

    if projections is not None and not projections.empty:
        try:
            anchors = load_serving_anchors("serving", season=season, week=week)
        except (FileNotFoundError, ValueError) as exc:
            problems.append(f"preseason serving anchors are not loadable: {exc}")
        else:
            if set(anchors["game_id"]) != set(projections["game_id"]):
                problems.append(
                    "preseason serving anchors and projections cover different games"
                )
            else:
                details.append(
                    f"preseason anchors: {len(anchors)} games round-trip cleanly"
                )
    return problems, warnings, details


def resolve_kickoff_target(
    season: int,
    frames: dict[str, pd.DataFrame],
    *,
    as_of=None,
    game_ids: list[int] | tuple[int, ...] | None = None,
    cluster_minutes: float = 15.0,
    schedule: pd.DataFrame | None = None,
) -> KickoffTarget:
    """Resolve explicit games or the next stored market-covered kickoff cluster."""
    if cluster_minutes < 0:
        raise ValueError("--cluster-minutes must be nonnegative")
    now = _utc(as_of)
    schedule = (
        load_division_one_schedule(season) if schedule is None else schedule
    ).copy()
    schedule["game_id"] = pd.to_numeric(schedule["game_id"], errors="coerce")
    schedule["kickoff"] = pd.to_datetime(
        schedule["start_date"], utc=True, errors="coerce"
    )
    schedule = schedule.dropna(subset=["game_id", "kickoff"])
    schedule["game_id"] = schedule["game_id"].astype(int)
    schedule = schedule.drop_duplicates("game_id")

    if game_ids:
        requested = tuple(dict.fromkeys(int(game_id) for game_id in game_ids))
        selected = schedule[schedule["game_id"].isin(requested)].copy()
        missing = sorted(set(requested) - set(selected["game_id"]))
        if missing:
            raise ValueError(
                "target games are absent from the Division I schedule: "
                + ", ".join(str(game_id) for game_id in missing)
            )
    else:
        offers = frames.get("offers", pd.DataFrame())
        if offers.empty:
            raise ValueError(
                "no stored Odds API offers; pass --game-id or capture a preflight poll"
            )
        covered = offers[
            offers["market"].eq("spreads")
            & offers["selection"].eq("home")
            & offers["game_id"].notna()
        ]
        covered_ids = set(covered["game_id"].astype(int))
        upcoming = schedule[
            schedule["game_id"].isin(covered_ids) & schedule["kickoff"].gt(now)
        ].copy()
        if upcoming.empty:
            raise ValueError("no future market-covered kickoff remains in the schedule")
        first = upcoming["kickoff"].min()
        selected = upcoming[
            upcoming["kickoff"].le(first + pd.Timedelta(minutes=cluster_minutes))
        ].copy()

    selected = selected.sort_values(["kickoff", "game_id"], kind="stable")
    first_kickoff = selected["kickoff"].min()
    last_kickoff = selected["kickoff"].max()
    if (last_kickoff - first_kickoff).total_seconds() > cluster_minutes * 60:
        raise ValueError(
            "target games span more than --cluster-minutes; run separate windows"
        )
    labels = tuple(
        f"{row.away_team} at {row.home_team} ({int(row.game_id)})"
        for row in selected.itertuples()
    )
    return KickoffTarget(
        game_ids=tuple(int(value) for value in selected["game_id"]),
        first_kickoff=first_kickoff,
        last_kickoff=last_kickoff,
        labels=labels,
        kickoffs=tuple(selected["kickoff"]),
    )


def plan_kickoff_window(
    target: KickoffTarget,
    *,
    as_of=None,
    lead_minutes: float = 5.0,
    post_minutes: float = 5.0,
    interval_seconds: float = 120.0,
) -> KickoffWindowPlan:
    if lead_minutes <= 0:
        raise ValueError("--lead-minutes must be positive")
    if post_minutes <= 0:
        raise ValueError("--post-minutes must be positive")
    if interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive")
    now = _utc(as_of)
    planned_start = target.first_kickoff - pd.Timedelta(minutes=lead_minutes)
    starts_at = max(now, planned_start)
    if starts_at >= target.first_kickoff:
        raise ValueError(
            "target kickoff has started; a new pregame capture cannot be proven"
        )
    ends_at = target.last_kickoff + pd.Timedelta(minutes=post_minutes)
    duration_seconds = (ends_at - starts_at).total_seconds()
    polls = ceil(duration_seconds / interval_seconds) + 1
    return KickoffWindowPlan(
        starts_at=starts_at,
        ends_at=ends_at,
        polls=polls,
    )


def plan_kickoff_poll_markets(
    target: KickoffTarget,
    plan: KickoffWindowPlan,
    interval_seconds: float,
) -> tuple[tuple[str, ...], ...]:
    """Request totals only in each game's final scheduled pregame poll."""
    if interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive")
    if target.kickoffs:
        if len(target.kickoffs) != len(target.game_ids):
            raise ValueError("target kickoff timestamps do not match target games")
        kickoffs = tuple(_utc(value) for value in target.kickoffs)
    elif len(target.game_ids) == 1:
        kickoffs = (target.first_kickoff,)
    else:
        raise ValueError("multi-game target lacks per-game kickoff timestamps")

    poll_times = tuple(
        plan.starts_at + pd.Timedelta(seconds=index * interval_seconds)
        for index in range(plan.polls)
    )
    totals_polls: set[int] = set()
    for kickoff in kickoffs:
        eligible = [
            index for index, poll_time in enumerate(poll_times) if poll_time < kickoff
        ]
        if not eligible:
            raise ValueError(
                f"no scheduled poll occurs before kickoff {kickoff.isoformat()}"
            )
        totals_polls.add(eligible[-1])
    return tuple(
        CLOSING_MARKETS if index in totals_polls else LIVE_MARKETS
        for index in range(plan.polls)
    )


class KickoffPollPlanner:
    """Re-plan each poll from the provider's current kickoff times.

    The frozen schedule only seeds the window. Providers move
    ``commence_time`` by minutes near kickoff, so the closing-total pull and
    the end of the window follow the latest stored provider state instead:
    totals are requested whenever the next poll could fall at or after a
    pending game's current kickoff, and polling stops once every target has
    been observed live or final, or when the poll cap is reached.
    """

    def __init__(
        self,
        target: KickoffTarget,
        *,
        interval_seconds: float,
        post_minutes: float,
        max_polls: int,
        progress=log.info,
        now=lambda: datetime.now(timezone.utc),
    ) -> None:
        if len(target.kickoffs) != len(target.game_ids):
            raise ValueError("target kickoff timestamps do not match target games")
        self.interval = pd.Timedelta(seconds=interval_seconds)
        self.post = pd.Timedelta(minutes=post_minutes)
        self.max_polls = max_polls
        self.progress = progress
        self.now = now
        self.labels = dict(zip(target.game_ids, target.labels))
        self.kickoffs = {
            game_id: _utc(kickoff)
            for game_id, kickoff in zip(target.game_ids, target.kickoffs)
        }
        self.pending = set(target.game_ids)

    def observe(self, snapshot) -> None:
        events = snapshot.events
        if events.empty or "game_id" not in events:
            return
        events = events[events["game_id"].isin(self.pending)]
        for row in events.itertuples(index=False):
            game_id = int(row.game_id)
            commence = pd.to_datetime(row.commence_time, utc=True, errors="coerce")
            if pd.notna(commence) and commence != self.kickoffs[game_id]:
                self.progress(
                    f"{self.labels[game_id]} kickoff moved from "
                    f"{self.kickoffs[game_id].isoformat()} to {commence.isoformat()}"
                )
                self.kickoffs[game_id] = commence
            if row.phase in ("live", "final"):
                self.pending.discard(game_id)

    def __call__(self, poll_index: int, latest) -> tuple[tuple[str, ...], ...]:
        if latest is not None:
            self.observe(latest)
        if not self.pending or poll_index >= self.max_polls:
            return ()
        poll_time = _utc(self.now())
        remaining = self.max_polls - poll_index
        last_kickoff = max(self.kickoffs[game_id] for game_id in self.pending)
        span = (last_kickoff + self.post - poll_time).total_seconds()
        planned = min(remaining, max(1, ceil(span / self.interval.total_seconds()) + 1))
        poll_times = [poll_time + index * self.interval for index in range(planned)]
        totals_polls: set[int] = set()
        for game_id in self.pending:
            kickoff = self.kickoffs[game_id]
            eligible = [
                index for index, time in enumerate(poll_times) if time < kickoff
            ]
            if eligible:
                totals_polls.add(eligible[-1])
        return tuple(
            CLOSING_MARKETS if index in totals_polls else LIVE_MARKETS
            for index in range(planned)
        )


def _latest_successful_poll(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.Series | None, pd.DataFrame]:
    polls = frames.get("polls", pd.DataFrame())
    if polls.empty:
        return None, pd.DataFrame()
    successful = polls[polls["poll_status"].eq("ok")].copy()
    if successful.empty:
        return None, pd.DataFrame()
    successful["odds_fetched_at"] = pd.to_datetime(
        successful["odds_fetched_at"], utc=True, errors="coerce"
    )
    successful = successful.dropna(subset=["odds_fetched_at"])
    if successful.empty:
        return None, pd.DataFrame()
    latest = successful.sort_values("odds_fetched_at", kind="stable").iloc[-1]
    offers = frames.get("offers", pd.DataFrame())
    latest_offers = (
        offers[offers["snapshot_id"].eq(latest["snapshot_id"])].copy()
        if not offers.empty
        else offers
    )
    return latest, latest_offers


def _quota_problems(
    frames: dict[str, pd.DataFrame],
    planned_markets: tuple[tuple[str, ...], ...],
    min_quota: int,
    extension_polls: int = 0,
) -> tuple[list[str], list[str]]:
    latest, _ = _latest_successful_poll(frames)
    if latest is None:
        return ["no successful Odds API poll is stored"], []
    remaining_values = [
        int(value)
        for value in (
            latest.get("odds_requests_remaining"),
            latest.get("scores_requests_remaining"),
        )
        if pd.notna(value)
    ]
    if not remaining_values:
        return ["latest Odds API poll lacks quota provenance"], []
    remaining = min(remaining_values)
    planned_cost = sum(live_poll_quota_cost(markets) for markets in planned_markets)
    totals_polls = sum("totals" in markets for markets in planned_markets)
    extension_cost = extension_polls * live_poll_quota_cost(CLOSING_MARKETS)
    expected_after = remaining - planned_cost - extension_cost
    details = [
        f"last stored Odds API quota: {remaining} remaining, about "
        f"{planned_cost} for {len(planned_markets)} planned polls including "
        f"{totals_polls} closing-total pull(s), up to {extension_cost} more for "
        f"{extension_polls} kickoff-drift polls, at least {expected_after} after"
    ]
    if expected_after < min_quota:
        return [
            f"planned window would leave about {expected_after} requests; "
            f"minimum is {min_quota}"
        ], details
    return [], details


def check_live_preflight(
    frames: dict[str, pd.DataFrame],
    target: KickoffTarget,
    *,
    as_of=None,
    planned_markets: tuple[tuple[str, ...], ...],
    min_quota: int = 50,
    max_poll_age_minutes: float = 15.0,
    max_offer_staleness_seconds: float = 300.0,
    min_providers: int = 2,
) -> tuple[list[str], list[str], list[str]]:
    now = _utc(as_of)
    problems: list[str] = []
    warnings: list[str] = []
    details: list[str] = []
    latest, latest_offers = _latest_successful_poll(frames)
    if latest is None:
        return ["no successful Odds API poll is stored"], warnings, details

    fetched_at = _utc(latest["odds_fetched_at"])
    age_minutes = (now - fetched_at).total_seconds() / 60.0
    if age_minutes < -5:
        problems.append("latest Odds API poll timestamp is in the future")
    elif age_minutes > max_poll_age_minutes:
        problems.append(
            f"latest Odds API poll is {age_minutes:.1f}m old; maximum is "
            f"{max_poll_age_minutes:.1f}m"
        )
    details.append(
        f"latest Odds API poll: {fetched_at.isoformat()} ({age_minutes:.1f}m old)"
    )

    quota_problems, quota_details = _quota_problems(frames, planned_markets, min_quota)
    problems.extend(quota_problems)
    details.extend(quota_details)

    configured = latest.get("configured_bookmakers")
    try:
        configured_bookmakers = json.loads(configured) if pd.notna(configured) else []
    except (TypeError, json.JSONDecodeError):
        configured_bookmakers = []
    if not configured_bookmakers:
        warnings.append(
            "configured bookmaker execution eligibility is not verified; "
            "the capture is suitable only for the market anchor"
        )

    required_offer_columns = {
        "game_id",
        "market",
        "selection",
        "phase",
        "provider_key",
        "staleness_seconds",
    }
    missing_offer_columns = sorted(required_offer_columns - set(latest_offers.columns))
    if missing_offer_columns:
        problems.append(
            "latest poll has no usable offers: missing "
            + ", ".join(missing_offer_columns)
        )
        return problems, warnings, details
    spreads = latest_offers[
        latest_offers["market"].eq("spreads")
        & latest_offers["selection"].eq("home")
        & latest_offers["phase"].eq("pregame")
    ].copy()
    for game_id, label in zip(target.game_ids, target.labels):
        game = spreads[spreads["game_id"].eq(game_id)]
        if game.empty:
            problems.append(f"latest poll has no pregame home spread for {label}")
            continue
        providers = int(game["provider_key"].nunique())
        staleness = pd.to_numeric(game["staleness_seconds"], errors="coerce").max()
        if providers < min_providers:
            problems.append(
                f"{label} has {providers} spread providers; minimum is {min_providers}"
            )
        if pd.isna(staleness):
            problems.append(f"{label} has no provider freshness measurement")
        elif float(staleness) > max_offer_staleness_seconds:
            problems.append(
                f"{label} maximum provider staleness is {float(staleness):.0f}s; "
                f"maximum is {max_offer_staleness_seconds:.0f}s"
            )
        else:
            details.append(
                f"{label}: {providers} spread providers, "
                f"{float(staleness):.0f}s maximum staleness"
            )
    return problems, warnings, details


def check_kickoff_readiness(
    season: int,
    week: int = 1,
    *,
    as_of=None,
    game_ids: list[int] | tuple[int, ...] | None = None,
    cluster_minutes: float = 15.0,
    lead_minutes: float = 5.0,
    post_minutes: float = 5.0,
    interval_seconds: float = 120.0,
    min_quota: int = 50,
    forecast_directory: str | None = None,
    max_forecast_age_hours: float = 48.0,
    max_source_age_hours: float = 48.0,
    max_poll_age_minutes: float = 15.0,
    max_offer_staleness_seconds: float = 300.0,
    min_providers: int = 2,
) -> KickoffReadiness:
    """Run the read-only authoritative gate for the next kickoff window."""
    now = _utc(as_of)
    problems, warnings, details = check_forecast_readiness(
        season,
        week,
        as_of=now,
        forecast_directory=forecast_directory,
        max_forecast_age_hours=max_forecast_age_hours,
        max_source_age_hours=max_source_age_hours,
    )
    if forecast_directory and problems:
        return KickoffReadiness(None, tuple(problems), tuple(warnings), tuple(details))
    try:
        OddsAPIClient().ensure_single_quota_region()
    except OddsAPIError as exc:
        problems.append(str(exc))
    live_problems, frames = verify_live_snapshots(season)
    problems.extend(live_problems)
    target = None
    try:
        target = resolve_kickoff_target(
            season,
            frames,
            as_of=now,
            game_ids=game_ids,
            cluster_minutes=cluster_minutes,
            schedule=(
                load_weekly_capture_forecast(forecast_directory, season)[
                    "schedule_coverage"
                ]
                if forecast_directory
                else None
            ),
        )
        plan = plan_kickoff_window(
            target,
            as_of=now,
            lead_minutes=lead_minutes,
            post_minutes=post_minutes,
            interval_seconds=interval_seconds,
        )
        market_plan = plan_kickoff_poll_markets(target, plan, interval_seconds)
    except ValueError as exc:
        problems.append(str(exc))
    else:
        details.append(
            f"target window: {target.first_kickoff.isoformat()} to "
            f"{target.last_kickoff.isoformat()} ({len(target.game_ids)} games)"
        )
        details.append(
            f"poll plan: {plan.polls} polls from {plan.starts_at.isoformat()} "
            f"through at least {plan.ends_at.isoformat()}"
        )
        preflight_problems, preflight_warnings, preflight_details = (
            check_live_preflight(
                frames,
                target,
                as_of=now,
                planned_markets=market_plan,
                min_quota=min_quota,
                max_poll_age_minutes=max_poll_age_minutes,
                max_offer_staleness_seconds=max_offer_staleness_seconds,
                min_providers=min_providers,
            )
        )
        problems.extend(preflight_problems)
        warnings.extend(preflight_warnings)
        details.extend(preflight_details)
    return KickoffReadiness(
        target=target,
        problems=tuple(dict.fromkeys(problems)),
        warnings=tuple(dict.fromkeys(warnings)),
        details=tuple(details),
    )


def format_readiness(result: KickoffReadiness) -> str:
    lines = ["READY" if result.ready else "NOT READY"]
    if result.target is not None:
        lines.extend(f"TARGET: {label}" for label in result.target.labels)
    lines.extend(f"OK: {detail}" for detail in result.details)
    lines.extend(f"WARNING: {warning}" for warning in result.warnings)
    lines.extend(f"BLOCKER: {problem}" for problem in result.problems)
    return "\n".join(lines)


def _paired_total_provider_updates(
    totals: pd.DataFrame, spread_providers: set[str]
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    required = {
        "provider_key",
        "provider_last_update",
        "selection",
        "point",
        "price",
    }
    if totals.empty or not required.issubset(totals.columns):
        return {}
    valid: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for provider, group in totals.groupby("provider_key", dropna=True):
        provider_key = str(provider)
        if provider_key not in spread_providers:
            continue
        candidate = group.copy()
        candidate["provider_last_update"] = pd.to_datetime(
            candidate["provider_last_update"], utc=True, errors="coerce"
        )
        candidate["point"] = pd.to_numeric(candidate["point"], errors="coerce")
        candidate["price"] = pd.to_numeric(candidate["price"], errors="coerce")
        candidate = candidate.dropna(
            subset=["provider_last_update", "selection", "point", "price"]
        )
        candidate = candidate[
            candidate["point"].map(lambda value: isfinite(float(value)))
            & candidate["price"].map(lambda value: isfinite(float(value)))
        ]
        candidate = candidate.sort_values(
            "provider_last_update", kind="stable"
        ).drop_duplicates("selection", keep="last")
        paired = candidate[candidate["selection"].isin(["over", "under"])]
        if set(paired["selection"]) != {"over", "under"}:
            continue
        if paired["point"].nunique() != 1:
            continue
        valid[provider_key] = (
            paired["provider_last_update"].min(),
            paired["provider_last_update"].max(),
        )
    return valid


def validate_completed_window(
    target: KickoffTarget,
    frames: dict[str, pd.DataFrame],
    anchors: pd.DataFrame,
    *,
    max_offer_staleness_seconds: float = 300.0,
    min_providers: int = 2,
    schedule: pd.DataFrame | None = None,
) -> tuple[list[str], list[str]]:
    """Prove each target froze a fresh pregame line after a live phase poll.

    A frozen forecast schedule must be passed on fresh runners, which restore
    only the weekly artifact and hold no stored Division I schedule.
    """
    problems: list[str] = []
    details: list[str] = []
    events = frames["events"].copy()
    offers = frames["offers"].copy()
    events["fetched_at"] = pd.to_datetime(
        events["fetched_at"], utc=True, errors="coerce"
    )
    offers["fetched_at"] = pd.to_datetime(
        offers["fetched_at"], utc=True, errors="coerce"
    )
    kickoff_by_game = dict.fromkeys(target.game_ids)
    if schedule is None:
        schedule = load_division_one_schedule(int(anchors["season"].iloc[0]))
    schedule = schedule.copy()
    schedule["start_date"] = pd.to_datetime(
        schedule["start_date"], utc=True, errors="coerce"
    )
    kickoff_by_game.update(
        schedule[schedule["game_id"].isin(target.game_ids)]
        .set_index("game_id")["start_date"]
        .to_dict()
    )
    provider_kickoffs = _provider_kickoffs(events, target.game_ids)

    for game_id, label in zip(target.game_ids, target.labels):
        scheduled = kickoff_by_game.get(game_id)
        kickoff = provider_kickoffs.get(game_id, scheduled)
        if kickoff is None or pd.isna(kickoff):
            problems.append(f"{label} has no kickoff timestamp")
            continue
        if scheduled is not None and pd.notna(scheduled) and kickoff != scheduled:
            drift = (kickoff - scheduled).total_seconds()
            details.append(
                f"{label}: provider kickoff {kickoff.isoformat()} is "
                f"{drift:+.0f}s from the frozen schedule"
            )
        row = anchors[anchors["game_id"].eq(game_id)]
        if row.empty:
            problems.append(f"{label} has no market anchor")
            continue
        if len(row) != 1:
            problems.append(f"{label} has more than one market anchor")
            continue
        anchor = row.iloc[0]
        closing_fetched_at = _utc(anchor["closing_fetched_at"])
        if closing_fetched_at >= kickoff:
            problems.append(f"{label} selected a post-kickoff snapshot")
        selected = offers[
            offers["game_id"].eq(game_id)
            & offers["snapshot_id"].eq(anchor["closing_snapshot_id"])
            & offers["market"].eq("spreads")
            & offers["selection"].eq("home")
            & offers["phase"].eq("pregame")
        ]
        providers = int(selected["provider_key"].nunique())
        if providers < min_providers:
            problems.append(
                f"{label} closing snapshot has {providers} providers; "
                f"minimum is {min_providers}"
            )
        provider_updates = pd.to_datetime(
            selected["provider_last_update"], utc=True, errors="coerce"
        )
        # A book that has not moved its number is still offering it, so the
        # consensus keeps every pregame provider; freshness is proven by the
        # count of providers that refreshed inside the staleness limit.
        fresh_spreads = 0
        if provider_updates.isna().any() or provider_updates.empty:
            problems.append(f"{label} closing providers lack update timestamps")
        else:
            if provider_updates.ge(kickoff).any():
                problems.append(
                    f"{label} selected a provider update at or after kickoff"
                )
            fresh_spreads = int(
                (kickoff - provider_updates)
                .dt.total_seconds()
                .le(max_offer_staleness_seconds)
                .sum()
            )
            if fresh_spreads < min_providers:
                problems.append(
                    f"{label} closing snapshot has {fresh_spreads} providers "
                    f"updated within {max_offer_staleness_seconds:.0f}s of "
                    f"kickoff; minimum is {min_providers}"
                )
        # Totals close in the last pregame poll that requested them, which
        # is the spread snapshot unless kickoff moved after that pull.
        pregame_totals = offers[
            offers["game_id"].eq(game_id)
            & offers["market"].eq("totals")
            & offers["phase"].eq("pregame")
            & offers["fetched_at"].le(closing_fetched_at)
        ]
        totals_snapshot = (
            pregame_totals.sort_values("fetched_at", kind="stable")["snapshot_id"].iloc[
                -1
            ]
            if not pregame_totals.empty
            else anchor["closing_snapshot_id"]
        )
        paired_total_updates = _paired_total_provider_updates(
            pregame_totals[pregame_totals["snapshot_id"].eq(totals_snapshot)],
            set(selected["provider_key"].dropna().astype(str)),
        )
        fresh_totals = sum(
            (kickoff - oldest).total_seconds() <= max_offer_staleness_seconds
            for oldest, _ in paired_total_updates.values()
        )
        if fresh_totals < min_providers:
            problems.append(
                f"{label} closing totals have {fresh_totals} paired providers "
                f"updated within {max_offer_staleness_seconds:.0f}s of kickoff; "
                f"minimum is {min_providers}"
            )
        if (
            paired_total_updates
            and max(newest for _, newest in paired_total_updates.values()) >= kickoff
        ):
            problems.append(
                f"{label} selected a total provider update at or after kickoff"
            )
        post_kickoff = events[
            events["game_id"].eq(game_id)
            & events["fetched_at"].ge(kickoff)
            & events["phase"].isin(["live", "final"])
        ]
        if post_kickoff.empty:
            problems.append(
                f"{label} has no stored post-kickoff live or final provider state"
            )
        if not any(label in problem for problem in problems):
            live_offers = int(
                offers[
                    offers["game_id"].eq(game_id)
                    & offers["fetched_at"].ge(kickoff)
                    & offers["phase"].eq("live")
                ].shape[0]
            )
            details.append(
                f"{label}: froze {anchor['closing_snapshot_id']} at "
                f"{closing_fetched_at.isoformat()} from {providers} spread "
                f"providers ({fresh_spreads} fresh) and {len(paired_total_updates)} "
                f"paired-total providers ({fresh_totals} fresh) in "
                f"{totals_snapshot}; ignored {live_offers} live offers"
            )
    return problems, details


def _provider_kickoffs(
    events: pd.DataFrame, game_ids: tuple[int, ...]
) -> dict[int, pd.Timestamp]:
    """Kickoff per game as the provider reported it when the game went live.

    The frozen schedule is only a fallback: the closing line is the last
    pregame quote before the kickoff the provider actually committed to.
    """
    if events.empty or "commence_time" not in events:
        return {}
    started = events[
        events["game_id"].isin(game_ids) & events["phase"].isin(["live", "final"])
    ].copy()
    if started.empty:
        return {}
    started["commence_time"] = pd.to_datetime(
        started["commence_time"], utc=True, errors="coerce"
    )
    started = started.dropna(subset=["commence_time", "fetched_at"])
    started = started.sort_values("fetched_at", kind="stable").drop_duplicates(
        "game_id"
    )
    return {
        int(game_id): commence
        for game_id, commence in zip(started["game_id"], started["commence_time"])
    }


def _write_market_anchors(season: int, built: pd.DataFrame) -> None:
    expected = built[SERVING_ANCHOR_COLUMNS].reset_index(drop=True)
    artifact = serving_anchor_artifact(season, MARKET_ANCHOR_WEEK)
    store.write_processed(built, *artifact)
    stored = load_serving_anchors("serving", season=season, week=MARKET_ANCHOR_WEEK)
    if not stored.equals(expected):
        raise ValueError(
            f"stored {'/'.join(artifact)} does not round-trip through the "
            "serving anchor loader"
        )


def run_kickoff_window(
    season: int,
    week: int = 1,
    *,
    game_ids: list[int] | tuple[int, ...] | None = None,
    cluster_minutes: float = 15.0,
    lead_minutes: float = 5.0,
    post_minutes: float = 5.0,
    interval_seconds: float = 120.0,
    lookback_hours: float = 1.0,
    lookahead_hours: float = 2.0,
    min_quota: int = 50,
    max_failures: int = 3,
    max_wait_hours: float = 2.0,
    max_extension_minutes: float = 30.0,
    forecast_directory: str | None = None,
    max_forecast_age_hours: float = 48.0,
    max_source_age_hours: float = 48.0,
    max_offer_staleness_seconds: float = 300.0,
    min_providers: int = 2,
    progress=log.info,
    now=lambda: datetime.now(timezone.utc),
    sleep=time.sleep,
) -> KickoffRunResult:
    """Wait for one window, poll across kickoff, validate, and write anchors."""
    try:
        OddsAPIClient().ensure_single_quota_region()
    except OddsAPIError as exc:
        raise ValueError(str(exc)) from exc
    current = _utc(now())
    problems, warnings, details = check_forecast_readiness(
        season,
        week,
        as_of=current,
        forecast_directory=forecast_directory,
        max_forecast_age_hours=max_forecast_age_hours,
        max_source_age_hours=max_source_age_hours,
    )
    if problems:
        raise ValueError("; ".join(problems))
    for warning in warnings:
        progress(f"WARNING: {warning}")
    for detail in details:
        progress(f"OK: {detail}")

    schedule = (
        load_weekly_capture_forecast(forecast_directory, season)["schedule_coverage"]
        if forecast_directory
        else load_division_one_schedule(season)
    )

    replay_problems, frames = verify_live_snapshots(season)
    if replay_problems:
        raise ValueError("; ".join(replay_problems))
    target = resolve_kickoff_target(
        season,
        frames,
        as_of=current,
        game_ids=game_ids,
        cluster_minutes=cluster_minutes,
        schedule=schedule,
    )
    plan = plan_kickoff_window(
        target,
        as_of=current,
        lead_minutes=lead_minutes,
        post_minutes=post_minutes,
        interval_seconds=interval_seconds,
    )
    market_plan = plan_kickoff_poll_markets(target, plan, interval_seconds)
    if max_extension_minutes < 0:
        raise ValueError("--max-extension-minutes must be nonnegative")
    extension_polls = ceil(max_extension_minutes * 60 / interval_seconds)
    quota_problems, quota_details = _quota_problems(
        frames, market_plan, min_quota, extension_polls
    )
    if quota_problems:
        raise ValueError("; ".join(quota_problems))
    for detail in quota_details:
        progress(f"OK: {detail}")

    wait_seconds = (plan.starts_at - current).total_seconds()
    if wait_seconds > max_wait_hours * 3600:
        raise ValueError(
            f"polling window starts in {wait_seconds / 3600:.1f}h; maximum wait "
            f"is {max_wait_hours:.1f}h"
        )
    progress(
        f"targeting {len(target.game_ids)} game(s) from "
        f"{target.first_kickoff.isoformat()} through "
        f"{target.last_kickoff.isoformat()}"
    )
    if wait_seconds > 0:
        progress(f"waiting {wait_seconds / 60:.1f}m for polling window")
    while True:
        remaining = (plan.starts_at - _utc(now())).total_seconds()
        if remaining <= 0:
            break
        sleep(min(remaining, 60.0))

    current = _utc(now())
    active_plan = plan_kickoff_window(
        target,
        as_of=current,
        lead_minutes=lead_minutes,
        post_minutes=post_minutes,
        interval_seconds=interval_seconds,
    )
    planner = KickoffPollPlanner(
        target,
        interval_seconds=interval_seconds,
        post_minutes=post_minutes,
        max_polls=active_plan.polls + extension_polls,
        progress=progress,
        now=now,
    )
    completed = run_live_polling(
        season,
        schedule,
        polls=planner.max_polls,
        interval_seconds=interval_seconds,
        lookback_hours=lookback_hours,
        lookahead_hours=lookahead_hours,
        days_from=None,
        min_quota=min_quota,
        max_failures=max_failures,
        required_game_ids=target.game_ids,
        poll_markets=planner,
        progress=progress,
        sleep=sleep,
    )
    if planner.pending:
        unsettled = ", ".join(
            planner.labels[game_id] for game_id in sorted(planner.pending)
        )
        raise ValueError(
            f"{unsettled} never reached a live or final provider state after "
            f"{completed} polls; market anchors were not changed"
        )

    replay_problems, frames = verify_live_snapshots(season)
    if replay_problems:
        raise ValueError("; ".join(replay_problems))
    built = build_live_market_anchors(
        season,
        schedule=schedule if forecast_directory else None,
    )
    built = built[built["game_id"].isin(target.game_ids)].reset_index(drop=True)
    validation_problems, target_details = validate_completed_window(
        target,
        frames,
        built,
        max_offer_staleness_seconds=max_offer_staleness_seconds,
        min_providers=min_providers,
        schedule=schedule,
    )
    if validation_problems:
        raise ValueError(
            "; ".join(validation_problems) + "; market anchors were not changed"
        )
    _write_market_anchors(season, built)
    return KickoffRunResult(
        plan=active_plan,
        anchor_count=len(built),
        target_details=tuple(target_details),
        polls_completed=completed,
    )
