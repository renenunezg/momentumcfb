"""Bounded Tier 2 feed pilot; append-only evidence, no pregame writes.

The frozen baseline scores the provider's current situation. Historical play
prefixes are retained for process diagnostics; final-game aggregates never
enter a reconstructed earlier state. Source age is not guaranteed latency.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backend.cfbd.snapshots import capture, read_snapshot
from backend.features.ingame import build_process_evidence
from backend.model.ingame import build_serving_inputs, win_probability

ACTIVE = {"in progress", "in_progress", "inprogress", "live"}
FINAL = {"final", "completed"}


def live_plays(payload, season, week):
    teams = payload.get("teams", [])
    home = next((t for t in teams if t.get("homeAway") == "home"), None)
    away = next((t for t in teams if t.get("homeAway") == "away"), None)
    if home is None or away is None or len(teams) != 2:
        raise ValueError("live payload requires exactly one home and away team")
    names = {int(t["teamId"]): t["team"] for t in teams}
    rows: list[dict] = []
    seen: dict[str, str] = {}
    previous_score = (0, 0)
    for drive_number, drive in enumerate(payload.get("drives", []), 1):
        for play_number, play in enumerate(drive.get("plays", []), 1):
            identity = str(play["id"])
            encoded = json.dumps(play, sort_keys=True)
            if identity in seen:
                if seen[identity] != encoded:
                    raise ValueError("conflicting duplicate live play ID")
                continue
            seen[identity] = encoded
            team_id = int(play["teamId"])
            if team_id not in names:
                raise ValueError("unknown live play team")
            offense = names[team_id]
            is_home = team_id == int(home["teamId"])
            try:
                minutes, seconds = map(int, play["clock"].split(":"))
                clock = {"minutes": minutes, "seconds": seconds}
            except (ValueError, AttributeError):
                clock = None
            rows.append(
                dict(
                    id=identity,
                    game_id=int(payload["id"]),
                    season=season,
                    week=week,
                    season_type="regular",
                    drive_id=str(drive["id"]),
                    drive_number=drive_number,
                    play_number=play_number,
                    game_play_number=len(rows) + 1,
                    home=home["team"],
                    away=away["team"],
                    offense=offense,
                    defense=away["team"] if is_home else home["team"],
                    offense_score=play.get("homeScore" if is_home else "awayScore"),
                    defense_score=play.get("awayScore" if is_home else "homeScore"),
                    period=play.get("period"),
                    clock=clock,
                    wallclock=play.get("wallClock"),
                    down=play.get("down"),
                    distance=play.get("distance"),
                    yards_to_goal=play.get("yardsToGoal"),
                    yards_gained=play.get("yardsGained"),
                    scoring=(play.get("homeScore"), play.get("awayScore"))
                    != previous_score,
                    play_type=play.get("playType"),
                    play_text=play.get("playText"),
                    ppa=play.get("epa"),
                    pbp_source="cfbd_live",
                    raw_play_json=encoded,
                )
            )
            previous_score = (play.get("homeScore"), play.get("awayScore"))
    return pd.DataFrame(rows)


def score_snapshot(receipt, anchor, params, *, max_age_seconds=180):
    payload = receipt["payload"]
    result = dict(
        game_id=int(payload["id"]),
        fetched_at=receipt["fetched_at"],
        payload_sha256=receipt["payload_sha256"],
        status=payload.get("status"),
        win_probability=None,
        reason="inactive",
        source_age_seconds=None,
        request_seconds=receipt["request_seconds"],
        model_version="ingame_baseline_v1",
    )
    if str(payload.get("status", "")).lower() not in ACTIVE:
        return result
    if len(anchor) != 1 or int(anchor.iloc[0].game_id) != result["game_id"]:
        result["reason"] = "missing_pregame_anchor"
        return result
    # Require the original forecast receipt and kickoff, not just a line value.
    a = anchor.iloc[0]
    kickoff = pd.to_datetime(a.get("start_date"), utc=True, errors="coerce")
    published = pd.to_datetime(a.get("as_of"), utc=True, errors="coerce")
    fetched = pd.Timestamp(receipt["fetched_at"])
    if (
        pd.isna(kickoff)
        or pd.isna(published)
        or published >= kickoff
        or kickoff > fetched
    ):
        result["reason"] = "invalid_pregame_anchor"
        return result
    teams = payload.get("teams", [])
    home: dict = next((x for x in teams if x.get("homeAway") == "home"), {})
    away: dict = next((x for x in teams if x.get("homeAway") == "away"), {})
    if home.get("teamId") != a.get("home_team_id") or away.get("teamId") != a.get(
        "away_team_id"
    ):
        result["reason"] = "anchor_team_mismatch"
        return result
    plays = [p for d in payload.get("drives", []) for p in d.get("plays", [])]
    times = pd.to_datetime(
        [p.get("wallClock") for p in plays], utc=True, errors="coerce"
    )
    latest = times.max() if len(times) else pd.NaT
    age = (fetched - latest).total_seconds() if pd.notna(latest) else None
    result["source_age_seconds"] = age
    if age is None or not 0 <= age <= max_age_seconds:
        result["reason"] = "stale_or_unknown_source_time"
        return result
    period = payload.get("period")
    if not isinstance(period, int) or not 1 <= period <= 4:
        result["reason"] = "unsupported_period"
        return result
    try:
        minutes, seconds = map(int, payload["clock"].split(":"))
    except (ValueError, AttributeError, KeyError):
        result["reason"] = "missing_clock"
        return result
    if not 0 <= minutes * 60 + seconds <= 900 or not 0 <= seconds < 60:
        result["reason"] = "invalid_clock"
        return result
    possession = payload.get("possession")
    if possession not in (home.get("team"), away.get("team")):
        result["reason"] = "unknown_possession"
        return result
    values = [
        home.get("points"),
        away.get("points"),
        payload.get("yardsToGoal"),
        payload.get("down"),
        payload.get("distance"),
    ]
    if any(
        v is None or not isinstance(v, (int, float)) or not np.isfinite(v)
        for v in values
    ):
        result["reason"] = "incomplete_situation"
        return result
    if (
        min(values[:2]) < 0
        or not 0 <= values[2] <= 100
        or not 1 <= values[3] <= 4
        or values[4] < 0
    ):
        result["reason"] = "invalid_situation"
        return result
    state = pd.DataFrame(
        [
            dict(
                game_id=result["game_id"],
                seconds_remaining=(4 - period) * 900 + minutes * 60 + seconds,
                is_overtime=False,
                play_category="scrimmage",
                offense_is_home=possession == home["team"],
                yards_to_goal=values[2],
                home_margin=values[0] - values[1],
            )
        ]
    )
    inputs = build_serving_inputs(state, anchor)
    result.update(
        win_probability=float(win_probability(inputs, params)[0]),
        reason="scored",
        anchor_as_of=str(a.as_of),
        anchor_model_version=a.get("model_version"),
        anchor_source="frozen_pure_model",
    )
    return result


def run_pilot(
    client,
    anchors,
    params,
    destination,
    *,
    season,
    week,
    game_ids=(),
    polls=1,
    interval=60,
    max_games=1,
    compare_espn=False,
    root=None,
    sleep=time.sleep,
    now=lambda: datetime.now(timezone.utc),
):
    if not 1 <= polls <= 50 or not 1 <= max_games <= 3 or not 5 <= interval <= 60:
        raise ValueError("pilot requires 1-50 polls, 1-3 games, 5-60 second interval")
    if len(set(game_ids)) > max_games:
        raise ValueError("game allowlist exceeds pilot maximum")
    client.ensure_budget(polls * (1 + max_games))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    chosen = set(map(int, game_ids))
    summaries = []
    comparisons = []
    for poll in range(polls):
        board_path = capture(client, "/scoreboard", {}, root=root)
        board_receipt = read_snapshot(board_path)
        board = board_receipt["payload"]
        active = {
            int(g["id"]): g for g in board if str(g.get("status", "")).lower() in ACTIVE
        }
        if not chosen:
            chosen = set(sorted(set(active) & set(anchors.game_id))[:max_games])
        if compare_espn:
            from backend.serving.feed_comparison import observe_comparison

            for game_id in sorted(chosen):
                comparison = observe_comparison(
                    board_receipt, anchors, game_id, root=root
                )
                comparison.update(poll=poll, cfbd_snapshot=str(board_path))
                comparisons.append(comparison)
                # Persist every pair so a later feed failure cannot erase it.
                (
                    destination / f"comparison_{game_id}_{board_path.stem}.json"
                ).write_text(json.dumps(comparison, allow_nan=False) + "\n")
        for game_id in sorted(chosen & set(active)):
            path = capture(
                client,
                "/live/plays",
                {"gameId": game_id},
                root=root,
                object_response=True,
            )
            receipt = read_snapshot(path)
            if int(receipt["payload"]["id"]) != game_id:
                raise ValueError("live response game differs from request")
            frame = live_plays(receipt["payload"], season, week)
            output = destination / f"{game_id}_{path.stem}"
            frame.to_parquet(output.with_suffix(".parquet"), index=False)
            if not frame.empty:
                build_process_evidence(frame).to_parquet(
                    output.with_name(output.name + "_process.parquet"), index=False
                )
            summary = score_snapshot(
                receipt, anchors[anchors.game_id.eq(game_id)], params
            )
            summary.update(snapshot=str(path), poll=poll, plays=len(frame))
            output.with_suffix(".json").write_text(
                json.dumps(summary, allow_nan=False) + "\n"
            )
            summaries.append(summary)
        final_ids = {
            int(g["id"]) for g in board if str(g.get("status", "")).lower() in FINAL
        }
        if chosen and chosen <= final_ids:
            break
        if poll + 1 < polls:
            sleep(interval)
    result = pd.DataFrame(summaries)
    stamp = pd.Timestamp(now()).strftime("%Y%m%dT%H%M%S%fZ")
    result.to_parquet(destination / f"pilot_{stamp}.parquet", index=False)
    if comparisons:
        pd.DataFrame(comparisons).to_parquet(
            destination / f"comparisons_{stamp}.parquet", index=False
        )
    return result
