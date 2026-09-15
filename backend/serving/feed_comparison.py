"""Paired upstream observations; clock differences are not latency estimates."""

from datetime import datetime, timezone
from time import perf_counter

import pandas as pd
import requests

from backend.cfbd.snapshots import read_snapshot, save_snapshot, snapshot_root

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
)


def capture_espn(game_id, kickoff, *, root=None):
    params = {
        "dates": pd.Timestamp(kickoff).strftime("%Y%m%d"),
        "groups": 80,
        "limit": 400,
    }
    requested = datetime.now(timezone.utc)
    started = perf_counter()
    response = requests.get(ESPN_SCOREBOARD, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()
    # Archive the upstream response, including missing-game evidence.
    return save_snapshot(
        root or snapshot_root(),
        "/espn/scoreboard",
        params,
        payload,
        requested,
        datetime.now(timezone.utc),
        request_seconds=perf_counter() - started,
    )


def compare_scoreboards(cfbd_receipt, espn_receipt, game_id):
    cfbd = next((g for g in cfbd_receipt["payload"] if int(g["id"]) == game_id), None)
    espn = next(
        (
            g
            for g in espn_receipt["payload"].get("events", [])
            if str(g.get("id")) == str(game_id)
        ),
        None,
    )
    result = dict(
        game_id=game_id,
        cfbd_fetched_at=cfbd_receipt["fetched_at"],
        espn_fetched_at=espn_receipt["fetched_at"],
        request_gap_seconds=(
            pd.Timestamp(espn_receipt["fetched_at"])
            - pd.Timestamp(cfbd_receipt["fetched_at"])
        ).total_seconds(),
        reason="missing_game",
        scores_agree=None,
    )
    if cfbd is None or espn is None:
        return result
    competition = espn["competitions"][0]
    competitors = {c["homeAway"]: c for c in competition["competitors"]}
    for side in ("home", "away"):
        if str(competitors[side]["id"]) != str(cfbd[f"{side}Team"]["id"]):
            result["reason"] = "team_mismatch"
            return result
    status = competition.get("status", espn.get("status", {}))
    result.update(
        reason="paired",
        cfbd_status=cfbd.get("status"),
        espn_status=status.get("type", {}).get("state"),
        cfbd_clock=cfbd.get("clock"),
        espn_clock=status.get("displayClock"),
        cfbd_period=cfbd.get("period"),
        espn_period=status.get("period"),
    )
    for side in ("home", "away"):
        result[f"cfbd_{side}_score"] = cfbd[f"{side}Team"].get("points")
        score = competitors[side].get("score")
        result[f"espn_{side}_score"] = float(score) if score is not None else None
    comparable = all(
        result[f"{source}_{side}_score"] is not None
        for source in ("cfbd", "espn")
        for side in ("home", "away")
    )
    if comparable:
        result["scores_agree"] = all(
            result[f"cfbd_{side}_score"] == result[f"espn_{side}_score"]
            for side in ("home", "away")
        )
    return result


def observe_comparison(board_receipt, anchors, game_id, *, root=None):
    anchor = anchors[anchors.game_id.eq(game_id)]
    if len(anchor) != 1 or pd.isna(anchor.iloc[0].get("start_date")):
        return dict(game_id=game_id, reason="missing_comparison_kickoff")
    try:
        path = capture_espn(game_id, anchor.iloc[0].start_date, root=root)
        result = compare_scoreboards(board_receipt, read_snapshot(path), game_id)
        result["espn_snapshot"] = str(path)
        return result
    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        # A comparison outage must not discard the independent CFBD evidence.
        return dict(
            game_id=game_id, reason="espn_comparison_failed", error=type(exc).__name__
        )
