"""Opponent-adjusted player value above a positional replacement.

Every credited play is measured against what an average player produces
against that opponent unit, using the unit rating fit strictly before the
game's week so a player's own game never feeds the rating that adjusts it.
Rates shrink toward the position mean before a replacement baseline is
subtracted, and the season is published as one cumulative snapshot per week.
"""

import logging
import re
from datetime import datetime

import numpy as np
import pandas as pd

from backend.etl import store
from backend.features.ingame import build_game_states
from backend.features.possessions import classify_plays
from backend.features.units import build_unit_games
from backend.model.ingame import (
    build_serving_inputs,
    load_baseline_params,
    win_probability,
)
from backend.model.unit_ratings import fit_unit_ratings
from backend.model.weekly import load_weekly_games
from backend.players.credit import assign_play_credit
from backend.players.ingest import read_season_source, read_weekly

log = logging.getLogger(__name__)

MODEL_VERSION = "cfb_player_value_v1"
# Games are the opportunity unit for replacement and shrinkage: a defender's
# credited plays are disruption events, not snaps, so plays cannot be the
# denominator on one side of the ball and the opportunity count on the other.
PRIOR_GAMES = 4.0
REPLACEMENT_PERCENTILE = 0.30
QUALIFYING_GAMES = 6
MIN_QUALIFIED_PLAYERS = 10
FCS_OPPONENT_WEIGHT = 0.5
OPPONENT_EFFECT_PRIOR_GAMES = 3.0

POSITION_GROUPS = {
    "QB": "QB",
    "RB": "RB",
    "FB": "RB",
    "HB": "RB",
    "TB": "RB",
    "WR": "WR",
    "TE": "TE",
    "OL": "OL",
    "OT": "OL",
    "OG": "OL",
    "C": "OL",
    "G": "OL",
    "T": "OL",
    "DL": "DL",
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "EDGE": "DL",
    "LB": "LB",
    "OLB": "LB",
    "ILB": "LB",
    "MLB": "LB",
    "DB": "DB",
    "CB": "DB",
    "S": "DB",
    "FS": "DB",
    "SS": "DB",
    "NB": "DB",
    "PK": "K",
    "K": "K",
    "P": "P",
    "LS": "LS",
}
ROLE_GROUPS = {
    "passer": "QB",
    "rusher": "RB",
    "receiver": "WR",
    "target": "WR",
    "fumbler": "RB",
    "returner": "WR",
    "sacker": "DL",
    "interceptor": "DB",
    "pass_defender": "DB",
    "fumble_forcer": "LB",
    "fumble_recoverer": "LB",
}
CHANNEL_UNITS = {
    "rush": ("rush_offense", "rush_defense", "rush_plays"),
    "pass": ("pass_offense", "pass_defense", "pass_plays"),
}

SNAPSHOT_COLUMNS = [
    "season",
    "week",
    "as_of",
    "model_version",
    "athlete_id",
    "athlete_name",
    "team",
    "team_id",
    "classification",
    "position",
    "position_group",
    "games",
    "plays",
    "raw_epa",
    "adjusted_epa",
    "adjusted_rate",
    "adjusted_per_game",
    "shrunk_per_game",
    "replacement_per_game",
    "value_above_replacement",
    "wpa",
    "fcs_play_share",
    "overall_rank",
    "position_rank",
]


def season_unit_effects(games: pd.DataFrame, unit_games: pd.DataFrame) -> pd.DataFrame:
    """Per-play opponent effects for every (model_week, team), fit pregame.

    Ratings are PPA per game; dividing by the training window's average plays
    per game on that channel turns them into per-play effects. A week with no
    completed prior game carries the previous week's fit, and the first week
    of a season carries zero (an average opponent).
    """
    weeks = sorted(int(week) for week in games["model_week"].dropna().unique())
    teams = pd.concat([games["home_team"], games["away_team"]]).dropna().unique()
    previous = pd.DataFrame(
        {
            "team": teams,
            "rush_offense": 0.0,
            "pass_offense": 0.0,
            "rush_defense": 0.0,
            "pass_defense": 0.0,
        }
    )
    frames = []
    for week in weeks:
        if week < 1:
            continue
        window = games[games["model_week"].lt(week)]
        completed = window[window["completed"].fillna(False).astype(bool)]
        current = None
        if not completed.empty:
            latest = pd.to_datetime(completed["start_date"], utc=True).max()
            as_of = (latest + pd.Timedelta(days=1)).to_pydatetime()
            try:
                fitted = fit_unit_ratings(unit_games, games, week, as_of).frame
            except ValueError as exc:
                log.info(f"unit fit week {week} skipped: {exc}")
            else:
                window_units = unit_games[
                    unit_games["game_id"].isin(completed["game_id"])
                ]
                current = fitted[
                    [
                        "team",
                        *sum(
                            (
                                [offense, defense]
                                for offense, defense, _ in CHANNEL_UNITS.values()
                            ),
                            [],
                        ),
                    ]
                ].copy()
                # A four-game unit rating swings far more than a season's,
                # so the per-play effect is shrunk toward an average opponent
                # by games played until the rating has earned its weight.
                played = pd.concat(
                    [completed["home_team"], completed["away_team"]]
                ).value_counts()
                games_played = current["team"].map(played).fillna(0.0)
                confidence = games_played / (games_played + OPPONENT_EFFECT_PRIOR_GAMES)
                for offense, defense, plays_column in CHANNEL_UNITS.values():
                    per_game = float(window_units[plays_column].mean())
                    if not np.isfinite(per_game) or per_game <= 0:
                        per_game = 1.0
                    current[offense] = current[offense] / per_game * confidence
                    current[defense] = current[defense] / per_game * confidence
        if current is None:
            current = previous.copy()
        current["model_week"] = week
        frames.append(current)
        previous = current.drop(columns="model_week")
    return pd.concat(frames, ignore_index=True)


def adjust_credit(
    credit: pd.DataFrame, effects: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    """Add the opponent-adjusted credit and the FCS weight to every row."""
    schedule = games[
        [
            "game_id",
            "model_week",
            "home_team",
            "away_team",
            "home_classification",
            "away_classification",
        ]
    ]
    out = credit.merge(schedule, on="game_id", how="inner")
    opponent_class = np.where(
        out["opponent"].eq(out["home_team"]),
        out["home_classification"],
        out["away_classification"],
    )
    out["weight"] = np.where(
        pd.Series(opponent_class).fillna("").str.lower().eq("fcs").to_numpy(),
        FCS_OPPONENT_WEIGHT,
        1.0,
    )
    out = out.merge(
        effects.rename(columns={"team": "opponent"}),
        on=["model_week", "opponent"],
        how="left",
    )
    effect = np.zeros(len(out))
    for channel, (offense, defense, _) in CHANNEL_UNITS.items():
        on_channel = out["channel"].eq(channel).to_numpy()
        is_offense = out["side"].eq("offense").to_numpy()
        # Offense: expected EPA against this defense is minus its counter
        # rating, so credit above expectation adds the counter back. Defense:
        # expected EPA allowed is the offense rating, so stopping a strong
        # offense is worth more than the raw play.
        effect = np.where(on_channel & is_offense, out[defense].fillna(0.0), effect)
        effect = np.where(on_channel & ~is_offense, out[offense].fillna(0.0), effect)
    out["adjusted_epa"] = (out["credit_epa"] + out["share"] * effect) * out["weight"]
    out["raw_epa"] = out["credit_epa"] * out["weight"]
    return out.drop(
        columns=[
            "home_team",
            "away_team",
            "home_classification",
            "away_classification",
            *sum(([o, d] for o, d, _ in CHANNEL_UNITS.values()), []),
        ]
    )


def _historical_win_probabilities(season: int) -> pd.DataFrame | None:
    try:
        stored = store.read_processed(
            "ingame",
            "baseline_predictions.parquet",
            columns=[
                "season",
                "game_id",
                "play_index",
                "source_play_id",
                "offense_is_home",
                "win_probability",
                "home_win",
            ],
        )
    except FileNotFoundError:
        return None
    stored = stored[stored["season"].eq(season)]
    if stored.empty:
        return None
    return stored.drop(columns="season")


def _projection_anchors(season: int, games: pd.DataFrame) -> pd.DataFrame:
    """Per game, the projection published for the week it was played."""
    frames = []
    # Serving anchors are the projections the live model actually served,
    # including the week-zero slate the week-one artifact covers; week 00 is
    # the market closing capture and is not a model projection.
    for name in store.processed_names("serving"):
        match = re.fullmatch(rf"anchors_{season}_(\d\d)", name)
        if match is None or int(match.group(1)) < 1:
            continue
        frame = store.read_processed(
            "serving",
            f"{name}.parquet",
            columns=["game_id", "home_margin", "margin_sd"],
        )
        frame["projection_week"] = int(match.group(1))
        frames.append(frame)
    for parts in (("preseason", "projections"), ("projections",)):
        for name in store.processed_names(*parts):
            if not name.startswith(f"{season}_"):
                continue
            frame = store.read_processed(
                *parts,
                f"{name}.parquet",
                columns=["game_id", "home_margin", "margin_sd"],
            )
            frame["projection_week"] = int(name.split("_")[1])
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no projection artifacts for {season}")
    projections = pd.concat(frames, ignore_index=True).merge(
        games[["game_id", "model_week"]], on="game_id", how="inner"
    )
    projections = projections[
        projections["projection_week"].le(projections["model_week"].clip(lower=1))
    ]
    latest = projections.sort_values("projection_week").drop_duplicates(
        "game_id", keep="last"
    )
    return latest[["game_id", "model_week", "home_margin", "margin_sd"]]


def _live_win_probabilities(
    season: int, plays: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    states = build_game_states(plays)
    anchors = _projection_anchors(season, games)
    inputs = build_serving_inputs(states, anchors)
    params = load_baseline_params(
        store.read_processed("ingame", "baseline_summary.parquet")
    )
    inputs["win_probability"] = win_probability(inputs, params)
    finals = games[["game_id", "home_points", "away_points"]]
    inputs = inputs.merge(finals, on="game_id", how="left")
    inputs["home_win"] = inputs["home_points"].gt(inputs["away_points"])
    return inputs[
        [
            "game_id",
            "play_index",
            "source_play_id",
            "offense_is_home",
            "win_probability",
            "home_win",
        ]
    ]


def play_wpa(states: pd.DataFrame) -> pd.DataFrame:
    """Win probability added per play, from the offense's point of view.

    A state is the situation before its play, so the play's effect is the
    next state's probability minus its own; the last play resolves to the
    final result.
    """
    ordered = states.sort_values(["game_id", "play_index"]).reset_index(drop=True)
    next_wp = ordered.groupby("game_id")["win_probability"].shift(-1)
    final = ordered["home_win"].astype(float)
    home_delta = next_wp.fillna(final) - ordered["win_probability"]
    ordered["wpa"] = np.where(ordered["offense_is_home"], home_delta, -home_delta)
    return ordered[["source_play_id", "wpa"]].rename(
        columns={"source_play_id": "play_id"}
    )


def finalize_credit(credit: pd.DataFrame, wpa: pd.DataFrame) -> pd.DataFrame:
    """Attach win probability added and zero out garbage-time EPA.

    Garbage time counts for win probability, which already discounts it,
    but not for EPA value or the play count behind the rate.
    """
    wpa = wpa.assign(play_id=wpa["play_id"].astype(str))
    out = credit.merge(wpa, on="play_id", how="left")
    sign = np.where(out["side"].eq("offense"), 1.0, -1.0)
    out["wpa"] = out["wpa"].fillna(0.0) * out["share"] * sign * out["weight"]
    competitive = out["is_competitive"].astype(float)
    out["plays"] = out["weight"] * competitive
    out["raw_epa"] = out["raw_epa"] * competitive
    out["adjusted_epa"] = out["adjusted_epa"] * competitive
    return out


def _event_values(plays: pd.DataFrame) -> tuple[float, float]:
    """Defensive value of an untagged tackle for loss and pass defended."""
    competitive = plays[plays["is_competitive"]]
    losses = competitive[competitive["is_rush"] & competitive["yards_gained"].lt(0)]
    incompletions = competitive[
        competitive["is_pass"] & competitive["play_type"].eq("Pass Incompletion")
    ]
    tfl = -float(losses["epa"].mean()) if not losses.empty else 0.0
    pd_value = -float(incompletions["epa"].mean()) if not incompletions.empty else 0.0
    return tfl, pd_value


def box_disruptions(
    box: pd.DataFrame,
    credit: pd.DataFrame,
    plays: pd.DataFrame,
    effects: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Non-sack tackles for loss and untagged passes defended from the box.

    The play feed tags sacks, interceptions, and some pass breakups; the box
    score carries the rest. Each untagged event is valued at the season's
    average EPA for that event and adjusted for the opponent offense.
    """
    defensive = box[box["category"].eq("defensive")].copy()
    defensive["stat"] = pd.to_numeric(defensive["stat"], errors="coerce").fillna(0.0)
    wide = defensive.pivot_table(
        index=["game_id", "team", "athlete_id", "athlete_name"],
        columns="stat_name",
        values="stat",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    for column in ("TFL", "SACKS", "PD"):
        if column not in wide:
            wide[column] = 0.0
    wide["athlete_id"] = wide["athlete_id"].astype(str)

    tagged = (
        credit[credit["role"].eq("pass_defender")]
        .groupby(["game_id", "athlete_id"])["share"]
        .size()
        .rename("tagged_pd")
        .reset_index()
    )
    wide = wide.merge(tagged, on=["game_id", "athlete_id"], how="left")
    wide["tfl_events"] = (wide["TFL"] - wide["SACKS"]).clip(lower=0.0)
    wide["pd_events"] = (wide["PD"] - wide["tagged_pd"].fillna(0.0)).clip(lower=0.0)
    wide = wide[(wide["tfl_events"] > 0) | (wide["pd_events"] > 0)]

    schedule = games[
        [
            "game_id",
            "model_week",
            "home_team",
            "away_team",
            "home_classification",
            "away_classification",
        ]
    ]
    out = wide.merge(schedule, on="game_id", how="inner")
    out["opponent"] = np.where(
        out["team"].eq(out["home_team"]), out["away_team"], out["home_team"]
    )
    opponent_class = np.where(
        out["opponent"].eq(out["home_team"]),
        out["home_classification"],
        out["away_classification"],
    )
    out["weight"] = np.where(
        pd.Series(opponent_class).fillna("").str.lower().eq("fcs").to_numpy(),
        FCS_OPPONENT_WEIGHT,
        1.0,
    )
    out = out.merge(
        effects.rename(columns={"team": "opponent"}),
        on=["model_week", "opponent"],
        how="left",
    )
    tfl_value, pd_value = _event_values(plays)
    raw = out["tfl_events"] * tfl_value + out["pd_events"] * pd_value
    adjusted = out["tfl_events"] * (tfl_value + out["rush_offense"].fillna(0.0)) + out[
        "pd_events"
    ] * (pd_value + out["pass_offense"].fillna(0.0))
    return pd.DataFrame(
        {
            "game_id": out["game_id"],
            "athlete_id": out["athlete_id"],
            "athlete_name": out["athlete_name"],
            "team": out["team"],
            "model_week": out["model_week"],
            "side": "defense",
            "role": "box_disruption",
            "plays": (out["tfl_events"] + out["pd_events"]) * out["weight"],
            "raw_epa": raw * out["weight"],
            "adjusted_epa": adjusted * out["weight"],
            "wpa": 0.0,
            "weight": out["weight"],
        }
    )


def _position_groups(roster: pd.DataFrame, credited: pd.DataFrame) -> pd.DataFrame:
    positions = roster[["id", "position"]].rename(columns={"id": "athlete_id"})
    positions["athlete_id"] = positions["athlete_id"].astype(str)
    positions = positions.drop_duplicates("athlete_id")
    positions["position_group"] = positions["position"].map(POSITION_GROUPS)
    # A player the roster misses or lists at an unmapped spot is grouped by
    # the role that carries most of their plays.
    dominant = (
        credited.groupby(["athlete_id", "role"])["plays"]
        .sum()
        .reset_index()
        .sort_values("plays", ascending=False)
        .drop_duplicates("athlete_id")
    )
    dominant["role_group"] = dominant["role"].map(ROLE_GROUPS).fillna("DL")
    out = dominant[["athlete_id", "role_group"]].merge(
        positions, on="athlete_id", how="left"
    )
    out["position_group"] = out["position_group"].fillna(out["role_group"])
    out["position"] = out["position"].fillna(out["role_group"])
    return out[["athlete_id", "position", "position_group"]]


def _rank_snapshot(week_frame: pd.DataFrame, week: int) -> pd.DataFrame:
    frame = week_frame.copy()
    frame["adjusted_rate"] = (frame["adjusted_epa"] / frame["plays"]).where(
        frame["plays"].gt(0), 0.0
    )
    frame["adjusted_per_game"] = frame["adjusted_epa"] / frame["games"]
    group_mean = frame.groupby("position_group").apply(
        lambda g: np.average(g["adjusted_per_game"], weights=g["games"]),
        include_groups=False,
    )
    frame["group_mean"] = frame["position_group"].map(group_mean)
    frame["shrunk_per_game"] = (
        frame["games"] * frame["adjusted_per_game"] + PRIOR_GAMES * frame["group_mean"]
    ) / (frame["games"] + PRIOR_GAMES)

    # Early in the season everyone qualifies; by mid-season a replacement
    # must have played half the weeks so a one-game cameo cannot set it.
    threshold = max(1, min(QUALIFYING_GAMES, week // 2))
    qualified = frame[frame["games"].ge(threshold)]
    by_group = qualified.groupby("position_group")["shrunk_per_game"]
    counts = by_group.size()
    replacement = by_group.quantile(REPLACEMENT_PERCENTILE)
    side_of_group = frame.drop_duplicates("position_group").set_index("position_group")[
        "side"
    ]
    pooled = qualified.groupby("side")["shrunk_per_game"].quantile(
        REPLACEMENT_PERCENTILE
    )
    thin = counts.index[counts.lt(MIN_QUALIFIED_PLAYERS)]
    for group in thin:
        replacement[group] = pooled.get(side_of_group.get(group), 0.0)
    frame["replacement_per_game"] = frame["position_group"].map(replacement)
    frame["replacement_per_game"] = (
        frame["replacement_per_game"].fillna(frame["side"].map(pooled)).fillna(0.0)
    )
    frame["value_above_replacement"] = frame["games"] * (
        frame["shrunk_per_game"] - frame["replacement_per_game"]
    )
    frame = frame.sort_values(
        ["value_above_replacement", "plays"], ascending=[False, False]
    ).reset_index(drop=True)
    frame["overall_rank"] = np.arange(1, len(frame) + 1)
    frame["position_rank"] = frame.groupby("position_group").cumcount() + 1
    frame["week"] = week
    return frame


def weekly_snapshots(
    credited: pd.DataFrame,
    positions: pd.DataFrame,
    teams: pd.DataFrame,
    season: int,
    as_of: datetime,
) -> pd.DataFrame:
    """One cumulative row per player per model week."""
    per_week = (
        credited.groupby(["athlete_id", "model_week"])
        .agg(
            plays=("plays", "sum"),
            raw_epa=("raw_epa", "sum"),
            adjusted_epa=("adjusted_epa", "sum"),
            wpa=("wpa", "sum"),
            fcs_plays=("fcs_plays", "sum"),
            games=("game_id", "nunique"),
        )
        .reset_index()
    )
    identity = (
        credited.sort_values("model_week")
        .groupby("athlete_id")
        .agg(athlete_name=("athlete_name", "last"), team=("team", "last"))
        .reset_index()
    )
    side = (
        credited.groupby(["athlete_id", "side"])["plays"]
        .sum()
        .reset_index()
        .sort_values("plays", ascending=False)
        .drop_duplicates("athlete_id")[["athlete_id", "side"]]
    )
    weeks = sorted(int(week) for week in per_week["model_week"].unique())
    snapshots = []
    for week in weeks:
        cumulative = (
            per_week[per_week["model_week"].le(week)]
            .groupby("athlete_id")
            .agg(
                plays=("plays", "sum"),
                raw_epa=("raw_epa", "sum"),
                adjusted_epa=("adjusted_epa", "sum"),
                wpa=("wpa", "sum"),
                fcs_plays=("fcs_plays", "sum"),
                games=("games", "sum"),
            )
            .reset_index()
        )
        cumulative = cumulative[cumulative["games"].gt(0)]
        cumulative = (
            cumulative.merge(identity, on="athlete_id")
            .merge(side, on="athlete_id")
            .merge(positions, on="athlete_id", how="left")
        )
        snapshots.append(_rank_snapshot(cumulative, week))
    out = pd.concat(snapshots, ignore_index=True)
    # A garbage-time-only appearance has games but no counted plays.
    out["fcs_play_share"] = (out["fcs_plays"] / out["plays"]).where(
        out["plays"].gt(0), 0.0
    )
    out["season"] = season
    out["as_of"] = as_of.isoformat()
    out["model_version"] = MODEL_VERSION
    catalog = teams[["school", "id", "classification"]].rename(
        columns={"school": "team", "id": "team_id"}
    )
    out = out.merge(catalog, on="team", how="left")
    return out[SNAPSHOT_COLUMNS]


def split_half_reliability(
    credited: pd.DataFrame, positions: pd.DataFrame
) -> pd.DataFrame:
    """Correlation of adjusted rates between odd and even games per group.

    This is the only criterion available for ever tuning the credit shares:
    there is no ground truth for responsibility, only whether the resulting
    rates describe a stable skill.
    """
    per_game = (
        credited.groupby(["athlete_id", "game_id"])
        .agg(plays=("plays", "sum"), adjusted_epa=("adjusted_epa", "sum"))
        .reset_index()
        .sort_values(["athlete_id", "game_id"])
    )
    per_game["half"] = per_game.groupby("athlete_id").cumcount() % 2
    halves = (
        per_game.groupby(["athlete_id", "half"])
        .agg(games=("game_id", "size"), adjusted_epa=("adjusted_epa", "sum"))
        .reset_index()
    )
    halves["rate"] = halves["adjusted_epa"] / halves["games"]
    empty = pd.DataFrame(
        columns=["position_group", "players", "split_half_correlation"]
    )
    if halves["half"].nunique() < 2:
        return empty
    wide = halves.pivot(index="athlete_id", columns="half", values=["rate", "games"])
    wide.columns = [f"{name}_{half}" for name, half in wide.columns]
    wide = wide.dropna()
    wide = wide[
        (wide["games_0"] >= QUALIFYING_GAMES / 2)
        & (wide["games_1"] >= QUALIFYING_GAMES / 2)
    ]
    wide = wide.reset_index().merge(positions, on="athlete_id", how="left")
    rows = []
    for group, frame in wide.groupby("position_group"):
        if len(frame) < MIN_QUALIFIED_PLAYERS:
            continue
        rows.append(
            {
                "position_group": group,
                "players": len(frame),
                "split_half_correlation": float(
                    np.corrcoef(frame["rate_0"], frame["rate_1"])[0, 1]
                ),
            }
        )
    return pd.DataFrame(rows, columns=empty.columns)


def build_player_values(season: int, as_of: datetime) -> dict[str, pd.DataFrame]:
    """Build every player-value artifact for one season from raw sources."""
    plays = classify_plays(store.read_season_pbp(season))
    games = load_weekly_games(season)
    unit_games = build_unit_games(plays)
    effects = season_unit_effects(games, unit_games)

    play_stats = read_weekly(season, "play_stats")
    credit = adjust_credit(assign_play_credit(play_stats, plays), effects, games)

    states = _historical_win_probabilities(season)
    if states is None:
        states = _live_win_probabilities(season, plays, games)
    credit = finalize_credit(credit, play_wpa(states))

    box = read_weekly(season, "box")
    disruptions = box_disruptions(box, credit, plays, effects, games)
    columns = [
        "game_id",
        "athlete_id",
        "athlete_name",
        "team",
        "model_week",
        "side",
        "role",
        "plays",
        "raw_epa",
        "adjusted_epa",
        "wpa",
        "weight",
    ]
    credited = pd.concat([credit[columns], disruptions[columns]], ignore_index=True)
    credited["fcs_plays"] = np.where(credited["weight"].lt(1.0), credited["plays"], 0.0)

    roster = read_season_source(season, "roster")
    teams = read_season_source(season, "teams")
    positions = _position_groups(roster, credited)
    snapshots = weekly_snapshots(credited, positions, teams, season, as_of)
    reliability = split_half_reliability(credited, positions)
    return {
        "player_values": snapshots,
        "unit_effects": effects,
        "reliability": reliability,
        "credited": credited,
    }
