import numpy as np
import pandas as pd

SCRIMMAGE_TYPES = frozenset(
    {
        "Fumble",
        "Fumble Recovery (Opponent)",
        "Fumble Recovery (Opponent) Touchdown",
        "Fumble Recovery (Own)",
        "Fumble Recovery (Own) Touchdown",
        "Fumble Return Touchdown",
        "Interception",
        "Interception Return",
        "Interception Return Touchdown",
        "Pass",
        "Pass Completion",
        "Pass Incompletion",
        "Pass Interception Return",
        "Pass Reception",
        "Passing Touchdown",
        "Rush",
        "Rushing Touchdown",
        "Sack",
        "Safety",
    }
)
RUSH_TYPES = frozenset({"Rush", "Rushing Touchdown"})
PASS_TYPES = frozenset(
    {
        "Interception",
        "Interception Return",
        "Interception Return Touchdown",
        "Pass",
        "Pass Completion",
        "Pass Incompletion",
        "Pass Interception Return",
        "Pass Reception",
        "Passing Touchdown",
        "Sack",
    }
)
TURNOVER_TYPES = frozenset(
    {
        "Fumble Recovery (Opponent)",
        "Fumble Recovery (Opponent) Touchdown",
        "Fumble Return Touchdown",
        "Interception",
        "Interception Return",
        "Interception Return Touchdown",
        "Pass Interception Return",
    }
)
SPECIAL_TEAMS_TYPES = frozenset(
    {
        "Blocked Field Goal",
        "Blocked Field Goal Touchdown",
        "Blocked Punt",
        "Blocked Punt Touchdown",
        "Defensive 2pt Conversion",
        "Field Goal Good",
        "Field Goal Missed",
        "Kickoff",
        "Kickoff Return (Offense)",
        "Kickoff Return Touchdown",
        "Missed Field Goal Return",
        "Missed Field Goal Return Touchdown",
        "Punt",
        "Punt (Safety)",
        "Punt Return",
        "Punt Return Touchdown",
        "Two Point Pass",
        "Two Point Rush",
    }
)
# On every punt and kick row the provider lists the kicking team as offense,
# including return touchdowns (verified numerically on 2019-2022 score deltas).
PUNT_TYPES = frozenset(
    {
        "Blocked Punt",
        "Blocked Punt Touchdown",
        "Punt",
        "Punt (Safety)",
        "Punt Return",
        "Punt Return Touchdown",
    }
)
MISSED_KICK_TYPES = frozenset(
    {
        "Blocked Field Goal",
        "Blocked Field Goal Touchdown",
        "Field Goal Missed",
        "Missed Field Goal Return",
        "Missed Field Goal Return Touchdown",
    }
)
ADMINISTRATIVE_TYPES = frozenset(
    {
        "End of Game",
        "End of Half",
        "End of Regulation",
        "End Period",
        "Timeout",
        "Uncategorized",
        "placeholder",
    }
)
PERIOD_BOUNDARY_TYPES = frozenset({"End of Game", "End of Half", "End of Regulation"})

GARBAGE_MARGIN = {1: 43, 2: 38, 3: 28}
GARBAGE_MARGIN_LATE = 22
EXPLOSIVE_EPA = 1.0

CLOCK_MANAGEMENT_PATTERN = r"\bkneel(?:ed|ing|s)?\b|takes? a knee|\bspik(?:e|ed|es|ing)\b"
SPECIAL_TEAMS_TEXT_PATTERN = (
    r"\bkickoff\b|\bpunt(?:ed|s|ing)?\b|\bfield goal\b"
)
ADMINISTRATIVE_TEXT_PATTERN = (
    r"^end of (?:the )?(?:first|second|third|fourth|[1-4](?:st|nd|rd|th)) quarter"
    r"|^end of (?:ot|overtime|game|half|regulation)"
)
OFFENSIVE_TOUCHDOWN_TYPES = frozenset({"Passing Touchdown", "Rushing Touchdown"})

REQUIRED_COLUMNS = frozenset(
    {
        "away",
        "clock",
        "defense",
        "defense_score",
        "distance",
        "down",
        "drive_id",
        "drive_number",
        "game_id",
        "home",
        "id",
        "offense",
        "offense_score",
        "period",
        "play_number",
        "play_text",
        "play_type",
        "scoring",
        "season",
        "season_type",
        "week",
        "yards_gained",
        "yards_to_goal",
    }
)


def _clock_seconds(value) -> float:
    if not isinstance(value, dict):
        return np.nan
    minutes = value.get("minutes")
    seconds = value.get("seconds")
    if minutes is None or seconds is None:
        return np.nan
    return float(minutes) * 60 + float(seconds)


def _source_flag(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[column].astype("boolean").fillna(False).astype(bool)


def classify_plays(plays: pd.DataFrame) -> pd.DataFrame:
    """Normalize source plays and assign mutually exclusive football categories."""
    missing = sorted(REQUIRED_COLUMNS - set(plays.columns))
    if missing:
        raise ValueError(f"plays are missing required columns: {', '.join(missing)}")

    df = plays.copy()
    if "epa" not in df:
        if "ppa" not in df:
            raise ValueError("plays are missing required efficiency column: epa or ppa")
        df["epa"] = df["ppa"]
    df["_source_order"] = np.arange(len(df))
    for column in (
        "defense_score",
        "distance",
        "down",
        "drive_number",
        "offense_score",
        "period",
        "play_number",
        "epa",
        "yards_gained",
        "yards_to_goal",
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["source_drive_id"] = df["drive_id"].astype("string")
    df["clock_seconds_remaining"] = df["clock"].map(_clock_seconds)
    df["regulation_seconds_elapsed"] = np.where(
        df["period"].between(1, 4),
        (df["period"] - 1) * 900 + (900 - df["clock_seconds_remaining"]),
        np.nan,
    )
    if "game_play_number" in df:
        df["game_play_number"] = pd.to_numeric(
            df["game_play_number"], errors="coerce"
        )
        sort_columns = ["game_id", "game_play_number", "_source_order"]
    else:
        sort_columns = [
            "game_id",
            "period",
            "drive_number",
            "play_number",
            "_source_order",
        ]
    df = df.sort_values(sort_columns, kind="stable", ignore_index=True)

    text = df["play_text"].fillna("")
    clock_management = text.str.contains(
        CLOCK_MANAGEMENT_PATTERN, case=False, regex=True
    )
    special_teams_text = text.str.contains(
        SPECIAL_TEAMS_TEXT_PATTERN, case=False, regex=True
    )
    administrative_text = text.str.contains(
        ADMINISTRATIVE_TEXT_PATTERN, case=False, regex=True
    )
    source_kneel = _source_flag(df, "is_kneel_source")
    source_penalty = _source_flag(df, "is_penalty_source")
    source_special_teams = _source_flag(df, "is_special_teams_source")
    source_scrimmage = _source_flag(df, "is_scrimmage_source")

    conditions = [
        source_kneel | clock_management,
        source_penalty | df["play_type"].eq("Penalty"),
        source_special_teams
        | df["play_type"].isin(SPECIAL_TEAMS_TYPES)
        | special_teams_text,
        df["play_type"].isin(ADMINISTRATIVE_TYPES) | administrative_text,
        source_scrimmage | df["play_type"].isin(SCRIMMAGE_TYPES),
    ]
    categories = [
        "clock_management",
        "penalty",
        "special_teams",
        "administrative",
        "scrimmage",
    ]
    df["play_category"] = np.select(conditions, categories, default="unknown")
    df["is_scrimmage"] = df["play_category"].eq("scrimmage")

    threshold = df["period"].map(GARBAGE_MARGIN).fillna(GARBAGE_MARGIN_LATE)
    score_margin = (df["offense_score"] - df["defense_score"]).abs()
    df["is_garbage_time"] = df["is_scrimmage"] & score_margin.gt(threshold)
    df["is_competitive"] = (
        df["is_scrimmage"] & df["epa"].notna() & ~df["is_garbage_time"]
    )
    df["is_non_turnover_efficiency"] = (
        df["is_competitive"] & ~df["play_type"].isin(TURNOVER_TYPES)
    )
    source_rush = _source_flag(df, "is_rush_source")
    source_pass = _source_flag(df, "is_pass_source")
    source_turnover = _source_flag(df, "is_turnover_source")
    source_sack = _source_flag(df, "is_sack_source")
    df["is_rush"] = df["is_competitive"] & (
        source_rush | df["play_type"].isin(RUSH_TYPES)
    )
    df["is_pass"] = df["is_competitive"] & (
        source_pass | df["play_type"].isin(PASS_TYPES)
    )
    df["is_turnover"] = df["is_scrimmage"] & (
        source_turnover | df["play_type"].isin(TURNOVER_TYPES)
    )
    df["is_sack"] = df["is_scrimmage"] & (
        source_sack | df["play_type"].eq("Sack")
    )
    df["is_explosive"] = df["is_competitive"] & df["epa"].gt(EXPLOSIVE_EPA)
    df["is_offensive_touchdown"] = (
        df["is_scrimmage"]
        & ~df["is_garbage_time"]
        & df["scoring"].fillna(False).astype(bool)
        & df["play_type"].isin(OFFENSIVE_TOUCHDOWN_TYPES)
    )

    success = (
        (df["down"].eq(1) & df["yards_gained"].ge(0.5 * df["distance"]))
        | (df["down"].eq(2) & df["yards_gained"].ge(0.7 * df["distance"]))
        | (df["down"].isin([3, 4]) & df["yards_gained"].ge(df["distance"]))
    )
    df["is_success"] = df["is_competitive"] & success

    boundary_event = (
        df["play_category"].eq("special_teams")
        | df["play_type"].isin(PERIOD_BOUNDARY_TYPES)
        | df["scoring"].fillna(False).astype(bool)
    )
    df["boundary_before"] = boundary_event.groupby(df["game_id"]).transform(
        lambda values: values.shift(fill_value=False).cumsum()
    )
    return df.drop(columns=["_source_order"])


def build_possessions(plays: pd.DataFrame) -> pd.DataFrame:
    """Build one row per reconstructed offensive scrimmage possession.

    A possession is a maximal chronological sequence of scrimmage plays by one
    offense, split by offense changes, source-drive changes, scores, kicks,
    punts, field-goal attempts, or halftime boundaries.
    """
    classified = classify_plays(plays)
    scrimmage = classified[classified["is_scrimmage"]].copy()
    if scrimmage.empty:
        return pd.DataFrame()

    by_game = scrimmage.groupby("game_id", sort=False)
    previous_offense = by_game["offense"].shift()
    previous_defense = by_game["defense"].shift()
    previous_boundary = by_game["boundary_before"].shift()
    previous_drive = by_game["source_drive_id"].shift()
    new_possession = (
        previous_offense.isna()
        | scrimmage["offense"].ne(previous_offense)
        | scrimmage["defense"].ne(previous_defense)
        | scrimmage["boundary_before"].ne(previous_boundary)
        | scrimmage["source_drive_id"].ne(previous_drive)
    )
    scrimmage["possession_number"] = new_possession.groupby(
        scrimmage["game_id"]
    ).cumsum().astype(int)
    scrimmage["possession_id"] = (
        scrimmage["game_id"].astype(str)
        + ":"
        + scrimmage["possession_number"].astype(str)
    )

    scrimmage["competitive_epa"] = scrimmage["epa"].where(
        scrimmage["is_competitive"]
    )
    scrimmage["non_turnover_epa"] = scrimmage["epa"].where(
        scrimmage["is_non_turnover_efficiency"]
    )
    scrimmage["competitive_rush"] = scrimmage["is_rush"].astype(int)
    scrimmage["competitive_pass"] = scrimmage["is_pass"].astype(int)
    scrimmage["competitive_early_down"] = (
        scrimmage["is_competitive"] & scrimmage["down"].isin([1, 2])
    ).astype(int)
    scrimmage["rush_epa"] = scrimmage["epa"].where(scrimmage["is_rush"])
    scrimmage["pass_epa"] = scrimmage["epa"].where(scrimmage["is_pass"])
    scrimmage["early_down_epa"] = scrimmage["epa"].where(
        scrimmage["is_competitive"] & scrimmage["down"].isin([1, 2])
    )
    scrimmage["opponent_40"] = (
        ~scrimmage["is_garbage_time"] & scrimmage["yards_to_goal"].le(40)
    )

    keys = ["game_id", "possession_number"]
    grouped = scrimmage.groupby(keys, sort=True)
    out = grouped.agg(
        possession_id=("possession_id", "first"),
        source_drive_id=("source_drive_id", "first"),
        season=("season", "first"),
        week=("week", "first"),
        season_type=("season_type", "first"),
        home_team=("home", "first"),
        away_team=("away", "first"),
        offense=("offense", "first"),
        defense=("defense", "first"),
        start_period=("period", "first"),
        end_period=("period", "last"),
        start_yards_to_goal=("yards_to_goal", "first"),
        start_elapsed_seconds=("regulation_seconds_elapsed", "first"),
        end_elapsed_seconds=("regulation_seconds_elapsed", "last"),
        scrimmage_plays=("id", "size"),
        competitive_plays=("is_competitive", "sum"),
        non_turnover_plays=("is_non_turnover_efficiency", "sum"),
        garbage_time_plays=("is_garbage_time", "sum"),
        successes=("is_success", "sum"),
        explosive_plays=("is_explosive", "sum"),
        rush_plays=("competitive_rush", "sum"),
        pass_plays=("competitive_pass", "sum"),
        early_down_plays=("competitive_early_down", "sum"),
        turnovers=("is_turnover", "sum"),
        sacks=("is_sack", "sum"),
        touchdowns=("is_offensive_touchdown", "sum"),
        reached_opponent_40=("opponent_40", "max"),
    ).reset_index()

    sums = grouped[
        [
            "competitive_epa",
            "non_turnover_epa",
            "rush_epa",
            "pass_epa",
            "early_down_epa",
        ]
    ].sum(min_count=1)
    sums = sums.rename(
        columns={
            "competitive_epa": "epa_total",
            "non_turnover_epa": "non_turnover_epa_total",
            "rush_epa": "rush_epa_total",
            "pass_epa": "pass_epa_total",
            "early_down_epa": "early_down_epa_total",
        }
    ).reset_index()
    out = out.merge(sums, on=keys, how="left", validate="one_to_one")

    out["epa_per_play"] = out["epa_total"] / out["competitive_plays"]
    out["non_turnover_epa_per_play"] = (
        out["non_turnover_epa_total"] / out["non_turnover_plays"]
    )
    out["success_rate"] = out["successes"] / out["competitive_plays"]
    out["explosive_rate"] = out["explosive_plays"] / out["competitive_plays"]
    out["rush_epa_per_play"] = out["rush_epa_total"] / out["rush_plays"]
    out["pass_epa_per_play"] = out["pass_epa_total"] / out["pass_plays"]
    out["early_down_epa_per_play"] = (
        out["early_down_epa_total"] / out["early_down_plays"]
    )
    span = (
        out["end_elapsed_seconds"] - out["start_elapsed_seconds"]
    ).where(out["start_period"].le(4) & out["end_period"].le(4))
    out["scrimmage_span_seconds"] = span.where(span.ge(0))
    out["timed_play_intervals"] = (out["scrimmage_plays"] - 1).clip(lower=0).where(
        out["scrimmage_span_seconds"].notna(), 0
    )
    out["seconds_per_play"] = (
        out["scrimmage_span_seconds"] / out["timed_play_intervals"]
    ).where(out["timed_play_intervals"].gt(0))
    out["opportunity_touchdowns"] = out["touchdowns"].where(
        out["reached_opponent_40"], 0
    )

    return out.drop(
        columns=[
            "start_elapsed_seconds",
            "end_elapsed_seconds",
        ]
    ).sort_values(keys, kind="stable", ignore_index=True)


def _aggregate_side(
    possessions: pd.DataFrame, team_column: str, opponent_column: str, prefix: str
) -> pd.DataFrame:
    work = possessions.copy()
    work["team"] = work[team_column]
    work["opponent"] = work[opponent_column]
    keys = ["season", "week", "season_type", "game_id", "team"]
    grouped = work.groupby(keys, sort=True)
    out = grouped.agg(
        opponent=("opponent", "first"),
        possessions=("possession_id", "size"),
        scrimmage_plays=("scrimmage_plays", "sum"),
        competitive_plays=("competitive_plays", "sum"),
        non_turnover_plays=("non_turnover_plays", "sum"),
        epa_total=("epa_total", "sum"),
        non_turnover_epa_total=("non_turnover_epa_total", "sum"),
        successes=("successes", "sum"),
        explosive_plays=("explosive_plays", "sum"),
        turnovers=("turnovers", "sum"),
        touchdowns=("touchdowns", "sum"),
        opportunities=("reached_opponent_40", "sum"),
        opportunity_touchdowns=("opportunity_touchdowns", "sum"),
        average_start_yards_to_goal=("start_yards_to_goal", "mean"),
        scrimmage_span_seconds=("scrimmage_span_seconds", "sum"),
        timed_play_intervals=("timed_play_intervals", "sum"),
    ).reset_index()
    out["epa_per_play"] = out["epa_total"] / out["competitive_plays"]
    out["non_turnover_epa_per_play"] = (
        out["non_turnover_epa_total"] / out["non_turnover_plays"]
    )
    out["success_rate"] = out["successes"] / out["competitive_plays"]
    out["explosive_rate"] = out["explosive_plays"] / out["competitive_plays"]
    out["touchdowns_per_opportunity"] = (
        out["opportunity_touchdowns"] / out["opportunities"]
    ).where(out["opportunities"].gt(0))
    out["plays_per_possession"] = out["scrimmage_plays"] / out["possessions"]
    out["seconds_per_play"] = (
        out["scrimmage_span_seconds"] / out["timed_play_intervals"]
    ).where(out["timed_play_intervals"].gt(0))

    value_columns = [column for column in out.columns if column not in keys + ["opponent"]]
    out = out.rename(columns={column: f"{prefix}_{column}" for column in value_columns})
    return out


def build_team_games(possessions: pd.DataFrame) -> pd.DataFrame:
    """Build one opponent-adjustable feature row per game and team."""
    if possessions.empty:
        return pd.DataFrame()

    offense = _aggregate_side(possessions, "offense", "defense", "offense")

    defense = _aggregate_side(possessions, "defense", "offense", "defense")
    defense = defense.rename(
        columns={
            "defense_epa_total": "defense_epa_total_allowed",
            "defense_non_turnover_epa_total": "defense_non_turnover_epa_total_allowed",
            "defense_successes": "defense_successes_allowed",
            "defense_explosive_plays": "defense_explosive_plays_allowed",
            "defense_turnovers": "defense_turnovers_forced",
            "defense_touchdowns": "defense_touchdowns_allowed",
            "defense_opportunities": "defense_opportunities_allowed",
            "defense_opportunity_touchdowns": "defense_opportunity_touchdowns_allowed",
            "defense_average_start_yards_to_goal": "defense_average_start_yards_to_goal_allowed",
            "defense_epa_per_play": "defense_epa_per_play_allowed",
            "defense_non_turnover_epa_per_play": "defense_non_turnover_epa_per_play_allowed",
            "defense_success_rate": "defense_success_rate_allowed",
            "defense_explosive_rate": "defense_explosive_rate_allowed",
            "defense_touchdowns_per_opportunity": "defense_touchdowns_per_opportunity_allowed",
        }
    )

    keys = ["season", "week", "season_type", "game_id", "team"]
    out = offense.merge(
        defense.drop(columns="opponent"),
        on=keys,
        how="outer",
        validate="one_to_one",
    )
    out["game_possessions"] = (
        out["offense_possessions"] + out["defense_possessions"]
    ) / 2
    out["possession_balance"] = (
        out["offense_possessions"] - out["defense_possessions"]
    )
    out["game_plays_per_possession"] = (
        out["offense_scrimmage_plays"] + out["defense_scrimmage_plays"]
    ) / (out["offense_possessions"] + out["defense_possessions"])
    return out.sort_values(keys, kind="stable", ignore_index=True)
