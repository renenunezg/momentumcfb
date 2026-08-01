import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as parquet
import requests

PBP_RELEASE_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "espn_cfb_pbp/play_by_play_{season}.parquet"
)

CORE_SOURCE_COLUMNS = frozenset(
    {
        "season",
        "seasonType",
        "week",
        "game_id",
        "game_play_number",
        "id",
        "period",
        "clock.minutes",
        "clock.seconds",
        "start.TimeSecsRem",
        "homeTeamId",
        "awayTeamId",
        "homeTeamName",
        "awayTeamName",
        "start.pos_team.id",
        "start.def_pos_team.id",
        "start.pos_team.name",
        "start.def_pos_team.name",
        "start.pos_team_score",
        "start.def_pos_team_score",
        "end.pos_team_score",
        "end.def_pos_team_score",
        "down",
        "distance",
        "start.yardsToEndzone",
        "end.down",
        "end.distance",
        "end.yardsToEndzone",
        "statYardage",
        "type.text",
        "text",
        "drive.id",
        "scoringPlay",
        "isPenalty",
        "isTurnover",
        "scrimmage_play",
        "sp",
        "kneel_down",
        "EPA_scrimmage",
    }
)

OPTIONAL_COLUMN_MAP = {
    "EPA_pass": "pass_epa",
    "EPA_rush": "rush_epa",
    "EPA_success": "epa_success",
    "EPA_explosive": "epa_explosive",
    "EPA_explosive_pass": "epa_explosive_pass",
    "EPA_explosive_rush": "epa_explosive_rush",
    "first_down_created": "first_down_created",
    "early_down": "early_down",
    "late_down": "late_down",
    "standard_down": "standard_down",
    "passing_down": "passing_down",
    "goal_to_go": "goal_to_go",
    "under_2": "under_two_minutes",
    "pass": "is_pass_source",
    "rush": "is_rush_source",
    "pass_attempt": "is_pass_attempt_source",
    "completion": "is_completion_source",
    "target": "is_target_source",
    "sack": "is_sack_source",
    "int": "is_interception_source",
    "turnover_vec": "is_turnover_play_source",
    "cpoe": "cpoe",
    "xpass": "xpass",
    "pass_oe": "pass_oe",
    "air_yards": "air_yards",
    "yards_after_catch": "yards_after_catch",
    "stuffed_run": "stuffed_run",
    "opportunity_run": "opportunity_run",
    "highlight_run": "highlight_run",
    "line_yards": "line_yards",
    "second_level_yards": "second_level_yards",
    "open_field_yards": "open_field_yards",
    "short_rush_success": "short_rush_success",
    "short_rush_attempt": "short_rush_attempt",
    "power_rush_success": "power_rush_success",
    "power_rush_attempt": "power_rush_attempt",
    "havoc": "havoc",
    "TFL": "tackle_for_loss",
    "TFL_pass": "pass_tackle_for_loss",
    "TFL_rush": "rush_tackle_for_loss",
    "forced_fumble": "forced_fumble",
    "pass_breakup": "pass_breakup",
    "qb_hurry": "qb_hurry",
    "scoring_opp": "scoring_opportunity",
    "rz_play": "red_zone_play",
    "penalty_flag": "penalty_flag",
    "penalty_declined": "penalty_declined",
    "penalty_no_play": "penalty_no_play",
    "penalty_offset": "penalty_offset",
    "penalty_yards_signed": "penalty_yards",
    "penalized_team": "penalized_team_id",
    "EPA_sp": "special_teams_epa",
    "EPA_fg": "field_goal_epa",
    "EPA_punt": "punt_epa",
    "EPA_kickoff": "kickoff_epa",
    "fg_attempt": "field_goal_attempt",
    "fg_made": "field_goal_made",
    "fg_make_prob": "field_goal_make_probability",
    "punt_play": "punt_play",
    "kickoff_play": "kickoff_play",
    "rusher_player_id": "rusher_player_id",
    "passer_player_id": "passer_player_id",
    "receiver_player_id": "receiver_player_id",
    "sack_player_id": "sack_player_id",
    "interception_player_id": "interception_player_id",
    "fumble_forced_player_id": "fumble_forced_player_id",
    "fumble_recovered_player_id": "fumble_recovered_player_id",
    "gameSpread": "game_spread",
    "homeTeamSpread": "home_team_spread",
    "overUnder": "over_under",
    "gameSpreadAvailable": "game_spread_available",
    "wp_before": "win_probability_before",
    "modified": "source_modified_at",
}

SEASON_TYPE_NAMES = {1: "preseason", 2: "regular", 3: "postseason"}


def season_pbp_url(season: int) -> str:
    return PBP_RELEASE_URL.format(season=season)


def download_season_pbp(
    season: int,
    destination: Path,
    session: requests.Session | None = None,
) -> Path:
    """Download and atomically replace a raw SportsDataverse season snapshot."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    http = session or requests.Session()

    try:
        with http.get(season_pbp_url(season), stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)

        columns = set(parquet.ParquetFile(temporary).schema.names)
        missing = sorted(CORE_SOURCE_COLUMNS - columns)
        if missing:
            raise ValueError(
                "SportsDataverse PBP is missing required columns: "
                + ", ".join(missing)
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return destination


def _drive_numbers(source: pd.DataFrame) -> pd.Series:
    return source.groupby("game_id", sort=False)["drive.id"].transform(
        lambda values: pd.factorize(values, sort=False)[0] + 1
    )


def normalize_pbp(source: pd.DataFrame) -> pd.DataFrame:
    """Convert SportsDataverse ESPN PBP into the project's stable play contract."""
    missing = sorted(CORE_SOURCE_COLUMNS - set(source.columns))
    if missing:
        raise ValueError(
            "SportsDataverse PBP is missing required columns: " + ", ".join(missing)
        )

    season_type = pd.to_numeric(source["seasonType"], errors="coerce").map(
        SEASON_TYPE_NAMES
    )
    out = pd.DataFrame(
        {
            "season": source["season"],
            "season_type": season_type.fillna(source["seasonType"].astype("string")),
            "week": source["week"],
            "game_id": source["game_id"],
            "id": source["id"],
            "game_play_number": source["game_play_number"],
            "play_number": source["game_play_number"],
            "period": source["period"],
            "clock_minutes": source["clock.minutes"],
            "clock_seconds": source["clock.seconds"],
            "seconds_remaining": source["start.TimeSecsRem"],
            "home_team_id": source["homeTeamId"],
            "away_team_id": source["awayTeamId"],
            "home": source["homeTeamName"],
            "away": source["awayTeamName"],
            "offense_id": source["start.pos_team.id"],
            "defense_id": source["start.def_pos_team.id"],
            "offense": source["start.pos_team.name"],
            "defense": source["start.def_pos_team.name"],
            "offense_score": source["start.pos_team_score"],
            "defense_score": source["start.def_pos_team_score"],
            "end_offense_score": source["end.pos_team_score"],
            "end_defense_score": source["end.def_pos_team_score"],
            "down": source["down"],
            "distance": source["distance"],
            "yards_to_goal": source["start.yardsToEndzone"],
            "end_down": source["end.down"],
            "end_distance": source["end.distance"],
            "end_yards_to_goal": source["end.yardsToEndzone"],
            "yards_gained": source["statYardage"],
            "play_type": source["type.text"],
            "play_text": source["text"],
            "drive_id": source["drive.id"].astype("string"),
            "drive_number": _drive_numbers(source),
            "scoring": source["scoringPlay"],
            "is_penalty_source": source["isPenalty"],
            "is_turnover_source": source["isTurnover"],
            "is_scrimmage_source": source["scrimmage_play"],
            "is_special_teams_source": source["sp"],
            "is_kneel_source": source["kneel_down"],
            "epa": source["EPA_scrimmage"],
            "ppa": source["EPA_scrimmage"],
            "pbp_source": "sportsdataverse",
        }
    )
    out["clock"] = [
        {"minutes": minutes, "seconds": seconds}
        for minutes, seconds in zip(
            out["clock_minutes"], out["clock_seconds"], strict=True
        )
    ]

    for source_column, output_column in OPTIONAL_COLUMN_MAP.items():
        out[output_column] = (
            source[source_column] if source_column in source else pd.NA
        )

    return out.sort_values(
        ["game_id", "game_play_number", "id"], kind="stable", ignore_index=True
    )
