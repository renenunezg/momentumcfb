import numpy as np
import pandas as pd

DIVISION_ONE = {"fbs", "fcs"}
SCORING_COLUMNS = [
    "home_points",
    "away_points",
    "game_possessions",
    "home_epa_per_possession",
    "away_epa_per_possession",
]
WEEK_ZERO_MINIMUM_GAP = pd.Timedelta(hours=48)


def _division_one_schedule(games: pd.DataFrame) -> pd.DataFrame:
    """Normalize the full D1 schedule and retain unplayed games."""
    schedule = games.copy()
    schedule = schedule[
        schedule["home_classification"].str.lower().isin(DIVISION_ONE)
        & schedule["away_classification"].str.lower().isin(DIVISION_ONE)
    ]
    schedule = schedule.rename(
        columns={
            "id": "game_id",
            "home_id": "home_team_id",
            "away_id": "away_team_id",
        }
    )
    schedule["start_date"] = pd.to_datetime(schedule["start_date"], utc=True)
    max_regular_week = schedule.loc[schedule["season_type"].eq("regular"), "week"].max()
    if pd.isna(max_regular_week):
        raise ValueError("the schedule has no regular-season games")
    schedule["model_week"] = np.where(
        schedule["season_type"].eq("regular"),
        schedule["week"],
        max_regular_week + schedule["week"],
    ).astype(int)
    return schedule.sort_values(
        ["start_date", "game_id"], kind="stable", ignore_index=True
    )


def _split_week_zero(schedule: pd.DataFrame) -> pd.DataFrame:
    """Give the distinct opening slate its own chronological model period.

    CFBD labels the late-August Week 0 slate and the following Week 1 slate
    as regular-season Week 1. The schedules are separated by a multi-day
    idle window. Keep the source ``week`` intact, but assign the earlier
    cluster model week 0 so its completed games can train the Week 1 fit.
    """
    out = schedule.copy()
    first_week = out[out["season_type"].eq("regular") & out["week"].eq(1)]
    starts = first_week["start_date"].drop_duplicates().sort_values()
    if len(starts) < 2:
        return out
    gaps = starts.diff()
    largest_gap_at = gaps.idxmax()
    if gaps.loc[largest_gap_at] < WEEK_ZERO_MINIMUM_GAP:
        return out
    week_one_start = starts.loc[largest_gap_at]
    week_zero = (
        out["season_type"].eq("regular")
        & out["week"].eq(1)
        & out["start_date"].lt(week_one_start)
    )
    out.loc[week_zero, "model_week"] = 0
    return out


def _join_scoring_features(
    schedule: pd.DataFrame, team_games: pd.DataFrame
) -> pd.DataFrame:
    """Join possession features while retaining scheduled future games.

    Completed games carry model inputs. Future games remain in the frame with
    missing scoring inputs so they can be projected without entering training.
    """
    feature_columns = [
        "game_id",
        "team",
        "offense_possessions",
        "offense_epa_total",
        "game_possessions",
    ]
    features = team_games[feature_columns].copy()
    home = features.rename(
        columns={
            "team": "home_team",
            "offense_possessions": "home_offense_possessions",
            "offense_epa_total": "home_offense_epa_total",
            "game_possessions": "home_game_possessions",
        }
    )
    away = features.rename(
        columns={
            "team": "away_team",
            "offense_possessions": "away_offense_possessions",
            "offense_epa_total": "away_offense_epa_total",
            "game_possessions": "away_game_possessions",
        }
    )
    out = schedule.merge(
        home, on=["game_id", "home_team"], how="left", validate="one_to_one"
    ).merge(away, on=["game_id", "away_team"], how="left", validate="one_to_one")
    out["game_possessions"] = out[
        ["home_game_possessions", "away_game_possessions"]
    ].mean(axis=1)
    out["home_epa_per_possession"] = (
        out["home_offense_epa_total"] / out["home_offense_possessions"]
    )
    out["away_epa_per_possession"] = (
        out["away_offense_epa_total"] / out["away_offense_possessions"]
    )
    return out.sort_values(["start_date", "game_id"], kind="stable", ignore_index=True)


def build_weekly_scoring_games(
    games: pd.DataFrame, team_games: pd.DataFrame
) -> pd.DataFrame:
    """Build the schedule-aware frame used by the in-season weekly fit."""
    schedule = _split_week_zero(_division_one_schedule(games))
    return _join_scoring_features(schedule, team_games)


def build_scoring_games(games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Join completed Division I schedules and possession features by game."""
    schedule = _division_one_schedule(games)
    out = _join_scoring_features(schedule, team_games)
    completed = out["completed"].fillna(False).astype(bool)
    return out[completed].dropna(subset=SCORING_COLUMNS).reset_index(drop=True)
