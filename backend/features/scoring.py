import numpy as np
import pandas as pd


def build_scoring_games(
    games: pd.DataFrame, team_games: pd.DataFrame
) -> pd.DataFrame:
    """Join completed Division I schedules and possession features by game."""
    schedule = games[games["completed"].fillna(False)].copy()
    division_one = {"fbs", "fcs"}
    schedule = schedule[
        schedule["home_classification"].str.lower().isin(division_one)
        & schedule["away_classification"].str.lower().isin(division_one)
    ]
    schedule = schedule.rename(
        columns={
            "id": "game_id",
            "home_id": "home_team_id",
            "away_id": "away_team_id",
        }
    )

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
        home, on=["game_id", "home_team"], how="inner", validate="one_to_one"
    ).merge(
        away, on=["game_id", "away_team"], how="inner", validate="one_to_one"
    )
    out["game_possessions"] = out[
        ["home_game_possessions", "away_game_possessions"]
    ].mean(axis=1)
    out["home_epa_per_possession"] = (
        out["home_offense_epa_total"] / out["home_offense_possessions"]
    )
    out["away_epa_per_possession"] = (
        out["away_offense_epa_total"] / out["away_offense_possessions"]
    )
    out["start_date"] = pd.to_datetime(out["start_date"], utc=True)
    max_regular_week = out.loc[
        out["season_type"].eq("regular"), "week"
    ].max()
    out["model_week"] = np.where(
        out["season_type"].eq("regular"),
        out["week"],
        max_regular_week + out["week"],
    ).astype(int)
    required = [
        "home_points",
        "away_points",
        "game_possessions",
        "home_epa_per_possession",
        "away_epa_per_possession",
    ]
    return out.dropna(subset=required).sort_values(
        ["start_date", "game_id"], kind="stable", ignore_index=True
    )
