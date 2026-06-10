import pandas as pd

GARBAGE_MARGIN = {1: 43, 2: 38, 3: 28}
GARBAGE_MARGIN_LATE = 22
EXPLOSIVE_EPA = 1.0

RUN_TYPES = {"Rush", "Rushing Touchdown"}
PASS_TYPES = {
    "Pass Reception",
    "Pass Completion",
    "Pass Incompletion",
    "Passing Touchdown",
    "Pass Interception Return",
    "Interception Return Touchdown",
    "Sack",
}


def flag_garbage_time(plays: pd.DataFrame) -> pd.Series:
    margin = (plays["offense_score"] - plays["defense_score"]).abs()
    threshold = plays["period"].map(GARBAGE_MARGIN).fillna(GARBAGE_MARGIN_LATE)
    return margin > threshold


def _aggregate(plays: pd.DataFrame) -> pd.DataFrame:
    off = (
        plays.groupby(["game_id", "offense"])
        .agg(
            off_epa=("ppa", "mean"),
            total_off_epa=("ppa", "sum"),
            plays=("ppa", "size"),
            explosive_rate=("ppa", lambda s: (s > EXPLOSIVE_EPA).mean()),
            season=("season", "first"),
            week=("week", "first"),
            season_type=("season_type", "first"),
        )
        .reset_index()
        .rename(columns={"offense": "team"})
    )
    deff = (
        plays.groupby(["game_id", "defense"])
        .agg(def_epa=("ppa", "mean"), total_def_epa=("ppa", "sum"))
        .reset_index()
        .rename(columns={"defense": "team"})
    )
    run = (
        plays[plays["play_type"].isin(RUN_TYPES)]
        .groupby(["game_id", "offense"])["ppa"]
        .mean()
        .rename("run_epa")
        .reset_index()
        .rename(columns={"offense": "team"})
    )
    pas = (
        plays[plays["play_type"].isin(PASS_TYPES)]
        .groupby(["game_id", "offense"])["ppa"]
        .mean()
        .rename("pass_epa")
        .reset_index()
        .rename(columns={"offense": "team"})
    )
    df = off.merge(deff, on=["game_id", "team"], how="left")
    df = df.merge(run, on=["game_id", "team"], how="left")
    df = df.merge(pas, on=["game_id", "team"], how="left")
    return df


def team_game_epa(plays: pd.DataFrame) -> pd.DataFrame:
    plays = plays.dropna(subset=["offense", "defense"]).copy()
    plays["ppa"] = pd.to_numeric(plays["ppa"], errors="coerce")
    plays = plays.dropna(subset=["ppa"])
    plays["garbage"] = flag_garbage_time(plays)

    frames = []
    for variant, subset in (("all", plays), ("filtered", plays[~plays["garbage"]])):
        agg = _aggregate(subset)
        agg["variant"] = variant
        frames.append(agg)
    return pd.concat(frames, ignore_index=True)
