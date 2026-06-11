import numpy as np
import pandas as pd

from backend.config import FCS_LABEL

GAME_COLS = [
    "game_id",
    "season",
    "week",
    "week_index",
    "season_type",
    "home_team",
    "away_team",
    "neutral_site",
    "home_points",
    "away_points",
    "margin",
    "closing_spread",
    "home_off_epa_total",
    "away_off_epa_total",
    "epa_margin",
]


def flatten_lines(lines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in lines.itertuples(index=False):
        spreads = [
            float(l["spread"])
            for l in rec.lines
            if l.get("spread") is not None
        ]
        if spreads:
            rows.append({"game_id": rec.game_id, "closing_spread": float(np.median(spreads))})
    return pd.DataFrame(rows, columns=["game_id", "closing_spread"])


def build_games(
    games: pd.DataFrame, lines: pd.DataFrame, epa: pd.DataFrame, variant: str
) -> pd.DataFrame:
    g = games[games["completed"]].copy()
    g = g.rename(columns={"id": "game_id"})
    g = g.dropna(subset=["home_points", "away_points"])
    g["margin"] = g["home_points"] - g["away_points"]

    g = g.merge(flatten_lines(lines), on="game_id", how="left")

    e = epa[epa["variant"] == variant][["game_id", "team", "total_off_epa"]]
    g = g.merge(
        e.rename(columns={"team": "home_team", "total_off_epa": "home_off_epa_total"}),
        on=["game_id", "home_team"],
        how="left",
    )
    g = g.merge(
        e.rename(columns={"team": "away_team", "total_off_epa": "away_off_epa_total"}),
        on=["game_id", "away_team"],
        how="left",
    )
    g["epa_margin"] = g["home_off_epa_total"] - g["away_off_epa_total"]

    for side in ("home", "away"):
        fcs = g[f"{side}_classification"].str.lower() != "fbs"
        g.loc[fcs, f"{side}_team"] = FCS_LABEL
    g = g[(g["home_team"] != FCS_LABEL) | (g["away_team"] != FCS_LABEL)]

    max_reg = g.loc[g["season_type"] == "regular", "week"].max()
    g["week_index"] = np.where(
        g["season_type"] == "regular", g["week"] - 1, max_reg + g["week"] - 1
    ).astype(int)

    return g[GAME_COLS].reset_index(drop=True)
