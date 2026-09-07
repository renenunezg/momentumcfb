"""Heisman vote-share model: a conditional logit over each season's candidates.

Within a season the predicted vote share is a softmax over the candidate
pool, which sums to one by construction and matches how a ballot behaves.
Features are per-game rates plus team record, poll rank, position, and
conference tier, so a mid-season row is on the same footing as a full
season. Training rows are the top-ten ballot finishers from the seed CSV
with everyone else in the pool at zero share.
"""

import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backend.players.heisman_seed import SEED_PATH
from backend.players.ingest import read_season_source, read_weekly

MODEL_VERSION = "cfb_heisman_share_v1"
CANDIDATE_POOL = 300
RIDGE_PENALTY = 0.5
RATE_PRIOR_GAMES = 4.0
POWER_CONFERENCES = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Pac-10", "Big East"}
POWER_INDEPENDENTS = {"Notre Dame"}
POSITION_DUMMIES = ("QB", "RB", "WR")

RATE_FEATURES = [
    "pass_yards",
    "pass_touchdowns",
    "interceptions",
    "rush_yards",
    "rush_touchdowns",
    "receiving_yards",
    "receiving_touchdowns",
    "total_touchdowns",
]
CONTEXT_FEATURES = ["win_pct", "rank_points", "power_conference", "games"]
# Defensive box stats only exist from 2014, so a defender's rates cannot be
# a feature across the training window; a defender dummy carries how rarely
# the vote goes that way instead.
DEFENSIVE_POSITIONS = {
    "DL",
    "DE",
    "DT",
    "NT",
    "EDGE",
    "LB",
    "OLB",
    "ILB",
    "MLB",
    "DB",
    "CB",
    "S",
    "FS",
    "SS",
}
FEATURES = (
    [f"{name}_per_game" for name in RATE_FEATURES]
    + CONTEXT_FEATURES
    + [f"is_{position}" for position in POSITION_DUMMIES]
    + ["is_defender"]
)

BOARD_COLUMNS = [
    "season",
    "week",
    "as_of",
    "model_version",
    "athlete_id",
    "athlete_name",
    "team",
    "position",
    "games",
    "predicted_share",
    "predicted_rank",
    "value_rank",
    "rank_gap",
    "win_pct",
    "ap_rank",
    "pass_yards",
    "pass_touchdowns",
    "interceptions",
    "rush_yards",
    "rush_touchdowns",
    "receiving_yards",
    "receiving_touchdowns",
    "total_touchdowns",
    "tackles",
    "sacks",
    "defensive_interceptions",
]

HISTORY_COLUMNS = [
    "season",
    "actual_winner",
    "actual_winner_team",
    "actual_share",
    "predicted_winner",
    "predicted_winner_team",
    "predicted_winner_share",
    "actual_winner_predicted_share",
    "actual_winner_predicted_rank",
    "winner_hit",
    "top_three_hit",
    "winner_value_rank",
]

_STAT_MAP = {
    ("passing", "YDS"): "pass_yards",
    ("passing", "TD"): "pass_touchdowns",
    ("passing", "INT"): "interceptions",
    ("rushing", "CAR"): "rush_attempts",
    ("rushing", "YDS"): "rush_yards",
    ("rushing", "TD"): "rush_touchdowns",
    ("receiving", "REC"): "receptions",
    ("receiving", "YDS"): "receiving_yards",
    ("receiving", "TD"): "receiving_touchdowns",
    ("defensive", "TOT"): "tackles",
    ("defensive", "SACKS"): "sacks",
    ("defensive", "TFL"): "tackles_for_loss",
    ("defensive", "PD"): "passes_defended",
    ("interceptions", "INT"): "defensive_interceptions",
}


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = re.sub(r"\([^)]*\)", "", text)  # Miami (FL) -> Miami
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", text.lower())
    return re.sub(r"[^a-z]", "", text)


def _regular_box(season: int, through_week: int | None) -> pd.DataFrame:
    box = read_weekly(season, "box")
    box = box[box["season_type"].eq("regular")]
    if through_week is not None:
        box = box[box["week"].le(through_week)]
    return box


def season_stats(season: int, through_week: int | None = None) -> pd.DataFrame:
    """Per-player box totals through a regular-season week (or the season)."""
    box = _regular_box(season, through_week)
    passing = box[box["category"].eq("passing") & box["stat_name"].eq("C/ATT")].copy()
    attempts = passing["stat"].str.split("/", expand=True)
    passing["pass_attempts"] = pd.to_numeric(attempts[1], errors="coerce")

    keyed = box.copy()
    keyed["feature"] = [
        _STAT_MAP.get((category, name))
        for category, name in zip(keyed["category"], keyed["stat_name"])
    ]
    keyed = keyed[keyed["feature"].notna()]
    keyed["value"] = pd.to_numeric(keyed["stat"], errors="coerce").fillna(0.0)
    totals = keyed.pivot_table(
        index=["athlete_id"],
        columns="feature",
        values="value",
        aggfunc="sum",
        fill_value=0.0,
    )
    for feature in set(_STAT_MAP.values()) - set(totals.columns):
        totals[feature] = 0.0
    extra = passing.groupby("athlete_id")[["pass_attempts"]].sum()
    totals = totals.join(extra, how="left").fillna(0.0)

    identity = (
        box.sort_values("week")
        .groupby("athlete_id")
        .agg(
            athlete_name=("athlete_name", "last"),
            team=("team", "last"),
            games=("game_id", "nunique"),
        )
    )
    stats = identity.join(totals, how="left").fillna(0.0).reset_index()
    stats["athlete_id"] = stats["athlete_id"].astype(str)
    stats["total_touchdowns"] = (
        stats["pass_touchdowns"]
        + stats["rush_touchdowns"]
        + stats["receiving_touchdowns"]
    )
    stats["offense_usage"] = (
        stats["pass_attempts"] + stats["rush_attempts"] + 3.0 * stats["receptions"]
    )
    stats["defense_usage"] = (
        stats["tackles"]
        + 3.0 * (stats["tackles_for_loss"] + stats["sacks"])
        + 3.0 * stats["defensive_interceptions"]
        + 2.0 * stats["passes_defended"]
    )
    stats["season"] = season
    return stats


def team_context(season: int, through_week: int | None = None) -> pd.DataFrame:
    """Win percentage, latest AP rank, and conference tier per team."""
    games = pd.read_parquet(
        f"backend/data/raw/games/{season}.parquet",
        columns=[
            "season_type",
            "week",
            "completed",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
        ],
    )
    games = games[
        games["season_type"].eq("regular")
        & games["completed"].fillna(False).astype(bool)
    ]
    if through_week is not None:
        games = games[games["week"].le(through_week)]
    home = pd.DataFrame(
        {
            "team": games["home_team"],
            "win": games["home_points"].gt(games["away_points"]),
        }
    )
    away = pd.DataFrame(
        {
            "team": games["away_team"],
            "win": games["away_points"].gt(games["home_points"]),
        }
    )
    record = pd.concat([home, away]).groupby("team")["win"].agg(["sum", "size"])
    record["win_pct"] = record["sum"] / record["size"]

    rankings = read_season_source(season, "rankings")
    ap = rankings[
        rankings["poll"].eq("AP Top 25") & rankings["season_type"].eq("regular")
    ]
    if through_week is not None:
        ap = ap[ap["week"].le(through_week)]
    latest = ap[ap["week"].eq(ap["week"].max())] if not ap.empty else ap
    rank = latest.set_index("school")["rank"]

    teams = read_season_source(season, "teams")
    context = teams[["school", "conference", "classification"]].rename(
        columns={"school": "team"}
    )
    context["win_pct"] = context["team"].map(record["win_pct"]).fillna(0.0)
    context["ap_rank"] = context["team"].map(rank)
    context["rank_points"] = (26 - context["ap_rank"]).fillna(0.0)
    context["power_conference"] = (
        context["conference"].isin(POWER_CONFERENCES)
        | context["team"].isin(POWER_INDEPENDENTS)
    ).astype(float)
    return context[
        [
            "team",
            "classification",
            "win_pct",
            "ap_rank",
            "rank_points",
            "power_conference",
        ]
    ]


def featured_players(season: int, through_week: int | None = None) -> pd.DataFrame:
    """Every FBS player with box stats, with model features attached."""
    stats = season_stats(season, through_week)
    context = team_context(season, through_week)
    roster = read_season_source(season, "roster")[["id", "position"]].rename(
        columns={"id": "athlete_id"}
    )
    roster["athlete_id"] = roster["athlete_id"].astype(str)
    pool = stats.merge(context, on="team", how="inner")
    pool = pool[pool["classification"].eq("fbs")]
    pool = pool.merge(roster.drop_duplicates("athlete_id"), on="athlete_id", how="left")
    pool["position"] = pool["position"].fillna("")
    for position in POSITION_DUMMIES:
        pool[f"is_{position}"] = pool["position"].eq(position).astype(float)
    pool["is_defender"] = pool["position"].isin(DEFENSIVE_POSITIONS).astype(float)
    # Per-game rates shrink toward the position mean with a prior of a few
    # games, so one big opener cannot read as a full season's pace. At a
    # season's end the prior is a small fraction of the sample.
    games = pool["games"].clip(lower=1)
    group = pool["position"].where(pool["position"].isin(POSITION_DUMMIES), "other")
    for name in RATE_FEATURES:
        rate = pool[name] / games
        prior = rate.groupby(group).transform("mean")
        pool[f"{name}_per_game"] = (games * rate + RATE_PRIOR_GAMES * prior) / (
            games + RATE_PRIOR_GAMES
        )
    return pool.reset_index(drop=True)


def candidate_pool(
    players: pd.DataFrame, keep_ids: pd.Series | None = None
) -> pd.DataFrame:
    """The most used players on each side of the ball, plus any forced ids.

    Half the pool comes from each side so a linebacker season is ranked
    against other defenders' usage, not against pass attempts. Ballot
    finishers outside the cut (a two-way fullback) are kept so the model
    always sees every vote-getter.
    """
    half = CANDIDATE_POOL // 2
    offense = players.sort_values("offense_usage", ascending=False).head(half)
    defense = players.sort_values("defense_usage", ascending=False).head(half)
    forced = (
        players[players["athlete_id"].isin(keep_ids)]
        if keep_ids is not None
        else players.iloc[0:0]
    )
    return (
        pd.concat([offense, defense, forced])
        .drop_duplicates("athlete_id")
        .reset_index(drop=True)
    )


def _roster_fallback(
    season: int, row, players: pd.DataFrame, context: pd.DataFrame
) -> pd.DataFrame | None:
    """A zero-stat candidate row for a ballot player the box never lists.

    Seasons before 2014 carry no defensive box, so a pure defender who
    finished on the ballot has a roster entry and nothing else.
    """
    roster = read_season_source(season, "roster")
    roster = roster.assign(
        key=(roster["first_name"].fillna("") + roster["last_name"].fillna("")).map(
            normalize_name
        ),
        team_key=roster["team"].map(normalize_name),
    )
    hit = roster[
        roster["key"].eq(normalize_name(row.player))
        & roster["team_key"].eq(normalize_name(row.school))
    ]
    if len(hit) != 1:
        return None
    template = players.iloc[0:1].copy()
    for column in template.columns:
        if pd.api.types.is_numeric_dtype(template[column]):
            template[column] = 0.0
    template["athlete_id"] = str(hit["id"].iloc[0])
    template["athlete_name"] = row.player
    template["team"] = hit["team"].iloc[0]
    template["position"] = hit["position"].iloc[0]
    template["season"] = season
    team = context[context["team"].eq(hit["team"].iloc[0])]
    if not team.empty:
        for column in ("win_pct", "ap_rank", "rank_points", "power_conference"):
            template[column] = team[column].iloc[0]
    template["games"] = float(players["games"].max())
    return template


def _match_seed(
    seed: pd.DataFrame, players: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach athlete ids to the season's ballot rows, failing loudly."""
    rows = seed[seed["season"].eq(season)].copy()
    names = players.assign(key=players["athlete_name"].map(normalize_name))
    context = team_context(season)
    matched = []
    for row in rows.itertuples(index=False):
        key = normalize_name(row.player)
        hit = names[names["key"].eq(key)]
        if len(hit) > 1:
            team_key = normalize_name(row.school)
            hit = hit[hit["team"].map(normalize_name).eq(team_key)]
        if hit.empty:
            # Initials and nicknames differ between sources; last name plus
            # first initial within the school is the fallback.
            parts = row.player.split()
            pattern = f"^{normalize_name(parts[0])[0]}.*{normalize_name(parts[-1])}$"
            same_school = names[
                names["team"].map(normalize_name).eq(normalize_name(row.school))
            ]
            hit = same_school[same_school["key"].str.match(pattern)]
            if hit.empty:
                # Wikipedia and CFBD spell some schools differently; a
                # unique name match across the pool still identifies him.
                hit = names[names["key"].str.match(pattern)]
        if hit.empty:
            fallback = _roster_fallback(season, row, players, context)
            if fallback is not None:
                players = pd.concat([players, fallback], ignore_index=True)
                names = players.assign(key=players["athlete_name"].map(normalize_name))
                hit = fallback
        if len(hit) != 1:
            raise ValueError(
                f"{season} ballot row {row.player} ({row.school}) matched "
                f"{len(hit)} candidates"
            )
        matched.append({"athlete_id": hit["athlete_id"].iloc[0], "points": row.points})
    return pd.DataFrame(matched), players


def load_seed() -> pd.DataFrame:
    return pd.read_csv(SEED_PATH)


def training_rows(seasons: list[int], seed: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for season in seasons:
        players = featured_players(season)
        ballot, players = _match_seed(seed, players, season)
        pool = candidate_pool(players, ballot["athlete_id"])
        pool = pool.merge(ballot, on="athlete_id", how="left")
        pool["points"] = pool["points"].fillna(0.0)
        pool["share"] = pool["points"] / pool["points"].sum()
        frames.append(pool)
    return pd.concat(frames, ignore_index=True)


class ShareModel:
    """Conditional logit with ridge-penalised coefficients."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray, beta: np.ndarray):
        self.mean = mean
        self.scale = scale
        self.beta = beta

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        return (frame[FEATURES].to_numpy(float) - self.mean) / self.scale

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        logits = self._design(frame) @ self.beta
        out = np.empty(len(frame))
        for _, index in frame.groupby("season").indices.items():
            block = logits[index]
            block = np.exp(block - block.max())
            out[index] = block / block.sum()
        return out

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": FEATURES, "coefficient": self.beta})


def fit_share_model(rows: pd.DataFrame) -> ShareModel:
    design = rows[FEATURES].to_numpy(float)
    mean = design.mean(axis=0)
    scale = design.std(axis=0)
    scale[scale == 0] = 1.0
    design = (design - mean) / scale
    share = rows["share"].to_numpy(float)
    groups = [index for _, index in rows.groupby("season").indices.items()]

    def loss_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ beta
        loss = RIDGE_PENALTY * float(beta @ beta)
        grad = 2.0 * RIDGE_PENALTY * beta
        for index in groups:
            block = logits[index]
            block = block - block.max()
            weights = np.exp(block)
            probability = weights / weights.sum()
            loss -= float(share[index] @ np.log(probability))
            grad += design[index].T @ (probability - share[index])
        return loss, grad

    result = minimize(
        loss_and_grad, np.zeros(len(FEATURES)), jac=True, method="L-BFGS-B"
    )
    if not result.success:
        raise RuntimeError(f"share model did not converge: {result.message}")
    return ShareModel(mean, scale, result.x)


def leave_one_season_out(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-season holdout: fit on the other seasons, score the held-out one."""
    records = []
    for season in sorted(rows["season"].unique()):
        held = rows[rows["season"].eq(season)].copy()
        model = fit_share_model(rows[rows["season"].ne(season)])
        held["predicted_share"] = model.predict(held)
        held = held.sort_values("predicted_share", ascending=False).reset_index(
            drop=True
        )
        held["predicted_rank"] = np.arange(1, len(held) + 1)
        actual = held.sort_values("share", ascending=False).iloc[0]
        predicted = held.iloc[0]
        actual_row = held[held["athlete_id"].eq(actual["athlete_id"])].iloc[0]
        records.append(
            {
                "season": int(season),
                "actual_winner": actual["athlete_name"],
                "actual_winner_team": actual["team"],
                "actual_share": float(actual["share"]),
                "predicted_winner": predicted["athlete_name"],
                "predicted_winner_team": predicted["team"],
                "predicted_winner_share": float(predicted["predicted_share"]),
                "actual_winner_predicted_share": float(actual_row["predicted_share"]),
                "actual_winner_predicted_rank": int(actual_row["predicted_rank"]),
                "winner_hit": bool(actual_row["predicted_rank"] == 1),
                "top_three_hit": bool(actual_row["predicted_rank"] <= 3),
            }
        )
    return pd.DataFrame(records)


def build_board(
    model: ShareModel,
    season: int,
    week: int,
    player_values: pd.DataFrame | None,
    as_of: datetime,
) -> pd.DataFrame:
    """Predicted shares through a week, with the value rank beside them."""
    pool = candidate_pool(featured_players(season, week))
    pool["predicted_share"] = model.predict(pool)
    pool = pool.sort_values("predicted_share", ascending=False).reset_index(drop=True)
    pool["predicted_rank"] = np.arange(1, len(pool) + 1)
    if player_values is not None and not player_values.empty:
        latest = player_values[player_values["week"].le(week)]
        latest = latest[latest["week"].eq(latest["week"].max())]
        pool = pool.merge(
            latest[["athlete_id", "overall_rank"]].rename(
                columns={"overall_rank": "value_rank"}
            ),
            on="athlete_id",
            how="left",
        )
    else:
        pool["value_rank"] = np.nan
    pool["rank_gap"] = pool["value_rank"] - pool["predicted_rank"]
    pool["season"] = season
    pool["week"] = week
    pool["as_of"] = as_of.isoformat()
    pool["model_version"] = MODEL_VERSION
    return pool[BOARD_COLUMNS]
