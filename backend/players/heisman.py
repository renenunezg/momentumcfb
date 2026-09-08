"""Heisman vote-share model: a conditional logit over each season's candidates.

Within a season the predicted vote share is a softmax over the candidate
pool, which sums to one by construction and matches how a ballot behaves.
Training and evaluation use historical weekly snapshots with candidate
pools chosen from statistics available at that week. Evaluation trains
only on earlier seasons and reports winners missing from the pool as misses.
Shares are conditional on the selected pool, not calibrated win probabilities.
"""

import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backend.etl import store
from backend.features.scoring import _division_one_schedule, _split_week_zero
from backend.players.heisman_seed import SEED_PATH
from backend.players.ingest import read_season_source, read_weekly

MODEL_VERSION = "cfb_heisman_share_v2"
EVALUATION_KIND = "expanding_window_model_weekly"
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


def _regular_box(
    season: int, through_week: int | None, box: pd.DataFrame | None = None
) -> pd.DataFrame:
    if box is None:
        box = _feature_sources(season)["box"]
    box = box[box["season_type"].eq("regular")]
    if through_week is not None:
        box = box[box["week"].le(through_week)]
    return box


def season_stats(
    season: int, through_week: int | None = None, *, box: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-player box totals through a regular-season week (or the season)."""
    box = _regular_box(season, through_week, box)
    passing = box[box["category"].eq("passing") & box["stat_name"].eq("C/ATT")].copy()
    passing["pass_attempts"] = pd.to_numeric(
        passing["stat"].str.partition("/", expand=False).str[2], errors="coerce"
    )

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


def _feature_sources(season: int) -> dict[str, pd.DataFrame]:
    """Use the same chronological weeks as team and player-value production."""
    games = store.read_games(season)
    schedule = _split_week_zero(_division_one_schedule(games))
    weeks = schedule.set_index("game_id")["model_week"]
    games = games[games["id"].isin(weeks.index)].copy()
    games["week"] = games["id"].map(weeks).astype(int)
    box = read_weekly(season, "box")
    box = box[box["game_id"].isin(weeks.index)].copy()
    box["week"] = box["game_id"].map(weeks).astype(int)
    return {
        "box": box,
        "games": games,
        **{
            name: read_season_source(season, name)
            for name in ("rankings", "teams", "roster")
        },
    }


def team_context(
    season: int,
    through_week: int | None = None,
    *,
    sources: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Win percentage, latest AP rank, and conference tier per team."""
    sources = _feature_sources(season) if sources is None else sources
    games = sources["games"]
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

    rankings = sources["rankings"]
    ap = rankings[
        rankings["poll"].eq("AP Top 25") & rankings["season_type"].eq("regular")
    ]
    if through_week is not None:
        ap = ap[ap["week"].le(through_week)]
    latest = ap[ap["week"].eq(ap["week"].max())] if not ap.empty else ap
    rank = latest.set_index("school")["rank"]

    teams = sources["teams"]
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


def featured_players(
    season: int,
    through_week: int | None = None,
    *,
    sources: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Every FBS player with box stats, with model features attached."""
    sources = _feature_sources(season) if sources is None else sources
    stats = season_stats(season, through_week, box=sources["box"])
    context = team_context(season, through_week, sources=sources)
    roster = sources["roster"][["id", "position"]].rename(columns={"id": "athlete_id"})
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


def candidate_pool(players: pd.DataFrame) -> pd.DataFrame:
    """The most used players on each side, without outcome-based additions.

    Half the pool comes from each side so a linebacker season is ranked
    against other defenders' usage, not against pass attempts.
    """
    half = CANDIDATE_POOL // 2
    offense = (
        players[players["offense_usage"].gt(0)]
        .sort_values(["offense_usage", "athlete_id"], ascending=[False, True])
        .head(half)
    )
    defense = (
        players[players["defense_usage"].gt(0)]
        .sort_values(["defense_usage", "athlete_id"], ascending=[False, True])
        .head(half)
    )
    return (
        pd.concat([offense, defense])
        .drop_duplicates("athlete_id")
        .reset_index(drop=True)
    )


def _match_seed(
    seed: pd.DataFrame, players: pd.DataFrame, season: int, *, strict: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match labels without adding candidates or changing snapshot features."""
    rows = seed[seed["season"].eq(season)].copy()
    names = players.assign(key=players["athlete_name"].map(normalize_name))
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
            parts = [
                part for word in row.player.split() if (part := normalize_name(word))
            ]
            if not parts:
                raise ValueError(f"invalid Heisman ballot player name {row.player!r}")
            pattern = f"^{parts[0][0]}.*{parts[-1]}$"
            same_school = names[
                names["team"].map(normalize_name).eq(normalize_name(row.school))
            ]
            hit = same_school[same_school["key"].str.match(pattern)]
            if hit.empty:
                # Wikipedia and CFBD spell some schools differently; a
                # unique name match across the pool still identifies him.
                hit = names[names["key"].str.match(pattern)]
        if len(hit) != 1 and not strict:
            continue
        if len(hit) != 1:
            raise ValueError(
                f"{season} ballot row {row.player} ({row.school}) matched "
                f"{len(hit)} candidates"
            )
        matched.append({"athlete_id": hit["athlete_id"].iloc[0], "points": row.points})
    return pd.DataFrame(matched, columns=["athlete_id", "points"]), players


def load_seed() -> pd.DataFrame:
    return pd.read_csv(SEED_PATH)


def training_rows(seasons: list[int], seed: pd.DataFrame) -> pd.DataFrame:
    """Snapshots for every cached regular week, using only that week's pool.

    Targets are eventual ballot points conditional on the selected pool.
    A snapshot containing no eventual vote recipient remains in evaluation
    but contributes no likelihood when fitting.
    """
    if not seasons:
        raise ValueError("no historical player seasons available for Heisman training")
    frames = []
    for season in seasons:
        sources = _feature_sources(season)
        regular = _regular_box(season, None, sources["box"])
        for week in sorted(regular["week"].unique()):
            players = featured_players(season, int(week), sources=sources)
            pool = candidate_pool(players)
            if pool.empty:
                continue
            ballot, _ = _match_seed(seed, pool, season, strict=False)
            pool = pool.merge(ballot, on="athlete_id", how="left")
            pool["points"] = pd.to_numeric(pool["points"], errors="raise").fillna(0.0)
            total = float(pool["points"].sum())
            pool["share"] = pool["points"] / total if total else 0.0
            pool["week"] = int(week)
            frames.append(pool)
    if not frames:
        raise ValueError("historical player seasons have no regular-week candidates")
    return pd.concat(frames, ignore_index=True)


def _snapshot_groups(frame: pd.DataFrame):
    columns = ["season", "week"] if "week" in frame else ["season"]
    return frame.groupby(columns).indices.values()


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
        for index in _snapshot_groups(frame):
            block = logits[index]
            block = np.exp(block - block.max())
            out[index] = block / block.sum()
        return out

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": FEATURES, "coefficient": self.beta})


def fit_share_model(rows: pd.DataFrame) -> ShareModel:
    if rows.empty:
        raise ValueError("no historical snapshots available for Heisman fitting")
    valid_groups = [
        index for index in _snapshot_groups(rows) if rows.iloc[index]["share"].sum() > 0
    ]
    if not valid_groups:
        raise ValueError("historical candidate pools contain no ballot recipients")
    rows = rows.iloc[np.concatenate(valid_groups)].reset_index(drop=True)
    design = rows[FEATURES].to_numpy(float)
    mean = design.mean(axis=0)
    scale = design.std(axis=0)
    scale[scale == 0] = 1.0
    design = (design - mean) / scale
    share = rows["share"].to_numpy(float)
    groups = list(_snapshot_groups(rows))
    # Each season has equal total weight despite differing numbers of weeks.
    snapshot_counts = (
        rows.groupby("season")["week"].nunique() if "week" in rows else None
    )
    group_weights = [
        1.0 / snapshot_counts.loc[rows.iloc[index[0]]["season"]]
        if snapshot_counts is not None
        else 1.0
        for index in groups
    ]

    def loss_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ beta
        loss = RIDGE_PENALTY * float(beta @ beta)
        grad = 2.0 * RIDGE_PENALTY * beta
        for index, weight in zip(groups, group_weights):
            block = logits[index]
            block = block - block.max()
            weights = np.exp(block)
            probability = weights / weights.sum()
            log_probability = block - np.log(weights.sum())
            loss -= weight * float(share[index] @ log_probability)
            grad += weight * (design[index].T @ (probability - share[index]))
        return loss, grad

    result = minimize(
        loss_and_grad, np.zeros(len(FEATURES)), jac=True, method="L-BFGS-B"
    )
    if not result.success:
        raise RuntimeError(f"share model did not converge: {result.message}")
    return ShareModel(mean, scale, result.x)


def chronological_weekly_evaluation(
    rows: pd.DataFrame, seed: pd.DataFrame
) -> pd.DataFrame:
    """Evaluate each weekly pool with a model trained on earlier seasons only."""
    records = []
    for season in sorted(rows["season"].unique()):
        train = rows[rows["season"].lt(season)]
        if train.empty:
            continue
        model = fit_share_model(train)
        ballot = seed[seed["season"].eq(season)]
        if ballot.empty:
            raise ValueError(f"no Heisman ballot for evaluation season {season}")
        actual = ballot.sort_values("points", ascending=False).iloc[0]
        total_points = float(ballot["points"].sum())
        for week, held in rows[rows["season"].eq(season)].groupby("week"):
            held = held.copy()
            held["predicted_share"] = model.predict(held)
            held = held.sort_values(
                ["predicted_share", "athlete_id"], ascending=[False, True]
            ).reset_index(drop=True)
            held["predicted_rank"] = np.arange(1, len(held) + 1)
            matched, _ = _match_seed(
                ballot[ballot["points"].eq(actual["points"])],
                held,
                int(season),
                strict=False,
            )
            winner = held[held["athlete_id"].isin(matched["athlete_id"])]
            winner_in_pool = not winner.empty
            actual_rank = (
                int(winner["predicted_rank"].iloc[0]) if winner_in_pool else None
            )
            predicted = held.iloc[0]
            records.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "evaluation_kind": EVALUATION_KIND,
                    "training_seasons": sorted(
                        int(value) for value in train["season"].unique()
                    ),
                    "candidate_count": len(held),
                    "winner_in_pool": winner_in_pool,
                    "ballot_share_covered": float(held["points"].sum() / total_points),
                    "actual_winner": actual["player"],
                    "actual_winner_team": actual["school"],
                    "actual_share": float(actual["points"] / total_points),
                    "predicted_winner": predicted["athlete_name"],
                    "predicted_winner_team": predicted["team"],
                    "predicted_winner_share": float(predicted["predicted_share"]),
                    "actual_winner_predicted_share": float(
                        winner["predicted_share"].iloc[0]
                    )
                    if winner_in_pool
                    else 0.0,
                    "actual_winner_predicted_rank": actual_rank,
                    "winner_hit": actual_rank == 1,
                    "top_three_hit": actual_rank is not None and actual_rank <= 3,
                    "winner_value_rank": None,
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            *HISTORY_COLUMNS,
            "week",
            "evaluation_kind",
            "training_seasons",
            "candidate_count",
            "winner_in_pool",
            "ballot_share_covered",
        ],
    )


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
