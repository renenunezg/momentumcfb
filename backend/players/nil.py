"""NIL market value: what landing a player would cost in revenue-share dollars,
and for quarterbacks what his projected wins are worth at the same market's prices.

Market demand is the talent percentile of the program a transfer lands at (an
FCS landing spot counts as 0). A portal entrant's demand is predicted from his
portal stars and rating, which ranked landing spots better than production on
the 2025 holdout; every other player's from his production (grade, WAR, games,
class, program tier and talent), which ranks them less well, so his price is
flagged ``kind = "production"``. At each position every player with production
is ranked by demand; the ranking fills the Power 4 market's rostered slots
first and the rest after, as counted by the College Sports Commission, and a
slot's quantile reads the CSC's 2026 revenue-share pay at that position and
tier. Rostered players without production are assumed to rank below everyone
with it.

Worth ranks quarterbacks by projected next-season WAR and reads the same slots,
so worth and price share one dollar scale; surplus is worth less price.

Pay is revenue share per season (calendar 2026 payments, with performance
incentives counted as if paid). Third-party NIL is left out: the CSC reports it
as full contract values, not per season. The CSC publishes up to the 90th
percentile; above it pay follows the log-slope between the paid cohort's 75th
and 90th percentiles, capped at its 99th percentile.

Offensive linemen have no individual production, so they are not priced.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from backend.config import PROCESSED_DIR, RAW_DIR
from backend.players.ingest import players_dir
from backend.players.traits.targets import athlete_ids, normalize_name
from backend.players.traits.transfers import match_transfers
from backend.players.traits.war import POINTS_PER_WIN
from backend.players.value import POSITION_GROUPS

log = logging.getLogger(__name__)

MODEL_VERSION = "cfb_nil_value_v1"
DATA_DIR = Path(__file__).parent / "data"
PAY_PATH = DATA_DIR / "csc_football_pay_2026.csv"
MODEL_PATH = DATA_DIR / "nil_model.json"
PRICED_GROUPS = ("QB", "RB", "WR", "TE", "DL", "LB", "DB")
WORTH_GROUPS = ("QB",)
PRICE_ROW, WORTH_ROW, SURPLUS_ROW = "nil_price", "nil_worth", "nil_surplus"
PORTAL_KIND, PRODUCTION_KIND, WORTH_KIND = "portal_rating", "production", "projected"
# Portal position labels the roster map lacks.
PORTAL_GROUPS = {
    **POSITION_GROUPS,
    "IOL": "OL",
    "OC": "OL",
    "SDE": "DL",
    "WDE": "DL",
    "DUAL": "QB",
    "PRO": "QB",
    "APB": "RB",
}
P4_CONFERENCES = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"}
RIDGE = 1.0
WORTH_RIDGE = 3.0
STARTER_GAMES = 8
FULL_SEASON_GAMES = 12
TAIL_Z_CAP = norm.ppf(0.99)
PAY_Z = norm.ppf([0.10, 0.25, 0.50, 0.75, 0.90])
BOOTSTRAP_DRAWS = 1000
# A worth model must beat transfer stars on transfers this clearly, and be no
# clearer than this worse than carrying last season forward.
BEATS_STARS = 0.05
CLEARLY_WORSE = 0.95


@dataclass(frozen=True)
class NilModel:
    dev_targets: tuple[int, ...]
    holdout_target: int
    rating: dict[str, list[float]]
    rating_sd: dict[str, float]
    production: dict[str, list[float]]
    production_sd: dict[str, float]
    worth: list[float]
    price_groups: tuple[str, ...]
    production_groups: tuple[str, ...]
    worth_passes: bool
    evaluation: dict
    pay_sha256: str


# Features ----------------------------------------------------------------


def _tiers(season: int) -> pd.Series:
    teams = pd.read_parquet(players_dir(season) / "teams.parquet")
    tier = np.where(
        teams["conference"].isin(P4_CONFERENCES) | teams["school"].eq("Notre Dame"),
        "P4",
        np.where(teams["classification"].eq("fbs"), "G5", teams["classification"]),
    )
    return pd.Series(tier, index=teams["school"]).groupby(level=0).first()


def _talent(season: int) -> pd.Series:
    talent = pd.read_parquet(RAW_DIR / "talent" / f"{season}.parquet")
    return talent.groupby("team")["talent"].max().rank(pct=True)


def _roster(season: int) -> pd.DataFrame:
    roster = pd.read_parquet(players_dir(season) / "roster.parquet")
    roster = roster.assign(
        athlete_id=athlete_ids(roster["id"]),
        key=(roster["first_name"].fillna("") + " " + roster["last_name"].fillna(""))
        .astype(str)
        .map(normalize_name),
    )
    roster["year"] = roster["year"].where(roster["year"].between(1, 6))
    return roster.dropna(subset=["athlete_id"])


def _portal(season: int) -> pd.DataFrame | None:
    path = players_dir(season) / "portal.parquet"
    if not path.exists():
        return None
    portal = pd.read_parquet(path)
    if portal.empty or "first_name" not in portal.columns:
        return None
    return portal.assign(
        key=(portal["first_name"].fillna("") + " " + portal["last_name"].fillna(""))
        .astype(str)
        .map(normalize_name)
    )


def _end_of_season(values: pd.DataFrame) -> pd.DataFrame:
    """Latest-week production per player at a priced group, games and WAR prorated
    to a full season from his team's games so far."""
    latest = values[values["week"].eq(values["week"].max())].astype({"athlete_id": str})
    latest = latest[latest["position_group"].isin(PRICED_GROUPS)]
    latest = latest.sort_values("games", ascending=False).drop_duplicates("athlete_id")
    played = latest.groupby("team")["games"].transform("max").clip(lower=1)
    scale = (FULL_SEASON_GAMES / played).clip(lower=1.0)
    war = latest["value_above_replacement"] / POINTS_PER_WIN
    return pd.DataFrame(
        {
            "athlete_id": latest["athlete_id"],
            "athlete_name": latest["athlete_name"],
            "team": latest["team"],
            "team_id": latest["team_id"],
            "classification": latest["classification"],
            "position_group": latest["position_group"],
            "games": latest["games"],
            "games_0": latest["games"] * scale,
            "war_0": war * scale,
        }
    ).reset_index(drop=True)


def production_frame(
    season: int, values: pd.DataFrame, grades: pd.DataFrame
) -> pd.DataFrame:
    """Every player with production at a priced group in ``season`` and the
    features demand and worth read: grade, prorated WAR and games, class, and his
    program's tier and talent percentile."""
    out = _end_of_season(values)
    grade = grades.astype({"athlete_id": str}).drop_duplicates("athlete_id")
    out["grade"] = out["athlete_id"].map(grade.set_index("athlete_id")["grade"])
    roster = _roster(season).drop_duplicates("athlete_id").set_index("athlete_id")
    out["class_year"] = out["athlete_id"].map(roster["year"])
    out["tier_0"] = out["team"].map(_tiers(season))
    out["talent_0"] = out["team"].map(_talent(season))
    return out


def production_features(frame: pd.DataFrame) -> np.ndarray:
    grade = frame["grade"].astype(float)
    p4 = frame["tier_0"].eq("P4").astype(float)
    centered = grade.fillna(40.0) - 50.0
    columns = [
        np.ones(len(frame)),
        centered,
        grade.isna().astype(float),
        frame["war_0"].fillna(0.0),
        frame["games_0"].astype(float),
        *(frame["class_year"].eq(c).astype(float) for c in (1, 2, 3)),
        p4,
        frame["tier_0"].eq("fcs").astype(float),
        frame["talent_0"].fillna(0.0),
        p4 * centered,
        p4 * frame["games_0"].astype(float),
    ]
    return np.column_stack(columns).astype(float)


def rating_features(frame: pd.DataFrame) -> np.ndarray:
    stars = frame["stars"].fillna(0.0).astype(float)
    rating = frame["rating"].astype(float)
    columns = [
        np.ones(len(frame)),
        *(stars.ge(k).astype(float) for k in (2, 3, 4, 5)),
        rating.fillna(0.0),
        rating.isna().astype(float),
    ]
    return np.column_stack(columns).astype(float)


def worth_features(frame: pd.DataFrame) -> np.ndarray:
    war = frame["war_0"].fillna(0.0).astype(float)
    columns = [
        np.ones(len(frame)),
        war,
        frame["mover"].astype(float) * war,
        frame["tier_0"].eq("fcs").astype(float) * war,
        frame["games_0"].ge(STARTER_GAMES).astype(float),
        frame["class_year"].eq(1).astype(float),
    ]
    return np.column_stack(columns).astype(float)


def _ridge(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    shrink = np.eye(x.shape[1]) * penalty
    shrink[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + shrink, x.T @ y)


# Training data ------------------------------------------------------------


def _season_values(season: int) -> pd.DataFrame:
    return pd.read_parquet(
        PROCESSED_DIR / "players" / "player_values" / f"{season}.parquet"
    )


def _season_grades(season: int) -> pd.DataFrame:
    from backend.players.traits.backtest import final_week_grades, with_fbs
    from backend.players.traits.grade import latest_grade_model_path, load_grade_model

    model = load_grade_model(
        latest_grade_model_path(PROCESSED_DIR / "players" / "models", 9999)
    )
    snapshot = pd.read_parquet(
        PROCESSED_DIR / "players" / "position_traits" / f"{season}.parquet"
    )
    return final_week_grades(with_fbs(snapshot, season), model)


def _transfers(target: int, features: pd.DataFrame) -> pd.DataFrame:
    """Players with production in ``target - 1`` who transferred for ``target``,
    with their portal stars and rating and their landing spot's talent."""
    portal = _portal(target)
    if portal is None:
        raise FileNotFoundError(f"no portal snapshot for {target}; run ingest-portal")
    roster_to, roster_from = _roster(target), _roster(target - 1)
    moves = match_transfers(
        portal.drop(columns="key"),
        roster_to.drop(columns=["athlete_id", "key"]),
        roster_from.drop(columns=["athlete_id", "key"]),
    )
    keys = roster_to.drop_duplicates("athlete_id").set_index("athlete_id")["key"]
    moves = (
        moves.assign(key=moves["athlete_id"].map(keys))
        .merge(
            portal[["key", "origin", "destination", "stars", "rating"]],
            on=["key", "origin", "destination"],
            how="left",
        )
        .drop_duplicates("athlete_id")
    )
    out = features.merge(
        moves[["athlete_id", "destination", "stars", "rating"]], on="athlete_id"
    )
    out["demand"] = out["destination"].map(_talent(target)).fillna(0.0)
    return out.assign(target=target)


def _worth_pairs(season: int, features: pd.DataFrame, movers: set) -> pd.DataFrame:
    """QBs with production in ``season`` still rostered in ``season + 1``, with
    their next-season WAR (0 when they did not play)."""
    nxt = _end_of_season(_season_values(season + 1))
    nxt_war = nxt.set_index("athlete_id")["war_0"]
    rostered = set(_roster(season + 1)["athlete_id"])
    qbs = features[features["position_group"].eq("QB")]
    qbs = qbs[qbs["athlete_id"].isin(rostered)].copy()
    qbs["mover"] = qbs["athlete_id"].isin(movers)
    qbs["war_t"] = qbs["athlete_id"].map(nxt_war).fillna(0.0)
    return qbs.assign(target=season + 1)


# Fitting -------------------------------------------------------------------


def _rho(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(spearmanr(prediction, target)[0])


def _rank_check(
    rows: pd.DataFrame, featurize, targets: tuple[int, ...]
) -> dict[str, float]:
    """Forward-chained Spearman of predicted against actual landing-spot talent:
    each target season predicted from the seasons before it."""
    predictions, actual = [], []
    for target in targets:
        train, test = rows[rows["target"].lt(target)], rows[rows["target"].eq(target)]
        if train.empty or test.empty:
            continue
        beta = _ridge(featurize(train), train["demand"].to_numpy(float), RIDGE)
        predictions.append(featurize(test) @ beta)
        actual.append(test["demand"].to_numpy(float))
    prediction, target = np.concatenate(predictions), np.concatenate(actual)
    return {"n": int(len(target)), "rho": _rho(prediction, target)}


def _worth_check(rows: pd.DataFrame, targets: tuple[int, ...]) -> dict:
    """Forward-chained next-season WAR error for transfers, against carrying last
    season forward and against a transfer-stars table, with the bootstrap share
    of resamples in which the model does worse."""
    parts = []
    for target in targets:
        train, test = rows[rows["target"].lt(target)], rows[rows["target"].eq(target)]
        beta = _ridge(
            worth_features(train), train["war_t"].to_numpy(float), WORTH_RIDGE
        )
        moved = train[train["mover"]]
        table = moved.groupby(moved["stars"].fillna(0.0).clip(upper=5))["war_t"].mean()
        stars = test["stars"].fillna(0.0).clip(upper=5).map(table)
        parts.append(
            test.assign(
                model=worth_features(test) @ beta,
                carry=test["war_0"],
                stars_baseline=stars.fillna(moved["war_t"].mean()),
            )
        )
    movers = pd.concat(parts)
    movers = movers[movers["mover"]]
    y = movers["war_t"].to_numpy(float)
    errors = {
        name: (y - movers[name].to_numpy(float)) ** 2
        for name in ("model", "carry", "stars_baseline")
    }
    draws = np.random.default_rng(0).integers(0, len(y), (BOOTSTRAP_DRAWS, len(y)))

    def worse(baseline: str) -> float:
        gap = errors["model"] - errors[baseline]
        return float(np.mean(gap[draws].sum(axis=1) > 0))

    return {
        "n": int(len(y)),
        **{f"rmse_{k}": float(np.sqrt(v.mean())) for k, v in errors.items()},
        "corr_model": float(np.corrcoef(y, movers["model"])[0, 1]),
        "worse_than_carry": worse("carry"),
        "worse_than_stars": worse("stars_baseline"),
    }


def fit_nil_model(dev_targets: tuple[int, ...], holdout_target: int) -> NilModel:
    """Evaluate on forward-chained dev target seasons and the holdout, then refit
    every passing model on all of them.

    A position's demand model passes when its forward-chained Spearman is
    positive on both; QB worth passes when it beats the stars table on transfers
    (``BEATS_STARS``) and is not clearly worse than carry (``CLEARLY_WORSE``) on
    both. The first dev target is training only.
    """
    targets = (*dev_targets, holdout_target)
    features = {
        t - 1: production_frame(t - 1, _season_values(t - 1), _season_grades(t - 1))
        for t in targets
    }
    transfers = pd.concat(
        [_transfers(t, features[t - 1]) for t in targets], ignore_index=True
    )
    movers = {
        t: set(transfers.loc[transfers["target"].eq(t), "athlete_id"]) for t in targets
    }
    worth_rows = pd.concat(
        [_worth_pairs(t - 1, features[t - 1], movers[t]) for t in targets],
        ignore_index=True,
    ).merge(
        transfers[["athlete_id", "target", "stars"]],
        on=["athlete_id", "target"],
        how="left",
    )
    checked_dev, checked_holdout = dev_targets[1:], (holdout_target,)
    evaluation: dict = {"rating": {}, "production": {}}
    rating: dict[str, list[float]] = {}
    production: dict[str, list[float]] = {}
    rating_sd: dict[str, float] = {}
    production_sd: dict[str, float] = {}
    price_groups: list[str] = []
    production_groups: list[str] = []
    for group in PRICED_GROUPS:
        rows = transfers[transfers["position_group"].eq(group)]
        y = rows["demand"].to_numpy(float)
        for name, featurize, coefs, sds, passing in (
            ("rating", rating_features, rating, rating_sd, price_groups),
            (
                "production",
                production_features,
                production,
                production_sd,
                production_groups,
            ),
        ):
            checks = {
                "dev": _rank_check(rows, featurize, checked_dev),
                "holdout": _rank_check(rows, featurize, checked_holdout),
            }
            evaluation[name][group] = checks
            if all(c["rho"] > 0 for c in checks.values()):
                passing.append(group)
            beta = _ridge(featurize(rows), y, RIDGE)
            coefs[group] = beta.tolist()
            sds[group] = float(np.std(y - featurize(rows) @ beta))
    worth_eval = {
        "dev": _worth_check(worth_rows, checked_dev),
        "holdout": _worth_check(worth_rows, checked_holdout),
    }
    evaluation["worth"] = worth_eval
    worth_passes = all(
        c["worse_than_stars"] < BEATS_STARS and c["worse_than_carry"] < CLEARLY_WORSE
        for c in worth_eval.values()
    )
    worth = _ridge(
        worth_features(worth_rows), worth_rows["war_t"].to_numpy(float), WORTH_RIDGE
    )
    return NilModel(
        dev_targets=tuple(dev_targets),
        holdout_target=holdout_target,
        rating=rating,
        rating_sd=rating_sd,
        production=production,
        production_sd=production_sd,
        worth=worth.tolist(),
        price_groups=tuple(price_groups),
        production_groups=tuple(production_groups),
        worth_passes=worth_passes,
        evaluation=evaluation,
        pay_sha256=hashlib.sha256(PAY_PATH.read_bytes()).hexdigest(),
    )


def write_nil_model(model: NilModel, path: Path = MODEL_PATH) -> Path:
    payload = {"model_version": MODEL_VERSION, **asdict(model)}
    path.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    return path


def load_nil_model(path: Path = MODEL_PATH) -> NilModel:
    payload = json.loads(path.read_text())
    if payload.pop("model_version") != MODEL_VERSION:
        raise ValueError(f"{path} is not a {MODEL_VERSION} model")
    if payload["pay_sha256"] != hashlib.sha256(PAY_PATH.read_bytes()).hexdigest():
        raise ValueError(f"{PAY_PATH} changed since {path} was fitted; refit it")
    for key in ("dev_targets", "price_groups", "production_groups"):
        payload[key] = tuple(payload[key])
    return NilModel(**payload)


# Pay -----------------------------------------------------------------------


def _pay_table() -> pd.DataFrame:
    pay = pd.read_csv(PAY_PATH)
    return pay[pay["source"].eq("revenue_share")].set_index(
        ["position", "tier", "cohort"]
    )


def revenue_share(group: str, tier: str, quantile: np.ndarray) -> np.ndarray:
    """Revenue-share pay per season, in millions, at rostered ``quantile`` of
    ``group`` in ``tier`` ("A4" or "non_A4").

    The unpaid share of the rostered cohort earns 0; within the paid cohort log
    pay is interpolated in normal z between the published percentiles, extended
    past them at the end segments' slopes, with z capped at the 99th percentile.
    """
    table = _pay_table()
    rostered = table.loc[(group, tier, "with_zero")]
    paid = table.loc[(group, tier, "without_zero")]
    share = paid["n"] / rostered["n"]
    quantile = np.asarray(quantile, float)
    within = (quantile - (1.0 - share)) / share
    z = norm.ppf(np.clip(within, 1e-6, 1 - 1e-6)).clip(max=TAIL_Z_CAP)
    logs = np.log(paid[["p10", "p25", "p50", "p75", "p90"]].to_numpy(float))
    low = logs[0] + (z - PAY_Z[0]) * (logs[1] - logs[0]) / (PAY_Z[1] - PAY_Z[0])
    high = logs[4] + (z - PAY_Z[4]) * (logs[4] - logs[3]) / (PAY_Z[4] - PAY_Z[3])
    inside = np.interp(z, PAY_Z, logs)
    pay = np.exp(np.where(z < PAY_Z[0], low, np.where(z > PAY_Z[4], high, inside)))
    return np.where(within > 0, pay, 0.0) / 1e6


def _slots(group: str) -> tuple[int, int]:
    table = _pay_table()
    return (
        int(table.loc[(group, "A4", "with_zero"), "n"]),
        int(table.loc[(group, "non_A4", "with_zero"), "n"]),
    )


def _pay_by_score(group: str, ranked: np.ndarray, scores: np.ndarray) -> tuple:
    """Pay and market quantile for each of ``scores`` placed in the descending
    ranking ``ranked``: Power 4 slots first, then the rest."""
    a4, rest = _slots(group)
    order = np.sort(ranked)
    # Rank 1 is the highest score; a score between two players takes a fractional rank.
    rank = len(order) - np.interp(scores, order, np.arange(len(order))) + 0.5
    rank = np.clip(rank, 0.5, None)
    in_a4 = rank <= a4
    quantile = np.where(in_a4, 1 - rank / a4, 1 - (rank - a4) / rest).clip(0.0, 1.0)
    pay = np.where(
        in_a4,
        revenue_share(group, "A4", quantile),
        revenue_share(group, "non_A4", quantile),
    )
    return pay, np.clip(1 - rank / (a4 + rest), 0.0, 1.0)


# Scoring -------------------------------------------------------------------


def _portal_entrants(season: int) -> pd.DataFrame:
    """Entrants of the portal that follows ``season``, tied to an athlete id by a
    unique name on the origin roster (destinations may not exist yet)."""
    portal = _portal(season + 1)
    if portal is None:
        return pd.DataFrame(columns=["athlete_id", "stars", "rating"])
    roster = _roster(season)
    lookup = roster.groupby(["key", "team"])["athlete_id"].agg(["nunique", "first"])
    lookup = lookup[lookup["nunique"].eq(1)]["first"].rename("athlete_id")
    entrants = portal.merge(
        lookup.reset_index().rename(columns={"team": "origin"}), on=["key", "origin"]
    )
    log.info(
        f"nil {season}: {len(entrants)} of {len(portal)} portal entries tied to a roster id"
    )
    return entrants.drop_duplicates("athlete_id")[["athlete_id", "stars", "rating"]]


def score_nil(
    season: int, values: pd.DataFrame, grades: pd.DataFrame, model: NilModel
) -> pd.DataFrame:
    """Price, and for passing worth groups worth and surplus, per player with
    production at a priced group: ``athlete_id, position_group, trait, value,
    uncertainty, percentile, kind`` with values in millions of dollars per season.

    Entrants of the next portal are priced from their portal rating; everyone
    else at a group whose production model passed, from production.
    """
    frame = production_frame(season, values, grades)
    entrants = _portal_entrants(season)
    frame = frame.merge(entrants, on="athlete_id", how="left")
    frame["mover"] = frame["athlete_id"].isin(entrants["athlete_id"])
    rows = []
    for group in model.price_groups:
        part = frame[frame["position_group"].eq(group)].reset_index(drop=True)
        if part.empty:
            continue
        portal = part["mover"].to_numpy()
        demand = np.full(len(part), np.nan)
        spread = np.full(len(part), np.nan)
        if portal.any():
            demand[portal] = rating_features(part[portal]) @ np.array(
                model.rating[group]
            )
            spread[portal] = model.rating_sd[group]
        if group in model.production_groups and (~portal).any():
            demand[~portal] = production_features(part[~portal]) @ np.array(
                model.production[group]
            )
            spread[~portal] = model.production_sd[group]
        scored = ~np.isnan(demand)
        part, demand, spread, portal = (
            part[scored].reset_index(drop=True),
            demand[scored],
            spread[scored],
            portal[scored],
        )
        if part.empty:
            continue
        price, market = _pay_by_score(group, demand, demand)
        low, _ = _pay_by_score(group, demand, demand - spread)
        high, _ = _pay_by_score(group, demand, demand + spread)
        rows.append(
            pd.DataFrame(
                {
                    "athlete_id": part["athlete_id"],
                    "position_group": group,
                    "trait": PRICE_ROW,
                    "value": price,
                    "uncertainty": (high - low) / 2,
                    "percentile": 100 * market,
                    "kind": np.where(portal, PORTAL_KIND, PRODUCTION_KIND),
                }
            )
        )
        if group in WORTH_GROUPS and model.worth_passes:
            projected = worth_features(part) @ np.array(model.worth)
            worth, worth_market = _pay_by_score(group, projected, projected)
            rows.append(
                pd.DataFrame(
                    {
                        "athlete_id": part["athlete_id"],
                        "position_group": group,
                        "trait": WORTH_ROW,
                        "value": worth,
                        "uncertainty": np.nan,
                        "percentile": 100 * worth_market,
                        "kind": WORTH_KIND,
                    }
                )
            )
            rows.append(
                pd.DataFrame(
                    {
                        "athlete_id": part["athlete_id"],
                        "position_group": group,
                        "trait": SURPLUS_ROW,
                        "value": worth - price,
                        "uncertainty": np.nan,
                        "percentile": np.nan,
                        "kind": WORTH_KIND,
                    }
                )
            )
    if not rows:
        raise ValueError(f"nil {season}: no player could be priced")
    out = pd.concat(rows, ignore_index=True)
    identity = frame.set_index("athlete_id")
    for column in ("athlete_name", "team", "team_id", "games"):
        out[column] = out["athlete_id"].map(identity[column])
    return out


def nil_artifact(season: int) -> tuple[str, ...]:
    return ("players", "nil_values", f"{season}.parquet")


def nil_trait_rows(
    nil: pd.DataFrame, published: pd.DataFrame, as_of: datetime
) -> pd.DataFrame:
    """``nil`` rows shaped as ``cfb.position_traits`` rows of the published
    snapshot's week, for athletes the snapshot publishes."""
    week = published["week"].max()
    shown = set(published["athlete_id"].astype(str))
    rows = nil[nil["athlete_id"].isin(shown)]
    return rows.assign(
        season=published["season"].iloc[0],
        week=week,
        as_of=as_of.isoformat(),
        model_version=MODEL_VERSION,
        in_grade=False,
        qualified=True,
        points_above_replacement=np.nan,
        fbs=True,
        poor_coverage=False,
    )
