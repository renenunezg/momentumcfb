"""The position grade: a 20 to 80 scouting number built from a player's traits.

Each trait's weight at a position comes from player-level evidence on the
development seasons only: its split-half reliability at 8 games (how well a
player's odd games predict his even games) times its portability (the share of
its year-over-year persistence that survives a transfer). Unreliable traits
(reliability under ``NOISE_RELIABILITY``), traits whose reliability cannot be
measured, and traits that do not travel (portability at or below zero) get no
weight and stay on the card as descriptive. A trait whose portability cannot be
measured keeps its reliability weight. Weights sum to 1 within each position.

A player's ``composite`` is the weighted sum of his oriented trait z-scores
among qualified FBS players at the group that week, with a trait he has no
value for counted at the position average (z = 0). He is graded only when he
has his position's primary exposure trait and his traits cover at least
``MIN_WEIGHT_COVERAGE`` of the position's weight. The ``grade`` puts the
composite on the 20 to 80 scale (mean 50, sd 10 among graded players). Nothing
is fitted to team margins or to any outcome.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import PROCESSED_DIR
from backend.players.traits import MODEL_VERSION, TRAITS

NOISE_RELIABILITY = 0.10
WEIGHT_GAMES = 8
# The trait whose denominator measures a player's usage at the position. Without
# it the player has no measured role there (a wildcat snap, a trick play), so he
# is not graded however extreme his other values are.
PRIMARY_TRAITS = {
    "QB": "qb.explosive",
    "RB": "rb.yards_over_expected",
    "WR": "rec.value_per_target",
    "TE": "rec.value_per_target",
    "DL": "def.every_week",
    "LB": "def.every_week",
    "DB": "def.every_week",
    "OL": "ol.line_yards",
    "K": "k.field_goal",
    "P": "p.punt_average",
}
# The least share of the position's total trait weight a player's measured traits
# must carry for a grade. Missing traits count as average (z = 0), so below this
# the composite would be mostly the average standing in for him.
MIN_WEIGHT_COVERAGE = 0.6
# A trait counts toward the coverage denominator only where at least this share
# of the week's qualified FBS players at the group have a value for it. Structural
# gaps (``qb.floor`` before four games, prior-week baselines in week one, receiver
# independence before any receiver has other-passer targets) then drop out for
# everyone, while one player with an unusually thin profile still fails the bar.
MIN_TRAIT_AVAILABILITY = 0.5
GRADE_MEAN, GRADE_SD, GRADE_MIN, GRADE_MAX = 50.0, 10.0, 20.0, 80.0
WEIGHT_COLUMNS = [
    "position_group",
    "trait",
    "sign",
    "sign_source",
    "reliability",
    "portability",
    "weight",
    "status",
]
ELIGIBILITY_COLUMNS = [
    "athlete_id",
    "position_group",
    "games",
    "has_primary",
    "coverage",
    "graded",
    "composite",
]
ORIENTATION_COLUMNS = ["position_group", "trait", "sign", "sign_source", "spearman_epa", "players"]
RELIABILITY_COLUMNS = ["position_group", "games", "players", "correlation"]
OUTPUT_COLUMNS = [
    "athlete_id",
    "position_group",
    "games",
    "composite",
    "grade",
    "grade_percentile",
    "uncertainty",
]
GRADE_NAME = re.compile(r"position_grade_(\d{4})\.json")


# Weights


def group_traits() -> pd.DataFrame:
    """``position_group, trait`` for every registered trait at each of its groups."""
    return pd.DataFrame(
        [(group, name) for name, spec in TRAITS.items() for group in spec.position_groups],
        columns=["position_group", "trait"],
    )


def _spearman(left: pd.Series, right: pd.Series) -> float:
    both = left.notna() & right.notna()
    left, right = left[both].rank(), right[both].rank()
    if len(left) < 3 or not (left.std() > 0 and right.std() > 0):
        return np.nan
    return float(left.corr(right))


def orientations(final_weeks: pd.DataFrame) -> pd.DataFrame:
    """The sign each trait enters the grade with, per group.

    ``higher_is_better`` True is +1 and False is -1 (``sign_source`` ``registered``).
    ``None`` takes the sign of the trait's Spearman correlation with v4
    ``shrunk_per_game`` among the rows of ``final_weeks`` at the group
    (``sign_source`` ``epa_spearman``; +1 when the correlation is missing).
    ``final_weeks`` holds ``athlete_id, position_group, trait, value, shrunk_per_game``
    for qualified players at the final week of each development season.
    """
    rows = []
    for group, trait in group_traits().itertuples(index=False):
        spec = TRAITS[trait]
        part = final_weeks[final_weeks["position_group"].eq(group) & final_weeks["trait"].eq(trait)]
        part = part.dropna(subset=["value", "shrunk_per_game"])
        rho = _spearman(part["value"], part["shrunk_per_game"]) if len(part) else np.nan
        if spec.higher_is_better is None:
            sign, source = (-1.0 if rho < 0 else 1.0), "epa_spearman"
        else:
            sign, source = (1.0 if spec.higher_is_better else -1.0), "registered"
        rows.append(
            {
                "position_group": group,
                "trait": trait,
                "sign": sign,
                "sign_source": source,
                "spearman_epa": rho,
                "players": len(part),
            }
        )
    return pd.DataFrame(rows, columns=ORIENTATION_COLUMNS)


def trait_weights(
    reliability: pd.DataFrame, portability: pd.DataFrame, orientation: pd.DataFrame
) -> pd.DataFrame:
    """One row per group and trait with its weight in the grade and why.

    ``reliability`` is the pooled split-half table (``position_group, trait,
    correlation`` and optionally ``games``, of which only ``WEIGHT_GAMES`` is read);
    ``portability`` is ``transfers.portability`` output, whose ``portable_share`` is
    clipped to [0, 1] and is unmeasured where NaN; ``orientation`` is
    ``orientations`` output. ``weight = max(reliability, 0) × (portability or 1)``,
    normalized within the group over weighted traits. ``status`` is
    ``unmeasured_reliability`` when no reliability could be measured (too few
    players, or a ``latest`` trait with no per-game values), ``noise`` when it is
    under ``NOISE_RELIABILITY``, ``team-driven`` when portability is measured and at
    or below zero, ``unmeasured`` when weighted without a portability, else
    ``weighted``. Only ``weighted`` and ``unmeasured`` traits carry weight.
    """
    rel = reliability
    if "games" in rel.columns:
        rel = rel[rel["games"].eq(WEIGHT_GAMES)]
    rel = rel.drop_duplicates(["position_group", "trait"]).set_index(["position_group", "trait"])
    port = portability.drop_duplicates(["position_group", "trait"]).set_index(
        ["position_group", "trait"]
    )
    out = orientation[["position_group", "trait", "sign", "sign_source"]].copy()
    keys = pd.MultiIndex.from_frame(out[["position_group", "trait"]])
    out["reliability"] = rel["correlation"].reindex(keys).to_numpy(float)
    raw_share = port["portable_share"].reindex(keys).to_numpy(float)
    out["portability"] = np.clip(raw_share, 0.0, 1.0)

    unmeasured_reliability = out["reliability"].isna()
    noise = ~unmeasured_reliability & out["reliability"].lt(NOISE_RELIABILITY)
    measured = out["portability"].notna()
    usable = ~unmeasured_reliability & ~noise
    team_driven = usable & measured & (raw_share <= 0)
    out["status"] = np.select(
        [unmeasured_reliability, noise, team_driven, ~measured],
        ["unmeasured_reliability", "noise", "team-driven", "unmeasured"],
        "weighted",
    )
    raw = out["reliability"].clip(lower=0.0) * out["portability"].fillna(1.0)
    raw = raw.where(out["status"].isin(["weighted", "unmeasured"]), 0.0).fillna(0.0)
    total = raw.groupby(out["position_group"]).transform("sum")
    out["weight"] = (raw / total.where(total.gt(0))).fillna(0.0)
    return out[WEIGHT_COLUMNS].reset_index(drop=True)


# The model artifact


@dataclass
class GradeModel:
    """``weights`` is ``trait_weights`` output; ``reliability`` the composite's pooled
    split-half correlation per group at each game count (``RELIABILITY_COLUMNS``)."""

    weights: pd.DataFrame
    dev_seasons: list[int]
    reliability: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=RELIABILITY_COLUMNS)
    )
    model_version: str = MODEL_VERSION


def grade_model_path(cutoff_season: int) -> Path:
    return PROCESSED_DIR / "players" / "models" / f"position_grade_{cutoff_season}.json"


def grade_cutoff(path: Path) -> int:
    """The cutoff season in a ``position_grade_<cutoff>.json`` file name."""
    match = GRADE_NAME.fullmatch(Path(path).name)
    if not match:
        raise ValueError(f"{path} is not a position grade model")
    return int(match.group(1))


def latest_grade_model_path(models_dir: Path, season: int) -> Path:
    """The grade model in ``models_dir`` with the greatest cutoff before ``season``."""
    cutoffs = {
        int(match.group(1)): path
        for path in Path(models_dir).glob("position_grade_*.json")
        if (match := GRADE_NAME.fullmatch(path.name))
    }
    earlier = [cutoff for cutoff in cutoffs if cutoff < season]
    if not earlier:
        raise FileNotFoundError(
            f"no position grade model with a cutoff before {season} in {models_dir}; "
            "run position-traits-fit first"
        )
    return cutoffs[max(earlier)]


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def save_grade_model(model: GradeModel, path: Path) -> Path:
    payload = {
        "model_version": model.model_version,
        "dev_seasons": [int(season) for season in model.dev_seasons],
        "weights": _records(model.weights[WEIGHT_COLUMNS]),
        "reliability": _records(model.reliability[RELIABILITY_COLUMNS]),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=1))
    temporary.replace(path)
    return path


def grade_model_from_payload(payload: dict) -> GradeModel:
    weights = pd.DataFrame(payload["weights"], columns=WEIGHT_COLUMNS)
    for column in ("sign", "reliability", "portability", "weight"):
        weights[column] = weights[column].astype(float)
    reliability = pd.DataFrame(payload["reliability"], columns=RELIABILITY_COLUMNS)
    return GradeModel(
        weights=weights,
        dev_seasons=[int(season) for season in payload["dev_seasons"]],
        reliability=reliability.astype({"games": int, "correlation": float}),
        model_version=payload["model_version"],
    )


def load_grade_model(path: Path) -> GradeModel:
    return grade_model_from_payload(json.loads(Path(path).read_text()))


def validate_grade_model(model: GradeModel, season: int) -> None:
    """Refuse a model that saw ``season`` or a later one, or that another version made."""
    if model.model_version != MODEL_VERSION:
        raise ValueError(f"grade model is {model.model_version!r}, expected {MODEL_VERSION!r}")
    if not model.dev_seasons or max(model.dev_seasons) >= season:
        raise ValueError(f"grade model on {model.dev_seasons} cannot score {season}")


# Scoring


def _scaled(values: pd.Series, reference: pd.Series, keys: list[pd.Series]) -> pd.Series:
    """(value − mean) / sd with the mean and sd (ddof 1) of ``reference`` (NaN where not
    a reference row) within ``keys``; 0 where sd is 0 or missing, NaN where the value is."""
    mean = reference.groupby(keys).transform("mean")
    sd = reference.groupby(keys).transform("std")
    z = (values - mean) / sd.where(sd.gt(0))
    return z.where(values.isna() | z.notna(), 0.0)


def reliability_at(table: pd.DataFrame, games: pd.Series) -> np.ndarray:
    """Linear in games from r(0) = 0, flat past the last point, clipped to [0, 1]."""
    table = table.dropna(subset=["correlation"])
    if table.empty:
        return np.zeros(len(games))
    table = table.sort_values("games")
    x = np.concatenate([[0.0], table["games"].to_numpy(float)])
    r = np.concatenate([[0.0], table["correlation"].to_numpy(float)])
    return np.clip(np.interp(games.to_numpy(float), x, r), 0.0, 1.0)


def graded_groups(model: GradeModel) -> set[str]:
    """Groups with at least one trait of positive weight."""
    return set(model.weights.loc[model.weights["weight"].gt(0), "position_group"])


def grade_eligibility(snapshot_week: pd.DataFrame, model: GradeModel) -> pd.DataFrame:
    """Every qualified FBS player at a graded group in one snapshot week, graded or not.

    ``snapshot_week`` holds trait snapshot rows of one week with ``team`` and an
    ``fbs`` boolean. The pool is qualified FBS players at the group. Each weighted
    trait is z-scored on the pool's mean and sd (ddof 1; 0 where sd is 0 or
    missing) and multiplied by the model's sign. ``composite = Σ w_t z_t`` over the
    group's weighted traits, with ``z = 0`` for a trait the player has no value for.
    A trait is available at the group that week when at least
    ``MIN_TRAIT_AVAILABILITY`` of the pool has a value for it. ``coverage`` is the
    weight of the available traits he has a value for over the weight of every
    available trait; ``graded`` needs ``has_primary`` (a value for
    ``PRIMARY_TRAITS[group]``) and ``coverage`` of at least ``MIN_WEIGHT_COVERAGE``.
    ``composite`` is NaN where he is not graded.
    """
    missing = {"team", "fbs"} - set(snapshot_week.columns)
    if missing:
        raise ValueError(f"score_grades needs {sorted(missing)} on the snapshot")
    keys = ["athlete_id", "position_group"]
    rows = snapshot_week.astype({"athlete_id": str})
    rows = rows.assign(qualified=rows["qualified"].eq(True), fbs=rows["fbs"].eq(True))
    players = rows.groupby(keys, as_index=False).agg(
        games=("games", "max"), qualified=("qualified", "any"), fbs=("fbs", "any")
    )
    players = players[
        players["qualified"]
        & players["fbs"]
        & players["position_group"].isin(graded_groups(model))
    ][[*keys, "games"]].reset_index(drop=True)
    weights = model.weights[model.weights["weight"].gt(0)]
    pool = rows.merge(players[keys], on=keys).merge(
        weights[["position_group", "trait", "sign", "weight"]], on=["position_group", "trait"]
    )
    value = pool["value"].astype(float)
    z = _scaled(value, value, [pool["position_group"], pool["trait"]]) * pool["sign"]
    pool = pool.assign(z=z)[z.notna()]
    pool_size = players.groupby("position_group").size()
    share = pool.groupby(["position_group", "trait"])["athlete_id"].nunique().div(
        pool_size, level="position_group"
    )
    available = share[share.ge(MIN_TRAIT_AVAILABILITY)].index
    available_weight = (
        weights.set_index(["position_group", "trait"])["weight"]
        .reindex(available)
        .groupby(level="position_group")
        .sum()
    )
    in_available = pd.MultiIndex.from_frame(pool[["position_group", "trait"]]).isin(available)
    sums = (
        pool.assign(
            weighted=pool["weight"] * pool["z"],
            covered=pool["weight"].where(in_available, 0.0),
        )
        .groupby(keys)[["weighted", "covered"]]
        .sum()
        .reindex(pd.MultiIndex.from_frame(players[keys]))
        .fillna(0.0)
    )
    primary = pool[pool["trait"].eq(pool["position_group"].map(PRIMARY_TRAITS))]
    primary_keys = set(zip(primary["athlete_id"], primary["position_group"]))
    players["has_primary"] = [key in primary_keys for key in zip(players["athlete_id"], players["position_group"])]
    denominator = players["position_group"].map(available_weight)
    players["coverage"] = sums["covered"].to_numpy() / denominator.where(denominator.gt(0))
    players["graded"] = players["has_primary"] & players["coverage"].ge(MIN_WEIGHT_COVERAGE)
    players["composite"] = np.where(players["graded"], sums["weighted"].to_numpy(), np.nan)
    return players[ELIGIBILITY_COLUMNS]


COVERAGE_LOSS_COLUMNS = ["position_group", "qualified", "graded", "no_primary", "low_coverage"]


def coverage_losses(eligibility: pd.DataFrame) -> pd.DataFrame:
    """Per group: qualified FBS players, how many are graded, and how many lose the
    grade for lacking the primary trait or, with it, for coverage under
    ``MIN_WEIGHT_COVERAGE``."""
    frame = eligibility.assign(
        no_primary=~eligibility["has_primary"],
        low_coverage=eligibility["has_primary"] & ~eligibility["graded"],
    )
    out = frame.groupby("position_group").agg(
        qualified=("athlete_id", "size"),
        graded=("graded", "sum"),
        no_primary=("no_primary", "sum"),
        low_coverage=("low_coverage", "sum"),
    )
    return out.reset_index()[COVERAGE_LOSS_COLUMNS].astype(
        {name: int for name in COVERAGE_LOSS_COLUMNS[1:]}
    )


def score_grades(snapshot_week: pd.DataFrame, model: GradeModel) -> pd.DataFrame:
    """Each graded player's grade from one snapshot week (see ``grade_eligibility``).

    ``grade = 50 + 10 × (composite − mean) / sd`` over the graded players at the
    group, clipped to [20, 80]; ``grade_percentile`` ranks the composite (0 to 100,
    100 for a lone player); ``uncertainty = 10 × sqrt(1 − r)`` with ``r`` the model's
    composite reliability at the player's games.
    """
    out = grade_eligibility(snapshot_week, model)
    out = out[out["graded"]].reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    by_group = out.groupby("position_group")["composite"]
    mean, sd = by_group.transform("mean"), by_group.transform("std")
    standard = ((out["composite"] - mean) / sd.where(sd.gt(0))).fillna(0.0)
    out["grade"] = (GRADE_MEAN + GRADE_SD * standard).clip(GRADE_MIN, GRADE_MAX)
    rank = by_group.rank(method="average")
    n = by_group.transform("size")
    out["grade_percentile"] = (100 * (rank - 1) / (n - 1)).where(n.gt(1), 100.0)
    reliability = np.zeros(len(out))
    for group, index in out.groupby("position_group").groups.items():
        table = model.reliability[model.reliability["position_group"].eq(group)]
        reliability[out.index.get_indexer(index)] = reliability_at(table, out.loc[index, "games"])
    out["uncertainty"] = GRADE_SD * np.sqrt(1 - reliability)
    return out[OUTPUT_COLUMNS]
