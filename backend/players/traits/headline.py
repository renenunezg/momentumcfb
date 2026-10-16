"""The position trait headline: one joint ridge of team margin on roster traits.

Each completed FBS-versus-FBS game from week 2 on is one row. A feature
``"{group}:{trait}"`` (every registered trait at each of its groups, plus
``"{group}:epa"`` from v4 ``shrunk_per_game``) holds the home team's aggregate
minus the away team's, where a team's aggregate is ``Σ share_p × z_p`` over its
players at the group in the snapshot of the week before the game. The fit
standardizes each feature column, solves the ridge in closed form with the
intercept and ``home_field`` unpenalized, picks the penalty by leave-one-season-out
MAE, and drops features whose weight is under twice the larger of its
jackknife and analytic standard errors. A player's headline is their own share-1 version of
the same sum, in points per game above the group's replacement level.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve

from backend.config import PROCESSED_DIR
from backend.players.traits import MODEL_VERSION, TRAITS
from backend.players.traits.snapshots import UNIT_PREFIX
from backend.players.value import REPLACEMENT_PERCENTILE

GROUPS = ("QB", "RB", "WR", "TE", "DL", "LB", "DB", "OL", "K", "P")
EPA = "epa"
# The trait whose denominator measures a player's usage at the group; OL units have share 1.
EXPOSURE_TRAITS = {
    "QB": "qb.explosive",  # dropbacks
    "RB": "rb.yards_over_expected",  # carries
    "WR": "rec.value_per_target",  # targets
    "TE": "rec.value_per_target",
    "DL": "def.every_week",  # games
    "LB": "def.every_week",
    "DB": "def.every_week",
    "K": "k.field_goal",
    "P": "p.punt_average",
}
META_COLUMNS = ["game_id", "season", "model_week", "margin", "home_field"]
FIXED = ["intercept", "home_field"]  # unpenalized and never dropped
PENALTIES = np.logspace(-2, 5, 36)
DROP_MULTIPLE = 2.0
RELIABILITY_COLUMNS = ["position_group", "games", "correlation"]
OUTPUT_COLUMNS = [
    "athlete_id",
    "position_group",
    "games",
    "raw_score",
    "headline",
    "headline_percentile",
    "uncertainty",
    "replaces_value",
]


@dataclass
class HeadlineFit:
    """``weights`` and the three spreads are indexed ``intercept``, ``home_field`` and the
    kept features; feature weights are per fit-time standard deviation of the column,
    whose mean and sd are in ``feature_scale``. ``loso_sd`` is the raw sd (ddof 1) of each
    weight across the leave-one-season-out refits, ``jackknife_sd`` its delete-one-season
    jackknife scaling and ``analytic_se`` the ridge sandwich standard error.
    ``dropped_reason`` maps each dropped feature to ``"constant"`` or ``"insignificant"``."""

    weights: pd.Series
    penalty: float
    loso_sd: pd.Series
    jackknife_sd: pd.Series
    analytic_se: pd.Series
    dropped: list[str]
    dropped_reason: dict[str, str]
    dev_seasons: list[int]
    model_version: str
    feature_scale: pd.DataFrame
    reliability: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=RELIABILITY_COLUMNS)
    )
    replaces_value: dict[str, bool] = field(default_factory=lambda: dict.fromkeys(GROUPS, False))


def feature_names() -> list[str]:
    """Every registered trait at each of its groups, then the group's ``epa``, group by group."""
    names = []
    for group in GROUPS:
        names += [f"{group}:{name}" for name, spec in TRAITS.items() if group in spec.position_groups]
        names.append(f"{group}:{EPA}")
    return names


def exposure_shares(frames: pd.DataFrame, positions_map: pd.Series) -> pd.DataFrame:
    """``athlete_id, week, exposure``: cumulative usage through each model week.

    ``frames`` are the raw trait rows and ``positions_map`` maps athlete id to
    position group. Usage is the denominator of the group's exposure trait in
    ``EXPOSURE_TRAITS``, summed whether or not the row has a numerator. Weeks run
    from the first to the last model week in ``frames``, so a week nobody played
    repeats the prior total. OL units have exposure 1.0 every week.
    """
    rows = frames.astype({"athlete_id": str})
    groups = positions_map.copy()
    groups.index = groups.index.astype(str)
    weeks = range(int(rows["model_week"].min()), int(rows["model_week"].max()) + 1)
    units = rows["athlete_id"].str.startswith(UNIT_PREFIX)
    exposure_trait = rows["athlete_id"].map(groups).map(EXPOSURE_TRAITS)
    used = rows[~units & rows["trait"].eq(exposure_trait)]
    weekly = (
        used.assign(denominator=used["denominator"].fillna(0.0))
        .groupby(["athlete_id", "model_week"])["denominator"]
        .sum()
        .unstack("model_week")
        .reindex(columns=weeks)
        .fillna(0.0)
        .cumsum(axis=1)
    )
    unit_ids = sorted(rows.loc[units, "athlete_id"].unique())
    weekly = pd.concat([weekly, pd.DataFrame(1.0, index=unit_ids, columns=weeks)])
    weekly.index.name, weekly.columns.name = "athlete_id", "week"
    return weekly.stack().rename("exposure").reset_index()


def _scaled(values: pd.Series, reference: pd.Series, keys: list[pd.Series]) -> pd.Series:
    """(value − mean) / sd with the mean and sd (ddof 1) of ``reference`` (NaN where not
    a reference row) within ``keys``; 0 where the value is missing or sd is 0 or NaN."""
    mean = reference.groupby(keys).transform("mean")
    sd = reference.groupby(keys).transform("std")
    z = (values - mean) / sd.where(sd.gt(0))
    return z.fillna(0.0)


def standardized_traits(snapshot: pd.DataFrame) -> pd.DataFrame:
    """``athlete_id, week, position_group, trait, z`` on the league scale of qualified players.

    The scale is the qualified players' mean and sd at the group, trait and week;
    every player with a value is placed on it. Traits with ``higher_is_better``
    False are negated first; ``None`` keeps its sign.
    """
    sign = snapshot["trait"].map(
        {name: -1.0 if spec.higher_is_better is False else 1.0 for name, spec in TRAITS.items()}
    )
    signed = snapshot["value"].astype(float) * sign
    qualified = snapshot["qualified"].eq(True)
    keys = [snapshot[column] for column in ("week", "position_group", "trait")]
    out = snapshot[["athlete_id", "week", "position_group", "trait"]].astype({"athlete_id": str})
    return out.assign(z=_scaled(signed, signed.where(qualified), keys))


def standardized_epa(player_values: pd.DataFrame) -> pd.DataFrame:
    """``athlete_id, week, z``: v4 ``shrunk_per_game`` on the scale of rows with a game played."""
    value = player_values["shrunk_per_game"].astype(float)
    keys = [player_values[column] for column in ("week", "position_group")]
    out = player_values[["athlete_id", "week"]].astype({"athlete_id": str})
    return out.assign(z=_scaled(value, value.where(player_values["games"].ge(1)), keys))


def _group_players(snapshot: pd.DataFrame) -> pd.DataFrame:
    """One row per player, week and group with their share of the team's group exposure."""
    players = (
        snapshot[snapshot["position_group"].isin(GROUPS)]
        .astype({"athlete_id": str})
        .drop_duplicates(["week", "athlete_id", "position_group"])[
            ["week", "team", "position_group", "athlete_id", "exposure"]
        ]
    )
    exposure = players["exposure"].astype(float).fillna(0.0)
    total = exposure.groupby([players["week"], players["team"], players["position_group"]]).transform("sum")
    return players.assign(share=(exposure / total.where(total.gt(0))).fillna(0.0))


def team_aggregates(snapshot: pd.DataFrame, player_values: pd.DataFrame) -> pd.DataFrame:
    """``Σ share × z`` per (week, team) and feature; a missing z counts as 0."""
    players = _group_players(snapshot)
    keys = ["athlete_id", "week", "position_group"]
    traits = standardized_traits(snapshot).merge(players, on=keys)
    traits["feature"] = traits["position_group"] + ":" + traits["trait"]
    epa = players.merge(standardized_epa(player_values), on=["athlete_id", "week"], how="left")
    epa = epa.assign(feature=epa["position_group"] + f":{EPA}", z=epa["z"].fillna(0.0))
    rows = pd.concat([traits, epa], ignore_index=True)
    rows["contribution"] = rows["share"] * rows["z"]
    return (
        rows.groupby(["week", "team", "feature"])["contribution"]
        .sum()
        .unstack("feature")
        .reindex(columns=feature_names())
        .fillna(0.0)
    )


def design_matrix(
    snapshots: pd.DataFrame, player_values: pd.DataFrame, games: pd.DataFrame, season: int
) -> pd.DataFrame:
    """One row per completed FBS-versus-FBS game of ``season`` from model week 2.

    ``snapshots`` are trait snapshot rows (``TRAIT_SNAPSHOT_COLUMNS``) plus an
    ``exposure`` column: the player's cumulative exposure at the group through that
    week (see ``exposure_shares``; 1.0 for an OL unit). A game in week ``w`` reads the
    week ``w − 1`` snapshot; a team without rows that week has every aggregate 0, and
    a game where neither team has rows is dropped.
    """
    if "exposure" not in snapshots.columns:
        raise ValueError("design_matrix needs an exposure column on the snapshots")
    if "season" in snapshots.columns:
        snapshots = snapshots[snapshots["season"].eq(season)]
    if "season" in player_values.columns:
        player_values = player_values[player_values["season"].eq(season)]
    played = games[
        games["completed"].eq(True)
        & games["home_classification"].eq("fbs")
        & games["away_classification"].eq("fbs")
        & games["model_week"].ge(2)
    ].reset_index(drop=True)
    aggregates = team_aggregates(snapshots, player_values)
    prior = played["model_week"].astype(int) - 1
    home_keys = pd.MultiIndex.from_arrays([prior, played["home_team"]])
    away_keys = pd.MultiIndex.from_arrays([prior, played["away_team"]])
    known = home_keys.isin(aggregates.index) | away_keys.isin(aggregates.index)
    home = aggregates.reindex(home_keys).fillna(0.0).to_numpy()
    away = aggregates.reindex(away_keys).fillna(0.0).to_numpy()
    out = pd.DataFrame(home - away, columns=aggregates.columns)
    out.insert(0, "game_id", played["game_id"].to_numpy())
    out.insert(1, "season", season)
    out.insert(2, "model_week", played["model_week"].astype(int).to_numpy())
    out.insert(3, "margin", (played["home_points"] - played["away_points"]).to_numpy())
    out.insert(4, "home_field", (~played["neutral_site"].eq(True)).astype(int).to_numpy())
    return out[known].reset_index(drop=True)


def _ridge(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    """Solve (XᵀX + λD)β = Xᵀy where D leaves the leading intercept and home_field free."""
    diagonal = np.full(x.shape[1], penalty)
    diagonal[: len(FIXED)] = 0.0
    return solve(x.T @ x + np.diag(diagonal), x.T @ y, assume_a="sym")


def _leave_one_season_out(
    x: np.ndarray, y: np.ndarray, seasons: np.ndarray, dev_seasons: list[int], penalty: float
) -> tuple[float, np.ndarray]:
    """Pooled held-out MAE and the refit coefficients, one row per held-out season."""
    predictions = np.empty_like(y)
    betas = []
    for season in dev_seasons:
        held = seasons == season
        beta = _ridge(x[~held], y[~held], penalty)
        predictions[held] = x[held] @ beta
        betas.append(beta)
    return float(np.mean(np.abs(y - predictions))), np.array(betas)


def _design(rows: pd.DataFrame, features: list[str], scale: pd.DataFrame) -> np.ndarray:
    z = (rows[features] - scale.loc[features, "mean"]) / scale.loc[features, "sd"]
    return np.column_stack([np.ones(len(rows)), rows["home_field"].to_numpy(float), z.to_numpy(float)])


def _fit(
    rows: pd.DataFrame, features: list[str], scale: pd.DataFrame, dev_seasons: list[int], penalty: float
) -> dict[str, pd.Series]:
    """Full-fit weights with the LOSO spread, its jackknife scaling and the analytic SE.

    ``jackknife_sd = sqrt((k − 1) / k × Σ_i (β_i − β̄)²)`` over the k held-out refits.
    ``analytic_se`` is the root diagonal of ``σ² A⁻¹ XᵀX A⁻¹`` with ``A = XᵀX + λD`` (the
    fit's own penalty, free on intercept and home_field) and ``σ²`` the full fit's
    residual sum of squares over ``n − p``.
    """
    x, y = _design(rows, features, scale), rows["margin"].to_numpy(float)
    _, betas = _leave_one_season_out(x, y, rows["season"].to_numpy(), dev_seasons, penalty)
    beta = _ridge(x, y, penalty)
    k = len(betas)
    jackknife = np.sqrt((k - 1) / k * ((betas - betas.mean(axis=0)) ** 2).sum(axis=0))
    gram = x.T @ x
    diagonal = np.full(x.shape[1], penalty)
    diagonal[: len(FIXED)] = 0.0
    inverse = np.linalg.inv(gram + np.diag(diagonal))
    residual = y - x @ beta
    sigma2 = residual @ residual / (len(y) - x.shape[1])
    analytic = np.sqrt(np.diag(sigma2 * inverse @ gram @ inverse))
    index = [*FIXED, *features]
    return {
        "weights": pd.Series(beta, index=index),
        "loso_sd": pd.Series(betas.std(axis=0, ddof=1), index=index),
        "jackknife_sd": pd.Series(jackknife, index=index),
        "analytic_se": pd.Series(analytic, index=index),
    }


def fit_headline(matrix: pd.DataFrame, dev_seasons: list[int]) -> HeadlineFit:
    """Ridge of ``margin`` on the standardized feature columns of the ``dev_seasons`` rows.

    Feature columns are those whose name holds ``:``. Columns are standardized once
    over the dev rows (an outcome-free scale shared by every refit); a column with no
    spread is dropped up front as ``"constant"``. The penalty minimizes pooled
    leave-one-season-out MAE over ``PENALTIES``. A feature is then dropped as
    ``"insignificant"`` when ``|weight| < 2 × max(jackknife_sd, analytic_se)``, and the
    rest refit once at the same penalty with every spread recomputed.
    """
    dev_seasons = sorted(int(season) for season in set(dev_seasons))
    if len(dev_seasons) < 2:
        raise ValueError("leave-one-season-out needs at least two dev seasons")
    rows = matrix[matrix["season"].isin(dev_seasons)].reset_index(drop=True)
    if set(rows["season"]) != set(dev_seasons):
        raise ValueError(f"matrix lacks dev seasons {sorted(set(dev_seasons) - set(rows['season']))}")
    features = [column for column in rows.columns if ":" in column]
    scale = pd.DataFrame({"mean": rows[features].mean(), "sd": rows[features].std(ddof=0)})
    flat = [name for name in features if not scale.loc[name, "sd"] > 0]
    kept = [name for name in features if name not in flat]

    x, y = _design(rows, kept, scale), rows["margin"].to_numpy(float)
    seasons = rows["season"].to_numpy()
    errors = [_leave_one_season_out(x, y, seasons, dev_seasons, p)[0] for p in PENALTIES]
    penalty = float(PENALTIES[int(np.argmin(errors))])

    first = _fit(rows, kept, scale, dev_seasons, penalty)
    spread = np.maximum(first["jackknife_sd"], first["analytic_se"])
    noise = [name for name in kept if abs(first["weights"][name]) < DROP_MULTIPLE * spread[name]]
    kept = [name for name in kept if name not in noise]
    final = _fit(rows, kept, scale, dev_seasons, penalty)
    return HeadlineFit(
        **final,
        penalty=penalty,
        dropped=flat + noise,
        dropped_reason={**dict.fromkeys(flat, "constant"), **dict.fromkeys(noise, "insignificant")},
        dev_seasons=dev_seasons,
        model_version=MODEL_VERSION,
        feature_scale=scale.loc[kept].rename_axis("feature").reset_index(),
    )


def _reliability_at(table: pd.DataFrame, games: pd.Series) -> np.ndarray:
    """Linear in games from r(0) = 0, flat past the last point, clipped to [0, 1]."""
    if table.empty:
        return np.zeros(len(games))
    table = table.sort_values("games")
    x = np.concatenate([[0.0], table["games"].to_numpy(float)])
    r = np.concatenate([[0.0], table["correlation"].to_numpy(float)])
    return np.clip(np.interp(games.to_numpy(float), x, r), 0.0, 1.0)


def score_players(
    snapshot_week: pd.DataFrame, player_values_week: pd.DataFrame, fit: HeadlineFit
) -> pd.DataFrame:
    """Each player's headline from one snapshot week and the same week's v4 values.

    ``raw_score = Σ w_f × z_f / sd_f`` over the kept features of the player's group:
    a share-1 player's contribution to the fitted margin. Dropped features are not in
    ``feature_scale`` and carry weight 0. ``headline`` subtracts the
    ``REPLACEMENT_PERCENTILE`` quantile of raw among the group's qualified players;
    the percentile ranks qualified players only.
    """
    scale = fit.feature_scale.set_index("feature")
    per_point = fit.weights.reindex(scale.index) / scale["sd"]

    players = (
        snapshot_week.astype({"athlete_id": str})
        .assign(qualified=snapshot_week["qualified"].eq(True))
        .groupby(["athlete_id", "position_group"], as_index=False)
        .agg(games=("games", "max"), qualified=("qualified", "any"))
    )
    traits = standardized_traits(snapshot_week)
    traits["feature"] = traits["position_group"] + ":" + traits["trait"]
    epa = players[["athlete_id", "position_group"]].merge(
        standardized_epa(player_values_week)[["athlete_id", "z"]], on="athlete_id"
    )
    epa["feature"] = epa["position_group"] + f":{EPA}"
    rows = pd.concat([traits, epa], ignore_index=True)
    rows = rows[rows["feature"].isin(per_point.index)]
    points = (rows["z"] * rows["feature"].map(per_point)).groupby(
        [rows["athlete_id"], rows["position_group"]]
    ).sum()
    keys = pd.MultiIndex.from_frame(players[["athlete_id", "position_group"]])
    players["raw_score"] = points.reindex(keys).fillna(0.0).to_numpy()

    qualified_raw = players["raw_score"].where(players["qualified"])
    by_group = qualified_raw.groupby(players["position_group"])
    players["headline"] = players["raw_score"] - by_group.transform(
        lambda raw: raw.quantile(REPLACEMENT_PERCENTILE)
    )
    rank = by_group.rank(method="average")
    n = by_group.transform("count")
    players["headline_percentile"] = (100 * (rank - 1) / (n - 1)).where(n.gt(1), 100.0).where(
        players["qualified"]
    )
    spread = by_group.transform("std")
    reliability = np.empty(len(players))
    for group, index in players.groupby("position_group").groups.items():
        table = fit.reliability[fit.reliability["position_group"].eq(group)]
        reliability[players.index.get_indexer(index)] = _reliability_at(table, players.loc[index, "games"])
    players["uncertainty"] = spread * np.sqrt(1 - reliability)
    players["replaces_value"] = players["position_group"].map(fit.replaces_value).eq(True)
    return players[OUTPUT_COLUMNS]


FIT_NAME = re.compile(r"position_traits_(\d{4})\.json")


def headline_fit_path(cutoff_season: int) -> Path:
    return PROCESSED_DIR / "players" / "models" / f"position_traits_{cutoff_season}.json"


def fit_cutoff(path: Path) -> int:
    """The cutoff season in a ``position_traits_<cutoff>.json`` file name."""
    match = FIT_NAME.fullmatch(Path(path).name)
    if not match:
        raise ValueError(f"{path} is not a position trait fit")
    return int(match.group(1))


def latest_fit_path(models_dir: Path, season: int) -> Path:
    """The fit in ``models_dir`` with the greatest cutoff before ``season``."""
    cutoffs = {
        int(match.group(1)): path
        for path in Path(models_dir).glob("position_traits_*.json")
        if (match := FIT_NAME.fullmatch(path.name))
    }
    earlier = [cutoff for cutoff in cutoffs if cutoff < season]
    if not earlier:
        raise FileNotFoundError(
            f"no position trait headline fit with a cutoff before {season} in {models_dir}; "
            "run position-traits-fit first"
        )
    return cutoffs[max(earlier)]


def _floats(series: pd.Series) -> dict[str, float]:
    return {name: float(value) for name, value in series.items()}


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def save_headline_fit(fit: HeadlineFit, path: Path) -> Path:
    payload = {
        "model_version": fit.model_version,
        "dev_seasons": [int(season) for season in fit.dev_seasons],
        "penalty": float(fit.penalty),
        "weights": _floats(fit.weights),
        "loso_sd": _floats(fit.loso_sd),
        "jackknife_sd": _floats(fit.jackknife_sd),
        "analytic_se": _floats(fit.analytic_se),
        "dropped": list(fit.dropped),
        "dropped_reason": dict(fit.dropped_reason),
        "feature_scale": _records(fit.feature_scale),
        "reliability": _records(fit.reliability),
        "replaces_value": {group: bool(flag) for group, flag in fit.replaces_value.items()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False, indent=1))
    temporary.replace(path)
    return path


def load_headline_fit(path: Path) -> HeadlineFit:
    return headline_fit_from_payload(json.loads(Path(path).read_text()))


def headline_fit_from_payload(payload: dict) -> HeadlineFit:
    return HeadlineFit(
        weights=pd.Series(payload["weights"], dtype=float),
        penalty=float(payload["penalty"]),
        loso_sd=pd.Series(payload["loso_sd"], dtype=float),
        jackknife_sd=pd.Series(payload["jackknife_sd"], dtype=float),
        analytic_se=pd.Series(payload["analytic_se"], dtype=float),
        dropped=list(payload["dropped"]),
        dropped_reason=dict(payload["dropped_reason"]),
        dev_seasons=[int(season) for season in payload["dev_seasons"]],
        model_version=payload["model_version"],
        feature_scale=pd.DataFrame(payload["feature_scale"], columns=["feature", "mean", "sd"]),
        reliability=(
            pd.DataFrame(payload["reliability"])
            if payload["reliability"]
            else pd.DataFrame(columns=RELIABILITY_COLUMNS)
        ),
        replaces_value={group: bool(flag) for group, flag in payload["replaces_value"].items()},
    )


def validate_headline_fit(fit: HeadlineFit, season: int) -> None:
    """Refuse a fit that saw ``season`` or a later one, or that another model version made."""
    if fit.model_version != MODEL_VERSION:
        raise ValueError(f"headline fit is {fit.model_version!r}, expected {MODEL_VERSION!r}")
    if not fit.dev_seasons or max(fit.dev_seasons) >= season:
        raise ValueError(f"headline fit on {fit.dev_seasons} cannot score {season}")
