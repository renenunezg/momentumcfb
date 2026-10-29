"""Validation of the served position grade against the baselines it must beat.

Two endpoints, both Spearman rank correlations among qualified FBS players at a
position group, every predictor measured at the grade's week and the target later:

- ``next_season``: the final-week grade of season t against ``shrunk_per_game`` at
  the final week of season t + 1, among players graded at the final week of both
  seasons at the same group with a value in both.
- ``rest_of_season``: the grade at each of weeks 5 to 9 against the player's
  adjusted EPA credit per game over his later games of that season (cumulative
  ``adjusted_epa`` at the final value week minus at the grade's week, over the games
  between), among graded players with at least ``MIN_LATER_GAMES`` later games. The
  statistic is the mean of the five weekly correlations.

Baselines: the EPA credit the grade reads (``shrunk_per_game`` at the latest value
week at or before the grade's week), games played as the value model counts them,
value above replacement, and for quarterbacks ESPN's season QBR when a table is
supplied. The grade minus EPA credit difference carries a percentile bootstrap
interval that resamples players, every row of a player at once, because the two
correlations share the players.

Rows follow the ``cfb.player_grade_validation`` contract
(``sql/20261027220000_player_grade_validation.sql``). The grade is rescored from the
trait snapshots with the frozen model, so a vintage without grade rows can be
scored, and where the artifact carries the grades it served they must agree with
the rescored ones: the grade evaluated is the grade shown.
"""

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from backend.config import PROCESSED_DIR
from backend.etl import store
from backend.players import pipeline
from backend.players.traits import backtest, grade

log = logging.getLogger(__name__)

NEXT_SEASON = "next_season"
REST_OF_SEASON = "rest_of_season"
EVAL_KINDS = (NEXT_SEASON, REST_OF_SEASON)
REST_OF_SEASON_WEEKS = (5, 6, 7, 8, 9)
MIN_LATER_GAMES = 2
# Players a correlation needs before it counts; a sample floor, not a tuned value.
MIN_PLAYERS = 30
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20261027
# A season's QBR table whose median quarterback has fewer plays than this is a
# partial scrape, not a season.
MIN_QBR_SEASON_PLAYS = 200
HOLDOUT_INSPECTED = "2025 inspected during development"
HOLDOUT_PREREGISTERED = "pre-registered, untouched"
HOLDOUT_STATUSES = (HOLDOUT_INSPECTED, HOLDOUT_PREREGISTERED)
# The 2026 rest-of-season test, registered on 2026-10-27 while local 2026 data ended
# at week 1: the served model file (cfb_position_traits_v1.1, the 2026-10-17 weights
# under their own version string) and the value model that feeds it. A run that
# claims HOLDOUT_PREREGISTERED on any other model or input version is refused.
PREREGISTERED_SEASON = 2026
PREREGISTERED_MODEL_SHA = (
    "3d85cb825df4d23113b51ef05470bf285ac7ea47e13f34d7d160a6591b01d6cd"
)
PREREGISTERED_VALUE_VERSION = "cfb_player_value_v5"
# The version strings a grade model file's weights have been served under. The
# 2026-10-17 weights were served as cfb_position_traits_v1 (file 272d3e51) until
# the 2026-10-26 publish, then re-stamped cfb_position_traits_v1.1 (file 3d85cb82)
# with byte-identical weights; served rows published before the re-stamp read v1.
SERVED_AS = {
    PREREGISTERED_MODEL_SHA: ("cfb_position_traits_v1", "cfb_position_traits_v1.1"),
}
PREDICTORS = ["grade", "epa_credit", "games", "value_above_replacement", "qbr"]
REQUIRED = ["grade", "epa_credit", "games", "value_above_replacement", "target"]
SPEARMAN_COLUMNS = {
    "grade": "spearman_grade",
    "epa_credit": "spearman_epa_credit",
    "games": "spearman_games",
    "value_above_replacement": "spearman_value_above_replacement",
    "qbr": "spearman_qbr",
}
VALIDATION_COLUMNS = [
    "grade_model_version",
    "model_sha",
    "input_value_model_version",
    "eval_kind",
    "seasons",
    "position_group",
    "players",
    *SPEARMAN_COLUMNS.values(),
    "diff_grade_minus_epa",
    "diff_lo_95",
    "diff_hi_95",
    "holdout_status",
    "computed_at",
]
KEY_COLUMNS = [
    "grade_model_version",
    "input_value_model_version",
    "eval_kind",
    "seasons",
    "position_group",
]
PAIR_COLUMNS = ["cell", "athlete_id", "position_group", *PREDICTORS, "target"]
VALUE_COLUMNS = [
    "week",
    "athlete_id",
    "position_group",
    "games",
    "adjusted_epa",
    "shrunk_per_game",
    "value_above_replacement",
    "model_version",
]
ARTIFACT_DIR = ("players", "grade_validation")


# Inputs


def model_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_values(season: int, root: Path = PROCESSED_DIR) -> pd.DataFrame:
    """The season's weekly player values (``VALUE_COLUMNS``) under ``root``."""
    path = Path(root).joinpath(*pipeline.player_values_artifact(season))
    return pd.read_parquet(path, columns=VALUE_COLUMNS).astype({"athlete_id": str})


def value_version(values: pd.DataFrame) -> str:
    versions = sorted(values["model_version"].dropna().unique())
    if len(versions) != 1:
        raise ValueError(f"player values carry {versions}, expected one version")
    return str(versions[0])


def read_snapshot(season: int, root: Path = PROCESSED_DIR) -> pd.DataFrame:
    """The season's weekly trait snapshot under ``root``, ready for scoring.

    A served artifact also holds grade, war and ``epa.credit`` rows and an ``fbs``
    flag; a backtest snapshot holds trait rows alone. Only trait rows are kept, and
    the flag is added from the season's teams file where it is missing, the way the
    weekly runner adds it.
    """
    path = Path(root).joinpath(*pipeline.position_traits_artifact(season))
    snapshot = pd.read_parquet(path).astype({"athlete_id": str})
    traits = snapshot[~snapshot["trait"].isin(["grade", "war"])]
    if "fbs" not in traits.columns:
        traits = backtest.with_fbs(traits, season)
    return traits


def served_grades(season: int, root: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Grade rows the artifact carries (empty for a backtest snapshot)."""
    path = Path(root).joinpath(*pipeline.position_traits_artifact(season))
    rows = pd.read_parquet(
        path, columns=["week", "athlete_id", "position_group", "trait", "value"]
    )
    rows = rows[rows["trait"].eq("grade")].astype({"athlete_id": str})
    return rows.rename(columns={"value": "grade"})[
        ["week", "athlete_id", "position_group", "grade"]
    ]


def load_qbr(path: Path) -> pd.DataFrame:
    """``season, athlete_id, qbr`` from an nflverse college QBR season table.

    ESPN player ids are the athlete ids the trait pipeline uses. A season whose
    median quarterback has under ``MIN_QBR_SEASON_PLAYS`` plays is a partial scrape
    and is refused rather than compared with a full season of credit.
    """
    table = pd.read_csv(path, usecols=["season", "player_id", "qbr_total", "qb_plays"])
    plays = table.groupby("season")["qb_plays"].median()
    partial = plays[plays.lt(MIN_QBR_SEASON_PLAYS)]
    if not partial.empty:
        raise ValueError(
            f"QBR seasons {sorted(partial.index)} are partial scrapes (median plays "
            f"{partial.round().to_dict()}); drop them from {path}"
        )
    return table.assign(athlete_id=table["player_id"].astype(str)).rename(
        columns={"qbr_total": "qbr"}
    )[["season", "athlete_id", "qbr"]]


# Scoring


def rescored_grades(
    snapshot: pd.DataFrame,
    values: pd.DataFrame,
    model: grade.GradeModel,
    weeks: list[int],
) -> pd.DataFrame:
    """``week, athlete_id, position_group, grade`` from the frozen model at ``weeks``."""
    scored = grade.with_epa_credit(snapshot, values)
    parts = []
    for week in weeks:
        week_rows = scored[scored["week"].eq(week)]
        if week_rows.empty:
            continue
        graded = grade.score_grades(week_rows, model)
        parts.append(graded.assign(week=week))
    columns = ["week", "athlete_id", "position_group", "grade"]
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)[columns]


def check_served_grades(served: pd.DataFrame, rescored: pd.DataFrame) -> None:
    """Refuse an evaluation whose rescored grades differ from the grades served."""
    if served.empty:
        return
    keys = ["week", "athlete_id", "position_group"]
    both = rescored.merge(served, on=keys, suffixes=("", "_served"))
    if len(both) != len(rescored):
        raise ValueError(
            f"{len(rescored) - len(both)} rescored grades have no served row"
        )
    gap = (both["grade"] - both["grade_served"]).abs()
    if gap.max() > 1e-6:
        raise ValueError(
            f"{int(gap.gt(1e-6).sum())} served grades differ from the rescored ones "
            f"(largest gap {gap.max():.4f}); the artifact was scored with other inputs"
        )


def _at_week(values: pd.DataFrame, week: int) -> pd.DataFrame:
    """Each player's value row at the latest value week at or before ``week``."""
    rows = values[values["week"].le(week)].sort_values("week", kind="stable")
    return rows.drop_duplicates("athlete_id", keep="last").set_index("athlete_id")


def _with_predictors(grades: pd.DataFrame, at: pd.DataFrame) -> pd.DataFrame:
    out = grades.copy()
    out["epa_credit"] = out["athlete_id"].map(at["shrunk_per_game"])
    out["games"] = out["athlete_id"].map(at["games"])
    out["value_above_replacement"] = out["athlete_id"].map(
        at["value_above_replacement"]
    )
    return out


def next_season_pairs(
    snapshot: pd.DataFrame,
    values: pd.DataFrame,
    next_snapshot: pd.DataFrame,
    next_values: pd.DataFrame,
    model: grade.GradeModel,
    qbr: pd.Series | None = None,
) -> pd.DataFrame:
    """One row per player graded at the final week of both seasons at the same
    group (``PAIR_COLUMNS``): season-t predictors and the next season's final-week
    ``shrunk_per_game`` as ``target``. ``qbr`` maps athlete id to season-t QBR."""
    week, next_week = int(snapshot["week"].max()), int(next_snapshot["week"].max())
    now = rescored_grades(snapshot, values, model, [week])
    nxt = rescored_grades(next_snapshot, next_values, model, [next_week])
    both = now.merge(
        nxt[["athlete_id", "position_group"]], on=["athlete_id", "position_group"]
    )
    both = _with_predictors(both, _at_week(values, week))
    both["target"] = both["athlete_id"].map(
        _at_week(next_values, next_week)["shrunk_per_game"]
    )
    both["qbr"] = both["athlete_id"].map(qbr) if qbr is not None else np.nan
    both["cell"] = week
    return both.dropna(subset=REQUIRED).reset_index(drop=True)[PAIR_COLUMNS]


def rest_of_season_pairs(
    snapshot: pd.DataFrame,
    values: pd.DataFrame,
    model: grade.GradeModel,
    weeks: tuple[int, ...] = REST_OF_SEASON_WEEKS,
) -> pd.DataFrame:
    """One row per graded player and week in ``weeks`` (``PAIR_COLUMNS``): week-w
    predictors and his adjusted EPA credit per game over his later games as
    ``target``, where he has at least ``MIN_LATER_GAMES`` of them. ``cell`` is the
    week."""
    grades = rescored_grades(snapshot, values, model, list(weeks))
    final = _at_week(values, int(values["week"].max()))
    parts = []
    for week, rows in grades.groupby("week"):
        at = _at_week(values, int(week))
        rows = _with_predictors(rows, at)
        later_games = rows["athlete_id"].map(final["games"]) - rows["games"]
        later_epa = rows["athlete_id"].map(final["adjusted_epa"]) - rows[
            "athlete_id"
        ].map(at["adjusted_epa"])
        rows["target"] = (later_epa / later_games).where(
            later_games.ge(MIN_LATER_GAMES)
        )
        parts.append(rows.assign(cell=int(week), qbr=np.nan))
    if not parts:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    return out.dropna(subset=REQUIRED).reset_index(drop=True)[PAIR_COLUMNS]


# Statistics


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < MIN_PLAYERS:
        return np.nan
    a, b = rankdata(left), rankdata(right)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _cells(frame: pd.DataFrame, predictors: list[str]) -> list[dict]:
    """Per cell: the predictor arrays, the target and each row's athlete code."""
    codes = pd.factorize(frame["athlete_id"])[0]
    cells = []
    for _, index in frame.groupby("cell").indices.items():
        cells.append(
            {
                "y": frame["target"].to_numpy()[index],
                "athlete": codes[index],
                **{name: frame[name].to_numpy()[index] for name in predictors},
            }
        )
    return cells


def mean_spearman(cells: list[dict], predictor: str) -> float:
    """The mean over cells with at least ``MIN_PLAYERS`` players of the Spearman of
    ``predictor`` with the target; NaN when no cell qualifies."""
    rhos = [spearman(cell[predictor], cell["y"]) for cell in cells]
    rhos = [rho for rho in rhos if not np.isnan(rho)]
    return float(np.mean(rhos)) if rhos else np.nan


def bootstrap_difference(
    frame: pd.DataFrame,
    left: str = "grade",
    right: str = "epa_credit",
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """The 2.5th and 97.5th percentiles of ``mean_spearman(left) −
    mean_spearman(right)`` over ``draws`` resamples of players with replacement,
    every cell row of a resampled player entering as many times as he was drawn."""
    cells = _cells(frame, [left, right])
    athletes = int(frame["athlete_id"].nunique())
    rng = np.random.default_rng(seed)
    diffs = np.empty(draws)
    for draw in range(draws):
        counts = np.bincount(rng.integers(0, athletes, athletes), minlength=athletes)
        resampled = []
        for cell in cells:
            rows = np.repeat(np.arange(len(cell["y"])), counts[cell["athlete"]])
            resampled.append(
                {
                    "y": cell["y"][rows],
                    left: cell[left][rows],
                    right: cell[right][rows],
                }
            )
        diffs[draw] = mean_spearman(resampled, left) - mean_spearman(resampled, right)
    diffs = diffs[~np.isnan(diffs)]
    if not len(diffs):
        return np.nan, np.nan
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def validation_table(
    pairs: pd.DataFrame,
    eval_kind: str,
    seasons: str,
    grade_model_version: str,
    model_sha: str,
    input_value_model_version: str,
    holdout_status: str,
    computed_at: datetime,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """One ``VALIDATION_COLUMNS`` row per group with at least ``MIN_PLAYERS`` players.

    ``spearman_qbr`` is filled only where every row of the group has a QBR, so the
    QBR column never describes a different population from the other columns.
    """
    if eval_kind not in EVAL_KINDS:
        raise ValueError(f"eval_kind {eval_kind!r} is not one of {EVAL_KINDS}")
    rows = []
    for group, frame in pairs.groupby("position_group"):
        players = int(frame["athlete_id"].nunique())
        if players < MIN_PLAYERS:
            continue
        predictors = [
            name for name in PREDICTORS if name != "qbr" or frame["qbr"].notna().all()
        ]
        cells = _cells(frame, predictors)
        stats = {
            SPEARMAN_COLUMNS[name]: mean_spearman(cells, name)
            if name in predictors
            else np.nan
            for name in PREDICTORS
        }
        lo, hi = bootstrap_difference(frame, draws=draws, seed=seed)
        rows.append(
            {
                "grade_model_version": grade_model_version,
                "model_sha": model_sha,
                "input_value_model_version": input_value_model_version,
                "eval_kind": eval_kind,
                "seasons": seasons,
                "position_group": group,
                "players": players,
                **stats,
                "diff_grade_minus_epa": stats["spearman_grade"]
                - stats["spearman_epa_credit"],
                "diff_lo_95": lo,
                "diff_hi_95": hi,
                "holdout_status": holdout_status,
                "computed_at": computed_at.isoformat(),
            }
        )
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


# The command


def seasons_label(seasons: list[int]) -> str:
    return "-".join(str(season) for season in seasons)


def run_validation(
    eval_kind: str,
    seasons: list[int],
    holdout_status: str,
    qbr_path: Path | None = None,
    model_path: Path | None = None,
    root: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Evaluate the frozen grade model on local artifacts and write the rows.

    ``next_season`` takes exactly two consecutive seasons; ``rest_of_season`` any
    seasons, pooled into one statistic per group with (season, week) cells. The
    model defaults to the latest grade model before the last season and must not
    have seen it. Reads only parquet under ``root`` and the QBR table; no network.
    With ``qbr_path`` the quarterback population is the graded quarterbacks with a
    QBR, logged, so every QB column shares one population.
    """
    if holdout_status not in HOLDOUT_STATUSES:
        raise ValueError(
            f"holdout status {holdout_status!r} is not one of {HOLDOUT_STATUSES}"
        )
    seasons = sorted(int(season) for season in seasons)
    if eval_kind == NEXT_SEASON and (len(seasons) != 2 or seasons[1] != seasons[0] + 1):
        raise ValueError(f"{NEXT_SEASON} needs two consecutive seasons, got {seasons}")
    if eval_kind not in EVAL_KINDS:
        raise ValueError(f"eval_kind {eval_kind!r} is not one of {EVAL_KINDS}")
    models = Path(root) / "players" / "models"
    model_path = model_path or grade.latest_grade_model_path(models, max(seasons))
    model = grade.load_grade_model(model_path)
    grade.validate_grade_model(model, max(seasons))
    qbr = load_qbr(qbr_path) if qbr_path else None

    data = {}
    for season in seasons:
        snapshot, values = read_snapshot(season, root), read_values(season, root)
        weeks = (
            [int(snapshot["week"].max())]
            if eval_kind == NEXT_SEASON
            else list(REST_OF_SEASON_WEEKS)
        )
        check_served_grades(
            served_grades(season, root),
            rescored_grades(snapshot, values, model, weeks),
        )
        data[season] = (snapshot, values)
    versions = {value_version(values) for _, values in data.values()}
    if len(versions) != 1:
        raise ValueError(f"seasons carry different value versions {sorted(versions)}")
    if holdout_status == HOLDOUT_PREREGISTERED:
        frozen = (PREREGISTERED_MODEL_SHA, {PREREGISTERED_VALUE_VERSION})
        if (model_sha256(model_path), versions) != frozen or (
            PREREGISTERED_SEASON not in seasons
        ):
            raise ValueError(
                f"{HOLDOUT_PREREGISTERED!r} covers model {PREREGISTERED_MODEL_SHA[:8]} "
                f"on {PREREGISTERED_VALUE_VERSION} in {PREREGISTERED_SEASON}; got "
                f"{model_sha256(model_path)[:8]} on {sorted(versions)} in {seasons}"
            )

    if eval_kind == NEXT_SEASON:
        first, later = seasons
        season_qbr = None
        if qbr is not None:
            season_qbr = qbr[qbr["season"].eq(first)].set_index("athlete_id")["qbr"]
        pairs = next_season_pairs(*data[first], *data[later], model, season_qbr)
    else:
        pairs = pd.concat(
            [
                rest_of_season_pairs(snapshot, values, model).assign(
                    cell=lambda frame, season=season: season * 100 + frame["cell"]
                )
                for season, (snapshot, values) in data.items()
            ],
            ignore_index=True,
        )
    if qbr is not None:
        quarterbacks = pairs["position_group"].eq("QB")
        without = quarterbacks & pairs["qbr"].isna()
        log.info(
            "grade validation: %d of %d quarterback rows have a QBR; the rest leave",
            int((quarterbacks & ~without).sum()),
            int(quarterbacks.sum()),
        )
        pairs = pairs[~without]

    rows = validation_table(
        pairs,
        eval_kind,
        seasons_label(seasons),
        model.model_version,
        model_sha256(model_path),
        versions.pop(),
        holdout_status,
        datetime.now(timezone.utc),
    )
    out = Path(root).joinpath(
        *ARTIFACT_DIR, f"{eval_kind}_{seasons_label(seasons)}.parquet"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out, index=False)
    log.info("grade validation rows written to %s", out)
    return rows


def with_served_versions(rows: pd.DataFrame) -> pd.DataFrame:
    """Name every version string the evaluated weights were served under, so a
    served row stamped with an earlier string can be matched to its evaluation."""
    return rows.assign(
        served_model_versions=[
            ", ".join(SERVED_AS.get(sha, (version,)))
            for sha, version in zip(rows["model_sha"], rows["grade_model_version"])
        ]
    )


def read_validation_artifacts() -> pd.DataFrame:
    """Every validation row written locally, for the publish step."""
    names = store.processed_names(*ARTIFACT_DIR)
    if not names:
        return pd.DataFrame(columns=VALIDATION_COLUMNS)
    return pd.concat(
        [store.read_processed(*ARTIFACT_DIR, f"{name}.parquet") for name in names],
        ignore_index=True,
    )[VALIDATION_COLUMNS]
