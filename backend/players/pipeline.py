"""Season runners that turn raw player sources into published artifacts."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import PROCESSED_DIR, RAW_DIR
from backend.etl import store
from backend.players import artifacts, heisman, value
from backend.players.credit import DEFENSE_ROLES, OFFENSE_SHARES
from backend.players.ingest import players_dir, read_season_source
from backend.players.traits import MODEL_VERSION, backtest, expectations, grade, war
from backend.players.traits.snapshots import TRAIT_SNAPSHOT_COLUMNS

log = logging.getLogger(__name__)

META_COLUMNS = [
    "season",
    "as_of",
    "value_model_version",
    "heisman_model_version",
    "credit_shares",
    "prior_games",
    "replacement_percentile",
    "qualifying_games",
    "fcs_opponent_weight",
    "opponent_prior_regression",
    "opponent_prior_share",
    "reliability",
    "heisman_training_seasons",
    "heisman_winner_hit_rate",
    "heisman_top_three_rate",
    "heisman_coefficients",
    "heisman_evaluation_kind",
    "heisman_evaluation_week",
    "heisman_evaluation_seasons",
    "heisman_winner_pool_coverage",
    "heisman_ballot_share_covered",
]


def player_values_artifact(season: int) -> tuple[str, ...]:
    return ("players", "player_values", f"{season}.parquet")


def position_traits_artifact(season: int) -> tuple[str, ...]:
    return ("players", "position_traits", f"{season}.parquet")


def heisman_board_artifact(season: int, week: int) -> tuple[str, ...]:
    return ("players", "heisman_board", f"{season}_{week:02d}.parquet")


POSITION_TRAIT_COLUMNS = [
    *TRAIT_SNAPSHOT_COLUMNS,
    "uncertainty",
    "in_grade",
    "points_above_replacement",
    "fbs",
    "poor_coverage",
]
HISTORY_ARTIFACT = ("players", "heisman_history.parquet")
META_ARTIFACT = ("players", "player_model_meta.parquet")


def run_player_values(season: int, as_of: datetime) -> pd.DataFrame:
    outputs = value.build_player_values(season, as_of)
    store.write_processed(outputs["player_values"], *player_values_artifact(season))
    store.write_processed(
        outputs["unit_effects"], "players", "unit_effects", f"{season}.parquet"
    )
    store.write_processed(
        outputs["reliability"], "players", "reliability", f"{season}.parquet"
    )
    return outputs["player_values"]


def frozen_expectation_tables(model_path: Path) -> expectations.ExpectationTables:
    """The serving expectation tables saved with the grade model at ``model_path``."""
    path = expectations.expectation_tables_path(
        model_path.parent, grade.grade_cutoff(model_path)
    )
    if not path.exists():
        raise FileNotFoundError(
            f"no frozen expectation tables {path.name} beside {model_path.name} in "
            f"{model_path.parent}; run position-traits-fit and restore both artifacts"
        )
    return expectations.load_expectation_tables(path)


def latest_grade_model_path(season: int) -> Path:
    """The frozen grade model with the greatest cutoff before ``season``."""
    return grade.latest_grade_model_path(
        store.PROCESSED_DIR / "players" / "models", season
    )


def grade_rows(snapshot: pd.DataFrame, model: grade.GradeModel) -> pd.DataFrame:
    """One ``trait = "grade"`` snapshot-shaped row per qualified FBS player-week."""
    weeks = sorted(snapshot["week"].unique())
    scored = pd.concat(
        [
            grade.score_grades(snapshot[snapshot["week"].eq(week)], model).assign(
                week=week
            )
            for week in weeks
        ],
        ignore_index=True,
    )
    keys = ["week", "athlete_id", "position_group"]
    identity = snapshot.drop_duplicates(keys)[
        [*keys, "season", "athlete_name", "team", "team_id"]
    ]
    rows = scored.merge(identity, on=keys)
    return rows.assign(
        trait="grade",
        value=rows["grade"],
        percentile=rows["grade_percentile"],
        kind="graded",
        qualified=True,
    )


def war_rows(
    snapshot: pd.DataFrame, values: pd.DataFrame, war_frame: pd.DataFrame
) -> pd.DataFrame:
    """One ``trait = "war"`` row per qualified FBS player-week with a v4 value.

    A snapshot week reads the latest v4 week at or before it (a week without games
    has no v4 row of its own). The row takes the player's v4 position group and
    games, ``value`` is his wins above replacement, ``points_above_replacement`` his
    v4 points, and ``percentile`` ranks wins among the week's rows at the group.
    """
    flags = (
        snapshot.assign(pool=snapshot["qualified"].eq(True) & snapshot["fbs"].eq(True))
        .groupby(["week", "athlete_id"])["pool"]
        .any()
    )
    pool = flags[flags].reset_index()[["week", "athlete_id"]]
    v4_weeks = np.array(sorted(values["week"].astype(int).unique()))
    if pool.empty or not len(v4_weeks):
        return pd.DataFrame(
            columns=[*TRAIT_SNAPSHOT_COLUMNS, "points_above_replacement"]
        )
    position = (
        np.searchsorted(v4_weeks, pool["week"].astype(int).to_numpy(), side="right") - 1
    )
    pool = pool[position >= 0].assign(value_week=v4_weeks[position[position >= 0]])
    measured = (
        values[["week", "athlete_id", "position_group", "games"]]
        .astype({"athlete_id": str})
        .merge(war_frame.astype({"athlete_id": str}), on=["athlete_id", "week"])
    )
    rows = pool.merge(
        measured.rename(columns={"week": "value_week"}), on=["athlete_id", "value_week"]
    ).dropna(subset=["wins_above_replacement"])
    identity = snapshot.drop_duplicates(["week", "athlete_id"])[
        ["week", "athlete_id", "season", "athlete_name", "team", "team_id"]
    ]
    rows = rows.merge(identity, on=["week", "athlete_id"])
    by_group = rows.groupby(["week", "position_group"])["wins_above_replacement"]
    rank = by_group.rank(method="average")
    n = by_group.transform("size")
    return rows.assign(
        trait="war",
        value=rows["wins_above_replacement"],
        percentile=(100 * (rank - 1) / (n - 1)).where(n.gt(1), 100.0),
        kind="measured",
        qualified=True,
    )


def score_snapshot(
    snapshot: pd.DataFrame,
    model: grade.GradeModel,
    values: pd.DataFrame,
    as_of: datetime,
    poor: bool,
    fbs_schools: set[str],
) -> pd.DataFrame:
    """The scored artifact for one season's weekly trait snapshots (``POSITION_TRAIT_COLUMNS``).

    ``snapshot`` carries ``fbs`` (from ``fbs_schools``); ``values`` are the season's v4
    values, which also give WAR.
    """
    snapshot = grade.with_epa_credit(snapshot, values)
    scored = grade_rows(snapshot, model)
    measured = war_rows(snapshot, values, war.wins_above_replacement(values))
    weighted = model.weights[model.weights["weight"].ne(0)]
    weighted_keys = set(zip(weighted["position_group"], weighted["trait"]))
    traits = snapshot[TRAIT_SNAPSHOT_COLUMNS].assign(uncertainty=np.nan)
    traits["in_grade"] = [
        (group, trait) in weighted_keys
        for group, trait in zip(traits["position_group"], traits["trait"])
    ]
    scored = scored.assign(in_grade=True, as_of=None, model_version=None)
    measured = measured.assign(
        in_grade=False, uncertainty=np.nan, as_of=None, model_version=None
    )
    columns = [*TRAIT_SNAPSHOT_COLUMNS, "uncertainty", "in_grade"]
    out = pd.concat(
        [
            traits.assign(points_above_replacement=np.nan),
            scored[columns].assign(points_above_replacement=np.nan),
            measured[[*columns, "points_above_replacement"]],
        ],
        ignore_index=True,
    )
    return out.assign(
        as_of=as_of.isoformat(),
        model_version=MODEL_VERSION,
        fbs=out["team"].isin(fbs_schools),
        poor_coverage=poor,
    )[POSITION_TRAIT_COLUMNS]


def run_position_traits(season: int, as_of: datetime) -> pd.DataFrame:
    """Score the season's weekly trait snapshots with the latest earlier grade model.

    The model and its expectation tables are frozen: they are only loaded, never
    refit, and a model whose development seasons include ``season`` is refused. The
    season decides its own receptions-or-targets mode from its own target coverage;
    every row records it in ``poor_coverage``, which the table does not publish. Each
    row carries ``fbs`` from the team's classification; grade and war rows exist for
    qualified FBS players only, and only FBS rows are published. ``in_grade`` marks
    trait rows whose trait has positive weight at the group, and every grade row.
    War rows carry v4's wins above replacement (``war.wins_above_replacement``) and
    ``points_above_replacement``.
    """
    model_path = latest_grade_model_path(season)
    model = grade.load_grade_model(model_path)
    grade.validate_grade_model(model, season)
    tables = frozen_expectation_tables(model_path)
    teams = read_season_source(season, "teams")
    fbs_schools = set(teams.loc[teams["classification"].eq("fbs"), "school"])
    built = backtest.build_season_traits(season, tables=tables)
    coverage = built["target_coverage"]
    poor = backtest.poor_target_coverage(coverage)
    snapshot = built["snapshots"].astype({"athlete_id": str})
    # The grade pool is qualified FBS players, so the flag precedes scoring.
    snapshot = snapshot.assign(fbs=snapshot["team"].isin(fbs_schools))
    out = score_snapshot(
        snapshot,
        model,
        backtest.player_values(season),
        as_of,
        poor,
        fbs_schools,
    )
    store.write_processed(out, *position_traits_artifact(season))
    graded = sorted(grade.graded_groups(model))
    latest = snapshot["week"].max()
    losses = grade.coverage_losses(
        grade.grade_eligibility(snapshot[snapshot["week"].eq(latest)], model)
    )
    for row in losses.itertuples():
        log.info(
            f"position grade {season} week {latest} {row.position_group}: "
            f"{row.graded} of {row.qualified} qualified FBS players graded; "
            f"{row.no_primary} lack the primary trait, {row.low_coverage} cover under "
            f"{grade.MIN_WEIGHT_COVERAGE} of the weight"
        )
    scored = out[out["trait"].eq("grade")]
    log.info(
        f"position traits {season}: {scored['athlete_id'].nunique()} qualified FBS players "
        f"graded over {scored['week'].nunique()} weeks with the model through "
        f"{max(model.dev_seasons)}; graded groups {', '.join(graded) or 'none'}; "
        f"{int(out['trait'].eq('war').sum())} war rows; "
        f"{'receptions' if poor else 'targets'} mode, uncovered incompletions "
        f"{backtest.uncovered_share(coverage):.3f}"
    )
    return out


def _seed_seasons(seed: pd.DataFrame, before: int) -> list[int]:
    return sorted(
        int(season)
        for season in seed["season"].unique()
        if season < before and players_dir(int(season)).is_dir()
    )


def train_heisman_model(
    season: int, as_of: datetime
) -> tuple[heisman.ShareModel, list[int], pd.DataFrame]:
    """Explicit offline training from cached, complete historical sources."""
    seed = heisman.load_seed()
    expected = sorted(int(year) for year in seed["season"].unique() if year < season)
    training_seasons = _seed_seasons(seed, season)
    missing = sorted(set(expected) - set(training_seasons))
    if missing or len(training_seasons) < 2:
        raise ValueError(
            f"Heisman training requires cached historical player seasons; missing {missing}. "
            "Restore the historical inputs or a trained model artifact before inference."
        )
    rows = heisman.training_rows(training_seasons, seed)
    model = heisman.fit_share_model(rows)
    weekly_history = heisman.chronological_weekly_evaluation(rows, seed)
    # Value ranks must use the same weekly cutoff, never the final season rank.
    for year, indexes in weekly_history.groupby("season").groups.items():
        try:
            values = store.read_processed(*player_values_artifact(int(year)))
        except FileNotFoundError:
            continue
        for index in indexes:
            record = weekly_history.loc[index]
            snapshot = values[values["week"].eq(record["week"])]
            winner, _ = heisman._match_seed(
                seed[seed["player"].eq(record["actual_winner"])],
                snapshot,
                int(year),
                strict=False,
            )
            hit = snapshot[snapshot["athlete_id"].isin(winner["athlete_id"])]
            if not hit.empty:
                weekly_history.loc[index, "winner_value_rank"] = int(
                    hit["overall_rank"].iloc[0]
                )
    artifacts.save_heisman_model(
        model,
        training_seasons,
        weekly_history,
        cutoff_season=season,
        created_at=as_of,
        source_provenance={
            str(year): {
                "manifest_sha256": hashlib.sha256(
                    (players_dir(year) / "manifest.parquet").read_bytes()
                ).hexdigest(),
                "games_sha256": hashlib.sha256(
                    (RAW_DIR / "games" / f"{year}.parquet").read_bytes()
                ).hexdigest(),
            }
            for year in training_seasons
        },
    )
    return model, training_seasons, weekly_history


def run_heisman(season: int, week: int | None, as_of: datetime) -> pd.DataFrame:
    """Build a board using a portable model trained only on earlier seasons."""
    model, training_seasons, weekly_history = artifacts.load_heisman_model(season)
    try:
        player_values = store.read_processed(*player_values_artifact(season))
    except FileNotFoundError:
        player_values = None
    if week is None:
        if player_values is None or player_values.empty:
            raise ValueError(f"no player values for {season}; run player-values first")
        week = int(player_values["week"].max())
    # The serving table has one row per season. Match the current forecast
    # week exactly; unavailable weeks remain absent rather than using later data.
    history = weekly_history[weekly_history["week"].eq(week)].copy()
    store.write_processed(history[heisman.HISTORY_COLUMNS], *HISTORY_ARTIFACT)
    board = heisman.build_board(model, season, week, player_values, as_of)
    store.write_processed(board, *heisman_board_artifact(season, week))

    reliability = store.read_processed("players", "reliability", f"{season}.parquet")
    meta = pd.DataFrame(
        [
            {
                "season": season,
                "as_of": as_of.isoformat(),
                "value_model_version": value.MODEL_VERSION,
                "heisman_model_version": heisman.MODEL_VERSION,
                "credit_shares": json.dumps(
                    {
                        "offense": {
                            f"{kind}:{stat}": {"role": role, "share": share}
                            for (kind, stat), (role, share) in OFFENSE_SHARES.items()
                        },
                        "defense": DEFENSE_ROLES,
                    }
                ),
                "prior_games": value.PRIOR_GAMES,
                "replacement_percentile": value.REPLACEMENT_PERCENTILE,
                "qualifying_games": value.QUALIFYING_GAMES,
                "fcs_opponent_weight": value.FCS_OPPONENT_WEIGHT,
                "opponent_prior_regression": value.OPPONENT_PRIOR_REGRESSION,
                "opponent_prior_share": value.OPPONENT_PRIOR_SHARE,
                "reliability": reliability.to_json(orient="records"),
                "heisman_training_seasons": json.dumps(training_seasons),
                "heisman_winner_hit_rate": float(history["winner_hit"].mean()),
                "heisman_top_three_rate": float(history["top_three_hit"].mean()),
                "heisman_coefficients": model.coefficients().to_json(orient="records"),
                "heisman_evaluation_kind": heisman.EVALUATION_KIND,
                "heisman_evaluation_week": week,
                "heisman_evaluation_seasons": len(history),
                "heisman_winner_pool_coverage": float(history["winner_in_pool"].mean()),
                "heisman_ballot_share_covered": float(
                    history["ballot_share_covered"].mean()
                ),
            }
        ],
        columns=META_COLUMNS,
    )
    store.write_processed(meta, *META_ARTIFACT)
    log.info(
        f"heisman {season} week {week}: {len(board)} candidates; "
        f"chronological week-{week} winner hit rate "
        f"{history['winner_hit'].mean():.2f} over {len(history)} seasons"
    )
    return board


def read_player_artifacts(season: int) -> dict[str, pd.DataFrame]:
    values = store.read_processed(*player_values_artifact(season))
    boards = [
        store.read_processed("players", "heisman_board", f"{name}.parquet")
        for name in store.processed_names("players", "heisman_board")
        if name.startswith(f"{season}_")
    ]
    board = pd.concat(boards, ignore_index=True) if boards else pd.DataFrame()
    history = pd.read_parquet(PROCESSED_DIR.joinpath(*HISTORY_ARTIFACT))
    meta = pd.read_parquet(PROCESSED_DIR.joinpath(*META_ARTIFACT))
    return {
        "player_values": values,
        "heisman_board": board,
        "heisman_history": history,
        "player_model_meta": meta,
        "position_traits": _latest_trait_week(season),
    }


def _latest_trait_week(season: int) -> pd.DataFrame:
    """The newest weekly trait snapshot, FBS players only; the artifact keeps every week.

    An artifact without the ``fbs`` flag predates the filter and is not published.
    """
    if not PROCESSED_DIR.joinpath(*position_traits_artifact(season)).exists():
        return pd.DataFrame(columns=POSITION_TRAIT_COLUMNS)
    traits = store.read_processed(*position_traits_artifact(season))
    if "fbs" not in traits.columns:
        log.warning(
            f"position traits {season}: artifact has no fbs flag, not published"
        )
        return pd.DataFrame(columns=POSITION_TRAIT_COLUMNS)
    latest = traits["week"].eq(traits["week"].max()) & traits["fbs"].eq(True)
    return traits[latest].reset_index(drop=True)
