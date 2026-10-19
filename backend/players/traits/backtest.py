"""Season assembly and the offline backtest of the position grade.

``build_season_traits`` runs every trait builder for one season and writes the
weekly snapshots, their split-half reliability and the target coverage.
``run_backtest`` sets the grade's trait weights from player-level evidence on
chronological development seasons (pooled split-half reliability times transfer
portability), measures the composite's own reliability, and checks the grade on
a later validation season it never saw: does a player's grade predict his next
season's grade and production better than his EPA credit does, and how does it
rank against the draft. ``write_backtest`` saves the grade model, the frozen
expectation tables the weekly runner scores with, and one long evaluation frame,
and prints every table as Markdown.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import PROCESSED_DIR, RAW_DIR
from backend.etl import store
from backend.features.possessions import classify_plays
from backend.players import pipeline
from backend.players.ingest import read_season_source, read_weekly
from backend.players.traits import EPA_TRAIT, all_traits, grade, snapshots, war
from backend.players.traits.context import load_season_context
from backend.players.traits.defense import defense_traits
from backend.players.traits.expectations import (
    ExpectationTables,
    expectation_tables_path,
    fit_expectation_tables,
    save_expectation_tables,
)
from backend.players.traits.line import line_traits
from backend.players.traits.offense import (
    POOR_TARGET_COVERAGE,
    qb_traits,
    rb_traits,
    receiver_traits,
)
from backend.players.traits.punting import punting_traits
from backend.players.traits.targets import recover_targets, target_coverage
from backend.players.traits.transfers import (
    TRANSFER_COLUMNS,
    match_transfers,
    portability,
)
from backend.players.value import MIN_QUALIFIED_PLAYERS

log = logging.getLogger(__name__)

FIRST_PLAY_SEASON = 2019
FIRST_PORTAL_SEASON = 2021
RELIABILITY_GAMES = (4, 6, 8, 12)
VALUE_COLUMNS = [
    "week",
    "athlete_id",
    "position_group",
    "games",
    "shrunk_per_game",
    "adjusted_per_game",
    "replacement_per_game",
    "value_above_replacement",  # the draft check and wins above replacement
]
CREDITED_GAME_COLUMNS = ["athlete_id", "game_id", "model_week", "adjusted_epa"]
IDENTITY_COLUMNS = ["athlete_id", "athlete_name", "team"]
SPLIT_COLUMNS = ["position_group", "games", "players", "correlation"]
NEXT_SEASON_COLUMNS = [
    "season",
    "next_season",
    "position_group",
    "players",
    "spearman_grade_next_grade",
    "spearman_grade_next_epa",
    "spearman_epa_next_epa",
    "holdout",
]
DRAFT_COLUMNS = [
    "position_group",
    "players",
    "spearman_pick_v4",
    "spearman_pick_grade",
    "spearman_nfl_games_v4",
    "spearman_nfl_games_grade",
]
DRAFT_COHORT = PROCESSED_DIR / "draft" / "2027" / "historical_cohort.parquet"
MATURE_OUTCOME = "observed"
# Artifact name per season output; the first three are the brief's, the last two
# let a cached season feed the pooled reliability without rebuilding its context.
SEASON_ARTIFACTS = {
    "snapshots": "{season}",
    "reliability": "reliability_{season}",
    "target_coverage": "target_coverage_{season}",
    "frames": "frames_{season}",
    "credited": "credited_games_{season}",
}
TABLES = (
    "weights",
    "selection",
    "dev_pairs",
    "grade_coverage",
    "reliability",
    "portability",
    "portal",
    "target_coverage",
    "next_season",
    "draft_check",
)
PORTAL_COLUMNS = [
    "season",
    "portal_rows",
    "fbs_destination_rows",
    "matched",
    "matched_fbs",
    "rate",
]


def traits_artifact(name: str) -> tuple[str, ...]:
    return ("players", "position_traits", f"{name}.parquet")


def season_artifact(output: str, season: int) -> tuple[str, ...]:
    return traits_artifact(SEASON_ARTIFACTS[output].format(season=season))


# Season assembly


def earlier_play_seasons(season: int) -> list[int]:
    """Raw play-by-play seasons on disk from ``FIRST_PLAY_SEASON`` up to ``season`` − 1."""
    root = RAW_DIR / "pbp"
    if not root.is_dir():
        return []
    return sorted(
        int(path.name)
        for path in root.iterdir()
        if path.is_dir()
        and path.name.isdigit()
        and FIRST_PLAY_SEASON <= int(path.name) < season
    )


def season_identity(
    credit: pd.DataFrame, box: pd.DataFrame, games: pd.DataFrame, frames: pd.DataFrame
) -> pd.DataFrame:
    """``athlete_id, athlete_name, team`` from each player's latest credit or box row.

    Box rows take their model week from ``games``. Every OL unit in ``frames``
    is named ``"<team> offensive line"``.
    """
    box_weeks = box.merge(games[["game_id", "model_week"]], on="game_id", how="left")
    rows = pd.concat(
        [frame[[*IDENTITY_COLUMNS, "model_week"]] for frame in (credit, box_weeks)],
        ignore_index=True,
    )
    rows = rows.dropna(subset=["athlete_id"]).astype({"athlete_id": str})
    people = rows.sort_values(
        "model_week", kind="stable", na_position="first"
    ).drop_duplicates("athlete_id", keep="last")[IDENTITY_COLUMNS]
    unit_ids = sorted(
        {
            athlete
            for athlete in frames["athlete_id"].astype(str)
            if athlete.startswith(snapshots.UNIT_PREFIX)
        }
    )
    unit_teams = [athlete.removeprefix(snapshots.UNIT_PREFIX) for athlete in unit_ids]
    units = pd.DataFrame(
        {
            "athlete_id": unit_ids,
            "athlete_name": [f"{team} offensive line" for team in unit_teams],
            "team": unit_teams,
        }
    )
    return pd.concat([people, units], ignore_index=True)


def with_fbs(snapshot: pd.DataFrame, season: int) -> pd.DataFrame:
    """``snapshot`` plus ``fbs``: whether the row's team is FBS in the season's teams file.

    The grade pool is qualified FBS players, so every scoring path adds the flag
    the same way the weekly runner does.
    """
    teams = read_season_source(season, "teams")
    schools = set(teams.loc[teams["classification"].eq("fbs"), "school"])
    return snapshot.assign(fbs=snapshot["team"].isin(schools))


def credited_games(credited: pd.DataFrame) -> pd.DataFrame:
    """Adjusted EPA per player and game from v4's credited frame: its per-game increments."""
    return (
        credited.astype({"athlete_id": str})
        .groupby(["athlete_id", "game_id", "model_week"], as_index=False)[
            "adjusted_epa"
        ]
        .sum()[CREDITED_GAME_COLUMNS]
    )


def uncovered_share(coverage: pd.DataFrame) -> float:
    """Incompletions with no target from the feed or play text, over all incompletions."""
    incompletions = coverage["incompletions"].sum()
    return float(coverage["uncovered"].sum() / incompletions) if incompletions else 0.0


def poor_target_coverage(coverage: pd.DataFrame) -> bool:
    """Whether the uncovered share of incompletions exceeds ``POOR_TARGET_COVERAGE``."""
    return uncovered_share(coverage) > POOR_TARGET_COVERAGE


def fit_tables_before(season: int) -> ExpectationTables:
    """Expectation tables fit on every play season on disk from 2019 up to ``season`` − 1."""
    return fit_expectation_tables(
        {
            earlier_season: classify_plays(store.read_season_pbp(earlier_season))
            for earlier_season in earlier_play_seasons(season)
        },
        before_season=season,
    )


def build_season_traits(
    season: int, tables: ExpectationTables | None = None
) -> dict[str, pd.DataFrame]:
    """Every trait frame, weekly snapshot and reliability table for ``season``.

    With ``tables`` (the frozen serving tables) no expectation is refit; a table
    whose source seasons include ``season`` or a later one is refused. Without it
    the tables are fit on every earlier season on disk from 2019, as the backtest
    does. The season decides its own target mode from its own coverage
    (``poor_target_coverage``), served or backtested. Writes the snapshots,
    reliability and target coverage, plus the raw frames and per-game credit the
    backtest pools for reliability.
    """
    started = time.perf_counter()
    if tables is None:
        tables = fit_tables_before(season)
    elif any(int(source) >= season for source in tables.source_seasons):
        raise ValueError(
            f"expectation tables from {tables.source_seasons} cannot rate {season}"
        )
    ctx = load_season_context(season)
    recovered = recover_targets(
        ctx.plays, read_weekly(season, "play_stats"), ctx.roster
    )
    coverage = target_coverage(ctx.plays, recovered)
    poor = poor_target_coverage(coverage)
    log.info(
        "position traits %s: expectations from %s, uncovered incompletions %.3f%s",
        season,
        tables.source_seasons,
        uncovered_share(coverage),
        " (receptions mode)" if poor else " (targets mode)",
    )
    frames = pd.concat(
        [
            qb_traits(ctx, tables, recovered, poor_coverage=poor),
            rb_traits(ctx, tables, recovered, poor_coverage=poor),
            receiver_traits(ctx, tables, recovered, poor_coverage=poor),
            defense_traits(ctx),
            line_traits(ctx, tables),
            punting_traits(ctx),
        ],
        ignore_index=True,
    ).astype({"athlete_id": str})
    identity = season_identity(ctx.credit, ctx.box, ctx.games, frames)
    weekly = snapshots.weekly_trait_snapshots(
        frames,
        ctx.positions,
        ctx.games,
        ctx.teams,
        identity,
        season,
        datetime.now(timezone.utc),
    )
    outputs = {
        "frames": frames,
        "snapshots": weekly,
        "reliability": snapshots.split_half_reliability(
            frames, ctx.positions, games_at=RELIABILITY_GAMES
        ),
        "target_coverage": coverage,
        "credited": credited_games(ctx.credited),
    }
    for output, frame in outputs.items():
        store.write_processed(frame, *season_artifact(output, season))
    log.info(
        "position traits %s: %d frame rows, %d snapshot rows in %.1fs",
        season,
        len(frames),
        len(outputs["snapshots"]),
        time.perf_counter() - started,
    )
    return outputs


def load_season(season: int, rebuild: bool = False) -> dict[str, pd.DataFrame]:
    """The season's written outputs, building them when any is missing or ``rebuild``."""
    stored = set(store.processed_names("players", "position_traits"))
    names = {
        output: SEASON_ARTIFACTS[output].format(season=season)
        for output in SEASON_ARTIFACTS
    }
    if rebuild or not set(names.values()) <= stored:
        return build_season_traits(season)
    log.info("position traits %s: reading cached artifacts", season)
    return {
        output: store.read_processed(*season_artifact(output, season))
        for output in names
    }


def player_values(season: int) -> pd.DataFrame:
    values = store.read_processed(*pipeline.player_values_artifact(season))
    log.info(
        "player values %s: model versions %s",
        season,
        sorted(values["model_version"].unique()),
    )
    return values[VALUE_COLUMNS].astype({"athlete_id": str})


# Pooled development evidence


def _season_keys(values: pd.Series, season: int) -> pd.Series:
    """Season-suffixed ids, so one athlete's seasons stay apart when pooled; an OL
    unit id keeps its ``unit:`` prefix."""
    return values.astype(str) + f"@{season}"


def _people(snapshot: pd.DataFrame) -> pd.DataFrame:
    """``position_group, team, fbs`` per athlete, indexed by athlete id."""
    return (
        snapshot.astype({"athlete_id": str})
        .drop_duplicates("athlete_id")
        .set_index("athlete_id")[["position_group", "team", "fbs"]]
    )


def frames_with_epa(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The season's raw trait frames plus one ``epa.credit`` row per credited game of
    an athlete in the snapshot: the game's adjusted EPA over a denominator of 1, so
    its ``rate`` aggregate is adjusted EPA per game."""
    frames = data["frames"].astype({"athlete_id": str})
    credited = data["credited"].astype({"athlete_id": str})
    credited = credited[
        credited["athlete_id"].isin(set(data["snapshots"]["athlete_id"].astype(str)))
    ]
    epa = pd.DataFrame(
        {
            "athlete_id": credited["athlete_id"],
            "game_id": credited["game_id"],
            "model_week": credited["model_week"],
            "trait": EPA_TRAIT,
            "numerator": credited["adjusted_epa"].astype(float),
            "denominator": 1.0,
        }
    )
    return pd.concat([frames, epa], ignore_index=True)


def pooled_frames(
    seasons: dict[int, dict[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The seasons' raw trait frames (with ``epa.credit``) and ``athlete_id,
    position_group`` with season-suffixed ids; each athlete takes his snapshot group
    in that season."""
    frames, groups = [], []
    for season, data in seasons.items():
        frame = frames_with_epa(data)
        frames.append(
            frame.assign(athlete_id=_season_keys(frame["athlete_id"], season))
        )
        people = _people(data["snapshots"])["position_group"].reset_index()
        groups.append(
            people.assign(athlete_id=_season_keys(people["athlete_id"], season))
        )
    return pd.concat(frames, ignore_index=True), pd.concat(groups, ignore_index=True)


def pooled_trait_reliability(
    seasons: dict[int, dict[str, pd.DataFrame]], games_at: tuple[int, ...]
) -> pd.DataFrame:
    """Each trait's split-half reliability with the seasons pooled."""
    frames, groups = pooled_frames(seasons)
    return snapshots.split_half_reliability(frames, groups, games_at=games_at)


def final_week_rows(seasons: dict[int, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Qualified FBS trait rows at each season's final week with that week's v4
    ``shrunk_per_game``, season-suffixed ids; the input to ``grade.orientations``."""
    parts = []
    for season, data in seasons.items():
        snapshot = data["snapshots"]
        final = snapshot[
            snapshot["week"].eq(snapshot["week"].max())
            & snapshot["qualified"].eq(True)
            & snapshot["fbs"].eq(True)
        ].astype({"athlete_id": str})
        epa = _final_epa(data["values"])
        rows = final[["athlete_id", "position_group", "trait", "value"]].assign(
            shrunk_per_game=final["athlete_id"].map(epa)
        )
        parts.append(rows.assign(athlete_id=_season_keys(rows["athlete_id"], season)))
    return pd.concat(parts, ignore_index=True)


# Split-half reliability of the composite


def _alternate(games: pd.DataFrame) -> pd.DataFrame:
    """Each player's distinct games in week order, alternating between half 0 and 1."""
    games = (
        games[["athlete_id", "game_id", "model_week"]]
        .dropna(subset=["game_id"])
        .drop_duplicates(["athlete_id", "game_id"])
        .sort_values(["athlete_id", "model_week", "game_id"], kind="stable")
    )
    order = games.groupby("athlete_id").cumcount()
    return games.assign(
        half=order % 2,
        slot=order // 2,
        played=games.groupby("athlete_id")["game_id"].transform("size"),
    )


def _half_games(alternated: pd.DataFrame, n: int) -> pd.DataFrame:
    """Players with at least ``n`` games, and the first ``n // 2`` games of each half."""
    chosen = alternated[alternated["played"].ge(n) & alternated["slot"].lt(n // 2)]
    return chosen[["athlete_id", "game_id", "half"]]


def _pearson(left: pd.Series, right: pd.Series) -> float:
    """Pearson correlation; NaN when either side is constant."""
    if not (left.std() > 0 and right.std() > 0):
        return np.nan
    return float(left.corr(right))


def half_composites(
    frames: pd.DataFrame,
    people: pd.DataFrame,
    chosen: pd.DataFrame,
    n: int,
    model: grade.GradeModel,
) -> pd.DataFrame:
    """Each chosen athlete's composite in half 0 and half 1, indexed by athlete.

    Each half is scored as its own snapshot week: trait values by the registered
    aggregate over the half's games, every athlete qualified, and the grade's own
    pool (FBS athletes at the group, from ``people``). ``latest`` traits have no
    games and drop out.
    """
    rows = frames.merge(chosen, on=["athlete_id", "game_id"])
    rows = rows.assign(
        aggregate=rows["trait"].map(
            {name: spec.aggregate for name, spec in all_traits().items()}
        )
    )
    values = snapshots.aggregate(rows, ["athlete_id", "trait", "half"]).reset_index()
    half_snapshot = values.assign(
        position_group=values["athlete_id"].map(people["position_group"]),
        team=values["athlete_id"].map(people["team"]),
        fbs=values["athlete_id"].map(people["fbs"]).eq(True),
        games=n // 2,
        qualified=True,
    ).dropna(subset=["position_group"])
    columns = {}
    for half in (0, 1):
        scored = grade.score_grades(
            half_snapshot[half_snapshot["half"].eq(half)], model
        )
        columns[half] = scored.set_index("athlete_id")["composite"]
    return pd.DataFrame(columns)


def composite_split_half(
    seasons: dict[int, dict[str, pd.DataFrame]],
    model: grade.GradeModel,
    games_at: tuple[int, ...],
) -> pd.DataFrame:
    """Split-half reliability of the grade's composite per group at each game count.

    Each athlete's distinct games with a trait-frame row alternate between halves in
    week order; at ``n`` an athlete needs ``n`` games and each half keeps its first
    ``n // 2``. The seasons are pooled with season-suffixed ids and each athlete takes
    his snapshot group; a group needs ``MIN_QUALIFIED_PLAYERS`` athletes scored in
    both halves.
    """
    by_games: dict[int, list[pd.DataFrame]] = {n: [] for n in games_at}
    for season, data in seasons.items():
        frames = frames_with_epa(data)
        people = _people(data["snapshots"])
        alternated = _alternate(data["frames"].astype({"athlete_id": str}))
        for n in games_at:
            chosen = _half_games(alternated, n)
            if n < 2 or chosen.empty:
                continue
            halves = half_composites(frames, people, chosen, n, model)
            halves["position_group"] = halves.index.map(people["position_group"])
            by_games[n].append(
                halves.set_axis(_season_keys(halves.index.to_series(), season))
            )
    rows = []
    for n, parts in by_games.items():
        if not parts:
            continue
        wide = pd.concat(parts).dropna()
        for group, frame in wide.groupby("position_group"):
            if len(frame) < MIN_QUALIFIED_PLAYERS:
                continue
            rows.append(
                {
                    "position_group": group,
                    "games": n,
                    "players": len(frame),
                    "correlation": _pearson(frame[0], frame[1]),
                }
            )
    out = pd.DataFrame(rows, columns=SPLIT_COLUMNS)
    return out.sort_values(["position_group", "games"], ignore_index=True)


def reliability_table(
    seasons: dict[int, dict[str, pd.DataFrame]],
    pooled_traits: pd.DataFrame,
    composite: pd.DataFrame,
) -> pd.DataFrame:
    """Each dev season's trait reliability, the pooled trait rows the weights read,
    and the pooled composite curve (``trait`` ``grade``)."""
    parts = [
        data["reliability"].assign(season=str(season))
        for season, data in seasons.items()
    ]
    parts.append(pooled_traits.assign(season="pooled"))
    parts.append(composite.assign(trait="grade", season="pooled"))
    columns = ["season", *snapshots.RELIABILITY_COLUMNS]
    return pd.concat(
        [part[columns] for part in parts if len(part)], ignore_index=True
    ).reindex(columns=columns)


# Report-only tables


def matched_transfers(seasons) -> dict[int, pd.DataFrame]:
    """``match_transfers`` for every portal season from ``FIRST_PORTAL_SEASON``."""
    return {
        season: match_transfers(
            read_season_source(season, "portal"),
            read_season_source(season, "roster"),
            read_season_source(season - 1, "roster"),
        )
        for season in sorted(seasons)
        if season >= FIRST_PORTAL_SEASON
    }


def portability_table(
    snapshots_by_season: dict[int, pd.DataFrame],
    matched: dict[int, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    matched = matched_transfers(snapshots_by_season) if matched is None else matched
    transfers = (
        pd.concat(matched.values(), ignore_index=True)
        if matched
        else pd.DataFrame(columns=TRANSFER_COLUMNS)
    )
    return portability(snapshots_by_season, transfers)


def portal_match_table(
    seasons, matched: dict[int, pd.DataFrame] | None = None
) -> pd.DataFrame:
    """Per portal season: portal rows, rows whose destination is an FBS school in the
    season's teams file, rows ``match_transfers`` tied to one athlete, the matched rows
    with an FBS destination, and ``rate = matched_fbs / fbs_destination_rows``."""
    matched = matched_transfers(seasons) if matched is None else matched
    rows = []
    for season, transfers in sorted(matched.items()):
        portal = read_season_source(season, "portal")
        teams = read_season_source(season, "teams")
        fbs = set(teams.loc[teams["classification"].eq("fbs"), "school"])
        destination_rows = int(portal["destination"].isin(fbs).sum())
        matched_fbs = int(transfers["destination"].isin(fbs).sum())
        rows.append(
            {
                "season": season,
                "portal_rows": len(portal),
                "fbs_destination_rows": destination_rows,
                "matched": len(transfers),
                "matched_fbs": matched_fbs,
                "rate": matched_fbs / destination_rows if destination_rows else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=PORTAL_COLUMNS)


def _spearman(left: pd.Series, right: pd.Series) -> float:
    both = left.notna() & right.notna()
    if both.sum() < MIN_QUALIFIED_PLAYERS:
        return np.nan
    return _pearson(left[both].rank(), right[both].rank())


def _final_epa(values: pd.DataFrame) -> pd.Series:
    """v4 ``shrunk_per_game`` at the values' final week, by athlete id."""
    final = values[values["week"].eq(values["week"].max())].astype({"athlete_id": str})
    return final.drop_duplicates("athlete_id").set_index("athlete_id")[
        "shrunk_per_game"
    ]


def final_week_grades(snapshot: pd.DataFrame, model: grade.GradeModel) -> pd.DataFrame:
    """``athlete_id, position_group, games, grade`` at the snapshot's final week."""
    final = snapshot[snapshot["week"].eq(snapshot["week"].max())]
    return grade.score_grades(final, model)[
        ["athlete_id", "position_group", "games", "grade"]
    ]


def next_season_table(
    seasons: dict[int, dict[str, pd.DataFrame]],
    model: grade.GradeModel | dict[int, grade.GradeModel],
    validation_season: int,
) -> pd.DataFrame:
    """Does a final-week grade predict next season, compared with EPA credit alone?

    For each consecutive pair ``(t, t + 1)`` with ``t + 1 ≤ validation_season``, over
    players graded at the final week of both seasons at the same group: the Spearman
    of the season-``t`` grade with the ``t + 1`` grade and with the ``t + 1`` v4
    ``shrunk_per_game``, and of the season-``t`` v4 value with the ``t + 1`` v4 value
    (the EPA-only baseline). Where at least ``MIN_QUALIFIED_PLAYERS`` players have all
    four numbers every correlation uses only them; otherwise (OL has no v4 value, few
    punters have one) the grade rows are kept and the EPA columns are mostly empty.
    ``holdout`` is True only for the pair ending in the validation season. ``model``
    may map each pair's first season to the model that grades both of its seasons
    (cross-fitted weights).
    """
    epa = {season: _final_epa(data["values"]) for season, data in seasons.items()}
    rows = []
    for season in sorted(seasons):
        later = season + 1
        if later not in seasons or later > validation_season:
            continue
        pair_model = model[season] if isinstance(model, dict) else model
        now = final_week_grades(seasons[season]["snapshots"], pair_model)
        nxt = final_week_grades(seasons[later]["snapshots"], pair_model)
        both = now.merge(
            nxt, on=["athlete_id", "position_group"], suffixes=("", "_next")
        )
        both = both.assign(
            epa=both["athlete_id"].map(epa[season]),
            epa_next=both["athlete_id"].map(epa[later]),
        )
        for group, frame in both.groupby("position_group"):
            complete = frame.dropna(subset=["grade", "grade_next", "epa", "epa_next"])
            used = complete if len(complete) >= MIN_QUALIFIED_PLAYERS else frame
            rows.append(
                {
                    "season": season,
                    "next_season": later,
                    "position_group": group,
                    "players": len(used),
                    "spearman_grade_next_grade": _spearman(
                        used["grade"], used["grade_next"]
                    ),
                    "spearman_grade_next_epa": _spearman(
                        used["grade"], used["epa_next"]
                    ),
                    "spearman_epa_next_epa": _spearman(used["epa"], used["epa_next"]),
                    "holdout": later == validation_season
                    and season == validation_season - 1,
                }
            )
    return pd.DataFrame(rows, columns=NEXT_SEASON_COLUMNS)


def final_week_values(
    snapshot: pd.DataFrame, values: pd.DataFrame, model: grade.GradeModel
) -> pd.DataFrame:
    """``athlete_id, grade, value_above_replacement`` at each source's final week."""
    scored = (
        final_week_grades(snapshot, model)
        .sort_values("games", ascending=False, kind="stable")
        .drop_duplicates("athlete_id")[["athlete_id", "grade"]]
    )
    final_values = values[values["week"].eq(values["week"].max())]
    v4 = final_values.drop_duplicates("athlete_id")[
        ["athlete_id", "value_above_replacement"]
    ]
    return scored.merge(v4, on="athlete_id", how="outer")


def draft_check(
    snapshots_by_season: dict[int, pd.DataFrame],
    values_by_season: dict[int, pd.DataFrame],
    model: grade.GradeModel,
) -> pd.DataFrame:
    """Rank correlation of final-week grade and v4 value with draft pick and NFL snaps.

    Report-only: draft year ``y`` reads season ``y − 1``. Pick is negated so higher
    is better; NFL snaps count only for a mature (``observed``) outcome. Every
    correlation uses the players with both a grade and a v4 value. Dev seasons are
    scored in sample.
    """
    if not DRAFT_COHORT.exists():
        log.info("draft check skipped: %s is absent", DRAFT_COHORT)
        return pd.DataFrame(columns=DRAFT_COLUMNS)
    cohort = pd.read_parquet(
        DRAFT_COHORT,
        columns=[
            "draft_year",
            "pick",
            "athlete_id",
            "position_group",
            "nfl_first3_scrimmage_snaps",
            "outcome_status",
        ],
    ).astype({"athlete_id": str})
    parts = []
    for year, players in cohort.groupby("draft_year"):
        season = int(year) - 1
        if season in snapshots_by_season:
            scores = final_week_values(
                snapshots_by_season[season], values_by_season[season], model
            )
            parts.append(players.merge(scores, on="athlete_id", how="inner"))
    if not parts:
        log.info("draft check: no draft year has a season in the backtest")
        return pd.DataFrame(columns=DRAFT_COLUMNS)
    pooled = pd.concat(parts, ignore_index=True).dropna(
        subset=["grade", "value_above_replacement"]
    )
    rows = []
    for group, frame in pooled.groupby("position_group"):
        mature = frame[frame["outcome_status"].eq(MATURE_OUTCOME)]
        rows.append(
            {
                "position_group": group,
                "players": len(frame),
                "spearman_pick_v4": _spearman(
                    frame["value_above_replacement"], -frame["pick"]
                ),
                "spearman_pick_grade": _spearman(frame["grade"], -frame["pick"]),
                "spearman_nfl_games_v4": _spearman(
                    mature["value_above_replacement"],
                    mature["nfl_first3_scrimmage_snaps"],
                ),
                "spearman_nfl_games_grade": _spearman(
                    mature["grade"], mature["nfl_first3_scrimmage_snaps"]
                ),
            }
        )
    return pd.DataFrame(rows, columns=DRAFT_COLUMNS)


def grade_coverage_table(
    seasons: dict[int, dict[str, pd.DataFrame]], model: grade.GradeModel
) -> pd.DataFrame:
    """``grade.coverage_losses`` at each season's final week, with a ``season`` column."""
    parts = []
    for season, data in seasons.items():
        snapshot = data["snapshots"]
        final = snapshot[snapshot["week"].eq(snapshot["week"].max())]
        parts.append(
            grade.coverage_losses(grade.grade_eligibility(final, model)).assign(
                season=season
            )
        )
    out = pd.concat(parts, ignore_index=True)
    return out[["season", *grade.COVERAGE_LOSS_COLUMNS]]


def log_war_leaders(data: dict[str, pd.DataFrame], season: int, top: int = 10) -> None:
    """Log the season's final-week wins above replacement leaders among qualified
    FBS players, scored the way the weekly runner scores them."""
    snapshot = data["snapshots"].astype({"athlete_id": str})
    final = snapshot[snapshot["week"].eq(snapshot["week"].max())]
    values = data["values"]
    measured = pipeline.war_rows(
        final,
        values,
        war.wins_above_replacement(values),
    )
    leaders = measured.sort_values("value", ascending=False).head(top)
    for rank, row in enumerate(leaders.itertuples(), start=1):
        log.info(
            "wins above replacement %s week %s #%d: %s (%s, %s) %.2f wins, %.1f points, %d games",
            season,
            row.week,
            rank,
            row.athlete_name,
            row.team,
            row.position_group,
            row.value,
            row.points_above_replacement,
            row.games,
        )


def score_cached_season(season: int) -> pd.DataFrame:
    """Score an already-built season from its cached artifacts and write
    ``scored_<season>.parquet``.

    Uses the latest grade model whose cutoff precedes ``season``; reads the cached
    snapshot, target coverage and the season's v4 values. Nothing is rebuilt and no expectation table is read. The
    output has the weekly runner's columns (trait, grade and war rows). A cached
    snapshot that is itself a scored artifact keeps only its trait rows.
    """
    models = store.PROCESSED_DIR / "players" / "models"
    model = grade.load_grade_model(grade.latest_grade_model_path(models, season))
    grade.validate_grade_model(model, season)
    snapshot = store.read_processed(*season_artifact("snapshots", season))
    snapshot = snapshot[~snapshot["trait"].isin(["grade", "war"])]
    snapshot = with_fbs(
        snapshot[snapshots.TRAIT_SNAPSHOT_COLUMNS].astype({"athlete_id": str}), season
    )
    teams = read_season_source(season, "teams")
    out = pipeline.score_snapshot(
        snapshot,
        model,
        player_values(season),
        datetime.now(timezone.utc),
        poor_target_coverage(
            store.read_processed(*season_artifact("target_coverage", season))
        ),
        set(teams.loc[teams["classification"].eq("fbs"), "school"]),
    )
    store.write_processed(out, *traits_artifact(f"scored_{season}"))
    log.info(
        "position traits %s: scored the cached season with the model through %s; "
        "%d rows, %d grade rows, %d war rows",
        season,
        max(model.dev_seasons),
        len(out),
        int(out["trait"].eq("grade").sum()),
        int(out["trait"].eq("war").sum()),
    )
    return out


# Model selection on the development pairs


DEV_PAIR_COLUMNS = [
    "variant",
    "season",
    "next_season",
    "position_group",
    "players",
    "spearman_grade_next_epa",
    "spearman_epa_next_epa",
]
SELECTION_COLUMNS = [
    "position_group",
    "current",
    "epa_trait",
    "fitted",
    "shrinkage",
    "epa_alone",
    "epa_weight_share",
    "epa_kept",
    "fitted_kept",
    "fitted_penalty",
    "shrink_games",
    "shrink_kept",
]
MIN_FIT_ROWS = 30  # per development pair; a sample floor, not a tuned value


def dev_pair_table(
    dev: dict[int, dict[str, pd.DataFrame]],
    model: grade.GradeModel | dict[int, grade.GradeModel],
    variant: str,
) -> pd.DataFrame:
    """``next_season_table`` over the development pairs only, labelled ``variant``."""
    table = next_season_table(dev, model, max(dev))
    return table.assign(variant=variant)[DEV_PAIR_COLUMNS]


def pair_means(
    table: pd.DataFrame, column: str = "spearman_grade_next_epa"
) -> pd.Series:
    """The mean of ``column`` over the development pairs, per group."""
    return table.groupby("position_group")[column].mean()


def _replace_groups(base: pd.DataFrame, parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """``base`` weights with each group in ``parts`` replaced by its rows."""
    kept = base[~base["position_group"].isin(parts)]
    return pd.concat([kept, *parts.values()], ignore_index=True)[grade.WEIGHT_COLUMNS]


def _model(weights: pd.DataFrame, dev_seasons: list[int], **fields) -> grade.GradeModel:
    return grade.GradeModel(
        weights=weights.reset_index(drop=True), dev_seasons=dev_seasons, **fields
    )


def regression_pairs(
    dev: dict[int, dict[str, pd.DataFrame]],
    base: grade.GradeModel,
    group: str,
    candidates: pd.DataFrame,
) -> dict[int, tuple[pd.DataFrame, pd.Series]]:
    """``(X, y)`` per development pair for ``group``.

    The sample is the players graded under ``base`` at the final week of season t
    who have a v4 value at the same group at the final week of t + 1. X holds their
    capped oriented z-scores on ``candidates`` (0 where missing or unavailable that
    week), y the next season's v4 ``shrunk_per_game`` z-scored over the sample. A
    pair with fewer than ``MIN_FIT_ROWS`` rows is left out.
    """
    out = {}
    for season in sorted(dev):
        later = season + 1
        if later not in dev:
            continue
        snapshot = dev[season]["snapshots"]
        final = snapshot[snapshot["week"].eq(snapshot["week"].max())]
        graded = grade.score_grades(final, base)
        graded = graded[graded["position_group"].eq(group)]["athlete_id"]
        values = dev[later]["values"]
        nxt = values[values["week"].eq(values["week"].max())].astype(
            {"athlete_id": str}
        )
        nxt = nxt[nxt["position_group"].eq(group)].drop_duplicates("athlete_id")
        target = nxt.set_index("athlete_id")["shrunk_per_game"].reindex(graded).dropna()
        if len(target) < MIN_FIT_ROWS or not target.std() > 0:
            continue
        x = (
            grade.trait_z_frame(final, candidates, group)
            .reindex(target.index)
            .fillna(0.0)
        )
        out[season] = (x, (target - target.mean()) / target.std())
    return out


def _candidates(weights: pd.DataFrame, group: str, epa_kept: bool) -> pd.DataFrame:
    """The traits that may enter the fit at ``group``: a reliability on at least the
    player floor (``latest`` traits have none) and, for ``epa.credit``, the EPA trait
    kept there."""
    rows = weights[weights["position_group"].eq(group) & weights["reliability"].notna()]
    if not epa_kept:
        rows = rows[rows["trait"].ne(EPA_TRAIT)]
    return rows[["position_group", "trait", "sign"]]


def _lower_bounds(candidates: pd.DataFrame) -> pd.Series:
    """0 for a higher-is-better trait, unbounded below for ``False`` or ``None``."""
    specs = all_traits()
    return pd.Series(
        [
            0.0 if specs[trait].higher_is_better is True else -np.inf
            for trait in candidates["trait"]
        ],
        index=candidates["trait"].to_list(),
    )


def select_grade_model(
    dev: dict[int, dict[str, pd.DataFrame]],
    pooled: pd.DataFrame,
    portability: pd.DataFrame,
    orientation: pd.DataFrame,
) -> tuple[grade.GradeModel, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The grade model, chosen position by position on the development pairs.

    Starting from the current grade (reliability × portability weights without EPA
    credit), each change is evaluated by the mean over the development pairs of the
    Spearman of the season-t final-week grade with season t + 1 v4 EPA credit, and
    kept at a group only where that mean improves: (1) ``epa.credit`` as a trait in
    the reliability × portability weights; (2) weights fitted by a bounded ridge of
    next-season EPA on the traits, evaluated with weights fitted on the other pairs
    (leave-one-pair-out), kept as the fit on every pair; (3) shrinkage of the
    composite by games / (games + k_g), k_g from the composite's split-half curve.
    Returns the model, the composite reliability table, the long ``dev_pairs`` table
    and the per-group ``selection`` table.
    """
    dev_seasons = sorted(dev)
    with_epa = grade.trait_weights(pooled, portability, orientation)
    without = grade.trait_weights(
        pooled, portability, orientation[orientation["trait"].ne(EPA_TRAIT)]
    )
    current = _model(without, dev_seasons)
    epa_model = _model(with_epa, dev_seasons)
    tables = [
        dev_pair_table(dev, current, "current"),
        dev_pair_table(dev, epa_model, "epa_trait"),
    ]
    base_mean, epa_mean = pair_means(tables[0]), pair_means(tables[1])
    groups = sorted(grade.graded_groups(epa_model) | grade.graded_groups(current))
    epa_kept = {
        g: bool(epa_mean.get(g, np.nan) > base_mean.get(g, np.nan)) for g in groups
    }

    epa_rows = with_epa[with_epa["trait"].eq(EPA_TRAIT)]
    dropped_epa = epa_rows.assign(weight=0.0, status="not kept")
    step1_parts = {}
    for g in groups:
        if epa_kept[g]:
            step1_parts[g] = with_epa[with_epa["position_group"].eq(g)]
        else:
            step1_parts[g] = pd.concat(
                [
                    without[without["position_group"].eq(g)],
                    dropped_epa[dropped_epa["position_group"].eq(g)],
                ]
            )
    step1 = _replace_groups(
        pd.concat([without, dropped_epa], ignore_index=True), step1_parts
    )
    model1 = _model(step1, dev_seasons)
    step1_mean = pd.Series(
        {g: (epa_mean if epa_kept[g] else base_mean).get(g, np.nan) for g in groups}
    )

    # (2) fitted weights, cross-fitted for the evaluation
    designs, lowers, candidate_frames = {}, {}, {}
    for g in groups:
        candidates = _candidates(step1, g, epa_kept[g])
        if candidates.empty:
            continue
        pairs = regression_pairs(dev, model1, g, candidates)
        if len(pairs) < 2:
            continue
        designs[g], lowers[g], candidate_frames[g] = (
            pairs,
            _lower_bounds(candidates),
            candidates,
        )
    pair_seasons = [season for season in dev_seasons if season + 1 in dev]
    cross_models = {}
    for held in pair_seasons:
        parts = {}
        for g, pairs in designs.items():
            training = {
                season: frame for season, frame in pairs.items() if season != held
            }
            if not training:
                continue
            coefficients, _ = grade.fit_ridge_weights(training, lowers[g])
            parts[g] = grade.fitted_group_weights(step1, g, coefficients)
        cross_models[held] = _model(_replace_groups(step1, parts), dev_seasons)
    fitted_table = dev_pair_table(dev, cross_models, "fitted")
    fitted_mean = pair_means(fitted_table)
    fitted_kept = {
        g: g in designs and bool(fitted_mean.get(g, np.nan) > step1_mean.get(g, np.nan))
        for g in groups
    }
    final_parts, penalties = {}, {}
    for g in groups:
        if fitted_kept[g]:
            coefficients, penalties[g] = grade.fit_ridge_weights(designs[g], lowers[g])
            final_parts[g] = grade.fitted_group_weights(step1, g, coefficients)
    step2 = _replace_groups(step1, final_parts)
    model2 = _model(step2, dev_seasons)
    step2_mean = pd.Series(
        {
            g: (fitted_mean if fitted_kept[g] else step1_mean).get(g, np.nan)
            for g in groups
        }
    )
    composite = composite_split_half(dev, model2, RELIABILITY_GAMES)
    reliability = composite.dropna(subset=["correlation"])[
        grade.RELIABILITY_COLUMNS
    ].reset_index(drop=True)

    # (3) shrinkage, on the same cross-fitted weights where they were kept
    shrink = grade.shrink_games_from(reliability)
    shrink_models = {}
    for held in pair_seasons:
        cross = cross_models[held].weights
        parts = {
            g: cross[cross["position_group"].eq(g)] for g in groups if fitted_kept[g]
        }
        shrink_models[held] = _model(
            _replace_groups(step2, parts),
            dev_seasons,
            reliability=reliability,
            shrink_games=shrink,
        )
    shrink_table = dev_pair_table(dev, shrink_models, "shrinkage")
    shrink_mean = pair_means(shrink_table)
    shrink_kept = {
        g: bool(shrink_mean.get(g, np.nan) > step2_mean.get(g, np.nan)) for g in groups
    }
    model = _model(
        step2,
        dev_seasons,
        reliability=reliability,
        shrink_games={g: shrink[g] for g in groups if shrink_kept[g] and g in shrink},
    )

    epa_alone = pair_means(tables[0], "spearman_epa_next_epa")
    share = epa_rows.set_index("position_group")["weight"]
    selection = pd.DataFrame(
        [
            {
                "position_group": g,
                "current": base_mean.get(g, np.nan),
                "epa_trait": epa_mean.get(g, np.nan),
                "fitted": fitted_mean.get(g, np.nan) if g in designs else np.nan,
                "shrinkage": shrink_mean.get(g, np.nan),
                "epa_alone": epa_alone.get(g, np.nan),
                "epa_weight_share": share.get(g, np.nan),
                "epa_kept": epa_kept[g],
                "fitted_kept": fitted_kept[g],
                "fitted_penalty": penalties.get(g, np.nan),
                "shrink_games": shrink.get(g, np.nan),
                "shrink_kept": shrink_kept[g],
            }
            for g in groups
        ],
        columns=SELECTION_COLUMNS,
    )
    dev_pairs = pd.concat([*tables, fitted_table, shrink_table], ignore_index=True)
    for row in selection.itertuples():
        log.info(
            "position grade %s: dev-pair mean current %.4f, +epa %.4f, +fitted %.4f, "
            "+shrinkage %.4f, epa alone %.4f; kept epa %s, fitted %s, shrinkage %s (k %.2f)",
            row.position_group,
            row.current,
            row.epa_trait,
            row.fitted,
            row.shrinkage,
            row.epa_alone,
            row.epa_kept,
            row.fitted_kept,
            row.shrink_kept,
            row.shrink_games,
        )
    return model, composite, dev_pairs, selection


# The backtest


@dataclass
class BacktestResult:
    weights: pd.DataFrame
    selection: pd.DataFrame
    dev_pairs: pd.DataFrame
    grade_coverage: pd.DataFrame
    reliability: pd.DataFrame
    portability: pd.DataFrame
    portal: pd.DataFrame
    target_coverage: pd.DataFrame
    next_season: pd.DataFrame
    draft_check: pd.DataFrame
    model: grade.GradeModel
    expectations: ExpectationTables


def run_backtest(
    dev_seasons: list[int],
    validation_season: int,
    rebuild: bool = False,
    score_holdout: bool = True,
) -> BacktestResult:
    """Set the grade's weights on ``dev_seasons`` and judge it on ``validation_season``.

    Reliability, portability and orientation read the development seasons only; the
    validation season enters no weight and no reliability curve, and is only scored.
    ``rebuild`` rebuilds cached season artifacts. The result also carries the serving
    expectation tables, fit on every play season through ``validation_season``, for
    the weekly runner to score later seasons with. With ``score_holdout`` False the
    validation season's next-season row and the draft check are not computed, so the
    model can be developed without looking at them.
    """
    dev_seasons = sorted({int(season) for season in dev_seasons})
    if validation_season <= max(dev_seasons):
        raise ValueError(
            f"validation season {validation_season} must follow every dev season {dev_seasons}"
        )
    seasons = {}
    for season in [*dev_seasons, validation_season]:
        data = load_season(season, rebuild)
        values = player_values(season)
        seasons[season] = {
            **data,
            "snapshots": grade.with_epa_credit(
                with_fbs(data["snapshots"], season), values
            ),
            "values": values,
        }
    dev = {season: seasons[season] for season in dev_seasons}
    snapshots_by_season = {
        season: data["snapshots"] for season, data in seasons.items()
    }

    pooled = pooled_trait_reliability(dev, (grade.WEIGHT_GAMES,))
    matched = matched_transfers(snapshots_by_season)
    dev_portability = portability_table(
        {season: snapshots_by_season[season] for season in dev_seasons},
        {season: frame for season, frame in matched.items() if season in dev_seasons},
    )
    orientation = grade.orientations(final_week_rows(dev))
    for row in orientation[orientation["sign_source"].eq("epa_spearman")].itertuples():
        log.info(
            "position grade: %s %s takes sign %+.0f (Spearman with v4 %.3f over %d players)",
            row.position_group,
            row.trait,
            row.sign,
            row.spearman_epa,
            row.players,
        )
    model, composite, dev_pairs, selection = select_grade_model(
        dev, pooled, dev_portability, orientation
    )
    log_war_leaders(seasons[validation_season], validation_season)
    return BacktestResult(
        weights=model.weights,
        selection=selection,
        dev_pairs=dev_pairs,
        grade_coverage=grade_coverage_table(seasons, model),
        reliability=reliability_table(dev, pooled, composite),
        portability=dev_portability,
        portal=portal_match_table(snapshots_by_season, matched),
        target_coverage=pd.concat(
            [data["target_coverage"] for data in seasons.values()], ignore_index=True
        ),
        next_season=(
            next_season_table(seasons, model, validation_season)
            if score_holdout
            else pd.DataFrame(columns=NEXT_SEASON_COLUMNS)
        ),
        draft_check=(
            draft_check(
                snapshots_by_season,
                {season: data["values"] for season, data in seasons.items()},
                model,
            )
            if score_holdout
            else pd.DataFrame(columns=DRAFT_COLUMNS)
        ),
        model=model,
        expectations=fit_tables_before(validation_season + 1),
    )


# Output


def _cell(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def markdown_table(frame: pd.DataFrame) -> str:
    """A GitHub Markdown table of ``frame``; floats to four decimals, missing values blank."""
    if frame.empty:
        return "(no rows)"
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(_cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])


def _column_type(values: pd.Series) -> pd.Series:
    present = values.dropna()
    if present.empty:
        return values.astype(float)
    if present.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        return values.astype("boolean")
    if present.map(
        lambda value: (
            isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
        )
    ).all():
        return values.astype(float)
    return values.map(lambda value: None if pd.isna(value) else str(value)).astype(
        object
    )


def evaluation_frame(result: BacktestResult) -> pd.DataFrame:
    """Every result table stacked with a ``table`` column; a column whose values are
    not all numbers or all booleans across tables is stored as strings."""
    stacked = pd.concat(
        [getattr(result, name).assign(table=name).astype(object) for name in TABLES],
        ignore_index=True,
    )
    stacked = stacked[
        ["table", *[column for column in stacked.columns if column != "table"]]
    ]
    return stacked.apply(
        lambda column: column if column.name == "table" else _column_type(column)
    )


def write_backtest(result: BacktestResult, cutoff_season: int) -> Path:
    """Save the grade model JSON, the frozen serving expectation tables beside it and
    the long evaluation frame, and print every table as Markdown."""
    path = grade.save_grade_model(result.model, grade.grade_model_path(cutoff_season))
    tables = save_expectation_tables(
        result.expectations, expectation_tables_path(path.parent, cutoff_season)
    )
    log.info(
        "position grade expectation tables from %s written to %s",
        result.expectations.source_seasons,
        tables,
    )
    store.write_processed(
        evaluation_frame(result), *traits_artifact(f"evaluation_{cutoff_season}")
    )
    for name in TABLES:
        print(f"\n## {name}\n\n{markdown_table(getattr(result, name))}")
    return path
