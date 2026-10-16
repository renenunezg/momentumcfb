"""Season assembly and the offline backtest of the position trait headline.

``build_season_traits`` runs every trait builder for one season and writes the
weekly snapshots, their split-half reliability and the target coverage.
``run_backtest`` fits the headline on chronological development seasons, scores a
later validation season it never saw, and runs the per-group swap test that
decides where the headline may replace the v4 value. ``write_backtest`` saves
the fit and one long evaluation frame and prints every table as Markdown.
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
from backend.model.weekly import load_weekly_games
from backend.players import pipeline
from backend.players.ingest import read_season_source, read_weekly
from backend.players.traits import TRAITS, headline, snapshots
from backend.players.traits.context import load_season_context
from backend.players.traits.defense import defense_traits
from backend.players.traits.expectations import fit_expectation_tables
from backend.players.traits.line import kicking_traits, line_traits
from backend.players.traits.offense import (
    POOR_TARGET_COVERAGE,
    qb_traits,
    rb_traits,
    receiver_traits,
)
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
REPLACEMENT_GAMES = 6
WEEK_BUCKETS = {"2-5": (2, 5), "6-9": (6, 9), "10+": (10, None)}
VALUE_COLUMNS = [
    "week",
    "athlete_id",
    "position_group",
    "games",
    "shrunk_per_game",
    "adjusted_per_game",
    "value_above_replacement",  # read only by the report-only draft check
]
CREDITED_GAME_COLUMNS = ["athlete_id", "game_id", "model_week", "adjusted_epa"]
IDENTITY_COLUMNS = ["athlete_id", "athlete_name", "team"]
SPLIT_COLUMNS = ["position_group", "games", "players", "correlation"]
SWAP_COLUMNS = [
    "position_group",
    "mae_v4_only",
    "mae_with_traits",
    "improvement",
    "reliability_6_v4",
    "reliability_6_headline",
    "replaces_value",
]
DRAFT_COLUMNS = [
    "position_group",
    "players",
    "spearman_pick_v4",
    "spearman_pick_headline",
    "spearman_nfl_games_v4",
    "spearman_nfl_games_headline",
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
    "validation",
    "swap",
    "reliability",
    "portability",
    "target_coverage",
    "draft_check",
)


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


def with_exposure(
    snapshot: pd.DataFrame, frames: pd.DataFrame, positions_map: pd.Series
) -> pd.DataFrame:
    """``snapshot`` plus each player's cumulative ``exposure`` through the week.

    Exposure comes from ``headline.exposure_shares``; a snapshot week past the last
    week of ``frames`` keeps the latest total, a player without usage gets 0, and an
    OL unit 1.0.
    """
    shares = headline.exposure_shares(frames, positions_map)
    weeks = sorted(set(shares["week"].astype(int)) | set(snapshot["week"].astype(int)))
    wide = (
        shares.pivot(index="athlete_id", columns="week", values="exposure")
        .reindex(columns=weeks)
        .ffill(axis=1)
        .fillna(0.0)
    )
    wide.index.name, wide.columns.name = "athlete_id", "week"
    out = snapshot.astype({"athlete_id": str}).merge(
        wide.stack().rename("exposure").reset_index(),
        on=["athlete_id", "week"],
        how="left",
    )
    out["exposure"] = out["exposure"].fillna(0.0)
    out.loc[out["athlete_id"].str.startswith(snapshots.UNIT_PREFIX), "exposure"] = 1.0
    return out


def credited_games(credited: pd.DataFrame) -> pd.DataFrame:
    """Adjusted EPA per player and game from v4's credited frame: its per-game increments."""
    return (
        credited.astype({"athlete_id": str})
        .groupby(["athlete_id", "game_id", "model_week"], as_index=False)[
            "adjusted_epa"
        ]
        .sum()[CREDITED_GAME_COLUMNS]
    )


def _uncovered_share(coverage: pd.DataFrame) -> float:
    incompletions = coverage["incompletions"].sum()
    return float(coverage["uncovered"].sum() / incompletions) if incompletions else 0.0


def build_season_traits(season: int) -> dict[str, pd.DataFrame]:
    """Every trait frame, weekly snapshot and reliability table for ``season``.

    Expectation tables are fit on every earlier season on disk from 2019. Writes
    the snapshots (with ``exposure``), reliability and target coverage, plus the raw
    frames and per-game credit the backtest pools for reliability.
    """
    started = time.perf_counter()
    ctx = load_season_context(season)
    earlier = earlier_play_seasons(season)
    tables = fit_expectation_tables(
        {
            earlier_season: classify_plays(store.read_season_pbp(earlier_season))
            for earlier_season in earlier
        },
        before_season=season,
    )
    recovered = recover_targets(
        ctx.plays, read_weekly(season, "play_stats"), ctx.roster
    )
    coverage = target_coverage(ctx.plays, recovered)
    poor = _uncovered_share(coverage) > POOR_TARGET_COVERAGE
    log.info(
        "position traits %s: expectations from %s, uncovered incompletions %.3f%s",
        season,
        earlier,
        _uncovered_share(coverage),
        " (receptions mode)" if poor else "",
    )
    frames = pd.concat(
        [
            qb_traits(ctx, tables, recovered, poor_coverage=poor),
            rb_traits(ctx, tables, recovered, poor_coverage=poor),
            receiver_traits(ctx, tables, recovered, poor_coverage=poor),
            defense_traits(ctx),
            line_traits(ctx, tables),
            kicking_traits(ctx, tables),
        ],
        ignore_index=True,
    ).astype({"athlete_id": str})
    identity = season_identity(ctx.credit, ctx.box, ctx.games, frames)
    positions_map = snapshots.position_groups(frames, ctx.positions).set_index(
        "athlete_id"
    )["position_group"]
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
        "snapshots": with_exposure(weekly, frames, positions_map),
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


# Fits and validation


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    return [column for column in matrix.columns if ":" in column]


def epa_columns(matrix: pd.DataFrame) -> list[str]:
    return [
        column
        for column in feature_columns(matrix)
        if column.endswith(f":{headline.EPA}")
    ]


def fit_on(
    matrix: pd.DataFrame, features: list[str], dev_seasons: list[int]
) -> headline.HeadlineFit:
    return headline.fit_headline(
        matrix[[*headline.META_COLUMNS, *features]], dev_seasons
    )


def predict_margin(fit: headline.HeadlineFit, rows: pd.DataFrame) -> np.ndarray:
    """The fit's margin for each row, on the fit's dev-season feature scale."""
    scale = fit.feature_scale.set_index("feature")
    kept = list(scale.index)
    z = ((rows[kept] - scale["mean"]) / scale["sd"]).to_numpy(float)
    return (
        fit.weights["intercept"]
        + fit.weights["home_field"] * rows["home_field"].to_numpy(float)
        + z @ fit.weights.reindex(kept).to_numpy(float)
    )


def _mae(fit: headline.HeadlineFit, rows: pd.DataFrame) -> float:
    return float(
        np.mean(np.abs(rows["margin"].to_numpy(float) - predict_margin(fit, rows)))
    )


def validation_mae(
    fit: headline.HeadlineFit, rows: pd.DataFrame, variant: str
) -> pd.DataFrame:
    """``variant, week_bucket, games, mae`` over the validation rows by model week."""
    buckets = []
    for bucket, (first, last) in WEEK_BUCKETS.items():
        week = rows["model_week"]
        inside = week.ge(first) if last is None else week.between(first, last)
        buckets.append((bucket, rows[inside]))
    buckets.append(("all", rows))
    return pd.DataFrame(
        [
            {
                "variant": variant,
                "week_bucket": bucket,
                "games": len(part),
                "mae": _mae(fit, part) if len(part) else np.nan,
            }
            for bucket, part in buckets
        ]
    )


def swap_features(features: list[str], group: str) -> list[str]:
    """The swap test's columns for ``group``: its full feature set (traits and
    ``epa``) and only the ``epa`` column of every other group."""
    own = [name for name in features if name.startswith(f"{group}:")]
    others = [
        name
        for name in features
        if name.endswith(f":{headline.EPA}") and not name.startswith(f"{group}:")
    ]
    return others + own


def swap_maes(
    dev_matrix: pd.DataFrame,
    validation_rows: pd.DataFrame,
    dev_seasons: list[int],
    mae_v4_only: float,
) -> pd.DataFrame:
    """Per group: validation MAE of the dev fit on ``swap_features``."""
    rows = []
    for group in headline.GROUPS:
        features = swap_features(feature_columns(dev_matrix), group)
        fit = fit_on(dev_matrix, features, dev_seasons)
        rows.append(
            {
                "position_group": group,
                "mae_v4_only": mae_v4_only,
                "mae_with_traits": _mae(fit, validation_rows),
            }
        )
    return pd.DataFrame(rows)


def replacement_flags(swap: pd.DataFrame) -> pd.DataFrame:
    """``improvement = mae_v4_only − mae_with_traits``; the headline replaces the value
    only when it improves the MAE and is at least as reliable at six games.

    Both reliabilities come from ``paired_split_half`` (same athletes, same games). A
    NaN reliability compares False, so a group without a six-game v4 reliability
    (OL, K and P in v1) never replaces v4.
    """
    improvement = swap["mae_v4_only"] - swap["mae_with_traits"]
    return swap.assign(
        improvement=improvement,
        replaces_value=improvement.gt(0)
        & swap["reliability_6_headline"].ge(swap["reliability_6_v4"]),
    )


# Split-half reliability of v4 and the headline


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
    """Pearson correlation; NaN when either side is constant (say, a group whose every
    headline feature was dropped)."""
    if not (left.std() > 0 and right.std() > 0):
        return np.nan
    return float(left.corr(right))


def _season_keys(values: pd.Series, season: int) -> pd.Series:
    return f"{season}:" + values.astype(str)


def paired_games(frames: pd.DataFrame, credited: pd.DataFrame, n: int) -> pd.DataFrame:
    """``athlete_id, game_id, half``: the games both sides of the reliability use at ``n``.

    Each athlete's games are every distinct game with a trait-frame row (any trait,
    NaN numerator included), alternating between halves in week order; an athlete
    needs ``n`` such games and each half keeps its first ``n // 2``. An athlete with
    no credited row in those games has no v4 value and is left out of both sides.
    """
    chosen = _half_games(_alternate(frames), n)
    keys = credited[["athlete_id", "game_id"]].drop_duplicates()
    with_credit = chosen.merge(keys, on=["athlete_id", "game_id"])["athlete_id"]
    return chosen[chosen["athlete_id"].isin(set(with_credit))]


def _half_scores(
    frames: pd.DataFrame,
    credited: pd.DataFrame,
    groups: pd.Series,
    chosen: pd.DataFrame,
    n: int,
    fit: headline.HeadlineFit,
) -> pd.DataFrame:
    """Each chosen athlete's paired half values, indexed by athlete.

    Columns ``("headline", h)`` hold the half's ``raw_score`` and ``("v4", h)`` its
    adjusted EPA per game (0 for a chosen game without credit) for half ``h``. Each
    half is scored as its own snapshot week: trait values by the registered aggregate
    over the half's games and ``epa`` from the same per-game value. ``latest`` traits
    have no games and score 0.
    """
    rows = frames.merge(chosen, on=["athlete_id", "game_id"])
    rows = rows.assign(
        aggregate=rows["trait"].map(
            {name: spec.aggregate for name, spec in TRAITS.items()}
        )
    )
    values = snapshots.aggregate(rows, ["athlete_id", "trait", "half"]).reset_index()
    half_snapshot = values.assign(
        week=values["half"],
        position_group=values["athlete_id"].map(groups),
        games=n // 2,
        qualified=True,
    ).dropna(subset=["position_group"])

    players = chosen.drop_duplicates(["athlete_id", "half"])[["athlete_id", "half"]]
    epa = chosen.merge(credited, on=["athlete_id", "game_id"], how="left")
    epa = (
        epa.groupby(["athlete_id", "half"])["adjusted_epa"]
        .sum()
        .rename("total")
        .reset_index()
    )
    half_values = players.merge(epa, on=["athlete_id", "half"], how="left").assign(
        week=lambda f: f["half"],
        position_group=lambda f: f["athlete_id"].map(groups),
        games=n // 2,
        shrunk_per_game=lambda f: f["total"].fillna(0.0) / (n // 2),
    )
    columns = {}
    for half in (0, 1):
        scored = headline.score_players(
            half_snapshot[half_snapshot["week"].eq(half)],
            half_values[half_values["week"].eq(half)],
            fit,
        )
        columns[("headline", half)] = scored.drop_duplicates("athlete_id").set_index(
            "athlete_id"
        )["raw_score"]
        columns[("v4", half)] = half_values[half_values["week"].eq(half)].set_index(
            "athlete_id"
        )["shrunk_per_game"]
    return pd.DataFrame(columns)


def paired_split_half(
    seasons: dict[int, dict[str, pd.DataFrame]],
    fit: headline.HeadlineFit,
    games_at: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split-half reliability of the headline and of v4 on one paired sample.

    For each ``n`` both statistics use the same (season, athlete) set and the same
    games (``paired_games``): the headline side correlates the halves' ``raw_score``
    under ``fit``, the v4 side the halves' adjusted EPA per game. Dev seasons are
    pooled with season-prefixed ids, a group needs ``MIN_QUALIFIED_PLAYERS`` athletes
    with both sides in both halves, and each group takes its athletes' snapshot
    group. Returns ``(headline_rows, v4_rows)`` with equal ``players``.
    """
    by_games: dict[int, list[pd.DataFrame]] = {n: [] for n in games_at}
    groups_pooled = []
    for season, data in seasons.items():
        frames = data["frames"].astype({"athlete_id": str})
        credited = data["credited"].astype({"athlete_id": str})
        groups = (
            data["snapshots"]
            .astype({"athlete_id": str})
            .drop_duplicates("athlete_id")
            .set_index("athlete_id")["position_group"]
        )
        groups_pooled.append(
            groups.set_axis(_season_keys(groups.index.to_series(), season))
        )
        for n in games_at:
            chosen = paired_games(frames, credited, n)
            if n < 2 or chosen.empty:
                continue
            halves = _half_scores(frames, credited, groups, chosen, n, fit)
            by_games[n].append(
                halves.set_axis(_season_keys(halves.index.to_series(), season))
            )
    groups = pd.concat(groups_pooled) if groups_pooled else pd.Series(dtype=object)
    headline_rows, v4_rows = [], []
    for n, parts in by_games.items():
        if not parts:
            continue
        wide = pd.concat(parts).dropna()
        group = wide.index.to_series().map(groups)
        for name, frame in wide.groupby(group):
            if len(frame) < MIN_QUALIFIED_PLAYERS:
                continue
            for side, out in (("headline", headline_rows), ("v4", v4_rows)):
                out.append(
                    {
                        "position_group": name,
                        "games": n,
                        "players": len(frame),
                        "correlation": _pearson(frame[(side, 0)], frame[(side, 1)]),
                    }
                )
    return (
        pd.DataFrame(headline_rows, columns=SPLIT_COLUMNS),
        pd.DataFrame(v4_rows, columns=SPLIT_COLUMNS),
    )


def _at_games(table: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = table[table["games"].eq(REPLACEMENT_GAMES)]
    return rows.set_index("position_group")["correlation"].rename(name).reset_index()


def reliability_table(
    seasons: dict[int, dict[str, pd.DataFrame]],
    headline_rows: pd.DataFrame,
    v4_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Each dev season's trait reliability, then the pooled, paired headline and v4 rows."""
    parts = [
        data["reliability"].assign(season=str(season))
        for season, data in seasons.items()
    ]
    parts.append(headline_rows.assign(trait="headline", season="pooled"))
    parts.append(v4_rows.assign(trait=headline.EPA, season="pooled"))
    columns = ["season", *snapshots.RELIABILITY_COLUMNS]
    return pd.concat(
        [part[columns] for part in parts if len(part)], ignore_index=True
    ).reindex(columns=columns)


# Report-only tables


def portability_table(snapshots_by_season: dict[int, pd.DataFrame]) -> pd.DataFrame:
    matched = [
        match_transfers(
            read_season_source(season, "portal"),
            read_season_source(season, "roster"),
            read_season_source(season - 1, "roster"),
        )
        for season in sorted(snapshots_by_season)
        if season >= FIRST_PORTAL_SEASON
    ]
    transfers = (
        pd.concat(matched, ignore_index=True)
        if matched
        else pd.DataFrame(columns=TRANSFER_COLUMNS)
    )
    return portability(snapshots_by_season, transfers)


def _spearman(left: pd.Series, right: pd.Series) -> float:
    both = left.notna() & right.notna()
    if both.sum() < MIN_QUALIFIED_PLAYERS:
        return np.nan
    return _pearson(left[both].rank(), right[both].rank())


def final_week_values(
    snapshot: pd.DataFrame, values: pd.DataFrame, fit: headline.HeadlineFit
) -> pd.DataFrame:
    """``athlete_id, headline, value_above_replacement`` at each source's final week."""
    final = snapshot[snapshot["week"].eq(snapshot["week"].max())]
    final_values = values[values["week"].eq(values["week"].max())]
    scored = (
        headline.score_players(final, final_values, fit)
        .sort_values("games", ascending=False, kind="stable")
        .drop_duplicates("athlete_id")[["athlete_id", "headline"]]
    )
    v4 = final_values.drop_duplicates("athlete_id")[
        ["athlete_id", "value_above_replacement"]
    ]
    return scored.merge(v4, on="athlete_id", how="outer")


def draft_check(
    snapshots_by_season: dict[int, pd.DataFrame],
    values_by_season: dict[int, pd.DataFrame],
    fit: headline.HeadlineFit,
) -> pd.DataFrame:
    """Rank correlation of final-week headline and v4 value with draft pick and NFL snaps.

    Report-only: draft year ``y`` reads season ``y − 1``. Pick is negated so higher
    is better; NFL snaps count only for a mature (``observed``) outcome. Every
    correlation uses the players with both a headline and a v4 value. Dev seasons
    are scored in sample.
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
                snapshots_by_season[season], values_by_season[season], fit
            )
            parts.append(players.merge(scores, on="athlete_id", how="inner"))
    if not parts:
        log.info("draft check: no draft year has a season in the backtest")
        return pd.DataFrame(columns=DRAFT_COLUMNS)
    pooled = pd.concat(parts, ignore_index=True).dropna(
        subset=["headline", "value_above_replacement"]
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
                "spearman_pick_headline": _spearman(frame["headline"], -frame["pick"]),
                "spearman_nfl_games_v4": _spearman(
                    mature["value_above_replacement"],
                    mature["nfl_first3_scrimmage_snaps"],
                ),
                "spearman_nfl_games_headline": _spearman(
                    mature["headline"], mature["nfl_first3_scrimmage_snaps"]
                ),
            }
        )
    return pd.DataFrame(rows, columns=DRAFT_COLUMNS)


def weights_table(fit: headline.HeadlineFit) -> pd.DataFrame:
    kept = pd.DataFrame(
        {
            "feature": fit.weights.index,
            "weight": fit.weights.to_numpy(),
            "loso_sd": fit.loso_sd.to_numpy(),
            "dropped": False,
        }
    )
    dropped = pd.DataFrame(
        {"feature": fit.dropped, "weight": np.nan, "loso_sd": np.nan, "dropped": True}
    )
    return pd.concat([kept, dropped], ignore_index=True)


# The backtest


@dataclass
class BacktestResult:
    weights: pd.DataFrame
    validation: pd.DataFrame
    swap: pd.DataFrame
    reliability: pd.DataFrame
    portability: pd.DataFrame
    target_coverage: pd.DataFrame
    draft_check: pd.DataFrame
    fit: headline.HeadlineFit


def run_backtest(
    dev_seasons: list[int], validation_season: int, rebuild: bool = False
) -> BacktestResult:
    """Fit the headline on ``dev_seasons`` and judge it on the later ``validation_season``.

    The validation season's rows enter no fit, feature scale or reliability table;
    they are only scored. ``rebuild`` rebuilds cached season artifacts.
    """
    dev_seasons = sorted({int(season) for season in dev_seasons})
    if validation_season <= max(dev_seasons):
        raise ValueError(
            f"validation season {validation_season} must follow every dev season {dev_seasons}"
        )
    seasons = {}
    for season in [*dev_seasons, validation_season]:
        seasons[season] = {
            **load_season(season, rebuild),
            "values": player_values(season),
        }
    matrices = {
        season: headline.design_matrix(
            data["snapshots"], data["values"], load_weekly_games(season), season
        )
        for season, data in seasons.items()
    }
    dev_matrix = pd.concat(
        [matrices[season] for season in dev_seasons], ignore_index=True
    )
    validation_rows = matrices[validation_season]
    dev = {season: seasons[season] for season in dev_seasons}

    fit = fit_on(dev_matrix, feature_columns(dev_matrix), dev_seasons)
    v4_fit = fit_on(dev_matrix, epa_columns(dev_matrix), dev_seasons)
    validation = pd.concat(
        [
            validation_mae(fit, validation_rows, "traits"),
            validation_mae(v4_fit, validation_rows, "v4"),
        ],
        ignore_index=True,
    )

    headline_rows, v4_rows = paired_split_half(dev, fit, RELIABILITY_GAMES)
    swap = swap_maes(
        dev_matrix, validation_rows, dev_seasons, _mae(v4_fit, validation_rows)
    )
    swap = swap.merge(
        _at_games(v4_rows, "reliability_6_v4"), on="position_group", how="left"
    )
    swap = swap.merge(
        _at_games(headline_rows, "reliability_6_headline"),
        on="position_group",
        how="left",
    )
    swap = replacement_flags(swap)[SWAP_COLUMNS]

    fit.reliability = headline_rows.dropna(subset=["correlation"])[
        headline.RELIABILITY_COLUMNS
    ].reset_index(drop=True)
    fit.replaces_value = {
        group: bool(flag)
        for group, flag in zip(swap["position_group"], swap["replaces_value"])
    }
    snapshots_by_season = {
        season: data["snapshots"] for season, data in seasons.items()
    }
    return BacktestResult(
        weights=weights_table(fit),
        validation=validation,
        swap=swap,
        reliability=reliability_table(dev, headline_rows, v4_rows),
        portability=portability_table(snapshots_by_season),
        target_coverage=pd.concat(
            [data["target_coverage"] for data in seasons.values()], ignore_index=True
        ),
        draft_check=draft_check(
            snapshots_by_season,
            {season: data["values"] for season, data in seasons.items()},
            fit,
        ),
        fit=fit,
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


def write_backtest(
    result: BacktestResult, fit: headline.HeadlineFit, cutoff_season: int
) -> Path:
    """Save the fit JSON and the long evaluation frame, and print every table as Markdown."""
    path = headline.save_headline_fit(fit, headline.headline_fit_path(cutoff_season))
    store.write_processed(
        evaluation_frame(result), *traits_artifact(f"evaluation_{cutoff_season}")
    )
    for name in TABLES:
        print(f"\n## {name}\n\n{markdown_table(getattr(result, name))}")
    return path
