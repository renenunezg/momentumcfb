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
from backend.players.traits import MODEL_VERSION, backtest, expectations, headline
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
    "exposure",
    "uncertainty",
    "replaces_value",
    "in_headline",
    "fbs",
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


def frozen_expectation_tables(fit_path: Path) -> expectations.ExpectationTables:
    """The serving expectation tables saved with the fit at ``fit_path``."""
    path = expectations.expectation_tables_path(fit_path.parent, headline.fit_cutoff(fit_path))
    if not path.exists():
        raise FileNotFoundError(
            f"no frozen expectation tables {path.name} beside {fit_path.name} in "
            f"{fit_path.parent}; run position-traits-fit and restore both artifacts"
        )
    return expectations.load_expectation_tables(path)


def latest_headline_fit_path(season: int) -> Path:
    """The frozen headline fit with the greatest cutoff before ``season``."""
    return headline.latest_fit_path(store.PROCESSED_DIR / "players" / "models", season)


def headline_rows(
    snapshot: pd.DataFrame, player_values: pd.DataFrame, fit: headline.HeadlineFit
) -> pd.DataFrame:
    """One ``trait = "headline"`` snapshot-shaped row per scored player-week."""
    scored = pd.concat(
        [
            headline.score_players(
                snapshot[snapshot["week"].eq(week)],
                player_values[player_values["week"].eq(week)],
                fit,
            ).assign(week=week)
            for week in sorted(snapshot["week"].unique())
        ],
        ignore_index=True,
    )
    keys = ["week", "athlete_id", "position_group"]
    players = snapshot.groupby(keys, as_index=False).agg(
        qualified=("qualified", lambda flags: bool(flags.eq(True).any()))
    )
    identity = snapshot.drop_duplicates(keys)[
        [*keys, "season", "athlete_name", "team", "team_id"]
    ]
    rows = scored.merge(players, on=keys).merge(identity, on=keys)
    return rows.assign(
        trait="headline",
        value=rows["headline"],
        percentile=rows["headline_percentile"],
        kind="fitted",
    )


def run_position_traits(season: int, as_of: datetime) -> pd.DataFrame:
    """Score the season's weekly trait snapshots with the latest earlier headline fit.

    The fit and its expectation tables are frozen: they are only loaded, never refit,
    and a fit whose development seasons include ``season`` is refused; the fit's
    ``poor_coverage`` fixes the receptions-or-targets mode. Each row carries ``fbs``
    from the team's classification; headline rows exist for qualified FBS players
    only, and only FBS rows are published.
    """
    fit_path = latest_headline_fit_path(season)
    fit = headline.load_headline_fit(fit_path)
    headline.validate_headline_fit(fit, season)
    tables = frozen_expectation_tables(fit_path)
    teams = read_season_source(season, "teams")
    fbs_schools = set(teams.loc[teams["classification"].eq("fbs"), "school"])
    snapshot = backtest.build_season_traits(
        season, tables=tables, poor_coverage=fit.poor_coverage
    )["snapshots"].astype({"athlete_id": str})
    # The headline pool is qualified FBS players, so the flag precedes scoring.
    snapshot = snapshot.assign(fbs=snapshot["team"].isin(fbs_schools))
    values = backtest.player_values(season)
    scored = headline_rows(snapshot, values, fit)
    with_headline = headline.headline_groups(fit)
    traits = snapshot[[*TRAIT_SNAPSHOT_COLUMNS, "exposure"]].assign(uncertainty=np.nan)
    feature = traits["position_group"] + ":" + traits["trait"]
    traits["in_headline"] = (
        feature.isin(fit.weights.index)
        & ~feature.isin(fit.dropped)
        & traits["position_group"].isin(with_headline)
    )
    scored = scored.assign(in_headline=True, as_of=None, model_version=None)
    out = pd.concat(
        [traits, scored[[*TRAIT_SNAPSHOT_COLUMNS, "uncertainty", "in_headline"]]],
        ignore_index=True,
    )
    replaces = {
        group: bool(flag) and group in with_headline
        for group, flag in fit.replaces_value.items()
    }
    out = out.assign(
        replaces_value=out["position_group"].map(replaces).eq(True),
        as_of=as_of.isoformat(),
        model_version=MODEL_VERSION,
        fbs=out["team"].isin(fbs_schools),
    )[POSITION_TRAIT_COLUMNS]
    store.write_processed(out, *position_traits_artifact(season))
    without = [group for group in headline.GROUPS if group not in with_headline]
    log.info(
        f"position traits {season}: {scored['athlete_id'].nunique()} qualified FBS players "
        f"scored over {scored['week'].nunique()} weeks with the fit through "
        f"{max(fit.dev_seasons)}; {sum(replaces.values())} groups replace v4; "
        f"no headline (every feature dropped) at {', '.join(without) or 'no group'}"
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
        log.warning(f"position traits {season}: artifact has no fbs flag, not published")
        return pd.DataFrame(columns=POSITION_TRAIT_COLUMNS)
    latest = traits["week"].eq(traits["week"].max()) & traits["fbs"].eq(True)
    return traits[latest].reset_index(drop=True)
