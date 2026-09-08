"""Season runners that turn raw player sources into published artifacts."""

import hashlib
import json
import logging
from datetime import datetime

import pandas as pd

from backend.config import PROCESSED_DIR, RAW_DIR
from backend.etl import store
from backend.players import artifacts, heisman, value
from backend.players.credit import DEFENSE_ROLES, OFFENSE_SHARES
from backend.players.ingest import players_dir

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
    "opponent_effect_prior_games",
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


def heisman_board_artifact(season: int, week: int) -> tuple[str, ...]:
    return ("players", "heisman_board", f"{season}_{week:02d}.parquet")


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
                "opponent_effect_prior_games": value.OPPONENT_EFFECT_PRIOR_GAMES,
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
    }
