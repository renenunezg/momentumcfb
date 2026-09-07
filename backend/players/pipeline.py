"""Season runners that turn raw player sources into published artifacts."""

import json
import logging
from datetime import datetime

import pandas as pd

from backend.config import PROCESSED_DIR
from backend.etl import store
from backend.players import heisman, value
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


def _final_snapshot(season: int) -> pd.DataFrame | None:
    try:
        values = store.read_processed(*player_values_artifact(season))
    except FileNotFoundError:
        return None
    return values[values["week"].eq(values["week"].max())]


def _seed_seasons(seed: pd.DataFrame, before: int) -> list[int]:
    return sorted(
        int(season)
        for season in seed["season"].unique()
        if season < before and players_dir(int(season)).is_dir()
    )


def run_heisman(season: int, week: int | None, as_of: datetime) -> pd.DataFrame:
    """Fit the share model on past ballots, validate it, and build the board.

    The current season is never in the training set, so the board it
    produces is a genuine forecast rather than a fit to the season's votes.
    """
    seed = heisman.load_seed()
    training_seasons = _seed_seasons(seed, season)
    rows = heisman.training_rows(training_seasons, seed)
    model = heisman.fit_share_model(rows)
    history = heisman.leave_one_season_out(rows)

    winner_ranks = []
    for record in history.itertuples(index=False):
        snapshot = _final_snapshot(record.season)
        rank = None
        if snapshot is not None:
            winner = rows[
                rows["season"].eq(record.season)
                & rows["athlete_name"].eq(record.actual_winner)
            ]
            hit = snapshot[snapshot["athlete_id"].isin(winner["athlete_id"])]
            if not hit.empty:
                rank = int(hit["overall_rank"].iloc[0])
        winner_ranks.append(rank)
    history["winner_value_rank"] = pd.array(winner_ranks, dtype="Int64")
    history = history[heisman.HISTORY_COLUMNS]
    store.write_processed(history, *HISTORY_ARTIFACT)

    try:
        player_values = store.read_processed(*player_values_artifact(season))
    except FileNotFoundError:
        player_values = None
    if week is None:
        if player_values is None or player_values.empty:
            raise ValueError(f"no player values for {season}; run player-values first")
        week = int(player_values["week"].max())
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
            }
        ],
        columns=META_COLUMNS,
    )
    store.write_processed(meta, *META_ARTIFACT)
    log.info(
        f"heisman {season} week {week}: {len(board)} candidates; "
        f"leave-one-season-out winner hit rate "
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
