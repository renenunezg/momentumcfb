"""Player value and Heisman board command handlers."""

import logging
from argparse import Namespace

log = logging.getLogger(__name__)


def handle_ingest_players(args: Namespace) -> None:
    from backend.cfbd.client import CFBDClient
    from backend.players.ingest import ingest_player_sources

    client = CFBDClient()
    for season in args.seasons:
        manifest = ingest_player_sources(client, season, only_week=args.week)
        log.info(f"players {season}: {len(manifest)} sources snapshotted")


def handle_player_values(args: Namespace) -> None:
    from datetime import datetime, timezone

    from backend.players.pipeline import run_player_values

    for season in args.seasons:
        values = run_player_values(season, datetime.now(timezone.utc))
        latest = values[values["week"].eq(values["week"].max())]
        log.info(
            f"player values {season}: {values['athlete_id'].nunique()} players "
            f"through week {int(values['week'].max())}"
        )
        log.info(
            latest.head(10)[
                [
                    "overall_rank",
                    "athlete_name",
                    "team",
                    "position_group",
                    "plays",
                    "value_above_replacement",
                    "wpa",
                ]
            ].to_string(index=False)
        )


def handle_heisman(args: Namespace) -> None:
    from datetime import datetime, timezone

    from backend.players.pipeline import run_heisman

    board = run_heisman(args.season, args.week, datetime.now(timezone.utc))
    log.info(
        board.head(10)[
            [
                "predicted_rank",
                "athlete_name",
                "team",
                "position",
                "predicted_share",
                "value_rank",
            ]
        ].to_string(index=False)
    )


def handle_publish_players(args: Namespace) -> None:
    from backend.publish import publish_players

    stored = publish_players(args.season)
    for table, count in stored.items():
        log.info(f"cfb.{table}: {count} rows stored")
