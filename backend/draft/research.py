"""Source coverage and outcome cohorts, not a calibrated draft forecast.

Current college production, historical NFL opportunity, and roster evidence
remain separate. Missing measurements never become zero prospect grades.
"""

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from backend.draft.sources import receipt


def ids(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(r"\.0$", "", regex=True)


def name_key(value) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", text.lower())
    return re.sub(r"[^a-z]", "", text)


class Inputs:
    def __init__(self):
        self.sources: dict[str, dict] = {}

    def parquet(self, path: Path) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        source = {**receipt(path), "rows": len(frame), "columns": list(frame)}
        metadata = path.with_suffix(".source.json")
        if metadata.exists():
            source["retrieval"] = json.loads(metadata.read_text())
        self.sources[str(path.resolve())] = source
        return frame

    def picks(self, path: Path) -> pd.DataFrame:
        payload = json.loads(path.read_text())
        frame = pd.DataFrame(payload["rows"])
        self.sources[str(path.resolve())] = {
            **receipt(path),
            "rows": len(frame),
            "fetched_at": payload["fetched_at"],
            "source": payload["source"],
        }
        return frame


def watchlist(roster, catalog, values, as_of):
    roster = roster.copy()
    roster["athlete_id"] = ids(roster["id"])
    ambiguous = roster.athlete_id.duplicated(keep=False) | roster.athlete_id.isna()
    fbs = set(catalog.loc[catalog.classification.eq("fbs"), "school"])
    board = roster.loc[
        ~ambiguous & roster.team.isin(fbs),
        [
            "athlete_id",
            "first_name",
            "last_name",
            "team",
            "position",
            "year",
            "height",
            "weight",
        ],
    ].copy()
    board["athlete_name"] = board.first_name + " " + board.last_name
    board["class_year"] = board.year.where(board.year.between(1, 6))
    board["eligibility_status"] = "unverified"
    board["declaration_status"] = "unknown"
    board = board.drop(columns=["first_name", "last_name", "year"])

    values = values.copy()
    values["athlete_id"] = ids(values.athlete_id)
    values = values[pd.to_datetime(values.as_of, utc=True).le(as_of)]
    if values.empty:
        raise ValueError("No college player values available at the requested cutoff")
    latest_week = int(values.week.max())
    latest = values[values.week.eq(latest_week)].copy()
    if latest.athlete_id.duplicated().any():
        raise ValueError("Duplicate college player IDs in the latest value snapshot")
    latest = latest.rename(columns={"team": "value_team", "position": "value_position"})
    keep = [
        "athlete_id",
        "value_team",
        "value_position",
        "position_group",
        "week",
        "as_of",
        "model_version",
        "games",
        "value_above_replacement",
        "position_rank",
    ]
    board = board.merge(
        latest[keep], on="athlete_id", how="left", validate="one_to_one"
    )
    board["production_status"] = np.select(
        [board.week.isna(), board.team.ne(board.value_team)],
        ["no_individual_value", "team_mismatch"],
        default="observed",
    )
    board["nfl_potential"] = np.nan
    board["projected_pick"] = np.nan
    board["consensus_rank"] = np.nan
    return board, {
        "scope": "FBS roster exploration; draft eligibility is unverified",
        "players": len(board),
        "college_week": latest_week,
        "college_as_of": str(latest.as_of.max()),
        "college_snapshot_age_days": round(
            (as_of - pd.to_datetime(latest.as_of, utc=True).max()).total_seconds()
            / 86400,
            2,
        ),
        "ambiguous_roster_rows_quarantined": int(ambiguous.sum()),
        "invalid_class_years_all_rosters": int((~roster.year.between(1, 6)).sum()),
        "invalid_class_years_fbs": int(board.class_year.isna().sum()),
        "production_status": board.production_status.value_counts().to_dict(),
        "position_coverage": board.groupby("position", dropna=False)
        .agg(
            players=("athlete_id", "size"),
            measured=("value_above_replacement", "count"),
        )
        .reset_index()
        .to_dict("records"),
    }


def historical_cohort(inputs, cache, cfb_root, completed_season, nfl_root):
    files = sorted(cache.glob("cfbd_picks_*.json"))
    if not files:
        raise ValueError("Snapshot CFBD draft history before building research")
    cfbd = pd.concat([inputs.picks(path) for path in files], ignore_index=True)
    cfbd = cfbd.rename(
        columns={
            "year": "draft_year",
            "pick": "round_pick",
            "overall": "pick",
            "name": "college_name",
        }
    )
    cfbd["athlete_id"] = ids(cfbd.collegeAthleteId)
    nfl = inputs.parquet(cache / "draft_picks.parquet").rename(
        columns={"season": "draft_year", "position": "draft_position"}
    )
    # Draft year + overall pick is a natural bridge, verified independently by name.
    # CFBD nflAthleteId and nflverse gsis_id are different namespaces.
    joined = cfbd.merge(
        nfl[
            [
                "draft_year",
                "pick",
                "pfr_player_name",
                "gsis_id",
                "pfr_player_id",
                "draft_position",
                "team",
            ]
        ],
        on=["draft_year", "pick"],
        how="left",
        validate="one_to_one",
    )
    joined["identity_verified"] = (
        joined.college_name.map(name_key).eq(joined.pfr_player_name.map(name_key))
        & joined.pfr_player_name.notna()
    )
    joined.loc[~joined.identity_verified, ["gsis_id", "pfr_player_id"]] = None
    frames = []
    for year, picks in joined.groupby("draft_year"):
        path = cfb_root / "processed/players/player_values" / f"{year - 1}.parquet"
        picks = picks.copy()
        if path.exists():
            values = inputs.parquet(path)
            values["athlete_id"] = ids(values.athlete_id)
            if values.duplicated(["week", "athlete_id"]).any():
                raise ValueError(f"Duplicate historical player snapshot in {path}")
            # Retain players whose last observed snapshot predates the final week.
            values = values.sort_values("week").drop_duplicates(
                "athlete_id", keep="last"
            )
            columns = [
                "athlete_id",
                "model_version",
                "week",
                "as_of",
                "games",
                "value_above_replacement",
                "position_rank",
                "position_group",
            ]
            picks = picks.merge(
                values[columns], on="athlete_id", how="left", validate="many_to_one"
            )
        frames.append(picks)
    cohort = pd.concat(frames, ignore_index=True)
    combine = (
        inputs.parquet(cache / "combine.parquet")
        .drop(columns="draft_year")
        .rename(columns={"season": "draft_year", "pfr_id": "pfr_player_id"})
    )
    combine = combine[combine.pfr_player_id.notna()]
    duplicate = combine.duplicated(["draft_year", "pfr_player_id"], keep=False)
    measurements = [
        "ht",
        "wt",
        "forty",
        "bench",
        "vertical",
        "broad_jump",
        "cone",
        "shuttle",
    ]
    cohort = cohort.merge(
        combine.loc[~duplicate, ["draft_year", "pfr_player_id", *measurements]],
        on=["draft_year", "pfr_player_id"],
        how="left",
        validate="many_to_one",
    )
    cohort["nfl_first3_scrimmage_snaps"] = np.nan
    cohort["nfl_first3_special_teams_snaps"] = np.nan
    cohort["outcome_status"] = "immature"
    coverage = {}
    outcomes = {}
    first_year = int(cohort.draft_year.min())
    for season in range(first_year, completed_season + 1):
        snap_path = nfl_root / f"raw/awards/snaps/{season}.parquet"
        game_path = nfl_root / f"raw/awards/schedules/{season}.parquet"
        if not snap_path.exists() or not game_path.exists():
            coverage[str(season)] = {"complete": False, "reason": "missing_file"}
            continue
        snaps = inputs.parquet(snap_path)
        games = inputs.parquet(game_path)
        if snaps.empty or games.empty:
            coverage[str(season)] = {"complete": False, "reason": "empty_file"}
            continue
        regular = games[games.game_type.eq("REG")]
        expected = set(zip(regular.game_id, regular.home_team)) | set(
            zip(regular.game_id, regular.away_team)
        )
        snaps = snaps[snaps.game_type.eq("REG")]
        actual = set(zip(snaps.game_id, snaps.team))
        missing = expected - actual
        duplicate = snaps.duplicated(["game_id", "team", "pfr_player_id"]).any()
        complete = bool(expected) and not missing and not duplicate
        coverage[str(season)] = {
            "complete": complete,
            "expected_team_games": len(expected),
            "missing_team_games": len(missing),
            "duplicate_player_games": bool(duplicate),
        }
        if complete:
            outcomes[season] = snaps.groupby("pfr_player_id")[
                ["offense_snaps", "defense_snaps", "st_snaps"]
            ].sum()
    for year, rows in cohort.groupby("draft_year"):
        window = list(range(int(year), int(year) + 3))
        if window[-1] > completed_season:
            continue
        cohort.loc[rows.index, "outcome_status"] = "incomplete_sources"
        if not all(y in outcomes for y in window):
            continue
        cohort.loc[rows.index, "outcome_status"] = "unresolved_identity"
        total = pd.concat([outcomes[y] for y in window]).groupby(level=0).sum()
        eligible = rows[rows.identity_verified & rows.pfr_player_id.notna()]
        aligned = total.reindex(eligible.pfr_player_id).fillna(0)
        cohort.loc[eligible.index, "nfl_first3_scrimmage_snaps"] = (
            aligned.offense_snaps + aligned.defense_snaps
        ).to_numpy()
        cohort.loc[eligible.index, "nfl_first3_special_teams_snaps"] = (
            aligned.st_snaps.to_numpy()
        )
        cohort.loc[eligible.index, "outcome_status"] = "observed"
    summary = {
        "players": len(cohort),
        "identity_verified": int(cohort.identity_verified.sum()),
        "identity_review": cohort.loc[
            ~cohort.identity_verified,
            ["draft_year", "pick", "college_name", "pfr_player_name"],
        ].to_dict("records"),
        "college_value_matched": int(cohort.value_above_replacement.notna().sum()),
        "college_model_versions": cohort.model_version.value_counts(
            dropna=False
        ).to_dict(),
        "outcome_status": cohort.outcome_status.value_counts().to_dict(),
        "outcome_definition": "Observed regular-season scrimmage snaps in NFL years 1-3; opportunity, not player quality",
        "outcome_coverage": coverage,
        "coverage_by_position": cohort.groupby("draft_position")
        .agg(
            drafted=("pick", "size"),
            college_value=("value_above_replacement", "count"),
            forty=("forty", "count"),
            mature_outcomes=("nfl_first3_scrimmage_snaps", "count"),
        )
        .reset_index()
        .to_dict("records"),
        "training_ready": False,
        "training_blockers": [
            "Drafted-only sample excludes undrafted entrants",
            "Mixed college player-value model versions require a common rebuild",
            "Historical values are reconstructed, not original pre-draft publications",
            "Pre-draft rankings in draft records lack verified publication timestamps",
            "Snap totals measure opportunity and are affected by draft investment and team situation",
        ],
    }
    # Keep identifiers and permitted evidence, not unused biography or career totals.
    columns = [
        "draft_year",
        "pick",
        "round",
        "college_name",
        "collegeTeam",
        "athlete_id",
        "draft_position",
        "team",
        "identity_verified",
        "gsis_id",
        "pfr_player_id",
        "model_version",
        "week",
        "as_of",
        "games",
        "value_above_replacement",
        "position_rank",
        "position_group",
        *measurements,
        "nfl_first3_scrimmage_snaps",
        "nfl_first3_special_teams_snaps",
        "outcome_status",
    ]
    return cohort.reindex(columns=columns), summary


def team_context(inputs, cache, season, as_of):
    depth = inputs.parquet(cache / f"depth_charts_{season}.parquet")
    depth["depth_as_of"] = pd.to_datetime(depth.dt, utc=True)
    depth = depth[depth.depth_as_of.le(as_of)].copy()
    if depth.empty:
        raise ValueError("No NFL depth chart available at the requested cutoff")
    latest = depth.groupby("team").depth_as_of.transform("max")
    depth = depth[depth.depth_as_of.eq(latest)].copy()
    depth = depth.drop_duplicates(
        ["team", "pos_grp", "pos_slot", "pos_rank", "espn_id"]
    )
    contracts = inputs.parquet(cache / "historical_contracts.parquet")
    active = contracts[contracts.is_active.eq(True) & contracts.gsis_id.notna()].copy()
    ambiguous = active.gsis_id.duplicated(keep=False)
    contract_columns = ["gsis_id", "year_signed", "years", "apy_cap_pct"]
    depth = depth.merge(
        active.loc[~ambiguous, contract_columns],
        on="gsis_id",
        how="left",
        validate="many_to_one",
    )
    depth["need_grade"] = np.nan
    depth["need_status"] = "not_evaluated"
    depth["depth_age_days"] = (as_of - depth.depth_as_of).dt.total_seconds() / 86400
    selected = [
        "team",
        "pos_grp",
        "pos_name",
        "pos_abb",
        "pos_slot",
        "pos_rank",
        "player_name",
        "gsis_id",
        "espn_id",
        "depth_as_of",
        "depth_age_days",
        "year_signed",
        "years",
        "apy_cap_pct",
        "need_grade",
        "need_status",
    ]
    snaps = inputs.parquet(cache / f"snap_counts_{season}.parquet")
    roster = inputs.parquet(cache / f"roster_{season}.parquet")
    return depth[selected], {
        "teams": int(depth.team.nunique()),
        "depth_rows": len(depth),
        "earliest_team_snapshot": str(depth.depth_as_of.min()),
        "latest_team_snapshot": str(depth.depth_as_of.max()),
        "active_contract_matches": int(depth.year_signed.notna().sum()),
        "ambiguous_active_contract_rows": int(ambiguous.sum()),
        "current_snap_rows": len(snaps),
        "current_snap_max_week": int(snaps.week.max()) if len(snaps) else None,
        "current_roster_rows": len(roster),
        "needs_ready": False,
        "limitations": [
            "Depth charts describe listed roles, not proven starter quality",
            "Contract snapshots are current observations, not historical as-of records",
            "Signed year plus years does not establish expiry, options, void years or exit cost",
            "Future pick ownership and trade prices have not been sourced",
        ],
    }


def build(cfb_root, nfl_root, cache, output, season, as_of):
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise ValueError("as-of must include an explicit timezone")
    as_of = as_of.tz_convert("UTC")
    inputs = Inputs()
    board, board_audit = watchlist(
        inputs.parquet(cfb_root / f"raw/players/{season}/roster.parquet"),
        inputs.parquet(cfb_root / f"raw/players/{season}/teams.parquet"),
        inputs.parquet(cfb_root / f"processed/players/player_values/{season}.parquet"),
        as_of,
    )
    # A calendar-year change does not complete the NFL regular season/postseason.
    completed_season = min(season - 1, as_of.year - (2 if as_of.month < 3 else 1))
    cohort, history_audit = historical_cohort(
        inputs, cache, cfb_root, completed_season, nfl_root
    )
    teams, team_audit = team_context(inputs, cache, season, as_of)
    audit = {
        "schema_version": "draft_research_v1",
        "as_of": as_of.isoformat(),
        "draft_year": season + 1,
        "publication_ready": False,
        "watchlist": board_audit,
        "historical_cohort": history_audit,
        "team_context": team_audit,
        "sources": list(inputs.sources.values()),
    }
    output.mkdir(parents=True, exist_ok=True)
    board.to_parquet(output / "watchlist.parquet", index=False)
    cohort.to_parquet(output / "historical_cohort.parquet", index=False)
    teams.to_parquet(output / "team_context.parquet", index=False)
    # Pandas handles timestamps, nullable scalars, and NaN as JSON null.
    clean = json.loads(pd.Series([audit]).to_json(orient="values", date_format="iso"))[
        0
    ]
    (output / "audit.json").write_text(json.dumps(clean, indent=2, allow_nan=False))
    return clean
