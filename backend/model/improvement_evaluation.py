"""Small, fixed historical experiments for weekly scoring diagnostics.

Uses cached fall regular-season D1 games, with prior-season carryover means.
Rich preseason inputs are unavailable historically, so this research contract
cannot by itself authorize changing the live production priors.
2020-2022 selects candidates; 2023-2025 is historical validation, not a newly
untouched holdout. No 2026 outcomes enter fitting or selection.
"""

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backend.features.scoring import SCORING_COLUMNS
from backend.grading import _classification
from backend.model.calibration import _add_proper_scores, walk_forward_season
from backend.model.joint_scoring import DEFAULT_CONFIG, fit_joint_scoring
from backend.model.preseason import build_historical_carryover_priors
from backend.model.scoring_calibration import bounded_model_total
from backend.model.weekly import load_weekly_games

log = logging.getLogger(__name__)


def paired_validation_effects(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pair games, then resample season-week blocks to retain slate correlation."""
    validation = predictions[predictions.season.ge(2023)]
    baseline = validation[validation.candidate.eq("baseline")]
    rng = np.random.default_rng(20260914)
    rows = []
    for candidate in ("tighter_prior", "matchup_total_offset"):
        paired = validation[validation.candidate.eq(candidate)].merge(
            baseline,
            on=["season", "game_id"],
            suffixes=("", "_baseline"),
            validate="one_to_one",
        )
        if len(paired) != len(baseline):
            raise ValueError("candidate and baseline populations differ")
        for name, cohort in [("all", paired), *list(paired.groupby("matchup"))]:
            for metric in ("total_error", "margin_error", "joint_margin_total_nll"):
                values = cohort[metric]
                reference = cohort[f"{metric}_baseline"]
                delta = (
                    values - reference
                    if metric.endswith("nll")
                    else values.abs() - reference.abs()
                )
                blocks = (
                    cohort.assign(delta=delta)
                    .groupby(["season", "model_week"])
                    .delta.agg(["sum", "count"])
                )
                sampled = rng.integers(0, len(blocks), size=(10000, len(blocks)))
                effects = blocks["sum"].to_numpy()[sampled].sum(axis=1) / blocks[
                    "count"
                ].to_numpy()[sampled].sum(axis=1)
                lower, upper = np.quantile(effects, [0.025, 0.975])
                rows.append(
                    dict(
                        candidate=candidate,
                        matchup=name,
                        metric=metric,
                        games=len(cohort),
                        blocks=len(blocks),
                        delta=delta.mean(),
                        lower_95=lower,
                        upper_95=upper,
                    )
                )
    return pd.DataFrame(rows)


def historical_games(season: int) -> pd.DataFrame:
    games = load_weekly_games(season)
    # Exclude spring 2020 FCS and postseason provider-week ordering anomalies.
    return (
        games[
            games.season_type.eq("regular")
            & games.start_date.dt.year.eq(season)
            & games.start_date.dt.month.ge(8)
            & games.completed.fillna(False)
        ]
        .dropna(subset=SCORING_COLUMNS)
        .reset_index(drop=True)
    )


def _enrich(predictions: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    context = games[
        ["game_id", "home_classification", "away_classification", "game_possessions"]
    ]
    out = predictions.merge(context, on="game_id", validate="one_to_one")
    out["matchup"] = _classification(out)
    expected_ppp = out.model_total / out.expected_game_possessions
    out["pace_error_points"] = (
        out.expected_game_possessions - out.game_possessions
    ) * expected_ppp
    out["efficiency_error_points"] = (
        out.game_possessions * expected_ppp - out.actual_total
    )
    return out


def evaluate_weekly_improvements(destination: Path) -> pd.DataFrame:
    destination.mkdir(parents=True, exist_ok=True)
    reference_config = replace(DEFAULT_CONFIG, matchup_total_calibration=False)
    games_by_season = {year: historical_games(year) for year in range(2019, 2026)}
    frames = []
    for year in range(2020, 2026):
        previous = games_by_season[year - 1]
        fitted = fit_joint_scoring(
            previous,
            int(previous.model_week.max()) + 1,
            (previous.start_date.max() + pd.Timedelta(seconds=1)).to_pydatetime(),
            reference_config,
        )
        games = games_by_season[year]
        means = build_historical_carryover_priors(fitted, games)
        for name, config in (
            ("baseline", reference_config),
            (
                "tighter_prior",
                replace(
                    reference_config,
                    strength_prior_sd_ppp=DEFAULT_CONFIG.strength_prior_sd_ppp * 0.85,
                ),
            ),
        ):
            log.info("Historical replay %s %s (%s games)", year, name, len(games))
            prediction = _enrich(walk_forward_season(games, config, means), games)
            prediction["candidate"] = name
            frames.append(prediction)
    predictions = pd.concat(frames, ignore_index=True)
    baseline = predictions[predictions.candidate.eq("baseline")].copy()
    development = baseline[baseline.season.le(2022)]
    # A fixed 100-game zero-centered shrinkage prior limits sparse corrections.
    offsets = development.groupby("matchup").total_error.agg(["sum", "count"])
    offsets["total_adjustment"] = -offsets["sum"] / (offsets["count"] + 100)
    offsets.to_parquet(destination / "totals_offsets.parquet")
    corrected = baseline.copy()
    change = corrected.matchup.map(offsets.total_adjustment).fillna(0)
    corrected["model_total"] = bounded_model_total(
        corrected.model_total, corrected.home_margin, change
    )
    corrected["expected_home_points"] = (
        corrected.model_total + corrected.home_margin
    ) / 2
    corrected["expected_away_points"] = (
        corrected.model_total - corrected.home_margin
    ) / 2
    corrected["home_score_error"] = (
        corrected.expected_home_points - corrected.actual_home_points
    )
    corrected["away_score_error"] = (
        corrected.expected_away_points - corrected.actual_away_points
    )
    corrected["total_error"] = corrected.model_total - corrected.actual_total
    expected_ppp = corrected.model_total / corrected.expected_game_possessions
    corrected["pace_error_points"] = (
        corrected.expected_game_possessions - corrected.game_possessions
    ) * expected_ppp
    corrected["efficiency_error_points"] = (
        corrected.game_possessions * expected_ppp - corrected.actual_total
    )
    corrected["candidate"] = "matchup_total_offset"
    corrected = _add_proper_scores(corrected)
    predictions = pd.concat([predictions, corrected], ignore_index=True)
    predictions["split"] = np.where(
        predictions.season.le(2022), "development", "validation"
    )
    predictions["evaluation_contract"] = "fall_d1_historical_carryover_v1"
    predictions.to_parquet(destination / "predictions.parquet", index=False)
    rows = []
    for (split, candidate), frame in predictions.groupby(["split", "candidate"]):
        for group, cohort in [("all", frame), *list(frame.groupby("matchup"))]:
            for period, subset in [
                ("all_weeks", cohort),
                ("early_1_3", cohort[cohort.model_week.le(3)]),
            ]:
                if subset.empty:
                    continue
                rows.append(
                    {
                        "split": split,
                        "candidate": candidate,
                        "matchup": group,
                        "period": period,
                        "games": len(subset),
                        "margin_mae": subset.margin_error.abs().mean(),
                        "total_mae": subset.total_error.abs().mean(),
                        "margin_bias": subset.margin_error.mean(),
                        "total_bias": subset.total_error.mean(),
                        "joint_nll": subset.joint_margin_total_nll.mean(),
                        "pace_error_points": subset.pace_error_points.mean(),
                        "efficiency_error_points": subset.efficiency_error_points.mean(),
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_parquet(destination / "summary.parquet", index=False)
    effects = paired_validation_effects(predictions)
    effects.to_parquet(destination / "paired_validation.parquet", index=False)
    manifest = {
        "contract": "fall_d1_historical_carryover_v1",
        "development_seasons": [2020, 2021, 2022],
        "validation_seasons": [2023, 2024, 2025],
        "baseline_config": {
            key: str(value) if not np.isfinite(value) else value
            for key, value in asdict(reference_config).items()
        },
        "candidate_prior_sd_multiplier": 0.85,
        "totals_offset_zero_prior_games": 100,
        "available_scoring_games": {
            str(year): len(frame) for year, frame in games_by_season.items()
        },
        "limitations": [
            "Only fall regular-season D1 games with cached scoring features are included.",
            "Prior means use previous-season fits; rich production preseason inputs are unavailable.",
            "Historical validation seasons have been used in earlier model research.",
            "No 2026 outcomes are used for fitting, selection or validation.",
            "This command does not promote candidates or change frozen production forecasts.",
        ],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("%s", summary.query("matchup == 'all'").to_string(index=False))
    return summary
