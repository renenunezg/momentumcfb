"""Cached chronological checks of pure priors and market-line uncertainty.

2020-2022 is development; 2023-2025 is reused historical validation.
The pure-prior experiment uses historical carryover means, not reconstructed
rich preseason inputs. It cannot authorize a production prior change alone.
Market coverage evaluates future closing lines, not unobserved team strength
or realized scores. No 2026 results or provider requests enter this module.
"""

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from backend.etl import store
from backend.model.calibration import walk_forward_season
from backend.model.improvement_evaluation import historical_games
from backend.model.joint_scoring import DEFAULT_CONFIG, _team_catalog, fit_joint_scoring
from backend.model.market_history import (
    _fit_market_rating,
    fit_market_history,
    lined_games,
)
from backend.model.preseason import build_historical_carryover_priors
from backend.model.weekly import load_weekly_games

log = logging.getLogger(__name__)


def _pure_predictions() -> pd.DataFrame:
    frames = []
    previous = historical_games(2019)
    for year in range(2020, 2026):
        games = historical_games(year)
        fitted = fit_joint_scoring(
            previous,
            int(previous.model_week.max()) + 1,
            (previous.start_date.max() + pd.Timedelta(days=1)).to_pydatetime(),
        )
        means = build_historical_carryover_priors(fitted, games)
        for name, config in (
            ("baseline", DEFAULT_CONFIG),
            (
                "extra_prior_tightening",
                replace(
                    DEFAULT_CONFIG,
                    strength_prior_sd_ppp=DEFAULT_CONFIG.strength_prior_sd_ppp
                    / np.sqrt(1 + DEFAULT_CONFIG.strength_prior_correlation),
                ),
            ),
        ):
            prediction = walk_forward_season(games, config, means)
            prediction["candidate"] = name
            prediction = prediction.merge(
                games[["game_id", "home_classification", "away_classification"]],
                on="game_id",
                validate="one_to_one",
            )
            frames.append(prediction)
            log.info(
                "Pure uncertainty replay %s %s: %s games", year, name, len(prediction)
            )
        previous = games
    return pd.concat(frames, ignore_index=True)


def _market_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    carried: dict[str, float] = {}
    frames, ratings = [], []
    for year in range(2019, 2026):
        games = load_weekly_games(year)
        lines = store.read_lines(year)
        all_lined = lined_games(games, lines)
        # Score only chronological fall regular-season slates. The prior chain
        # retains all prior-season lines, exactly as build_market_prior does.
        targets = all_lined[
            all_lined.season_type.eq("regular")
            & all_lined.start_date.dt.year.eq(year)
            & all_lined.start_date.dt.month.ge(8)
        ]
        if year >= 2020:
            prior = pd.DataFrame(
                {"team": list(carried), "market_rating": list(carried.values())}
            )
            for week in sorted(targets.model_week.unique()):
                target = targets[targets.model_week.eq(week)]
                cutoff = target.start_date.min() - pd.Timedelta(hours=8)
                fitted = fit_market_history(games, lines, int(week), cutoff, prior)
                if fitted is None:
                    continue
                forecast = fitted.project(target)
                out = target[
                    [
                        "game_id",
                        "season",
                        "model_week",
                        "home_classification",
                        "away_classification",
                        "closing_spread",
                    ]
                ].merge(
                    forecast[
                        [
                            "market_history_home_margin",
                            "market_parameter_variance",
                            "market_observation_variance",
                        ]
                    ],
                    on="game_id",
                    validate="one_to_one",
                )
                out["line_error"] = -out.closing_spread - out.market_history_home_margin
                out["raw_line_sd"] = np.sqrt(
                    out.market_parameter_variance + out.market_observation_variance
                )
                out["as_of"] = cutoff
                out["last_training_kickoff"] = fitted.training.start_date.max()
                frames.append(out)
                ratings.append(fitted.ratings())
        catalog = _team_catalog(all_lined)
        final = _fit_market_rating(all_lined, catalog, carried, np.ones(len(all_lined)))
        carried.update(zip(catalog.team, final.rating.astype(float)))
        log.info("Market uncertainty replay %s: %s lines", year, len(targets))
    return pd.concat(frames, ignore_index=True), pd.concat(ratings, ignore_index=True)


def _cohorts(frame: pd.DataFrame):
    yield "all_d1", frame
    yield (
        "fbs_vs_fbs",
        frame[
            frame.home_classification.eq("fbs") & frame.away_classification.eq("fbs")
        ],
    )


def evaluate_rating_uncertainty(destination: Path) -> dict:
    """Save evidence and a dev-only SD scale; never rewrite model settings."""
    destination.mkdir(parents=True, exist_ok=True)
    pure = _pure_predictions()
    market, ratings = _market_predictions()
    development = market[market.season.le(2022)]
    # Gaussian predictive NLL has this exact positive scale optimum.
    sd_scale = float(
        np.sqrt(np.mean((development.line_error / development.raw_line_sd) ** 2))
    )
    rows = []
    for split, years in (("development", (2020, 2022)), ("validation", (2023, 2025))):
        for cohort, part in _cohorts(pure[pure.season.between(*years)]):
            for candidate, group in part.groupby("candidate"):
                rows.append(
                    dict(
                        source="pure_model",
                        split=split,
                        cohort=cohort,
                        candidate=candidate,
                        games=len(group),
                        margin_mae=float(group.margin_error.abs().mean()),
                        total_mae=float(group.total_error.abs().mean()),
                        nll=float(group.joint_margin_total_nll.mean()),
                    )
                )
        for cohort, part in _cohorts(market[market.season.between(*years)]):
            for name, scale in (("raw", 1.0), ("dev_scaled", sd_scale)):
                sd = part.raw_line_sd * scale
                rows.append(
                    dict(
                        source="future_market_line",
                        split=split,
                        cohort=cohort,
                        candidate=name,
                        games=len(part),
                        margin_mae=float(part.line_error.abs().mean()),
                        nll=float(
                            np.mean(
                                np.log(sd)
                                + 0.5 * (part.line_error / sd) ** 2
                                + 0.5 * np.log(2 * np.pi)
                            )
                        ),
                        coverage_80=float(
                            np.mean(part.line_error.abs() <= norm.ppf(0.9) * sd)
                        ),
                        mean_sd=float(sd.mean()),
                    )
                )
    report = pd.DataFrame(rows)
    manifest = {
        "development_seasons": [2020, 2021, 2022],
        "validation_seasons": [2023, 2024, 2025],
        "market_line_sd_scale": sd_scale,
        "scale_method": "gaussian_predictive_nll_positive_scale_mle",
        "market_rating_sd_method": "conditional_ridge_posterior_unscaled",
        "pure_prior_contract": (
            "The solver already preserves power variance when adding correlation; "
            "the candidate tests additional tightening, not a variance bug fix."
        ),
        "limitations": [
            "Historical validation seasons have been used in earlier research.",
            "Pure replay uses historical carryover, not rich preseason inputs.",
            "Market replay uses cached final closing lines, not timestamped receipts.",
            "Future-line coverage does not validate latent team-rating SD coverage.",
            "Moneylines and season win totals are not independent spread observations.",
        ],
    }
    pure.to_parquet(destination / "pure_predictions.parquet", index=False)
    market.to_parquet(destination / "market_predictions.parquet", index=False)
    ratings.to_parquet(destination / "market_ratings.parquet", index=False)
    report.to_parquet(destination / "summary.parquet", index=False)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("\n%s", report.to_string(index=False))
    return manifest
