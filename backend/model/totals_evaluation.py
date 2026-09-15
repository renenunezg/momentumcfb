"""Cached chronological totals experiments; never writes serving artifacts.

Run with ``python -m backend.model.totals_evaluation OUTPUT_DIRECTORY``.
Select on 2020-2022 before evaluating 2023-2025, which is reused historical
validation, not a newly untouched holdout. Historical carryover omits rich
live preseason inputs. Closing blends are retrospective benchmarks, not
executable price evidence. No 2026 outcomes select parameters.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from backend.etl import store
from backend.grading import _classification
from backend.model.improvement_evaluation import historical_games
from backend.model.joint_scoring import (
    DEFAULT_CONFIG,
    JointScoringPriors,
    fit_joint_scoring,
)
from backend.model.preseason import build_historical_carryover_priors
from backend.serving.market import flatten_closing_totals

SUPPORT_GAMES = (0, 25, 100, 300, 1000)
MARKET_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def replay():
    frames = []
    previous = None
    for season in range(2019, 2026):
        games = historical_games(season)
        if previous is not None:
            means = build_historical_carryover_priors(previous, games)
            old_pace = dict(
                zip(
                    previous.teams.team, previous.base_possessions + 0.5 * previous.pace
                )
            )
            names = dict(zip(games.home_team_id, games.home_team))
            names.update(zip(games.away_team_id, games.away_team))
            priors = JointScoringPriors(
                means,
                {k: (DEFAULT_CONFIG.strength_prior_sd_ppp,) * 2 for k in means},
                {k: old_pace.get(names[k], previous.base_possessions) for k in means},
                previous.base_possessions,
                previous.score_residual_covariance,
                previous.training_games,
                previous.season,
                previous.as_of,
            )
            closes = flatten_closing_totals(store.read_lines(season))
            for week in sorted(games.model_week.unique())[1:]:
                target = games[games.model_week.eq(week)]
                cutoff = target.start_date.min() - pd.Timedelta(microseconds=1)
                fitted = fit_joint_scoring(
                    games, int(week), cutoff.to_pydatetime(), priors=priors
                )
                out = pd.DataFrame(p.to_record() for p in fitted.project(target))
                context = target[
                    [
                        "game_id",
                        "home_points",
                        "away_points",
                        "home_classification",
                        "away_classification",
                    ]
                ]
                out = out.merge(context, on="game_id", validate="one_to_one")
                out = out.merge(
                    closes[["game_id", "closing_total"]],
                    on="game_id",
                    how="left",
                    validate="one_to_one",
                )
                out["actual_total"] = out.home_points + out.away_points
                out["matchup"] = _classification(out)
                out["model_week"] = week
                out["training_games"] = fitted.training_games
                out["base_ppp"] = fitted.base_ppp
                out["preseason_ppp"] = previous.base_ppp
                for support in SUPPORT_GAMES:
                    candidate = out.copy()
                    tested = replace(
                        fitted,
                        preseason_base_ppp=previous.base_ppp,
                        config=replace(fitted.config, scoring_prior_games=support),
                    )
                    records = pd.DataFrame(
                        p.to_record() for p in tested.project(target)
                    )
                    for column in (
                        "home_margin",
                        "home_spread",
                        "margin_sd",
                        "total_sd",
                        "margin_total_correlation",
                    ):
                        np.testing.assert_allclose(
                            records[column], out[column], rtol=0, atol=1e-12
                        )
                    for column in records:
                        candidate[column] = records[column].to_numpy()
                    candidate["candidate"] = support
                    frames.append(candidate)
            print(f"replayed {season}", flush=True)
        previous = fit_joint_scoring(
            games,
            int(games.model_week.max()) + 1,
            (games.start_date.max() + pd.Timedelta(days=1)).to_pydatetime(),
        )
    return pd.concat(frames, ignore_index=True)


def metrics(frame, mean, sd):
    error = mean - frame.actual_total
    scale = np.asarray(sd) * np.sqrt(
        (frame.degrees_of_freedom - 2) / frame.degrees_of_freedom
    )
    return dict(
        games=len(frame),
        mae=np.abs(error).mean(),
        rmse=np.sqrt(np.mean(error**2)),
        bias=error.mean(),
        nll=np.mean(-t.logpdf(error / scale, frame.degrees_of_freedom) + np.log(scale)),
        coverage80=np.mean(
            np.abs(error) <= t.ppf(0.9, frame.degrees_of_freedom) * scale
        ),
    )


def evaluate(destination):
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination / "replay.parquet"
    frame = pd.read_parquet(cache) if cache.exists() else replay()
    frame.to_parquet(cache, index=False)
    dev = frame[frame.season.le(2022)]
    ranking = dev.groupby("candidate").apply(
        lambda x: (x.model_total - x.actual_total).abs().mean(), include_groups=False
    )
    selected = int(ranking.idxmin())
    print(
        "Development scoring support MAE:",
        ranking.to_dict(),
        "selected",
        selected,
        flush=True,
    )
    base = dev[dev.candidate.eq(selected)]
    pure_scale = min(
        (0.85, 0.925, 1.0, 1.075, 1.15),
        key=lambda s: metrics(base, base.model_total, base.total_sd * s)["nll"],
    )
    market = base[base.closing_total.notna()]
    weight = min(
        MARKET_WEIGHTS,
        key=lambda w: metrics(
            market, (1 - w) * market.model_total + w * market.closing_total, 15.93
        )["mae"],
    )
    mean = (1 - weight) * market.model_total + weight * market.closing_total
    market_sd = min(
        (14.0, 15.0, 15.93, 17.0, 18.0), key=lambda sd: metrics(market, mean, sd)["nll"]
    )
    current_mean = 0.5 * (market.model_total + market.closing_total)
    current_sd = min(
        (14.0, 15.0, 15.93, 17.0, 18.0),
        key=lambda sd: metrics(market, current_mean, sd)["nll"],
    )
    selection = dict(
        support=selected,
        pure_sd_scale=pure_scale,
        market_weight=weight,
        market_sd=market_sd,
        retained_weight_sd=current_sd,
        development_seasons=[2020, 2021, 2022],
        validation_seasons=[2023, 2024, 2025],
    )
    (destination / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    ranking.to_csv(destination / "support_selection.csv")
    market_search = []
    for w in MARKET_WEIGHTS:
        for sd in (14.0, 15.0, 15.93, 17.0, 18.0):
            market_search.append(
                dict(
                    weight=w,
                    sd=sd,
                    **metrics(
                        market,
                        (1 - w) * market.model_total + w * market.closing_total,
                        sd,
                    ),
                )
            )
    pd.DataFrame(market_search).to_csv(
        destination / "market_selection.csv", index=False
    )
    print(
        "Frozen selection:",
        dict(
            support=selected,
            pure_sd_scale=pure_scale,
            market_weight=weight,
            market_sd=market_sd,
        ),
        flush=True,
    )
    rows = []
    for split, years in (
        ("development", (2020, 2021, 2022)),
        ("validation", (2023, 2024, 2025)),
    ):
        for support in sorted({0, selected}):
            subset = frame[frame.season.isin(years) & frame.candidate.eq(support)]
            groups = [("all", "all", subset)]
            groups += [
                (str(w), c, f)
                for (w, c), f in subset.groupby(["model_week", "matchup"])
            ]
            groups += [("all", c, f) for c, f in subset.groupby("matchup")]
            for week, matchup, group in groups:
                for label, cohort, prediction, sd in (
                    ("pure", group, group.model_total, group.total_sd),
                    (
                        "pure_calibrated_sd",
                        group,
                        group.model_total,
                        group.total_sd * pure_scale,
                    ),
                ):
                    rows.append(
                        dict(
                            split=split,
                            support=support,
                            week=week,
                            matchup=matchup,
                            layer=label,
                            **metrics(cohort, prediction, sd),
                        )
                    )
                cohort = group[group.closing_total.notna()]
                if cohort.empty:
                    continue
                for label, w, sd in (
                    ("pure_matched_market", 0.0, cohort.total_sd),
                    ("current_pricing", 0.5, 15.93),
                    ("current_weight_calibrated_sd", 0.5, current_sd),
                    ("selected_pricing", weight, market_sd),
                    ("closing_market", 1.0, market_sd),
                ):
                    rows.append(
                        dict(
                            split=split,
                            support=support,
                            week=week,
                            matchup=matchup,
                            layer=label,
                            **metrics(
                                cohort,
                                (1 - w) * cohort.model_total + w * cohort.closing_total,
                                sd,
                            ),
                        )
                    )
    summary = pd.DataFrame(rows)
    summary.to_csv(destination / "metrics.csv", index=False)
    validation = frame[frame.season.ge(2023)]
    paired = validation[validation.candidate.eq(selected)].merge(
        validation[validation.candidate.eq(0)],
        on=["season", "game_id"],
        suffixes=("", "_baseline"),
        validate="one_to_one",
    )
    paired["delta"] = (paired.model_total - paired.actual_total).abs() - (
        paired.model_total_baseline - paired.actual_total
    ).abs()
    effects = []
    for name, group in [("all", paired), *paired.groupby("matchup")]:
        blocks = group.groupby(["season", "model_week"]).delta.agg(["sum", "count"])
        sampled = np.random.default_rng(42).integers(
            len(blocks), size=(10000, len(blocks))
        )
        delta = blocks["sum"].to_numpy()[sampled].sum(1) / blocks["count"].to_numpy()[
            sampled
        ].sum(1)
        lower, upper = np.quantile(delta, [0.025, 0.975])
        effects.append(
            dict(
                matchup=name,
                games=len(group),
                delta=group.delta.mean(),
                lower95=lower,
                upper95=upper,
            )
        )
    pd.DataFrame(effects).to_csv(destination / "paired_effects.csv", index=False)
    drift = (
        frame[frame.candidate.eq(0)]
        .groupby(["season", "model_week", "matchup"])
        .agg(
            games=("game_id", "size"),
            base_ppp=("base_ppp", "first"),
            preseason_ppp=("preseason_ppp", "first"),
            training_games=("training_games", "first"),
        )
    )
    drift["ppp_drift"] = drift.base_ppp - drift.preseason_ppp
    drift.to_csv(destination / "drift.csv")
    print(
        summary[summary.week.eq("all") & summary.matchup.eq("all")].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    evaluate(Path(sys.argv[1]))
