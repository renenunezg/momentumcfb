"""Week 6 diagnosis and chronological spread/total research, never publishing.

Historical line prices are unavailable. Any historical return is a hypothetical
flat -110 benchmark at recorded Bovada closing lines, not an executable ledger.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import t

from backend.etl import store
from backend.model.distributions import marginal_cdf, marginal_scale
from backend.model.joint_scoring import _team_catalog
from backend.model.market_history import (
    _fit_market_rating,
    fit_market_history,
    history_weight,
    lined_games,
)
from backend.model.weekly import load_weekly_games
from examples.audit_moneyline_probabilities import interval

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/spread-total-audit-20261012"
SOURCE = ROOT / "output/moneyline-audit-20261012/current_v14_proxy.parquet"
WEEK = ROOT / "output/week6-policy-comparison-20261012"
DF = 500.0


def history_predictions(predictions):
    cache = OUT / "history_predictions.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    carried, frames = {}, []
    for year in range(2019, 2026):
        games, lines = load_weekly_games(year), store.read_lines(year)
        lined = lined_games(games, lines)
        prior = pd.DataFrame(
            {"team": list(carried), "market_rating": list(carried.values())}
        )
        for week, d in predictions[predictions.season.eq(year)].groupby("model_week"):
            cutoff = pd.Timestamp(d.as_of.iloc[0])
            fitted = fit_market_history(games, lines, int(week), cutoff, prior)
            if fitted is None:
                continue
            target = games[games.game_id.isin(d.game_id)]
            assert len(target) == len(d)
            assert (fitted.training.start_date < cutoff).all()
            a = fitted.project(target).reset_index()
            a["last_training_kickoff"] = fitted.training.start_date.max()
            frames.append(a)
        catalog = _team_catalog(lined)
        final = _fit_market_rating(lined, catalog, carried, np.ones(len(lined)))
        carried.update(zip(catalog.team, final.rating.astype(float)))
        print(f"Market history rebuilt {year}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(cache, index=False)
    return result


def load_data():
    d = pd.read_parquet(SOURCE)
    assert not d.duplicated(["season", "game_id"]).any()
    assert (
        pd.to_datetime(d.training_latest_start, utc=True)
        < pd.to_datetime(d.as_of, utc=True)
    ).all()
    assert (
        pd.to_datetime(d.as_of, utc=True) < pd.to_datetime(d.start_date, utc=True)
    ).all()
    history = history_predictions(d)
    d = d.merge(
        history[
            ["season", "game_id", "market_history_home_margin", "last_training_kickoff"]
        ],
        on=["season", "game_id"],
        how="left",
        validate="one_to_one",
    )
    d["model_side"] = d.home_margin
    has = d.market_history_home_margin.notna()
    h = d.model_week.map(history_weight)
    d.loc[has, "model_side"] = (
        (1 - h) * d.home_margin + h * d.market_history_home_margin
    )[has]
    rows = []
    for year in range(2020, 2026):
        for game in store.read_lines(year).itertuples():
            for offer in game.lines:
                rows.append(dict(season=year, game_id=game.game_id, **offer))
    quotes = pd.DataFrame(rows)
    cols = ["spread", "overUnder"]
    quotes[cols] = quotes[cols].apply(pd.to_numeric, errors="coerce")
    assert not quotes.duplicated(["season", "game_id", "provider"]).any()
    medians = (
        quotes.groupby(["season", "game_id"])[cols]
        .median()
        .rename(columns={"spread": "reference_spread", "overUnder": "reference_total"})
    )
    bovada = quotes[quotes.provider.eq("Bovada")][["season", "game_id", *cols]].rename(
        columns={"spread": "offer_spread", "overUnder": "offer_total"}
    )
    d = d.merge(
        medians, on=["season", "game_id"], how="left", validate="one_to_one"
    ).merge(bovada, on=["season", "game_id"], how="left", validate="one_to_one")
    return d


def probabilities(mean, sd, threshold):
    win = marginal_cdf(mean - np.floor(threshold) - 0.5, sd, DF)
    loss = marginal_cdf(np.ceil(threshold) - 0.5 - mean, sd, DF)
    push = np.where(threshold == np.floor(threshold), np.maximum(0, 1 - win - loss), 0)
    return np.column_stack([win, push, loss])


def parameters(train, market):
    sd = 15.35 if market == "spreads" else 15.93
    r, m, y = (
        train.reference.to_numpy(),
        train.model.to_numpy(),
        train.actual.to_numpy(),
    )

    def fit(fixed=None):
        def objective(x):
            mean = r + x[2] * (m - r) + x[0]
            scale = marginal_scale(np.exp(x[1]), DF)
            return -np.mean(t.logpdf((y - mean) / scale, DF) - np.log(scale))

        bounds = [
            (-10, 10),
            (np.log(5), np.log(35)),
            (0, 1) if fixed is None else (fixed, fixed),
        ]
        res = minimize(
            objective,
            [0, np.log(sd), 0.5 if fixed is None else fixed],
            bounds=bounds,
            method="L-BFGS-B",
        )
        assert res.success, res.message
        return dict(
            offset=float(res.x[0]), sd=float(np.exp(res.x[1])), weight=float(res.x[2])
        )

    return {
        "current_050": dict(offset=0, sd=sd, weight=0.5),
        "research_020": dict(offset=0, sd=sd, weight=0.2),
        "dev_calibrated_050": fit(0.5),
        "dev_fitted_blend": fit(),
        "market_reference": dict(offset=0, sd=sd, weight=0),
        "pure_model_side": dict(offset=0, sd=sd, weight=1),
    }


def evaluate(d, params):
    out = d.copy()
    mean = d.reference + params["weight"] * (d.model - d.reference) + params["offset"]
    pp = probabilities(mean.to_numpy(), params["sd"], d.threshold.to_numpy())
    balance = d.actual.to_numpy() - d.threshold.to_numpy()
    labels = np.where(balance > 0, 0, np.where(balance < 0, 2, 1))
    out["mean"] = mean
    out["nll"] = -np.log(np.clip(pp[np.arange(len(d)), labels], 1e-12, 1))
    out["brier"] = ((pp - np.eye(3)[labels]) ** 2).sum(axis=1)
    out["predicted_push"] = pp[:, 1]
    out["actual_push"] = labels == 1
    side = np.where(pp[:, 0] >= pp[:, 2], 1, -1)
    win = np.where(side == 1, pp[:, 0], pp[:, 2])
    loss = np.where(side == 1, pp[:, 2], pp[:, 0])
    out["side"] = side
    out["probability"] = win / (win + loss)
    out["expected_value"] = win * 100 / 110 - loss
    out["win"] = balance * side > 0
    out["loss"] = balance * side < 0
    out["profit"] = np.where(out.win, 100 / 110, np.where(out.loss, -1, 0))
    # At half lines: root is CDF^-1(110/210). Integer lines reserve push mass.
    scale = marginal_scale(params["sd"], DF)
    cost_half = scale * t.ppf(110 / 210, DF)
    cost_integer = brentq(
        lambda x: (
            (
                marginal_cdf(x - 0.5, params["sd"], DF)
                / (
                    marginal_cdf(x - 0.5, params["sd"], DF)
                    + marginal_cdf(-x - 0.5, params["sd"], DF)
                )
            )
            - 110 / 210
        ),
        -20,
        20,
    )
    out["edge_points"] = np.abs(mean - d.threshold) - np.where(
        d.threshold.eq(np.floor(d.threshold)), cost_integer, cost_half
    )
    return out


def score_summary(d):
    return dict(
        games=len(d),
        mae=float((d["mean"] - d.actual).abs().mean()),
        bias=float((d["mean"] - d.actual).mean()),
        rmse=float(np.sqrt(np.mean((d["mean"] - d.actual) ** 2))),
        log_loss=float(d.nll.mean()),
        brier=float(d.brier.mean()),
        predicted_push=float(d.predicted_push.mean()),
        actual_push=float(d.actual_push.mean()),
    )


def pick_summary(d):
    if d.empty:
        return dict(
            picks=0,
            wins=0,
            losses=0,
            pushes=0,
            profit=0,
            roi=None,
            predicted=None,
            win_rate=None,
            roi_ci=[None, None],
        )
    decisive = ~d.actual_push
    return dict(
        picks=len(d),
        wins=int(d.win.sum()),
        losses=int(d.loss.sum()),
        pushes=int(d.actual_push.sum()),
        profit=float(d.profit.sum()),
        roi=float(d.profit.mean()),
        predicted=float(d.probability.mean()),
        win_rate=float(d.loc[decisive, "win"].mean()),
        roi_ci=interval(d, d.profit),
    )


def week_diagnosis():
    published = pd.read_json(WEEK / "published.json")
    p = published[
        published.status.eq("recommended") & published.market.ne("h2h")
    ].copy()
    f = pd.read_parquet(WEEK / "inputs/projections.parquet")
    p = p.merge(
        f[["game_id", "home_margin", "model_total", "market_home_spread"]],
        on="game_id",
        suffixes=("", "_pure"),
        validate="many_to_one",
    )
    p["actual_margin"] = p.home_points - p.away_points
    p["actual_total"] = p.home_points + p.away_points
    p["priced_forecast"] = np.where(
        p.market.eq("spreads"), p.model_home_margin, p.model_total
    )
    p["actual"] = np.where(p.market.eq("spreads"), p.actual_margin, p.actual_total)
    p["reference"] = np.where(
        p.market.eq("spreads"), -p.market_home_spread, p.market_total
    )
    p["forecast_error"] = p.priced_forecast - p.actual
    p["market_error"] = p.reference - p.actual
    p["settlement_margin"] = np.where(
        p.market.eq("spreads"),
        p.actual_margin * np.where(p.side.eq("home"), 1, -1) + p.point,
        (p.actual_total - p.point) * np.where(p.side.eq("over"), 1, -1),
    )
    assert p.reason.eq("qualifying_edge").all()
    assert p.outcome.eq("win").equals(p.settlement_margin.gt(0))
    p.to_csv(OUT / "week6_picks.csv", index=False)
    summary = []
    for market, a in p.groupby("market"):
        summary.append(
            dict(
                market=market,
                picks=len(a),
                wins=int(a.outcome.eq("win").sum()),
                profit=float(a.profit_units.sum()),
                expected_wins=float(a.win_probability.sum()),
                expected_profit=float(a.expected_value_per_unit.sum()),
                forecast_mae=float(a.forecast_error.abs().mean()),
                market_mae=float(a.market_error.abs().mean()),
                average_miss_on_losses=float(
                    -a.loc[a.outcome.eq("loss"), "settlement_margin"].mean()
                ),
            )
        )
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = dict(
        development=[2020, 2021, 2022],
        validation=[2023, 2024, 2025],
        weight_search="bounded continuous 0 to 1, diagnostic only",
        objective="Student t predictive density NLL on development outcomes",
        candidates=[
            "current_050",
            "research_020",
            "dev_calibrated_050",
            "dev_fitted_blend",
            "market_reference",
            "pure_model_side",
        ],
        edge_thresholds=[2, 3, 4],
        execution="Bovada closing line, assumed -110, flat 1 unit, no volume floor reconstruction",
        production_changes=False,
    )
    (OUT / "analysis_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    weekly = week_diagnosis()
    all_data = load_data()
    all_data.to_parquet(OUT / "historical_inputs.parquet", index=False)
    scores, picks, segments, comparisons, parameters_by_market = [], [], [], [], {}
    for market in ["spreads", "totals"]:
        d = all_data.copy()
        d["reference"] = (
            -d.reference_spread if market == "spreads" else d.reference_total
        )
        d["threshold"] = -d.offer_spread if market == "spreads" else d.offer_total
        d["model"] = d.model_side if market == "spreads" else d.model_total
        d["actual"] = d.actual_margin if market == "spreads" else d.actual_total
        d = (
            d[
                np.isfinite(d[["reference", "threshold", "model", "actual"]]).all(
                    axis=1
                )
                & (d.threshold * 2).eq(np.round(d.threshold * 2))
            ]
            .copy()
            .reset_index(drop=True)
        )
        train = d[d.season.le(2022)]
        params = parameters(train, market)
        parameters_by_market[market] = params
        baseline = evaluate(d, params["current_050"])
        for method, param in params.items():
            a = evaluate(d, param)
            a["method"], a["market"] = method, market
            for split, mask in [
                ("development", a.season.le(2022)),
                ("validation", a.season.ge(2023)),
                *[(str(y), a.season.eq(y)) for y in [2023, 2024, 2025]],
            ]:
                group = a[mask]
                scores.append(
                    dict(
                        market=market,
                        method=method,
                        split=split,
                        **score_summary(group),
                    )
                )
                for gate in [2, 3, 4]:
                    chosen = group[
                        group.edge_points.ge(gate) & group.expected_value.gt(0)
                    ]
                    picks.append(
                        dict(
                            market=market,
                            method=method,
                            split=split,
                            gate=gate,
                            **pick_summary(chosen),
                        )
                    )
            validation = a[a.season.ge(2023)]
            delta = validation.nll - baseline.loc[validation.index, "nll"]
            comparisons.append(
                dict(
                    market=market,
                    method=method,
                    log_loss_change=float(delta.mean()),
                    ci=interval(validation, delta),
                )
            )
            chosen = validation[
                validation.edge_points.ge(2) & validation.expected_value.gt(0)
            ]
            for dimension, groups in [
                ("side", chosen.groupby("side")),
                (
                    "edge",
                    chosen.groupby(
                        pd.cut(chosen.edge_points, [2, 3, 4, 6, np.inf], right=False),
                        observed=True,
                    ),
                ),
                (
                    "week",
                    chosen.groupby(
                        np.where(chosen.model_week.le(3), "early_1_3", "later_4_plus")
                    ),
                ),
            ]:
                for name, group in groups:
                    segments.append(
                        dict(
                            market=market,
                            method=method,
                            dimension=dimension,
                            segment=str(name),
                            **pick_summary(group),
                        )
                    )
            a.to_parquet(OUT / f"{market}_{method}.parquet", index=False)
        print(market, params, flush=True)
    for name, rows in [("scores", scores), ("picks", picks), ("segments", segments)]:
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    report = dict(
        plan=plan,
        week6=weekly,
        parameters=parameters_by_market,
        comparisons=comparisons,
        limitations=[
            "Historical return benchmarks assume -110; actual spread and total odds and decision timestamps are unavailable.",
            "Historical carryover forecasts omit archived rich preseason and availability inputs.",
            "2023-2025 are reused validation seasons, not untouched holdouts.",
            "Current model architecture and fixed constants were developed using some of these historical years.",
            "Historical selection benchmarks omit the cross-market volume floor.",
        ],
        source_games=len(all_data),
    )
    (OUT / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
