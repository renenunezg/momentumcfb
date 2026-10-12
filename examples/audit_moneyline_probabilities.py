"""Read-only historical moneyline audit; closing-price returns are proxies.

Run with ``poetry run python -m examples.audit_moneyline_probabilities``.
No 2026 outcomes enter fitting; no production artifacts are written.
"""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import t

from backend.model.distributions import marginal_cdf, marginal_scale

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output/moneyline-audit-20261012"
SD, DF, GATE = 15.35, 500.0, 2.0
SCALE = float(marginal_scale(SD, DF))
DEVELOPMENT = (2021, 2022)
EVALUATION = (2023, 2024, 2025)


def implied(price):
    price = np.asarray(price, dtype=float)
    return np.where(price > 0, 100, -price) / (100 + np.abs(price))


def profit(price):
    price = np.asarray(price, dtype=float)
    return np.where(price > 0, price / 100, 100 / -price)


def losses(y, p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def interval(frame, values, *, denominator=None):
    """Resample forecast weeks, preserving within-week game dependence."""
    if frame.empty:
        return [None, None]
    d = (
        pd.DataFrame(
            dict(
                season=frame.season.to_numpy(),
                week=frame.model_week.to_numpy(),
                value=np.asarray(values),
                denominator=np.ones(len(frame)) if denominator is None else denominator,
            )
        )
        .groupby(["season", "week"])[["value", "denominator"]]
        .sum()
    )
    rng = np.random.default_rng(20261012)
    sample = rng.integers(0, len(d), (5000, len(d)))
    draws = d.value.to_numpy()[sample].sum(axis=1) / d.denominator.to_numpy()[
        sample
    ].sum(axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def load_quotes():
    rows, counts = [], []
    for year in (*DEVELOPMENT, *EVALUATION):
        lines = pd.read_parquet(ROOT / f"backend/data/raw/lines/{year}.parquet")
        for game in lines.itertuples():
            for offer in game.lines:
                rows.append(dict(season=year, game_id=game.game_id, **offer))
        counts.append(dict(season=year, raw_games=len(lines)))
    raw = pd.DataFrame(rows)
    cols = ["homeMoneyline", "awayMoneyline", "spread"]
    for col in cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    valid = np.isfinite(raw[cols]).all(axis=1)
    valid &= raw.homeMoneyline.abs().ge(100) & raw.awayMoneyline.abs().ge(100)
    q = raw[valid].copy()
    assert not q.duplicated(["season", "game_id", "provider"]).any()
    q["ih"], q["ia"] = implied(q.homeMoneyline), implied(q.awayMoneyline)
    q["overround"] = q.ih + q.ia
    # Abnormal/negative hold cannot establish a contemporaneous valid market.
    overround_ok = q.overround.between(1.0, 1.20)
    quality = dict(
        raw_provider_rows=len(raw),
        valid_price_spread_rows=len(q),
        invalid_overround_rows=int((~overround_ok).sum()),
        quote_timestamps_available=False,
    )
    q = q[overround_ok].copy()
    q["market_p"] = q.ih / q.overround
    q["market_power_p"] = [
        h ** brentq(lambda k: h**k + a**k - 1, 0.1, 10) for h, a in zip(q.ih, q.ia)
    ]
    return q, quality, counts


def load_forecasts(path, quotes):
    d = pd.read_parquet(path)
    assert not d.duplicated(["season", "game_id"]).any()
    dates = [
        pd.to_datetime(d[c], utc=True)
        for c in ["training_latest_start", "as_of", "start_date"]
    ]
    assert all(s.notna().all() for s in dates)
    assert ((dates[0] < dates[1]) & (dates[1] < dates[2])).all()
    assert d.season.max() <= 2025
    # Recompute consensus from the full raw spread feed, preserving quarter lines.
    spreads = []
    for year in (*DEVELOPMENT, *EVALUATION):
        raw = pd.read_parquet(ROOT / f"backend/data/raw/lines/{year}.parquet")
        for game in raw.itertuples():
            vals = [
                float(o["spread"])
                for o in game.lines
                if o.get("spread") is not None and np.isfinite(o["spread"])
            ]
            if vals:
                spreads.append(
                    dict(
                        season=year,
                        game_id=game.game_id,
                        market_margin=-float(np.median(vals)),
                    )
                )
    market = (
        quotes.groupby(["season", "game_id"])[["market_p", "market_power_p"]]
        .median()
        .reset_index()
    )
    d = d.merge(pd.DataFrame(spreads), validate="one_to_one").merge(
        market, validate="one_to_one"
    )
    d = d[d.actual_margin.ne(0)].copy().reset_index(drop=True)
    d["y"] = (d.actual_margin > 0).astype(int)
    d["residual"] = (d.home_margin - d.market_margin) / SD
    return d


def fit_models(d, years=DEVELOPMENT):
    train = d[d.season.isin(years)]
    y = train.y.to_numpy()
    m, r = (
        train.market_margin.to_numpy(),
        train.home_margin.to_numpy() - train.market_margin.to_numpy(),
    )
    weight = minimize_scalar(
        lambda w: losses(y, marginal_cdf(m + w * r, SD, DF)).mean(),
        bounds=(0, 1),
        method="bounded",
        options={"xatol": 1e-9},
    )

    def spread_objective(x):
        return losses(y, marginal_cdf(x[0] + m + x[2] * r, np.exp(x[1]), DF)).mean()

    spread = minimize(
        spread_objective,
        [0, np.log(SD), 0.2],
        method="L-BFGS-B",
        bounds=[(-5, 5), (np.log(5), np.log(30)), (0, 1)],
    )
    endpoints = [0.0, float(weight.x), 1.0]
    selected_weight = min(
        endpoints, key=lambda w: losses(y, marginal_cdf(m + w * r, SD, DF)).mean()
    )
    fitted = {"weight": selected_weight, "spread": spread.x.tolist()}
    assert weight.success and spread.success
    for name, include_residual in [("ml_calibrated", False), ("ml_plus_model", True)]:
        x = np.column_stack([np.ones(len(train)), logit(train.market_p)])
        if include_residual:
            x = np.column_stack([x, train.residual])
        initial = np.array([0.0, 1.0, 0.0] if include_residual else [0.0, 1.0])
        fit = minimize(
            lambda b: losses(y, expit(x @ b)).mean(),
            initial,
            method="L-BFGS-B",
            bounds=[(-2, 2), (0.2, 3)] + ([(-2, 2)] if include_residual else []),
        )
        assert fit.success
        fitted[name] = fit.x.tolist()
    return fitted


def probabilities(d, fitted):
    m, r = d.market_margin, d.home_margin - d.market_margin
    result = dict(
        market_ml=d.market_p.to_numpy(), market_power=d.market_power_p.to_numpy()
    )
    for name, w in [
        ("spread_only", 0),
        ("current_020", 0.2),
        ("old_050", 0.5),
        ("pure_fixed_sd", 1),
        ("dev_weight", fitted["weight"]),
    ]:
        result[name] = marginal_cdf(m + w * r, SD, DF)
    a, logs, w = fitted["spread"]
    result["dev_spread_calibration"] = marginal_cdf(a + m + w * r, np.exp(logs), DF)
    for name in ["ml_calibrated", "ml_plus_model"]:
        b = fitted[name]
        z = b[0] + b[1] * logit(d.market_p)
        if len(b) == 3:
            z = z + b[2] * d.residual
        result[name] = expit(z)
    return result


def select_picks(d, quotes, p, *, provider=None, gate=GATE, floor=0):
    """Numerical production gate only; no invented source/freshness eligibility."""
    base = d[
        ["season", "model_week", "game_id", "y", "home_team", "away_team", "market_p"]
    ].copy()
    base["p"] = p
    q = quotes if provider is None else quotes[quotes.provider.eq(provider)]
    joined = base.merge(
        q[["season", "game_id", "provider", "homeMoneyline", "awayMoneyline"]],
        on=["season", "game_id"],
        validate="one_to_many",
    )
    sides = []
    for side in ["home", "away"]:
        a = joined.copy()
        a["side"] = side
        a["price"] = a[f"{side}Moneyline"]
        if side == "away":
            a["p"] = 1 - a.p
            a["market_p"] = 1 - a.market_p
            a["y"] = 1 - a.y
        a["break_even"] = implied(a.price)
        a["edge_points"] = SCALE * (t.ppf(a.p, DF) - t.ppf(a.break_even, DF))
        a["ev"] = a.p * (1 + profit(a.price)) - 1
        a["market_ev"] = a.market_p * (1 + profit(a.price)) - 1
        a["profit"] = np.where(a.y.eq(1), profit(a.price), -1.0)
        sides.append(a)
    a = pd.concat(sides, ignore_index=True)
    a = a.sort_values(
        ["edge_points", "provider", "side"], ascending=[False, False, True]
    )
    a = a.drop_duplicates(["season", "game_id"], keep="first")
    chosen = a[(a.edge_points >= gate) & (a.ev > 0)].copy()
    chosen["reason"] = "gate"
    if floor:
        extra = []
        for key, group in a.groupby(["season", "model_week"]):
            existing = chosen[(chosen.season == key[0]) & (chosen.model_week == key[1])]
            candidates = group[
                (group.edge_points > 0)
                & (group.ev > 0)
                & ~group.game_id.isin(existing.game_id)
            ]
            if len(existing) < floor:
                extra.append(
                    candidates.head(floor - len(existing)).assign(reason="floor")
                )
        if extra:
            chosen = pd.concat([chosen, *extra], ignore_index=True)
    return chosen


def pick_summary(d):
    if d.empty:
        return dict(
            picks=0,
            wins=0,
            profit=0.0,
            roi=None,
            roi_ci=[None, None],
            probability=None,
            win_rate=None,
            bias=None,
            expected_profit=0.0,
        )
    return dict(
        picks=len(d),
        wins=int(d.y.sum()),
        profit=float(d.profit.sum()),
        roi=float(d.profit.mean()),
        roi_ci=interval(d, d.profit),
        probability=float(d.p.mean()),
        win_rate=float(d.y.mean()),
        bias=float((d.p - d.y).mean()),
        bias_ci=interval(d, d.p - d.y),
        expected_profit=float(d.ev.sum()),
        market_expected_profit=float(d.market_ev.sum()),
        dogs=int(d.price.gt(0).sum()),
        largest_win=float(d.profit.max()),
        profit_without_largest_win=float(d.profit.sum() - max(0, d.profit.max())),
    )


def verify_frozen_v8_math(games, picks):
    # Frozen v8 formula: keep this audit reproducible when the live policy changes.
    # Its parity with production v8 was checked before replacing that policy.
    indexed = games.set_index(["season", "game_id"])
    selected = picks[picks.method.eq("current_020") & picks.execution.eq("best_quote")]
    assert not selected.duplicated(["season", "game_id"]).any()
    errors = []
    for pick in selected.itertuples():
        game = indexed.loc[(pick.season, pick.game_id)]
        mean = game.market_margin + 0.2 * (game.home_margin - game.market_margin)
        offer = dict(market="h2h", side=pick.side, point=None)
        direction = 1 if offer["side"] == "home" else -1
        win = float(marginal_cdf(direction * mean, SD, DF))
        edge = brentq(
            lambda delta: (
                marginal_cdf(direction * mean - delta, SD, DF)
                - float(implied(pick.price))
            ),
            -200,
            200,
        )
        errors.append((abs(win - pick.p), abs(edge - pick.edge_points)))
    maximum = np.array(errors).max(axis=0)
    assert maximum[0] < 1e-12 and maximum[1] < 1e-8
    return dict(
        selected_decisions_checked=len(selected),
        max_probability_difference=float(maximum[0]),
        max_edge_point_difference=float(maximum[1]),
        duplicates=0,
    )


def run_source(label, path, quotes, destination):
    d = load_forecasts(path, quotes)
    fitted = fit_models(d)
    pp = probabilities(d, fitted)
    score_rows, picks_rows, selections, reliability = [], [], [], []
    for method, p in pp.items():
        d[f"p_{method}"] = p
        for split, mask in [
            ("development", d.season.isin(DEVELOPMENT)),
            ("evaluation", d.season.isin(EVALUATION)),
            *[(str(y), d.season.eq(y)) for y in EVALUATION],
        ]:
            a = d[mask]
            prob = np.asarray(p)[mask]
            delta = losses(a.y, prob) - losses(a.y, a.market_p)
            score_rows.append(
                dict(
                    source=label,
                    method=method,
                    split=split,
                    games=len(a),
                    log_loss=float(losses(a.y, prob).mean()),
                    brier=float(np.mean((a.y - prob) ** 2)),
                    log_loss_delta_vs_market=float(delta.mean()),
                    delta_ci=interval(a, delta),
                )
            )
        for execution, provider in [("best_quote", None), ("Bovada", "Bovada")]:
            selected = select_picks(d, quotes, p, provider=provider)
            selected["method"], selected["execution"], selected["source"] = (
                method,
                execution,
                label,
            )
            selections.append(selected)
            for split, a in [
                ("development", selected[selected.season.isin(DEVELOPMENT)]),
                ("evaluation", selected[selected.season.isin(EVALUATION)]),
                *[(str(y), selected[selected.season.eq(y)]) for y in EVALUATION],
            ]:
                picks_rows.append(
                    dict(
                        source=label,
                        method=method,
                        execution=execution,
                        split=split,
                        **pick_summary(a),
                    )
                )
        evaluation = d[d.season.isin(EVALUATION)].copy()
        evaluation["p"] = np.asarray(p)[d.season.isin(EVALUATION)]
        # One market-defined underdog per game avoids outcome-based selection.
        away_dog = evaluation.market_p.ge(0.5)
        evaluation.loc[away_dog, "p"] = 1 - evaluation.loc[away_dog, "p"]
        evaluation.loc[away_dog, "y"] = 1 - evaluation.loc[away_dog, "y"]
        evaluation["bucket"] = pd.cut(
            evaluation.p, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1], include_lowest=True
        )
        for bucket, a in evaluation.groupby("bucket", observed=True):
            reliability.append(
                dict(
                    source=label,
                    method=method,
                    bucket=str(bucket),
                    games=len(a),
                    predicted=float(a.p.mean()),
                    actual=float(a.y.mean()),
                    bias_ci=interval(a, a.p - a.y),
                )
            )
    d.to_parquet(destination / f"{label}_scored_games.parquet", index=False)
    all_picks = pd.concat(selections, ignore_index=True)
    all_picks.to_parquet(destination / f"{label}_proxy_picks.parquet", index=False)
    # Fixed policy sensitivities, descriptive only; do not choose a winning threshold.
    sensitivities = []
    for gate in [0.0, 1.0, 2.0, 3.0, 4.0]:
        s = select_picks(d, quotes, pp["current_020"], gate=gate)
        s = s[s.season.isin(EVALUATION)]
        sensitivities.append(
            dict(source=label, scenario=f"gate_{gate:g}", **pick_summary(s))
        )
    s = select_picks(d, quotes, pp["current_020"], floor=15)
    for reason in ["gate", "floor"]:
        a = s[s.season.isin(EVALUATION) & s.reason.eq(reason)]
        sensitivities.append(
            dict(
                source=label,
                scenario=f"moneyline_only_floor_{reason}",
                **pick_summary(a),
            )
        )
    selected = all_picks[
        all_picks.method.eq("current_020")
        & all_picks.execution.eq("best_quote")
        & all_picks.season.isin(EVALUATION)
    ].copy()
    selected_diagnostics = []
    segments = [("all", selected)]
    segments += [
        ("underdogs", selected[selected.price.gt(0)]),
        ("favorites", selected[selected.price.lt(0)]),
        ("weeks_1_4", selected[selected.model_week.le(4)]),
        ("weeks_5_plus", selected[selected.model_week.gt(4)]),
    ]
    segments += [
        (f"provider_{key}", group) for key, group in selected.groupby("provider")
    ]
    for name, a in segments:
        delta = losses(a.y, a.p) - losses(a.y, a.market_p)
        selected_diagnostics.append(
            dict(
                source=label,
                segment=name,
                **pick_summary(a),
                market_probability=float(a.market_p.mean()),
                paired_log_loss_delta=float(delta.mean()),
                paired_log_loss_ci=interval(a, delta),
            )
        )
    # A fixed expanding-window stress check; results never choose production rules.
    rolling = []
    rolling_parameters = []
    for year in EVALUATION:
        params = fit_models(d, tuple(range(2021, year)))
        target = d[d.season.eq(year)]
        rolling_parameters.append(dict(year=year, parameters=params))
        for method, prob in probabilities(target, params).items():
            a = target.copy()
            a["p"], a["method"] = prob, method
            rolling.append(a)
    rolling = pd.concat(rolling, ignore_index=True)
    rolling_scores = []
    for method, a in rolling.groupby("method"):
        delta = losses(a.y, a.p) - losses(a.y, a.market_p)
        rolling_scores.append(
            dict(
                source=label,
                method=method,
                games=len(a),
                log_loss=float(losses(a.y, a.p).mean()),
                market_delta=float(delta.mean()),
                delta_ci=interval(a, delta),
            )
        )
    rolling.to_parquet(
        destination / f"{label}_rolling_probabilities.parquet", index=False
    )
    return dict(
        source=label,
        path=str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        sha256=sha256(path.read_bytes()).hexdigest(),
        games=len(d),
        by_season=d.groupby("season").size().to_dict(),
        parameters=fitted,
        scores=score_rows,
        picks=picks_rows,
        reliability=reliability,
        sensitivities=sensitivities,
        selected_diagnostics=selected_diagnostics,
        rolling_parameters=rolling_parameters,
        rolling_scores=rolling_scores,
        verification=verify_frozen_v8_math(d, all_picks),
    )


def rebuild_current_proxy(destination):
    """Current fixed engine/calibration with retrospective carryover priors.

    This reproduces the source of moneyline margins under a research prior,
    not archived production forecasts or rich preseason/availability inputs.
    """
    from backend.features.scoring import load_scoring_team_games
    from backend.model.forecast_calibration import apply_forecast_calibration
    from backend.model.pick_calibration import build_research_forecasts
    from backend.model.process import build_process_prior
    from backend.model.weekly import load_weekly_games

    predictions = build_research_forecasts()
    predictions.to_parquet(
        destination / "current_engine_predictions.parquet", index=False
    )
    frames = []
    for year, frame in predictions.groupby("season"):
        games = load_weekly_games(int(year))
        teams = load_scoring_team_games(int(year))
        prior = build_process_prior(int(year))
        for week, f in frame.groupby("model_week"):
            frames.append(
                apply_forecast_calibration(
                    f,
                    games,
                    teams,
                    prior,
                    int(week),
                    pd.Timestamp(f.as_of.iloc[0]).to_pydatetime(),
                )
            )
    pd.concat(frames, ignore_index=True).to_parquet(
        destination / "current_v14_proxy.parquet", index=False
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).with_name("moneyline_audit_plan.json"),
    )
    parser.add_argument("--rebuild-current", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.rebuild_current or not (args.output / "current_v14_proxy.parquet").exists():
        rebuild_current_proxy(args.output)
    quotes, quality, coverage = load_quotes()
    quotes.to_parquet(args.output / "paired_quotes.parquet", index=False)
    sources = [
        (
            "v9_original",
            ROOT
            / "backend/data/processed/recommendation_calibration/predictions.parquet",
        ),
        ("v14_proxy", args.output / "current_v14_proxy.parquet"),
    ]
    reports = []
    for label, path in sources:
        print(f"Auditing {label}", flush=True)
        reports.append(run_source(label, path, quotes, args.output))
    report = dict(
        design=json.loads(args.plan.read_text()),
        quality=quality,
        coverage=coverage,
        sources=reports,
        source_hashes={
            str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
            for p in sorted(
                {
                    *ROOT.glob("backend/data/raw/lines/202[1-5].parquet"),
                    *ROOT.glob("backend/data/raw/games/20[12][0-9].parquet"),
                    *ROOT.glob("backend/data/processed/team_games/20[12][0-9].parquet"),
                    *ROOT.glob("backend/model/*.py"),
                    ROOT / "backend/recommendations.py",
                    Path(__file__),
                    Path(__file__).with_name("moneyline_audit_plan.json"),
                }
            )
            if "2026" not in p.name
        },
    )
    (args.output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "verification.json").write_text(
        json.dumps(
            {source["source"]: source["verification"] for source in reports}, indent=2
        )
        + "\n"
    )
    for key in [
        "scores",
        "picks",
        "reliability",
        "sensitivities",
        "selected_diagnostics",
        "rolling_scores",
    ]:
        pd.DataFrame([r for source in reports for r in source[key]]).to_csv(
            args.output / f"{key}.csv", index=False
        )
    print(json.dumps(quality), flush=True)
    for source in reports:
        print(source["source"], source["by_season"], source["parameters"], flush=True)
        for r in source["picks"]:
            if r["split"] == "evaluation" and r["execution"] == "best_quote":
                print(
                    r["method"],
                    r["picks"],
                    round(r.get("roi") or 0, 4),
                    r.get("roi_ci"),
                    round(r.get("bias") or 0, 4),
                    flush=True,
                )


if __name__ == "__main__":
    main()
