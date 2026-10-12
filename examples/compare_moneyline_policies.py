"""Compare fixed old/new 20% moneyline policies on identical historical cohorts.

Historical odds have no capture timestamps: returns are closing-price proxies.
Run: poetry run python -m examples.compare_moneyline_policies
"""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from backend.model.distributions import marginal_cdf
from backend.odds.markets import OFFER_COLUMNS
from backend.recommendations import H2H_MODEL_WEIGHT, build_recommendations
from examples.audit_moneyline_probabilities import (
    DEFAULT_OUTPUT,
    DF,
    EVALUATION,
    GATE,
    ROOT,
    SCALE,
    SD,
    interval,
    load_forecasts,
    load_quotes,
    losses,
    pick_summary,
    select_picks,
)

OUTPUT = ROOT / "output/moneyline-policy-comparison-20261012"


def paired_returns(games, old, new):
    """Paired season-week bootstrap; include weeks with zero selected bets."""
    d = games[["season", "model_week", "game_id"]].copy()
    for label, picks in [("old", old), ("new", new)]:
        p = picks[["season", "game_id", "profit"]].assign(stake=1)
        p = p.rename(columns={"profit": f"{label}_profit", "stake": f"{label}_stake"})
        d = d.merge(p, on=["season", "game_id"], how="left", validate="one_to_one")
    d = d.fillna(0)
    weekly = d.groupby(["season", "model_week"])[
        [f"{a}_{b}" for a in ["old", "new"] for b in ["profit", "stake"]]
    ].sum()
    rng = np.random.default_rng(20261012)
    samples = rng.integers(0, len(weekly), (5000, len(weekly)))
    sampled = {c: weekly[c].to_numpy()[samples].sum(axis=1) for c in weekly}
    delta = (
        sampled["new_profit"] / sampled["new_stake"]
        - sampled["old_profit"] / sampled["old_stake"]
    )
    return dict(
        roi_difference=float(new.profit.mean() - old.profit.mean()),
        roi_difference_ci=np.quantile(delta, [0.025, 0.975]).tolist(),
        profit_difference=float(new.profit.sum() - old.profit.sum()),
        profit_difference_ci=np.quantile(
            sampled["new_profit"] - sampled["old_profit"], [0.025, 0.975]
        ).tolist(),
    )


def production_parity(d, quotes, expected):
    """Verify math/selection with synthetic timestamp fixtures, not eligibility claims."""
    forecast = d[
        [
            "season",
            "model_week",
            "game_id",
            "home_team",
            "away_team",
            "home_margin",
            "model_total",
            "as_of",
            "start_date",
        ]
    ].copy()
    forecast["week"] = forecast.model_week
    forecast["pure_home_margin"] = forecast.home_margin
    forecast["market_informed_home_margin"] = forecast.home_margin
    forecast["model_version"] = "historical-parity-fixture"
    forecast["margin_sd"], forecast["total_sd"], forecast["degrees_of_freedom"] = (
        SD,
        15.93,
        DF,
    )
    forecast["home_missing_input_count"] = forecast["away_missing_input_count"] = 0
    q = quotes.merge(
        forecast[["season", "game_id", "as_of", "start_date"]],
        on=["season", "game_id"],
        validate="many_to_one",
    )
    offers = []
    for side in ["home", "away"]:
        a = q.copy()
        a["market"], a["selection"], a["point"], a["price"] = (
            "h2h",
            side,
            None,
            a[f"{side}Moneyline"],
        )
        a["provider_key"] = a.provider
        a["market_fetched_at"] = a["provider_last_update"] = a.as_of
        a["commence_time"] = a.start_date
        a["odds_api_event_id"] = "fixture-" + a.game_id.astype(str)
        a["match_score"], a["execution_eligibility_verified"] = 1.0, True
        offers.append(a.reindex(columns=OFFER_COLUMNS))
    offers = pd.concat(offers, ignore_index=True)
    decisions = []
    for as_of, group in forecast.groupby("as_of"):
        decisions.append(
            build_recommendations(
                group, offers[offers.game_id.isin(group.game_id)], decision_at=as_of
            )
        )
    result = pd.concat(decisions, ignore_index=True).query("market == 'h2h'")
    desired = expected.set_index(["season", "game_id"])
    actual = result.query("status == 'recommended'").set_index(["season", "game_id"])
    assert set(desired.index) == set(actual.index)
    desired, actual = desired.sort_index(), actual.sort_index()
    assert desired.side.equals(actual.side)
    assert np.allclose(desired.price, actual.price)
    assert desired.provider.equals(actual.provider)
    err = np.abs(desired.p.to_numpy() - actual.win_probability.to_numpy()).max()
    edge_err = np.abs(
        desired.edge_points.to_numpy() - actual.edge_points.to_numpy()
    ).max()
    assert err < 1e-12 and edge_err < 1e-9
    assert all(c["model_weight"] == 0.2 for c in result.pricing_context)
    return dict(
        games_checked=len(result),
        selected_picks_checked=len(actual),
        max_probability_error=float(err),
        max_edge_error=float(edge_err),
        provenance="Synthetic pregame timestamps test production math only; historical execution eligibility is unknown.",
    )


def run_source(label, path, quotes):
    d = load_forecasts(path, quotes)
    assert H2H_MODEL_WEIGHT == 0.2
    pp = dict(
        old_spread_020=marginal_cdf(
            0.8 * d.market_margin + 0.2 * d.home_margin, SD, DF
        ),
        new_paired_ml_020=marginal_cdf(
            0.8 * SCALE * t.ppf(d.market_p, DF) + 0.2 * d.home_margin, SD, DF
        ),
    )
    picks, summaries, scores = [], [], []
    for method, p in pp.items():
        d[f"p_{method}"] = p
        for execution, provider in [("best_quote", None), ("Bovada", "Bovada")]:
            a = select_picks(d, quotes, p, provider=provider)
            a["method"], a["execution"], a["source"] = method, execution, label
            picks.append(a)
            for split, mask in [
                ("evaluation", a.season.isin(EVALUATION)),
                ("development", a.season.isin([2021, 2022])),
                *[(str(y), a.season.eq(y)) for y in EVALUATION],
            ]:
                summaries.append(
                    dict(
                        source=label,
                        method=method,
                        execution=execution,
                        split=split,
                        **pick_summary(a[mask]),
                    )
                )
        for split, mask in [
            ("evaluation", d.season.isin(EVALUATION)),
            *[(str(y), d.season.eq(y)) for y in EVALUATION],
        ]:
            a = d[mask]
            p0 = np.asarray(p)[mask]
            scores.append(
                dict(
                    source=label,
                    method=method,
                    split=split,
                    games=len(a),
                    log_loss=float(losses(a.y, p0).mean()),
                    brier=float(np.mean((p0 - a.y) ** 2)),
                    market_log_loss=float(losses(a.y, a.market_p).mean()),
                    market_brier=float(np.mean((a.market_p - a.y) ** 2)),
                )
            )
    picks = pd.concat(picks, ignore_index=True)
    evaluation = d[d.season.isin(EVALUATION)]
    selected = picks[picks.season.isin(EVALUATION) & picks.execution.eq("best_quote")]
    old = selected[selected.method.eq("old_spread_020")]
    new = selected[selected.method.eq("new_paired_ml_020")]
    comparison = old.merge(
        new,
        on=["season", "model_week", "game_id"],
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
        validate="one_to_one",
    )
    comparison["change"] = (
        comparison["_merge"]
        .map({"left_only": "removed", "right_only": "added", "both": "retained"})
        .astype(str)
    )
    comparison.loc[
        (comparison["_merge"] == "both") & comparison.side_old.ne(comparison.side_new),
        "change",
    ] = "flipped_side"
    comparison["profit_delta"] = comparison.profit_new.fillna(
        0
    ) - comparison.profit_old.fillna(0)
    changes = []
    for change, a in comparison.groupby("change"):
        changes.append(
            dict(
                change=change,
                games=len(a),
                old_profit=float(a.profit_old.sum()),
                new_profit=float(a.profit_new.sum()),
                profit_delta=float(a.profit_delta.sum()),
            )
        )
    assert np.isclose(
        comparison.profit_delta.sum(), new.profit.sum() - old.profit.sum()
    )
    ll_delta = losses(evaluation.y, evaluation.p_new_paired_ml_020) - losses(
        evaluation.y, evaluation.p_old_spread_020
    )
    brier_delta = (evaluation.p_new_paired_ml_020 - evaluation.y) ** 2 - (
        evaluation.p_old_spread_020 - evaluation.y
    ) ** 2
    diagnostics = []
    for method, a in selected.groupby("method"):
        for segment, group in [
            ("underdogs", a[a.price.gt(0)]),
            ("favorites", a[a.price.lt(0)]),
            ("price_500_plus", a[a.price.ge(500)]),
            ("price_below_500", a[a.price.lt(500)]),
        ]:
            diagnostics.append(
                dict(method=method, segment=segment, **pick_summary(group))
            )
    new_all = picks[
        picks.method.eq("new_paired_ml_020") & picks.execution.eq("best_quote")
    ]
    parity = production_parity(d, quotes, new_all)
    for name, frame in [
        ("games", d),
        ("picks", picks),
        ("changes", comparison),
        ("segments", pd.DataFrame(diagnostics)),
    ]:
        frame.to_parquet(OUTPUT / f"{label}_{name}.parquet", index=False)
        if name != "games":
            frame.to_csv(OUTPUT / f"{label}_{name}.csv", index=False)
    return dict(
        source=label,
        input=str(path.relative_to(ROOT)),
        input_sha256=sha256(path.read_bytes()).hexdigest(),
        games=len(d),
        evaluation_games=len(evaluation),
        summaries=summaries,
        scores=scores,
        changes=changes,
        segments=diagnostics,
        paired=paired_returns(evaluation, old, new),
        probability_comparison=dict(
            log_loss_delta=float(ll_delta.mean()),
            log_loss_delta_ci=interval(evaluation, ll_delta),
            brier_delta=float(brier_delta.mean()),
            brier_delta_ci=interval(evaluation, brier_delta),
        ),
        production_parity=parity,
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    quotes, quality, coverage = load_quotes()
    sources = [
        ("v14_proxy", DEFAULT_OUTPUT / "current_v14_proxy.parquet"),
        (
            "v9_original",
            ROOT
            / "backend/data/processed/recommendation_calibration/predictions.parquet",
        ),
    ]
    reports = []
    for label, path in sources:
        print(f"Comparing {label}", flush=True)
        reports.append(run_source(label, path, quotes))
    result = dict(
        model_weight=0.2,
        gate_points=GATE,
        stakes="one unit per pick",
        evaluation_years=list(EVALUATION),
        quality=quality,
        coverage=coverage,
        limitations=[
            "Historical closing quotes have no decision timestamps; freshness and executability cannot be verified.",
            "Old cross-market volume floor cannot be reconstructed without historical spread/total prices; both headline policies use the fixed two-point gate only.",
            "2023-2025 are reused historical evaluation seasons, not pristine holdouts.",
            "Current engine replay uses retrospective historical carryover priors, not archived rich preseason or availability forecasts.",
        ],
        sources=reports,
        code_hashes={
            str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                ROOT / "examples/audit_moneyline_probabilities.py",
                ROOT / "backend/recommendations.py",
            ]
        },
    )
    (OUTPUT / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    pd.DataFrame([row for r in reports for row in r["summaries"]]).to_csv(
        OUTPUT / "summary.csv", index=False
    )
    for r in reports:
        print(
            r["source"],
            json.dumps(
                {
                    k: r[k]
                    for k in [
                        "changes",
                        "paired",
                        "probability_comparison",
                        "production_parity",
                    ]
                }
            ),
            flush=True,
        )
        print(
            pd.DataFrame(r["summaries"])
            .query("execution=='best_quote'")[
                [
                    "method",
                    "split",
                    "picks",
                    "wins",
                    "profit",
                    "roi",
                    "probability",
                    "win_rate",
                ]
            ]
            .to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
