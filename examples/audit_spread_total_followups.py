"""Development-selected key-score calibration and weekly weight sensitivities."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backend.model.distributions import marginal_cdf
from backend.recommendations import POLICY_VERSION, build_recommendations
from examples.audit_moneyline_probabilities import interval
from examples.audit_spread_total_picks import probabilities
from examples.replay_week6_policy import settle, summary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/spread-total-audit-20261012"
WEEK = ROOT / "output/week6-policy-comparison-20261012"


def score_mass(d, market):
    grid = np.arange(-80, 81) if market == "spreads" else np.arange(0, 151)
    groups = np.abs(grid) if market == "spreads" else grid
    sd = 15.35 if market == "spreads" else 15.93
    mean = d["mean"].to_numpy()[:, None]
    masses = marginal_cdf(grid[None, :] + 0.5 - mean, sd, 500) - marginal_cdf(
        grid[None, :] - 0.5 - mean, sd, 500
    )
    return grid, groups, masses, sd


def fit_keys(d, market, penalty):
    grid, groups, masses, sd = score_mass(d, market)
    gm = np.column_stack(
        [masses[:, groups == g].sum(axis=1) for g in range(groups.max() + 1)]
    )
    y = d.actual.to_numpy()
    base = marginal_cdf(y + 0.5 - d["mean"], sd, 500) - marginal_cdf(
        y - 0.5 - d["mean"], sd, 500
    )
    obs = np.abs(y).astype(int) if market == "spreads" else y.astype(int)
    valid = np.isin(y, grid)
    counts = np.bincount(obs[valid], minlength=len(gm[0]))

    def objective(logw):
        w = np.exp(logw)
        normalizer = 1 + gm @ (w - 1)
        value = (
            np.log(normalizer).sum()
            - np.log(np.maximum(base, 1e-15)).sum()
            - counts @ logw
            + penalty / 2 * (logw @ logw)
        )
        grad = w * (gm.T @ (1 / normalizer)) - counts + penalty * logw
        return value, grad

    res = minimize(
        objective,
        np.zeros(gm.shape[1]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-3, 3)] * gm.shape[1],
    )
    assert res.success, res.message
    return np.exp(res.x)


def key_probabilities(d, market, weights):
    grid, groups, masses, sd = score_mass(d, market)
    adj = masses * (weights[groups] - 1)
    norm = 1 + adj.sum(axis=1)
    threshold = d.threshold.to_numpy()
    base = probabilities(d["mean"].to_numpy(), sd, threshold)
    for i, mask in enumerate(
        [
            grid[None, :] > threshold[:, None],
            grid[None, :] == threshold[:, None],
            grid[None, :] < threshold[:, None],
        ]
    ):
        base[:, i] = (base[:, i] + (adj * mask).sum(axis=1)) / norm
    assert np.allclose(base.sum(axis=1), 1) and base.min() >= 0
    labels = np.where(d.actual > threshold, 0, np.where(d.actual < threshold, 2, 1))
    loss = -np.log(np.maximum(base[np.arange(len(d)), labels], 1e-12))
    return base, loss


def keys():
    reports = []
    for market in ["spreads", "totals"]:
        d = pd.read_parquet(OUT / f"{market}_current_050.parquet")
        fit = d[d.season.le(2021)]
        select = d[d.season.eq(2022)]
        validation = d[d.season.ge(2023)]
        weights = {p: fit_keys(fit, market, p) for p in [10, 50, 200]}
        scores = {
            p: float(key_probabilities(select, market, w)[1].mean())
            for p, w in weights.items()
        }
        penalty = min(scores, key=scores.get)
        w = fit_keys(d[d.season.le(2022)], market, penalty)
        pp, ll = key_probabilities(validation, market, w)
        delta = ll - validation.nll
        item = dict(
            market=market,
            selected_penalty=penalty,
            selection_scores=scores,
            selection_baseline=float(select.nll.mean()),
            baseline_log_loss=float(validation.nll.mean()),
            key_log_loss=float(ll.mean()),
            delta=float(delta.mean()),
            delta_ci=interval(validation, delta),
            predicted_push=float(pp[:, 1].mean()),
            actual_push=float(validation.actual_push.mean()),
            weights=w.tolist(),
        )
        if market == "spreads":
            item["key_lines"] = [
                dict(
                    line=k,
                    games=int(validation.threshold.abs().eq(k).sum()),
                    before=float(
                        validation.loc[
                            validation.threshold.abs().eq(k), "predicted_push"
                        ].mean()
                    ),
                    after=float(pp[validation.threshold.abs().eq(k), 1].mean()),
                    actual=float(
                        validation.loc[
                            validation.threshold.abs().eq(k), "actual_push"
                        ].mean()
                    ),
                )
                for k in [3, 7, 10, 14]
            ]
        reports.append(item)
        pd.DataFrame(
            dict(
                game_id=validation.game_id,
                season=validation.season,
                model_week=validation.model_week,
                old_log_loss=validation.nll,
                new_log_loss=ll,
                predicted_push=pp[:, 1],
                actual_push=validation.actual_push,
            )
        ).to_csv(OUT / f"{market}_key_validation.csv", index=False)
    (OUT / "key_calibration.json").write_text(json.dumps(reports, indent=2) + "\n")
    print("KEY CALIBRATION", json.dumps(reports), flush=True)


def combined_spread_keys():
    d = pd.read_parquet(OUT / "spreads_research_020.parquet")
    fit, select, validation = (
        d[d.season.le(2021)],
        d[d.season.eq(2022)],
        d[d.season.ge(2023)],
    )
    scores = {
        p: float(
            key_probabilities(select, "spreads", fit_keys(fit, "spreads", p))[1].mean()
        )
        for p in [10, 50, 200]
    }
    penalty = min(scores, key=scores.get)
    weights = fit_keys(d[d.season.le(2022)], "spreads", penalty)
    pp, loss = key_probabilities(validation, "spreads", weights)
    baseline = pd.read_parquet(OUT / "spreads_current_050.parquet").query(
        "season >= 2023"
    )
    delta = loss - baseline.nll
    report = dict(
        candidate="20% model/history component plus development-selected key-score weights",
        selected_penalty=penalty,
        selection_scores=scores,
        selection_baseline=float(select.nll.mean()),
        validation_current=float(baseline.nll.mean()),
        validation_candidate=float(loss.mean()),
        change=float(delta.mean()),
        change_ci=interval(validation, delta),
        predicted_push=float(pp[:, 1].mean()),
        actual_push=float(validation.actual_push.mean()),
        weights=weights.tolist(),
    )
    (OUT / "combined_spread_candidate.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def weekly_sensitivity():
    # This earlier diagnostic alters the published margin. v10 deliberately
    # prices spreads from separate frozen inputs, so it cannot stand in for v9.
    if POLICY_VERSION != "cfb-picks-v9":
        raise ValueError(
            "Legacy weekly sensitivity requires cfb-picks-v9; replay the current "
            "policy with examples.replay_week6_policy --output-dir instead."
        )
    f = pd.read_parquet(WEEK / "inputs/projections.parquet")
    offers = pd.read_parquet(WEEK / "inputs/market_offers.parquet")
    published = pd.read_json(WEEK / "published.json")
    decision = published.decision_at.iloc[0]
    scores = published[["game_id", "home_points", "away_points"]].drop_duplicates()
    rows = []
    for market in ["spreads", "totals"]:
        altered = f.copy()
        if market == "spreads":
            reference = -f.market_home_spread
            altered["market_informed_home_margin"] = reference + 0.4 * (
                f.market_informed_home_margin - reference
            )
        else:
            reference = 2 * f.market_informed_total - f.model_total
            altered["market_informed_total"] = reference + 0.2 * (
                f.model_total - reference
            )
        decisions = build_recommendations(altered, offers, decision_at=decision)
        picks = decisions[decisions.status.eq("recommended")].drop(
            columns=["home_points", "away_points"], errors="ignore"
        )
        picks = settle(picks, scores)
        picks.to_csv(OUT / f"week6_{market}_020_sensitivity.csv", index=False)
        rows.append(
            dict(
                changed_market=market,
                summary=summary(picks),
                normal_gate=int(picks.reason.eq("qualifying_edge").sum()),
                floor=int(picks.reason.eq("volume_floor").sum()),
                by_market={m: summary(g) for m, g in picks.groupby("market")},
            )
        )
        print("WEEK SENSITIVITY", rows[-1], flush=True)
    (OUT / "week6_weight_sensitivity.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-week-sensitivity", action="store_true")
    args = parser.parse_args()
    if args.legacy_week_sensitivity:
        weekly_sensitivity()
    else:
        keys()
        combined_spread_keys()
