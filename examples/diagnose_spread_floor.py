"""Diagnose frozen v10 spread tiers and compare research floor allocation rules.

Run: poetry run python -m examples.diagnose_spread_floor
This is a closing-line proxy audit, not a policy fit or an executable ledger.
No production policy, model weight, or recommendation gate is changed.
"""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from backend.model.spread_pricing import spread_win_loss
from backend.recommendations import grade_recommendations
from examples.backtest_combined_pick_policies import settle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/combined-policy-backtest-20261012"
OUT = ROOT / "output/spread-floor-diagnosis-20261012"
VARIANTS = ("points", "ev", "balanced_ev", "capped_ev", "soft_cap")


def summary(frame):
    resolved = frame.outcome.isin(["win", "loss"])
    return dict(
        n=len(frame),
        wins=int(frame.outcome.eq("win").sum()),
        losses=int(frame.outcome.eq("loss").sum()),
        pushes=int(frame.outcome.eq("push").sum()),
        profit=float(frame.profit_units.sum()),
        roi=float(frame.profit_units.mean()),
        observed=float(frame.loc[resolved, "outcome"].eq("win").mean()),
        predicted=float((frame.win_probability / (1 - frame.push_probability)).mean()),
        expected_profit=float(frame.expected_value_per_unit.sum()),
    )


def spread_diagnostics(decisions, forecasts):
    spread = decisions.query("market == 'spreads'").merge(
        forecasts[
            ["season", "game_id", "reference_spread", "model_side", "season_type"]
        ],
        on=["season", "game_id"],
        validate="one_to_one",
    )
    spread["reference_margin"] = -spread.reference_spread
    spread["direction"] = np.where(spread.side.eq("home"), 1, -1)
    spread["model_shift"] = spread.direction * (
        spread.model_home_margin - spread.reference_margin
    )
    spread["line_advantage"] = spread.direction * spread.reference_margin + spread.point
    spread["favorite"] = np.where(spread.point < 0, "favorite", "underdog")
    spread["has_book_difference"] = spread.line_advantage.abs() > 0.01
    spread["week_band"] = np.where(spread.week <= 5, "early", "late")
    spread["line_band"] = pd.cut(
        spread.point.abs(), [0, 3, 7, 14, 100], include_lowest=True
    ).astype(str)
    spread["key_line"] = spread.point.abs().isin([3, 7, 10, 14])
    spread["mean_error"] = spread.direction * (
        spread.actual_home_points - spread.actual_away_points - spread.model_home_margin
    )
    baseline = np.array(
        [
            spread_win_loss(
                row.direction * row.reference_margin, 15.35, 500, -row.point
            )
            for row in spread.itertuples()
        ]
    )
    spread["baseline_conditional"] = baseline[:, 0] / baseline.sum(axis=1)
    spread["predicted_conditional"] = spread.win_probability / (
        1 - spread.push_probability
    )
    rows = []
    for period, mask in [
        ("development", spread.season.le(2022)),
        ("evaluation", spread.season.ge(2023)),
    ]:
        for tier, group in spread[mask].groupby("qualification"):
            cuts = [("tier", tier, group)]
            if tier != "no_play":
                for column in [
                    "season",
                    "season_type",
                    "favorite",
                    "side",
                    "line_band",
                    "key_line",
                    "has_book_difference",
                    "week_band",
                ]:
                    cuts.extend(
                        (column, label, g) for label, g in group.groupby(column)
                    )
            for cut, label, g in cuts:
                rows.append(
                    dict(
                        split=period,
                        tier=tier,
                        cut=cut,
                        label=str(label),
                        **summary(g),
                        model_shift=float(g.model_shift.mean()),
                        line_advantage=float(g.line_advantage.mean()),
                        baseline=float(g.baseline_conditional.mean()),
                        mean_error=float(g.mean_error.mean()),
                    )
                )
    spread.to_parquet(OUT / "spread_diagnostics.parquet", index=False)
    pd.DataFrame(rows).to_csv(OUT / "spread_decomposition.csv", index=False)
    return spread


def select_floor(batch, variant):
    qualified = batch[batch.reason.eq("qualifying_edge")]
    eligible = batch[
        batch.reason.isin(["below_edge_threshold", "volume_floor"])
        & batch.market.ne("h2h")
        & batch.edge_points.gt(0)
        & batch.expected_value_per_unit.gt(0)
    ]
    needed = max(0, 15 - len(qualified))
    if variant in ("points", "ev", "soft_cap"):
        ranking = "expected_value_per_unit" if variant == "ev" else "edge_points"
        chosen = eligible.sort_values(
            [ranking, "game_id"], ascending=[False, True]
        ).head(needed)
        if variant == "soft_cap":
            # Exposure limit only. Keep the original shortlist; never replace a
            # capped total with an otherwise unselected spread (or vice versa).
            chosen = chosen.groupby("market", sort=False).head((needed + 1) // 2)
    else:
        queues = {
            market: list(
                g.sort_values(
                    ["expected_value_per_unit", "game_id"], ascending=[False, True]
                ).index
            )
            for market, g in eligible.groupby("market")
        }
        counts = (
            qualified.market.value_counts().to_dict()
            if variant == "balanced_ev"
            else {}
        )
        caps = {market: needed // 2 for market in ("spreads", "totals")}
        if needed % 2 and queues:
            best = max(
                queues,
                key=lambda m: eligible.loc[queues[m][0], "expected_value_per_unit"],
            )
            caps[best] += 1
        indices = []
        while len(indices) < needed:
            available = [
                m
                for m in queues
                if queues[m]
                and (variant == "balanced_ev" or counts.get(m, 0) < caps[m])
            ]
            if not available:
                break
            if variant == "balanced_ev":
                market = min(
                    available,
                    key=lambda m: (
                        counts.get(m, 0),
                        -eligible.loc[queues[m][0], "expected_value_per_unit"],
                        m,
                    ),
                )
            else:
                market = max(
                    available,
                    key=lambda m: eligible.loc[queues[m][0], "expected_value_per_unit"],
                )
            indices.append(queues[market].pop(0))
            counts[market] = counts.get(market, 0) + 1
        chosen = eligible.loc[indices]
    return pd.concat(
        [qualified.assign(tier="qualified"), chosen.assign(tier="exploratory")]
    ).assign(variant=variant, stake_units=1.0)


def floor_comparison(decisions, forecasts):
    picks = pd.concat(
        [
            select_floor(batch, variant)
            for _, batch in decisions.groupby(["season", "week"])
            for variant in VARIANTS
        ],
        ignore_index=True,
    )
    # Outcomes have already been attached for diagnosis, but select_floor uses
    # only forecast/quote fields, reasons, counts, and deterministic tie breaks.
    shuffled = decisions.copy()
    shuffled["profit_units"] = -1000.0
    shuffled["outcome"] = "loss"
    for _, batch in shuffled.groupby(["season", "week"]):
        original = decisions.loc[batch.index]
        for variant in VARIANTS:
            assert select_floor(batch, variant).index.equals(
                select_floor(original, variant).index
            )
    keys = ["season", "week", "game_id", "market"]
    baseline = picks[picks.variant.eq("points")].set_index(keys).sort_index()
    original = (
        decisions[decisions.status.eq("recommended")].set_index(keys).sort_index()
    )
    assert baseline.index.equals(original.index)
    assert np.allclose(baseline.profit_units, original.profit_units)
    actual_games = (
        forecasts[
            [
                "game_id",
                "start_date",
                "home_team",
                "away_team",
                "actual_home_points",
                "actual_away_points",
            ]
        ]
        .rename(
            columns={
                "game_id": "id",
                "actual_home_points": "home_points",
                "actual_away_points": "away_points",
            }
        )
        .assign(completed=True)
    )
    for variant, group in picks.groupby("variant"):
        graded = grade_recommendations(
            group.assign(status="recommended", outcome="pending"), actual_games
        )
        graded = graded.set_index(["game_id", "market"]).sort_index()
        expected = group.set_index(["game_id", "market"]).sort_index()
        assert graded.outcome.equals(expected.outcome)
        assert np.allclose(graded.profit_units, expected.profit_units)
        assert (
            group[group.tier.eq("qualified")]
            .set_index(keys)
            .index.equals(
                picks.query("variant == 'points' and tier == 'qualified'")
                .set_index(keys)
                .index
            )
        )
    picks.to_parquet(OUT / "floor_candidate_picks.parquet", index=False)
    rows = []
    splits = [
        ("development", picks.season.le(2022)),
        ("evaluation", picks.season.ge(2023)),
        *[(str(y), picks.season.eq(y)) for y in range(2020, 2026)],
    ]
    for split, mask in splits:
        for variant, group in picks[mask].groupby("variant"):
            for tier in ("all", "qualified", "exploratory"):
                for market in ("all", "h2h", "spreads", "totals"):
                    g = group if tier == "all" else group[group.tier.eq(tier)]
                    g = g if market == "all" else g[g.market.eq(market)]
                    rows.append(
                        dict(
                            split=split,
                            variant=variant,
                            tier=tier,
                            market=market,
                            **summary(g),
                        )
                    )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "floor_candidates.csv", index=False)
    return picks, table


def uncertainty(spread, forecasts):
    group = spread.query("season >= 2023 and qualification != 'no_play'")
    weeks = pd.MultiIndex.from_frame(
        forecasts.query("season >= 2023")[["season", "week"]].drop_duplicates()
    )
    sample = np.random.default_rng(20261012).integers(
        0, len(weeks), (10000, len(weeks))
    )
    rows = []
    for kind, population in [
        ("all", group),
        ("regular", group[group.season_type.eq("regular")]),
    ]:
        draws = {}
        for tier in ("qualified", "exploratory"):
            g = population[population.qualification.eq(tier)].assign(
                win=lambda x: x.outcome.eq("win").astype(int),
                resolved=lambda x: x.outcome.isin(["win", "loss"]).astype(int),
            )
            w = (
                g.groupby(["season", "week"])[["win", "resolved"]]
                .sum()
                .reindex(weeks, fill_value=0)
            )
            wins, n = [w[c].to_numpy()[sample].sum(axis=1) for c in ("win", "resolved")]
            draws[tier] = np.divide(wins, n, out=np.full(len(n), np.nan), where=n > 0)
        rows.append(
            dict(
                population=kind,
                qualified_minus_exploratory_win_rate_interval=np.nanquantile(
                    draws["qualified"] - draws["exploratory"], [0.025, 0.975]
                ).tolist(),
            )
        )
    floor_weeks = group[group.qualification.eq("exploratory")][
        ["season", "week"]
    ].drop_duplicates()
    matched = group.merge(floor_weeks, on=["season", "week"], validate="many_to_one")
    matched_rows = [
        dict(tier=tier, **summary(g)) for tier, g in matched.groupby("qualification")
    ]
    result = dict(intervals=rows, same_floor_weeks=matched_rows)
    (OUT / "spread_uncertainty.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [
        SOURCE / name
        for name in ("decisions.parquet", "forecasts.parquet", "picks.parquet")
    ]
    paths += [
        Path(__file__),
        ROOT / "backend/model/spread_pricing.py",
        ROOT / "backend/model/spread_pricing.json",
    ]
    manifest = {
        str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths
    }
    (OUT / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    forecasts = pd.read_parquet(SOURCE / "forecasts.parquet")
    decisions = (
        pd.read_parquet(SOURCE / "decisions.parquet").query("policy == 'new'").copy()
    )
    assert decisions.policy_version.eq("cfb-picks-v10").all()
    for context in decisions[decisions.market.isin(["h2h", "spreads"])].pricing_context:
        assert context["model_weight"] == 0.2
    # Counterfactual settlement of every priced candidate allows comparison of
    # allocation rules. This does not change their recorded recommended status.
    decisions = settle(
        decisions[decisions.win_probability.notna() & decisions.price.notna()].assign(
            stake_units=1.0
        ),
        forecasts,
    )
    spread = spread_diagnostics(decisions, forecasts)
    picks, table = floor_comparison(decisions, forecasts)
    intervals = uncertainty(spread, forecasts)
    report = dict(
        objective="Explain spread tier reversal and evaluate floor allocation mechanics without fitting a production policy.",
        recommendations_enabled=True,
        model_weight=0.2,
        policy_changed=False,
        development="2020-2022; includes data used to fit existing key calibration; 2020 has no moneyline quotes",
        evaluation="2023-2025 reused evaluation seasons, not an untouched holdout",
        soft_cap="After the current floor shortlist, allow at most ceil(shortfall/2) exploratory picks per spread/total market; leave unused slots empty. Structural exposure scenario, added after observing replacement tests; not selected by validation ROI.",
        uncertainty=intervals,
        validation=dict(
            baseline_reproduced=True,
            qualified_picks_preserved=True,
            selection_independent_of_outcomes=True,
            production_grading_matched_rows=len(picks),
        ),
        limitations=json.loads((SOURCE / "results.json").read_text())["limitations"],
    )
    (OUT / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        table.query(
            "split in ['development','evaluation'] and tier == 'exploratory' and market != 'h2h'"
        ).to_string(index=False)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
