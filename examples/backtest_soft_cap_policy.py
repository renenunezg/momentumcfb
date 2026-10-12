"""Replay v11 through production selection against frozen v7/v10 history."""

import json
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from backend.recommendations import (
    POLICY_VERSION,
    build_recommendations,
    grade_recommendations,
)
from examples.backtest_combined_pick_policies import paired_intervals, settle, summarize

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/combined-policy-backtest-20261012"
OUT = ROOT / "output/soft-cap-backtest-20261012"


def run_week(job):
    forecasts, offers = job
    d = build_recommendations(forecasts, offers, decision_at=forecasts.as_of.iloc[0])
    d["policy"] = "new"
    d["qualification"] = np.select(
        [d.reason.eq("qualifying_edge"), d.reason.eq("volume_floor")],
        ["qualified", "exploratory"],
        default="no_play",
    )
    return d


def main():
    assert POLICY_VERSION == "cfb-picks-v11"
    OUT.mkdir(exist_ok=True, parents=True)
    paths = [
        SOURCE / name
        for name in [
            "forecasts.parquet",
            "synthetic_eligibility_offers.parquet",
            "decisions.parquet",
            "picks.parquet",
        ]
    ]
    paths += [
        Path(__file__),
        ROOT / "backend/recommendations.py",
        ROOT / "backend/model/spread_pricing.py",
        ROOT / "backend/model/spread_pricing.json",
    ]
    (OUT / "input_manifest.json").write_text(
        json.dumps(
            {
                str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                for p in paths
            },
            indent=2,
        )
        + "\n"
    )
    forecasts = pd.read_parquet(SOURCE / "forecasts.parquet")
    offers = pd.read_parquet(SOURCE / "synthetic_eligibility_offers.parquet")
    baseline = pd.read_parquet(SOURCE / "decisions.parquet").query("policy=='new'")
    jobs = [
        (g, offers[offers.season.eq(y) & offers.game_id.isin(g.game_id)])
        for (y, w), g in forecasts.groupby(["season", "model_week"])
    ]
    chunks = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i, d in enumerate(pool.map(run_week, jobs), 1):
            chunks.append(d)
            print(f"{i}/{len(jobs)} weeks", flush=True)
    decisions = pd.concat(chunks, ignore_index=True)
    decisions.to_parquet(OUT / "decisions.parquet", index=False)
    keys = ["season", "game_id", "market"]
    a = baseline.set_index(keys).sort_index()
    b = decisions.set_index(keys).sort_index()
    assert a.index.equals(b.index)
    for column in [
        "side",
        "point",
        "price",
        "edge_points",
        "win_probability",
        "push_probability",
        "expected_value_per_unit",
    ]:
        pd.testing.assert_series_equal(a[column], b[column])
    assert a.reason.eq("qualifying_edge").equals(b.reason.eq("qualifying_edge"))
    for (year, week), g in decisions.groupby(["season", "week"]):
        old = baseline[baseline.season.eq(year) & baseline.week.eq(week)]
        q = old[old.reason.eq("qualifying_edge")]
        slots = max(0, 15 - len(q))
        original_floor = old[old.reason.eq("volume_floor")].sort_values(
            ["edge_points", "game_id", "market"], ascending=[False, True, True]
        )
        keep = original_floor.groupby("market", sort=False).head((slots + 1) // 2)
        expected = set(pd.concat([q, keep]).set_index(keys).index)
        assert set(g[g.status.eq("recommended")].set_index(keys).index) == expected
    for row in decisions.itertuples():
        c = row.pricing_context
        assert c["is_postseason"] == (
            forecasts.set_index("game_id").loc[row.game_id, "season_type"]
            == "postseason"
        )
        if row.market in ("h2h", "spreads") and "model_weight" in c:
            assert c["model_weight"] == 0.2
        if "spread_edge_breakdown" in c:
            e = c["spread_edge_breakdown"]
            assert np.isclose(
                e["model_contribution_points"], 0.2 * e["model_disagreement_points"]
            )
            assert np.isclose(
                row.edge_points,
                e["model_contribution_points"]
                + e["quote_difference_points"]
                + e["price_distribution_adjustment_points"],
            )
    picks = settle(decisions[decisions.status.eq("recommended")], forecasts)
    games = (
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
    graded = (
        grade_recommendations(picks.assign(outcome="pending"), games)
        .set_index(["game_id", "market"])
        .sort_index()
    )
    expected = picks.set_index(["game_id", "market"]).sort_index()
    assert graded.outcome.equals(expected.outcome)
    assert np.allclose(graded.profit_units, expected.profit_units)
    picks.to_parquet(OUT / "picks.parquet", index=False)
    export = picks.drop(columns=["decision_forecast", "pricing_context"]).copy()
    for column in [
        "season_type",
        "is_postseason",
        "research_flags",
        "floor_shortfall",
        "floor_market_cap",
    ]:
        export[column] = [c[column] for c in picks.pricing_context]
    for column in [
        "model_disagreement_points",
        "model_contribution_points",
        "quote_difference_points",
        "price_distribution_adjustment_points",
    ]:
        export[column] = [
            c.get("spread_edge_breakdown", {}).get(column)
            for c in picks.pricing_context
        ]
    export.to_csv(OUT / "picks.csv", index=False)
    old_picks = pd.read_parquet(SOURCE / "picks.parquet")
    results = []
    for label, data, policy in [
        ("v7", old_picks, "old"),
        ("v10", old_picks, "new"),
        ("v11", picks, "new"),
    ]:
        t = summarize(data)
        results.append(t[t.policy.eq(policy)].assign(policy=label))
    summary = pd.concat(results, ignore_index=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    intervals = paired_intervals(
        forecasts,
        pd.concat([old_picks.query("policy=='new'").assign(policy="old"), picks]),
    )
    (OUT / "results.json").write_text(
        json.dumps(
            dict(
                policy=POLICY_VERSION,
                summary=json.loads(summary.to_json(orient="records")),
                paired_v11_vs_v10_intervals=intervals,
                verification=dict(
                    weeks=len(jobs),
                    production_grading_rows=len(picks),
                    qualified_unchanged=True,
                    all_pricing_unchanged=True,
                    soft_cap_matches_independent_shortlist=True,
                    metadata_reconciles=True,
                ),
                limitations=json.loads((SOURCE / "results.json").read_text())[
                    "limitations"
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    print(summary.query("split=='evaluation'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
