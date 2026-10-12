"""Frozen v7 versus v10 across all markets with their shared weekly volume floor.

Closing-quote proxy, not an executable historical ledger. Spread/total prices
are assumed -110; fixture timestamps exercise selection, not historical freshness.
Run: poetry run python -m examples.backtest_combined_pick_policies
"""

import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from backend.model.market_history import history_weight
from backend.odds.markets import OFFER_COLUMNS
from backend.recommendations import build_recommendations

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/combined-policy-backtest-20261012"
SOURCE = ROOT / "output/spread-total-audit-20261012/historical_inputs.parquet"
OLD = (
    ROOT / "output/week6-policy-comparison-20261012/inputs/original_recommendations.py"
)
EVAL = [2023, 2024, 2025]


def old_policy():
    spec = importlib.util.spec_from_file_location("archived_policy_v7", OLD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.POLICY_VERSION == "cfb-picks-v7"
    return module


def inputs():
    d = pd.read_parquet(SOURCE)
    assert len(d) == 4227 and not d.duplicated(["season", "game_id"]).any()
    dates = {
        c: pd.to_datetime(d[c], utc=True)
        for c in [
            "as_of",
            "start_date",
            "training_latest_start",
            "last_training_kickoff",
        ]
    }
    assert (dates["training_latest_start"] < dates["as_of"]).all()
    assert (dates["last_training_kickoff"] < dates["as_of"]).all()
    assert (dates["as_of"] < dates["start_date"]).all()
    assert d.groupby(["season", "model_week"]).as_of.nunique().eq(1).all()
    d["week"] = d.model_week
    d["pure_home_margin"] = d.home_margin
    d["market_home_spread"] = d.reference_spread
    d["market_history_weight"] = d.model_week.map(history_weight)
    d["market_informed_home_margin"] = 0.5 * d.model_side - 0.5 * d.reference_spread
    d["market_informed_total"] = 0.5 * d.model_total + 0.5 * d.reference_total
    d["market_total_weight"] = 0.5
    d["market_weight"] = 0.5
    d["home_missing_input_count"] = d["away_missing_input_count"] = 0
    d["degrees_of_freedom"] = 500.0
    d["model_version"] = "v14_historical_proxy_synthetic_eligibility"
    records = []
    for year in sorted(d.season.unique()):
        raw = pd.read_parquet(ROOT / f"backend/data/raw/lines/{year}.parquet")
        records.extend(
            dict(season=year, game_id=r.game_id, **q)
            for r in raw.itertuples()
            for q in r.lines
        )
    q = pd.DataFrame(records)
    assert not q.duplicated(["season", "game_id", "provider"]).any()
    for col in ["homeMoneyline", "awayMoneyline"]:
        q[col] = pd.to_numeric(q[col], errors="coerce")
    q = q.merge(
        d[["season", "game_id", "as_of", "start_date"]],
        on=["season", "game_id"],
        validate="many_to_one",
    )
    both = (
        np.isfinite(q[["homeMoneyline", "awayMoneyline"]]).all(axis=1)
        & q.homeMoneyline.abs().ge(100)
        & q.awayMoneyline.abs().ge(100)
    )

    def implied(v):
        return np.where(v > 0, 100, -v) / (100 + np.abs(v))

    hold = implied(q.loc[both, "homeMoneyline"]) + implied(q.loc[both, "awayMoneyline"])
    quality = dict(
        games=len(d),
        evaluation_games=int(d.season.isin(EVAL).sum()),
        weeks=int(d.groupby(["season", "model_week"]).ngroups),
        paired_moneyline_book_rows=int(both.sum()),
        abnormal_hold_pairs=int(((hold < 1) | (hold > 1.2)).sum()),
        missing_bovada_spread=int(d.offer_spread.isna().sum()),
        missing_bovada_total=int(d.offer_total.isna().sum()),
    )
    rows = []
    for side in ["home", "away"]:
        r = q.copy()
        r["market"], r["selection"], r["point"] = "h2h", side, np.nan
        r["price"] = r[f"{side}Moneyline"]
        rows.append(r)
    for market, col, sides in [
        ("spreads", "offer_spread", ["home", "away"]),
        ("totals", "offer_total", ["over", "under"]),
    ]:
        for side in sides:
            r = d[["season", "game_id", "as_of", "start_date", col]].copy()
            r["market"], r["selection"], r["provider"] = market, side, "Bovada"
            r["point"] = r[col] * (-1 if side == "away" else 1)
            r["price"] = -110.0
            rows.append(r)
    offers = pd.concat(rows, ignore_index=True)
    offers["provider_key"] = offers.provider
    offers["market_fetched_at"] = offers["provider_last_update"] = offers.as_of
    offers["commence_time"] = offers.start_date
    offers["odds_api_event_id"] = "SYNTHETIC-ELIGIBILITY-" + offers.game_id.astype(str)
    offers["match_score"], offers["execution_eligibility_verified"] = 1.0, True
    offers = offers.reindex(columns=["season", *OFFER_COLUMNS])
    return d, offers, quality


def run_week(job):
    season, week, forecasts, offers = job
    paths = [
        OUT / "batches" / f"{season}-{week}-{name}.parquet" for name in ["old", "new"]
    ]
    if all(p.exists() for p in paths):
        return season, week, "cached"
    old = old_policy()
    for name, builder, path in zip(
        ["old", "new"], [old.build_recommendations, build_recommendations], paths
    ):
        decisions = builder(forecasts, offers, decision_at=forecasts.as_of.iloc[0])
        decisions["policy"] = name
        decisions["qualification"] = np.select(
            [
                decisions.reason.eq("qualifying_edge"),
                decisions.reason.eq("volume_floor"),
            ],
            ["qualified", "exploratory"],
            default="no_play",
        )
        decisions.to_parquet(path, index=False)
    return season, week, "ran"


def settle(decisions, games):
    d = decisions.merge(
        games[["season", "game_id", "actual_home_points", "actual_away_points"]],
        on=["season", "game_id"],
        validate="many_to_one",
    )
    margin, total = (
        d.actual_home_points - d.actual_away_points,
        d.actual_home_points + d.actual_away_points,
    )
    side_margin = np.where(d.side.eq("home"), margin, -margin)
    balance = np.where(
        d.market.eq("h2h"),
        side_margin,
        np.where(
            d.market.eq("spreads"),
            side_margin + d.point,
            (total - d.point) * np.where(d.side.eq("over"), 1, -1),
        ),
    )
    d["outcome"] = np.where(
        balance > 0,
        "win",
        np.where(balance < 0, "loss", np.where(d.market.eq("h2h"), "void", "push")),
    )
    d["profit_units"] = np.where(
        balance > 0,
        np.where(d.price > 0, d.price / 100, 100 / -d.price),
        np.where(balance < 0, -1, 0),
    )
    assert np.isfinite(balance).all()
    assert d.stake_units.eq(1).all()
    return d


def metrics(d):
    wins, losses, pushes, voids = [
        int(d.outcome.eq(x).sum()) for x in ["win", "loss", "push", "void"]
    ]
    staked = len(d) - voids
    return dict(
        picks=len(d),
        games=d.game_id.nunique(),
        wins=wins,
        losses=losses,
        pushes=pushes,
        voids=voids,
        win_rate=wins / (wins + losses) if wins + losses else None,
        profit=float(d.profit_units.sum()),
        staked=staked,
        roi=float(d.profit_units.sum() / staked) if staked else None,
        expected_profit=float(d.expected_value_per_unit.sum()),
    )


def summarize(picks):
    rows = []
    splits = [
        ("evaluation", picks.season.isin(EVAL)),
        ("development", picks.season.le(2022)),
        *[(str(y), picks.season.eq(y)) for y in range(2020, 2026)],
    ]
    for split, mask in splits:
        for policy in ["old", "new"]:
            for market in ["all", "h2h", "spreads", "totals"]:
                for tier in ["all", "qualified", "exploratory"]:
                    g = picks[mask & picks.policy.eq(policy)]
                    if market != "all":
                        g = g[g.market.eq(market)]
                    if tier != "all":
                        g = g[g.qualification.eq(tier)]
                    rows.append(
                        dict(
                            split=split,
                            policy=policy,
                            market=market,
                            qualification=tier,
                            **metrics(g),
                        )
                    )
    return pd.DataFrame(rows)


def paired_intervals(games, picks):
    weeks = pd.MultiIndex.from_frame(
        games[games.season.isin(EVAL)][["season", "model_week"]]
        .drop_duplicates()
        .rename(columns={"model_week": "week"})
    )
    rng = np.random.default_rng(20261012)
    sample = rng.integers(0, len(weeks), (5000, len(weeks)))
    rows = []
    for market in ["all", "h2h", "spreads", "totals"]:
        for tier in ["all", "qualified", "exploratory"]:
            g = picks[picks.season.isin(EVAL)]
            if market != "all":
                g = g[g.market.eq(market)]
            if tier != "all":
                g = g[g.qualification.eq(tier)]
            draws = {}
            for policy in ["old", "new"]:
                a = g[g.policy.eq(policy)].assign(
                    staked=lambda x: x.outcome.ne("void").astype(int)
                )
                a = (
                    a.groupby(["season", "week"])[["profit_units", "staked"]]
                    .sum()
                    .reindex(weeks, fill_value=0)
                )
                profit, stake = [
                    a[c].to_numpy()[sample].sum(axis=1)
                    for c in ["profit_units", "staked"]
                ]
                roi = np.divide(
                    profit, stake, out=np.full(len(stake), np.nan), where=stake > 0
                )
                draws[policy] = (profit, roi)
                rows.append(
                    dict(
                        market=market,
                        qualification=tier,
                        comparison=policy,
                        roi_ci=np.nanquantile(roi, [0.025, 0.975]).tolist()
                        if np.isfinite(roi).any()
                        else [None, None],
                    )
                )
            rows.append(
                dict(
                    market=market,
                    qualification=tier,
                    comparison="new_minus_old",
                    profit_delta_ci=np.quantile(
                        draws["new"][0] - draws["old"][0], [0.025, 0.975]
                    ).tolist(),
                    roi_delta_ci=np.nanquantile(
                        draws["new"][1] - draws["old"][1], [0.025, 0.975]
                    ).tolist()
                    if np.isfinite(draws["new"][1]).any()
                    and np.isfinite(draws["old"][1]).any()
                    else [None, None],
                )
            )
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "batches").mkdir(exist_ok=True)
    paths = [
        Path(__file__),
        SOURCE,
        OLD,
        ROOT / "backend/recommendations.py",
        ROOT / "backend/model/spread_pricing.py",
        ROOT / "backend/model/spread_pricing.json",
        ROOT / "backend/odds/markets.py",
        ROOT / "backend/model/distributions.py",
        *[
            ROOT / f"backend/data/raw/lines/{year}.parquet"
            for year in range(2020, 2026)
        ],
    ]
    fingerprint = {
        str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths
    }
    manifest = OUT / "input_manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == fingerprint, (
            "Source/code changed; use a fresh output directory."
        )
    manifest.write_text(json.dumps(fingerprint, indent=2) + "\n")
    d, offers, quality = inputs()
    d.to_parquet(OUT / "forecasts.parquet", index=False)
    offers.to_parquet(OUT / "synthetic_eligibility_offers.parquet", index=False)
    (OUT / "analysis_plan.json").write_text(
        json.dumps(
            dict(
                old="archived cfb-picks-v7",
                new="local cfb-picks-v10",
                evaluation=EVAL,
                development=[2020, 2021, 2022],
                stakes="flat one unit per pick; shared game outcomes stay together",
                batches="one season/model_week batch with 15-pick floor shared across all markets",
                moneyline="all archived book prices; both policies receive identical raw pairs; new policy applies its pairing and hold gates",
                spread_total="Bovada closing lines, assumed -110 on both sides",
                chronology="forecast model training predates as_of and kickoff; quote availability is unknown; eligibility timestamps are synthetic",
                fit="no refitting or threshold tuning",
                bootstrap="5000 paired season-week resamples; includes zero-pick weeks",
                quality=quality,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(quality), flush=True)
    jobs = [
        (
            int(s),
            int(w),
            g,
            offers[offers.season.eq(s) & offers.game_id.isin(g.game_id)],
        )
        for (s, w), g in d.groupby(["season", "model_week"])
    ]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i, result in enumerate(pool.map(run_week, jobs), 1):
            print(i, len(jobs), result, flush=True)
    decisions = pd.concat(
        [pd.read_parquet(p) for p in sorted((OUT / "batches").glob("*.parquet"))],
        ignore_index=True,
    )
    assert len(decisions) == len(d) * 3 * 2
    assert not decisions.duplicated(["policy", "season", "game_id", "market"]).any()
    picks = settle(decisions.query("status == 'recommended'").copy(), d)
    assert (
        not picks.query("policy == 'new' and market == 'h2h'")
        .qualification.eq("exploratory")
        .any()
    )
    assert picks.query("qualification == 'qualified'").edge_points.ge(2).all()
    assert (
        picks.query("qualification == 'exploratory'")
        .edge_points.between(0, 2, inclusive="neither")
        .all()
    )
    assert picks.expected_value_per_unit.gt(0).all()
    decisions.to_parquet(OUT / "decisions.parquet", index=False)
    picks.to_parquet(OUT / "picks.parquet", index=False)
    picks.drop(
        columns=["pricing_context", "decision_forecast"], errors="ignore"
    ).to_csv(OUT / "picks.csv", index=False)
    table = summarize(picks)
    table.to_csv(OUT / "summary.csv", index=False)
    weekly = (
        picks.groupby(["policy", "season", "week", "market", "qualification"])
        .agg(
            picks=("game_id", "size"),
            wins=("outcome", lambda v: v.eq("win").sum()),
            profit=("profit_units", "sum"),
        )
        .reset_index()
    )
    weekly.to_csv(OUT / "weekly.csv", index=False)
    result = dict(
        quality=quality,
        summary=json.loads(table.to_json(orient="records")),
        paired_week_intervals=paired_intervals(d, picks),
        limitations=[
            "Closing-quote proxy only; archived moneyline timestamps and spread/total prices are unavailable.",
            "Assumed -110 spread/total prices and Bovada-only lines can change cross-market quota allocation.",
            "Historical quote freshness, availability, matching confidence and model-input completeness are assumed, not verified.",
            "Historical model replay omits archived rich preseason and availability forecasts.",
            "2023-2025 are reused evaluation seasons; model architecture and fixed dispersion used some of these years.",
            "The new key-number weights were fitted on 2020-2022; development results are not independent validation.",
        ],
    )
    (OUT / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(table.query("split == 'evaluation'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
