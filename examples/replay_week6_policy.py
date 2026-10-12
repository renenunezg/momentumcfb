"""Replay the October 5 archived inputs against published Week 6 settlements."""

import argparse
import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from backend.recommendations import H2H_MODEL_WEIGHT, build_recommendations

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/week6-policy-comparison-20261012"


def settle(picks, scores):
    d = picks.merge(scores, on="game_id", validate="many_to_one")
    margin = d.home_points - d.away_points
    total = d.home_points + d.away_points
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
    d["outcome"] = np.where(balance > 0, "win", np.where(balance < 0, "loss", "push"))
    d["profit_units"] = np.where(
        balance > 0,
        np.where(d.price > 0, d.price / 100, 100 / -d.price),
        np.where(balance < 0, -1, 0),
    )
    return d


def summary(d):
    return dict(
        picks=len(d),
        wins=int(d.outcome.eq("win").sum()),
        losses=int(d.outcome.eq("loss").sum()),
        pushes=int(d.outcome.eq("push").sum()),
        profit=float(d.profit_units.sum()),
        roi=float(d.profit_units.mean()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    published = pd.read_json(OUT / "published.json")
    forecasts = pd.read_parquet(OUT / "inputs/projections.parquet")
    offers = pd.read_parquet(OUT / "inputs/market_offers.parquet")
    assert published.decision_at.nunique() == 1
    decision = published.decision_at.iloc[0]
    assert H2H_MODEL_WEIGHT == 0.2
    assert set(forecasts.game_id) == set(published.game_id)
    assert pd.to_datetime(forecasts.as_of, utc=True).nunique() == 1
    assert (
        pd.to_datetime(forecasts.as_of, utc=True).iloc[0]
        == pd.to_datetime(published.forecast_as_of, utc=True).iloc[0]
    )
    spec = importlib.util.spec_from_file_location(
        "original_policy", OUT / "inputs/original_recommendations.py"
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    old_replay = original.build_recommendations(forecasts, offers, decision_at=decision)
    new_replay = build_recommendations(forecasts, offers, decision_at=decision)
    old = published[published.status.eq("recommended")].copy()
    replay = old_replay[old_replay.status.eq("recommended")]
    keys = ["game_id", "market"]
    a, b = old.set_index(keys).sort_index(), replay.set_index(keys).sort_index()
    assert a.index.equals(b.index)
    for col in ["side", "selection", "provider_key", "reason"]:
        assert a[col].equals(b[col]), col
    for col in ["point", "price", "win_probability", "edge_points"]:
        assert np.allclose(a[col], b[col], equal_nan=True), col
    scores = published[["game_id", "home_points", "away_points"]].drop_duplicates()
    assert not scores.game_id.duplicated().any()
    assert scores.notna().all().all()
    new = new_replay[new_replay.status.eq("recommended")].copy()
    new = settle(
        new.drop(columns=["home_points", "away_points"], errors="ignore"), scores
    )
    check_old = settle(old.drop(columns=["home_points", "away_points"]), scores)
    assert np.allclose(check_old.profit_units, old.profit_units)
    assert check_old.outcome.tolist() == old.outcome.tolist()
    assert old.outcome.eq("win").sum() == 5 and old.outcome.eq("loss").sum() == 15
    comparison = old.merge(
        new,
        on=keys,
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
    both = comparison["_merge"].eq("both")
    same = (
        comparison.side_old.eq(comparison.side_new)
        & comparison.price_old.eq(comparison.price_new)
        & (
            comparison.point_old.eq(comparison.point_new)
            | (comparison.point_old.isna() & comparison.point_new.isna())
        )
    )
    comparison.loc[both & ~same, "change"] = "changed"
    comparison["profit_delta"] = comparison.profit_units_new.fillna(
        0
    ) - comparison.profit_units_old.fillna(0)
    assert np.isclose(
        comparison.profit_delta.sum(), new.profit_units.sum() - old.profit_units.sum()
    )
    for name, frame in [
        ("old_picks", old),
        ("new_picks", new),
        ("all_new_decisions", new_replay),
        ("changes", comparison),
    ]:
        frame.to_csv(destination / f"{name}.csv", index=False)
    report = dict(
        model_weight=H2H_MODEL_WEIGHT,
        new_policy=str(new_replay.policy_version.iloc[0]),
        by_qualification={
            reason: summary(group) for reason, group in new.groupby("reason")
        },
        decision_at=str(decision),
        artifact_run="37352643455",
        original_sha="37114e6529b42a4966c3b54978d5371cdc1b72ad",
        original_policy=original.POLICY_VERSION,
        forecast_games=len(forecasts),
        offers=len(offers),
        old=summary(old),
        new=summary(new),
        by_market={
            m: {
                "old": summary(old[old.market.eq(m)]),
                "new": summary(new[new.market.eq(m)]),
            }
            for m in ["h2h", "spreads", "totals"]
        },
        exposure=new.groupby(["game_id", "home_team", "away_team"])
        .size()
        .reset_index(name="picks")
        .query("picks>1")
        .to_dict("records"),
        input_hashes={
            str(p.relative_to(OUT)): sha256(p.read_bytes()).hexdigest()
            for p in (OUT / "inputs").iterdir()
            if p.is_file()
        },
    )
    (destination / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(
        comparison[
            [
                "game_id",
                "market",
                "change",
                "selection_old",
                "point_old",
                "price_old",
                "selection_new",
                "point_new",
                "price_new",
                "outcome_old",
                "outcome_new",
                "profit_delta",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
