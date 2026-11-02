"""Development-fitted spread model weight and key weights for cfb-picks-v12.

The weight minimizes win/push/loss log loss at the posted Bovada line on
2020-2022 regular and postseason games, around the CFBD median closing spread.
Key weights are then refit at that weight exactly as the 0.20 calibration was
(fit 2020-2021, penalty selected on 2022, final fit 2020-2022). 2023-2025 is
scored once against the frozen 0.20 calibration and never used for selection.
Run: poetry run python -m examples.fit_spread_cover_weight
"""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from examples.audit_moneyline_probabilities import interval
from examples.audit_spread_total_followups import fit_keys, key_probabilities
from examples.audit_spread_total_picks import probabilities

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/spread-total-audit-20261012/historical_inputs.parquet"
OUT = ROOT / "output/spread-cover-weight-20261102"
CALIBRATION = ROOT / "backend/model/spread_pricing.json"
PREVIOUS = ROOT / "output/spread-total-audit-20261012/combined_spread_candidate.json"
VERSION = "spread_keys_cover_v2"


def inputs():
    d = pd.read_parquet(SOURCE)
    d["reference"] = -d.reference_spread
    d["threshold"] = -d.offer_spread
    d["model"] = d.model_side
    d["actual"] = d.actual_margin
    keep = np.isfinite(d[["reference", "threshold", "model", "actual"]]).all(axis=1)
    keep &= (d.threshold * 2).eq(np.round(d.threshold * 2))
    return d[keep].reset_index(drop=True)


def with_mean(d, weight):
    return d.assign(mean=d.reference + weight * (d.model - d.reference))


def cover_loss(d, weight):
    """Win/push/loss log loss at the posted line without key weights."""
    p = probabilities(
        with_mean(d, weight)["mean"].to_numpy(), 15.35, d.threshold.to_numpy()
    )
    balance = (d.actual - d.threshold).to_numpy()
    labels = np.where(balance > 0, 0, np.where(balance < 0, 2, 1))
    return -np.log(np.clip(p[np.arange(len(d)), labels], 1e-12, 1))


def main():
    d = inputs()
    development, validation = d[d.season.le(2022)], d[d.season.ge(2023)]
    weight = float(
        minimize_scalar(
            lambda w: cover_loss(development, w).mean(),
            bounds=(0, 1),
            method="bounded",
            options=dict(xatol=1e-5),
        ).x
    )
    weight = round(weight, 4)
    dw = with_mean(d, weight)
    fit, select = dw[dw.season.le(2021)], dw[dw.season.eq(2022)]
    scores = {
        p: float(
            key_probabilities(select, "spreads", fit_keys(fit, "spreads", p))[1].mean()
        )
        for p in [10, 50, 200]
    }
    penalty = min(scores, key=scores.get)
    weights = fit_keys(dw[dw.season.le(2022)], "spreads", penalty)
    previous = np.asarray(json.loads(PREVIOUS.read_text())["weights"])
    current = key_probabilities(with_mean(validation, 0.2), "spreads", previous)[1]
    candidate = key_probabilities(with_mean(validation, weight), "spreads", weights)[1]
    report = dict(
        objective="win/push/loss log loss at the Bovada line, development 2020-2022",
        development_games=len(development),
        weight=weight,
        development_loss={
            str(w): float(cover_loss(development, w).mean()) for w in (0, weight, 0.2)
        },
        selected_penalty=penalty,
        selection_scores=scores,
        validation_games=len(validation),
        validation_current=float(current.mean()),
        validation_candidate=float(candidate.mean()),
        change=float((candidate - current).mean()),
        change_ci=interval(validation, candidate - current),
        weights=weights.tolist(),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    result = OUT / "result.json"
    result.write_text(json.dumps(report, indent=2) + "\n")
    CALIBRATION.write_text(
        json.dumps(
            dict(
                version=VERSION,
                model_weight=weight,
                margin_sd=15.35,
                degrees_of_freedom=500,
                grid_max=80,
                fit_seasons=[2020, 2021, 2022],
                selection_season=2022,
                penalty=penalty,
                weights=weights.tolist(),
                research_result_sha256=sha256(result.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "weights"}, indent=2))


if __name__ == "__main__":
    main()
