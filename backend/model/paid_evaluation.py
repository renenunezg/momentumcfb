"""Offline incremental-value experiment with prior-season CFBD metrics.

Run: python -m backend.model.paid_evaluation REPLAY.parquet OUTPUT_DIRECTORY
Fit 2020-2021, select 2022, refit through 2022, evaluate 2023-2025 once.
The validation seasons have been used by earlier model investigations.
Provider history is today's revised vintage, not frozen historical receipts.
The cached baseline uses historical carryover rather than rich live preseason
inputs. Results establish research value, not production activation authority.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backend.cfbd.snapshots import receipts
from backend.etl import store
from backend.model.paid_benchmarks import ADJUSTED_ENDPOINTS

RIDGES = (10.0, 100.0, 1000.0)


def history(endpoint, *, root=None):
    latest = {}
    for path, receipt in receipts(endpoint, root=root):
        year = receipt["params"].get("year")
        if year is None or year > 2025 or not receipt["payload"]:
            continue
        if year not in latest or receipt["fetched_at"] > latest[year][1]["fetched_at"]:
            latest[year] = path, receipt
    frames = []
    for year, (path, receipt) in sorted(latest.items()):
        frame = pd.json_normalize(receipt["payload"])
        if not frame.year.eq(year).all():
            raise ValueError("adjusted source year mismatch")
        frame["source_snapshot"] = str(path)
        frame["source_fetched_at"] = receipt["fetched_at"]
        frames.append(frame)
    if not frames:
        raise ValueError(f"no history for {endpoint}")
    return pd.concat(frames, ignore_index=True)


def select_adjustment(raw, residual, seasons):
    """Return a development-selected adjustment, including an exact no-op."""
    fit, select = seasons <= 2021, seasons == 2022
    if not fit.any() or not select.any() or not (seasons >= 2023).any():
        raise ValueError("missing chronological development or validation period")
    if not np.isfinite(raw).all() or not np.isfinite(residual).all():
        raise ValueError("nonfinite experiment input")

    def train(mask, ridge):
        center, scale = raw[mask].mean(0), raw[mask].std(0)
        scale[scale == 0] = 1
        x = np.column_stack([np.ones(len(raw)), (raw - center) / scale])
        penalty = np.diag([0.0] + [ridge] * raw.shape[1])
        coef = np.linalg.solve(
            x[mask].T @ x[mask] + penalty, x[mask].T @ residual[mask]
        )
        return x @ coef

    scores = {"none": float(np.abs(residual[select]).mean())}
    for ridge in RIDGES:
        scores[str(ridge)] = float(
            np.abs(train(fit, ridge)[select] - residual[select]).mean()
        )
    selected = min(scores, key=scores.get)
    adjustment = (
        np.zeros(len(raw))
        if selected == "none"
        else train(seasons <= 2022, float(selected))
    )
    return adjustment, dict(selected=selected, selection_mae=scores)


def team_inputs(predictions, *, root=None):
    out = predictions.copy()
    if "candidate" in out:
        out = out[out.candidate.eq(100)].copy()
    out = out[out.season.between(2020, 2025)].copy()
    if out.duplicated(["season", "game_id"]).any():
        raise ValueError("one frozen baseline per game required")
    source = history(ADJUSTED_ENDPOINTS[0], root=root)
    if source.duplicated(["year", "teamId"]).any():
        raise ValueError("duplicate adjusted team season")
    source["season"] = source.year + 1
    columns = [
        "epa.total",
        "epaAllowed.total",
        "successRate.total",
        "successRateAllowed.total",
        "explosiveness",
        "explosivenessAllowed",
    ]
    for side in ("home", "away"):
        renamed = (
            source[
                [
                    "season",
                    "teamId",
                    "year",
                    "source_snapshot",
                    "source_fetched_at",
                    *columns,
                ]
            ]
            .rename(
                columns={
                    c: f"{side}_prior_{c}"
                    for c in ["year", "source_snapshot", "source_fetched_at", *columns]
                }
            )
            .rename(columns={"teamId": f"{side}_team_id"})
        )
        out = out.merge(
            renamed,
            on=["season", f"{side}_team_id"],
            how="left",
            validate="many_to_one",
        )
        present = out[f"{side}_prior_year"].notna()
        if not (
            out.loc[present, f"{side}_prior_year"] < out.loc[present, "season"]
        ).all():
            raise ValueError("same-season adjusted metric leakage")
    needed = [f"{side}_prior_{c}" for side in ("home", "away") for c in columns]
    out["paid_covered"] = out[needed].notna().all(axis=1)
    return out


def _paired_interval(frame, target):
    # Block bootstrap weeks within seasons, retaining each season's presence.
    delta = (frame[f"paid_{target}"] - frame[f"actual_{target}"]).abs() - (
        frame[f"control_{target}"] - frame[f"actual_{target}"]
    ).abs()
    blocks = (
        frame[["season", "model_week"]]
        .assign(delta=delta)
        .groupby(["season", "model_week"])
        .delta.agg(["sum", "count"])
    )
    rng = np.random.default_rng(202209)
    sums, counts = np.zeros(2000), np.zeros(2000)
    for _, group in blocks.groupby(level=0):
        samples = rng.integers(0, len(group), size=(2000, len(group)))
        sums += group["sum"].to_numpy()[samples].sum(axis=1)
        counts += group["count"].to_numpy()[samples].sum(axis=1)
    low, high = np.quantile(sums / counts, [0.025, 0.975])
    return dict(
        paid_minus_control_mae=float(delta.mean()),
        block_ci_low=float(low),
        block_ci_high=float(high),
    )


def evaluate_teams(predictions, destination, *, root=None):
    out = team_inputs(predictions, root=root)
    out["actual_margin"] = out.home_points - out.away_points
    out["baseline_margin"] = out.home_margin
    out["baseline_total"] = out.model_total
    covered = out[out.paid_covered].copy()
    # Identical model-only calibration control isolates new information from
    # generic residual calibration. All model columns are pregame predictions.
    base = covered[
        ["home_margin", "model_total", "expected_game_possessions", "home_field_points"]
    ].to_numpy()
    decay = 1 / np.sqrt(covered.model_week.to_numpy())
    base = np.column_stack([base, decay])
    selection = {}
    for target in ("margin", "total"):
        features = []
        for offense, allowed in (
            ("epa.total", "epaAllowed.total"),
            ("successRate.total", "successRateAllowed.total"),
            ("explosiveness", "explosivenessAllowed"),
        ):
            home = covered[f"home_prior_{offense}"] + covered[f"away_prior_{allowed}"]
            away = covered[f"away_prior_{offense}"] + covered[f"home_prior_{allowed}"]
            value = home - away if target == "margin" else home + away
            features.append(value.to_numpy() * decay)
        augmented = np.column_stack([base, *features])
        residual = (
            covered[f"actual_{target}"] - covered[f"baseline_{target}"]
        ).to_numpy()
        for label, matrix in (("control", base), ("paid", augmented)):
            adjustment, selected = select_adjustment(
                matrix, residual, covered.season.to_numpy()
            )
            out[f"{label}_{target}"] = out[f"baseline_{target}"]
            out.loc[covered.index, f"{label}_{target}"] += adjustment
            selection[f"{label}_{target}"] = selected
    rows, effects = [], []
    subset = out[out.season.ge(2023)]
    groups = [
        ("all", subset),
        ("covered", subset[subset.paid_covered]),
        ("weeks_2_to_4", subset[subset.model_week.le(4)]),
    ]
    groups += [(f"matchup:{key}", value) for key, value in subset.groupby("matchup")]
    groups += [(f"season:{key}", value) for key, value in subset.groupby("season")]
    groups += [(f"week:{key}", value) for key, value in subset.groupby("model_week")]
    for segment, frame in groups:
        for target in ("margin", "total"):
            for model in ("baseline", "control", "paid"):
                err = frame[f"{model}_{target}"] - frame[f"actual_{target}"]
                rows.append(
                    dict(
                        segment=segment,
                        target=target,
                        model=model,
                        games=len(frame),
                        covered=int(frame.paid_covered.sum()),
                        mae=float(err.abs().mean()),
                        rmse=float(np.sqrt((err**2).mean())),
                        bias=float(err.mean()),
                        evidence="pure_model_revised_prior_season_data",
                    )
                )
            if len(frame):
                effects.append(
                    dict(
                        segment=segment,
                        target=target,
                        **_paired_interval(frame, target),
                    )
                )
    # Historical lines have no executable quote receipt. Keep this explicitly
    # retrospective and never use the close for feature or parameter selection.
    market_rows = []
    for target in ("total", "margin"):
        if target == "margin":
            from backend.serving.market import flatten_closing_lines

            closes = pd.concat(
                [
                    flatten_closing_lines(store.read_lines(y)).assign(season=y)
                    for y in range(2020, 2026)
                ]
            )
            out = out.merge(
                closes[["season", "game_id", "closing_spread"]],
                on=["season", "game_id"],
                how="left",
                validate="one_to_one",
            )
            out["closing_margin"] = -out.closing_spread
        valid = out[out.season.ge(2023) & out[f"closing_{target}"].notna()]
        for model in ("baseline", "control", "paid", "closing"):
            prediction = (
                valid[f"closing_{target}"]
                if model == "closing"
                else 0.5 * (valid[f"{model}_{target}"] + valid[f"closing_{target}"])
            )
            err = prediction - valid[f"actual_{target}"]
            market_rows.append(
                dict(
                    target=target,
                    model=model,
                    games=len(valid),
                    mae=float(err.abs().mean()),
                    rmse=float(np.sqrt((err**2).mean())),
                    bias=float(err.mean()),
                    evidence="closing_market"
                    if model == "closing"
                    else "retrospective_50pct_closing_blend_not_executable",
                )
            )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    out["evidence_kind"] = "prior_season_revised_vintage_not_frozen_historical_inputs"
    out.to_parquet(destination / "team_predictions.parquet", index=False)
    pd.DataFrame(rows).to_csv(destination / "team_metrics.csv", index=False)
    pd.DataFrame(effects).to_csv(destination / "team_effects.csv", index=False)
    pd.DataFrame(market_rows).to_csv(destination / "market_metrics.csv", index=False)
    (destination / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return pd.DataFrame(rows)


def evaluate_players(destination, *, root=None):
    """Predict next-season efficiency among returning observed players only.

    This conditions on next-year participation. It is neither a roster forecast
    nor a sanctioned replacement for split-half Heisman model validation.
    """
    seasons = []
    for year in range(2019, 2026):
        values = store.read_processed("players", "player_values", f"{year}.parquet")
        values = values[values.week.eq(values.week.max())].copy()
        # A transfer can have multiple team rows; combine actual rate numerators.
        grouped = values.groupby("athlete_id").agg(
            plays=("plays", "sum"), adjusted_epa=("adjusted_epa", "sum")
        )
        grouped["own_rate"] = grouped.adjusted_epa / grouped.plays
        grouped["season"] = year
        seasons.append(grouped.reset_index())
    own = pd.concat(seasons, ignore_index=True)
    own["athlete_id"] = own.athlete_id.astype(str)
    rows, effects = [], []
    for endpoint in ADJUSTED_ENDPOINTS[1:3]:
        source = history(endpoint, root=root)
        source["athlete_id"] = source.athleteId.astype(str)
        source["weighted"] = source.wepa * source.plays
        source = (
            source.groupby(["year", "athlete_id"])
            .agg(weighted=("weighted", "sum"), provider_plays=("plays", "sum"))
            .reset_index()
        )
        source["prior_wepa"] = source.weighted / source.provider_plays
        source["season"] = source.year
        joined = source.merge(own, on=["season", "athlete_id"], validate="one_to_one")
        joined["prior_year"] = joined.season
        joined["season"] += 1
        joined = joined.merge(
            own[["season", "athlete_id", "own_rate", "plays"]].rename(
                columns={"own_rate": "next_rate", "plays": "next_plays"}
            ),
            on=["season", "athlete_id"],
            validate="one_to_one",
        )
        joined = joined[
            joined.season.between(2020, 2025)
            & joined.plays.gt(0)
            & joined.next_plays.gt(0)
        ].copy()
        base = np.column_stack([joined.own_rate, np.log1p(joined.plays)])
        augmented = np.column_stack(
            [base, joined.prior_wepa, np.log1p(joined.provider_plays)]
        )
        residual = (joined.next_rate - joined.own_rate).to_numpy()
        for model, design in (("control", base), ("paid", augmented)):
            adjustment, selection = select_adjustment(
                design, residual, joined.season.to_numpy()
            )
            joined[f"{model}_rate"] = joined.own_rate + adjustment
            validation = joined[joined.season.ge(2023)]
            err = validation[f"{model}_rate"] - validation.next_rate
            rows.append(
                dict(
                    position=endpoint.rsplit("/", 1)[-1],
                    model=model,
                    players=len(validation),
                    mae=float(err.abs().mean()),
                    rmse=float(np.sqrt((err**2).mean())),
                    selected=selection["selected"],
                    evidence="next_season_observed_returners_not_heisman_validation",
                )
            )
        validation = joined[joined.season.ge(2023)].copy()
        validation["delta"] = (validation.paid_rate - validation.next_rate).abs() - (
            validation.control_rate - validation.next_rate
        ).abs()
        blocks = validation.groupby("athlete_id").delta.agg(["sum", "count"])
        rng = np.random.default_rng(202209)
        indices = rng.integers(0, len(blocks), size=(2000, len(blocks)))
        sampled = blocks["sum"].to_numpy()[indices].sum(axis=1) / blocks[
            "count"
        ].to_numpy()[indices].sum(axis=1)
        low, high = np.quantile(sampled, [0.025, 0.975])
        effects.append(
            dict(
                position=endpoint.rsplit("/", 1)[-1],
                paid_minus_control_mae=float(validation.delta.mean()),
                athlete_block_ci_low=float(low),
                athlete_block_ci_high=float(high),
                by_season=validation.groupby("season").delta.mean().to_dict(),
            )
        )
        joined.to_parquet(
            Path(destination) / f"{endpoint.rsplit('/', 1)[-1]}_predictions.parquet",
            index=False,
        )
    result = pd.DataFrame(rows)
    result.to_csv(Path(destination) / "player_metrics.csv", index=False)
    (Path(destination) / "player_effects.json").write_text(
        json.dumps(effects, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    predictions, destination = sys.argv[1:]
    summary = evaluate_teams(pd.read_parquet(predictions), destination)
    print(summary[summary.segment.isin(["all", "covered"])].to_string(index=False))
    print(evaluate_players(destination).to_string(index=False))
