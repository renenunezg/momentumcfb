"""Offline weekly CFP snapshot. Reads frozen inputs; never writes to serving systems."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backend.etl.ingest import to_snake
from backend.features.scoring import build_weekly_scoring_games
from backend.model.forecast_calibration import apply_forecast_calibration
from backend.model.joint_scoring import fit_joint_scoring
from backend.model.preseason import scoring_priors_from_ratings
from backend.playoffs import selection as c
from backend.playoffs.game_model import ForecastState, probability
from backend.playoffs.serving_payload import compact

ROOT = Path(__file__).resolve().parent
RULES_URL = "https://collegefootballplayoff.com/sports/2024/5/29/12-team-format"
FLEX_URL = "https://pac-12.com/news/2026/2/9/general-the-new-pac-12-announces-its-2026-football-schedule.aspx"


class WeeklyState(ForecastState):
    """Production weekly fit and calibration, with a fixed model information set."""

    def project(self, schedule):
        raw = pd.DataFrame(p.to_record() for p in self.fit.project(schedule))
        return apply_forecast_calibration(
            raw,
            self.known,
            self.team_games,
            self.process_prior,
            self.forecast_week,
            self.cutoff.to_pydatetime(),
        )


def read_state(archive, raw):
    def read(name):
        return pd.read_parquet(archive / f"{name}.parquet")

    expected = read("projections")
    cutoff = pd.Timestamp(expected.as_of.iloc[0])
    week = int(expected.week.iloc[0])
    training = read("training_games")
    schedule = build_weekly_scoring_games(raw, read("process_training_features"))
    # Replay the archived model information set exactly; fresh scores only condition
    # the season simulation, and cannot leak into the frozen weekly strength fit.
    future = schedule[~schedule.game_id.isin(training.game_id)].copy()
    future["completed"] = False
    future["model_week"] = np.maximum(future.model_week, week)
    games = pd.concat([training, future], ignore_index=True)
    priors = scoring_priors_from_ratings(
        read("preseason_ratings"), read("score_noise_prior"), read("srs_prior")
    )
    fit = fit_joint_scoring(games, week, cutoff.to_pydatetime(), priors=priors)
    state = WeeklyState(
        2026,
        cutoff,
        games,
        fit,
        read("process_training_features"),
        read("process_prior"),
    )
    state.forecast_week = week
    reproduced = state.project(read("forecast_schedule")).set_index("game_id")
    expected = expected.set_index("game_id")
    cols = ["home_margin", "model_total", "margin_sd"]
    delta = float(
        np.max(
            abs(
                reproduced.loc[expected.index, cols].to_numpy()
                - expected[cols].to_numpy()
            )
        )
    )
    if delta > 1e-7:
        raise ValueError(f"Weekly model replay differs from archive by {delta}")
    return state, schedule, delta


def prepare_schedule(schedule, cutoff):
    m = c.m
    ids = set(map(int, m.MEM[2026]))
    schedule = schedule[
        schedule.home_team_id.isin(ids) | schedule.away_team_id.isin(ids)
    ].copy()
    # Selection Sunday precedes Army-Navy; that result must not affect selection.
    schedule = schedule[schedule.start_date.lt("2026-12-07")].copy()
    additions = []
    for g in c.e.current_games():
        if not str(g["id"]).startswith("flex-"):
            continue
        hi, ai = int(g["hid"]), int(g["aid"])
        exists = (
            schedule.home_team_id.eq(hi)
            & schedule.away_team_id.eq(ai)
            & schedule.start_date.ge("2026-11-27")
        )
        if exists.any():
            continue
        additions.append(
            dict(
                game_id=990000000 + len(additions),
                season=2026,
                week=13,
                model_week=13,
                season_type="regular",
                start_date=pd.Timestamp(g["date"]),
                home_team_id=hi,
                away_team_id=ai,
                home_team=m.NAMES[g["h"]],
                away_team=m.NAMES[g["a"]],
                neutral_site=False,
                completed=False,
                home_points=np.nan,
                away_points=np.nan,
                conference_game=False,
            )
        )
    if additions:
        schedule = pd.concat([schedule, pd.DataFrame(additions)], ignore_index=True)
    schedule = schedule.sort_values(["start_date", "game_id"]).reset_index(drop=True)
    if schedule.game_id.duplicated().any():
        raise ValueError("Duplicate schedule games")
    reg = []
    for r in schedule.itertuples():
        h, a = m.IX.get(str(r.home_team_id), -1), m.IX.get(str(r.away_team_id), -1)
        done = bool(r.completed) and r.start_date < cutoff
        conf = bool(r.conference_game) and h >= 0 and a >= 0
        if (
            conf
            and m.MEM[2026][str(r.home_team_id)]["conf"]
            != m.MEM[2026][str(r.away_team_id)]["conf"]
        ):
            raise ValueError(f"Conflicting conference membership: {r.game_id}")
        margin = float(r.home_points - r.away_points) if done else np.nan
        if done and (not np.isfinite(margin) or margin == 0):
            raise ValueError(f"Invalid final score: {r.game_id}")
        reg.append(
            dict(
                id=str(r.game_id),
                date=r.start_date.isoformat(),
                h=h,
                a=a,
                hid=str(r.home_team_id),
                aid=str(r.away_team_id),
                loc=0.0 if r.neutral_site else m.HFA,
                margin=margin,
                title=False,
                conference=conf,
                conf=m.MEM[2026][str(r.home_team_id)]["conf"] if conf else None,
            )
        )
    counts = {
        i: sum(i in (g["h"], g["a"]) for g in reg) for i in map(m.IX.get, m.MEM[2026])
    }
    for i, count in counts.items():
        wanted = (
            11
            if m.NAMES[i] in ["Army", "Navy"]
            else 13
            if m.NAMES[i] == "San José State"
            else 12
        )
        if count != wanted:
            raise ValueError(
                f"Incomplete/extra schedule: {m.NAMES[i]} {count} != {wanted}"
            )
    return schedule, reg, additions


def seeded_field(score, champs, eligible):
    qualified, rank = c.e.field(score, champs, eligible, 2026)
    order = np.argsort(np.where(qualified, rank, 10000), axis=1)[:, :12]
    assert np.all(qualified.sum(1) == 12)
    assert np.all(np.diff(np.sort(order, axis=1), axis=1) > 0)
    return qualified, order


def run_simulation(state, schedule, reg, n, seed, independent=False, random_ties=False):
    m, e = c.m, c.e
    eligible = np.array([m.IX[i] for i in m.MEM[2026]])
    rng = np.random.default_rng(seed)
    draws = state.draw_state(rng, n)
    future = np.flatnonzero([not np.isfinite(g["margin"]) for g in reg])
    projections = state.project(schedule.iloc[future])
    state.validate_marginal_variance(projections)
    margins = np.tile([g["margin"] for g in reg], (n, 1))
    margins[:, future] = state.sample(projections, rng, draws, independent=independent)
    p = probability(projections)
    se = np.sqrt(np.maximum(p * (1 - p), 1 / n) / n)
    marginal_z = float(np.max(abs((margins[:, future] > 0).mean(0) - p) / se))
    A, offset = m.matrix(reg)
    prior = m.prior_for(2026, e.CFG["rho"], e.CFG["ridge"])
    inverse = np.linalg.inv(A.T @ A + e.CFG["ridge"] * np.eye(m.T))
    rating = (np.clip(margins, -35, 35) - offset) @ A @ inverse + e.CFG[
        "ridge"
    ] * prior @ inverse
    feat, wins, losses = m.features(reg, margins, rating, eligible, e.CFG["sd"])
    proxy = feat[:, :, c.SELECTION["cols"]] @ c.BETA
    if random_ties:
        pairs, _, tie_counts = e.title_participants(
            2026, reg, margins, proxy, wins, random_ties=True
        )
    else:
        pairs, tie_counts = c.corrected_pairs(2026, reg, margins, proxy, wins)
    titles = c.title_projections(state, 2026)
    played = np.zeros((n, m.T), bool)
    opp = np.full((n, m.T), -1, int)
    won = np.zeros((n, m.T), int)
    loc = np.zeros((n, m.T))
    champs = np.zeros((n, m.T), bool)
    rr = np.arange(n)
    for conf, pair in pairs.items():
        h, a = pair.T
        frame = titles.loc[list(zip(h, a))].reset_index(drop=True)
        margin = state.sample(frame, rng, draws, per_path=True, independent=independent)
        champion = np.where(margin > 0, h, a)
        champs[rr, champion] = True
        played[rr, h] = played[rr, a] = True
        opp[rr, h], opp[rr, a] = a, h
        won[rr, champion] = 1
        field = np.where(frame.neutral_site, 0.0, m.HFA)
        loc[rr, h], loc[rr, a] = field, -field
    feat, final_wins, final_losses = m.features(
        reg, margins, rating, eligible, e.CFG["sd"], (played, opp, won, loc)
    )
    noise = rng.gumbel(size=(n, m.T))
    score = c.RECORD_BETA * feat[:, :, 0] + noise
    q, seeds = seeded_field(score, champs, eligible)
    alt, _ = seeded_field(
        feat[:, :, c.SELECTION["cols"]] @ c.BETA + noise, champs, eligible
    )
    counts = np.zeros((m.T, 12), int)
    for slot in range(12):
        counts[:, slot] = np.bincount(seeds[:, slot], minlength=m.T)
    # Choose one legal realization with high support at every seed. Never assemble
    # mutually incompatible conference winners from independent marginal modes.
    support = np.log(np.maximum(counts / n, 1 / n))
    field_p = np.clip(q.mean(0), 1 / n, 1 - 1 / n)
    field_support = q @ np.log(field_p / (1 - field_p))
    candidates = np.flatnonzero(
        np.isclose(field_support, field_support.max(), rtol=0, atol=1e-10)
    )
    seed_support = support[seeds, np.arange(12)].sum(1)
    representative = int(candidates[np.argmax(seed_support[candidates])])
    catalog = state.fit.teams.set_index("team_id")

    def matchup(h, a, neutral):
        unique, inv = np.unique(np.stack([h, a], axis=1), axis=0, return_inverse=True)
        records = []
        for j, (hi, ai) in enumerate(unique):
            hid, aid = int(m.IDS[hi]), int(m.IDS[ai])
            records.append(
                dict(
                    game_id=980000000 + j,
                    season=2026,
                    week=16,
                    home_team_id=hid,
                    away_team_id=aid,
                    home_team=catalog.loc[hid, "team"],
                    away_team=catalog.loc[aid, "team"],
                    neutral_site=neutral,
                )
            )
        projected = state.project(pd.DataFrame(records))
        state.validate_marginal_variance(projected)
        frame = projected.iloc[inv].reset_index(drop=True)
        margin = state.sample(frame, rng, draws, per_path=True, independent=independent)
        return np.where(margin > 0, h, a)

    r1 = np.stack(
        [
            matchup(seeds[:, hi - 1], seeds[:, lo - 1], False)
            for hi, lo in [(8, 9), (5, 12), (7, 10), (6, 11)]
        ],
        axis=1,
    )
    qf = np.stack(
        [
            matchup(seeds[:, bye - 1], r1[:, j], True)
            for j, bye in enumerate([1, 4, 2, 3])
        ],
        axis=1,
    )
    sf = np.stack(
        [matchup(qf[:, 0], qf[:, 1], True), matchup(qf[:, 2], qf[:, 3], True)], axis=1
    )
    winner = matchup(sf[:, 0], sf[:, 1], True)
    rounds = [np.concatenate([seeds[:, :4], r1], axis=1), qf, sf, winner[:, None]]
    round_p = [np.bincount(x.ravel(), minlength=m.T) / n for x in rounds]
    for arr, total in zip(round_p, [8, 4, 2, 1]):
        np.testing.assert_allclose(arr.sum(), total, atol=1e-12)
    p = q.mean(0)
    bye = counts[:, :4].sum(1) / n
    for earlier, later in zip([p, *round_p[:-1]], round_p):
        assert np.all(later <= earlier + 1e-12)
    np.testing.assert_allclose(p.sum(), 12, atol=1e-12)
    np.testing.assert_allclose(bye.sum(), 4, atol=1e-12)
    return dict(
        eligible=eligible,
        probability=p,
        bye=bye,
        rounds=round_p,
        alt=alt.mean(0),
        seed_probability=counts / n,
        title=champs.mean(0),
        title_game=played.mean(0),
        expected_wins=final_wins.mean(0),
        expected_losses=final_losses.mean(0),
        representative=seeds[representative],
        representative_champs=champs[representative],
        half_difference=float(np.max(abs(q[: n // 2].mean(0) - q[n // 2 :].mean(0)))),
        marginal_z=marginal_z,
        tie_counts=tie_counts,
        projections=projections,
        future=future,
        margins=margins,
    )


def build(args):
    started = time.monotonic()
    inputs, output = args.inputs.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = to_snake(pd.DataFrame(json.loads((inputs / "games.json").read_text())))
    retrieved = json.loads((inputs / "retrieved.json").read_text())
    cutoff = pd.Timestamp(retrieved["retrieved_at"])
    state, schedule, replay_delta = read_state(args.archive, raw)
    schedule, reg, additions = prepare_schedule(schedule, cutoff)
    print(
        "Frozen weekly fit reproduced; simulating",
        args.n,
        "seasons;",
        len(reg),
        "games",
        flush=True,
    )
    sim = run_simulation(state, schedule, reg, args.n, args.seed)
    print(
        "Primary simulation complete; running dependence/tiebreak sensitivity",
        flush=True,
    )
    sensitivity = run_simulation(
        state,
        schedule,
        reg,
        args.sensitivity_n,
        args.seed + 1,
        independent=True,
        random_ties=True,
    )
    rankings = json.loads((inputs / "rankings.json").read_text())
    ap_week = max(
        w["week"] for w in rankings if any(p["poll"] == "AP Top 25" for p in w["polls"])
    )
    ap = {
        r["teamId"]: r["rank"]
        for w in rankings
        if w["week"] == ap_week
        for poll in w["polls"]
        if poll["poll"] == "AP Top 25"
        for r in poll["ranks"]
    }
    m = c.m
    teams = []
    for i in sim["eligible"]:
        known = [g for g in reg if i in (g["h"], g["a"]) and np.isfinite(g["margin"])]
        w = sum((g["margin"] > 0) == (g["h"] == i) for g in known)
        conf_games = [g for g in known if g["conference"]]
        cw = sum((g["margin"] > 0) == (g["h"] == i) for g in conf_games)
        p = float(sim["probability"][i])
        conf = m.MEM[2026][m.IDS[i]]["conf"]
        remaining = []
        for j, g in enumerate(reg):
            if i not in (g["h"], g["a"]) or np.isfinite(g["margin"]):
                continue
            home = bool(g["h"] == i)
            row = sim["projections"].iloc[list(sim["future"]).index(j)]
            winp = float(probability(pd.DataFrame([row]))[0])
            remaining.append(
                dict(
                    game_id=int(g["id"]),
                    opponent_id=int(g["aid"] if home else g["hid"]),
                    opponent=schedule.iloc[j].away_team
                    if home
                    else schedule.iloc[j].home_team,
                    home=home,
                    neutral=bool(schedule.iloc[j].neutral_site),
                    date=g["date"],
                    win_probability=winp if home else 1 - winp,
                    provisional=str(g["id"]).startswith("990"),
                )
            )
        teams.append(
            dict(
                id=int(m.IDS[i]),
                team=m.NAMES[i],
                conference=c.e.LEAGUES.get(conf, "Independent"),
                ap_rank=ap.get(int(m.IDS[i])),
                wins=int(w),
                losses=len(known) - int(w),
                conference_wins=int(cw),
                conference_losses=len(conf_games) - int(cw),
                playoff=p,
                bye=float(sim["bye"][i]),
                quarterfinal=float(sim["rounds"][0][i]),
                semifinal=float(sim["rounds"][1][i]),
                final=float(sim["rounds"][2][i]),
                champion=float(sim["rounds"][3][i]),
                conference_champion=float(sim["title"][i]),
                conference_title_game=float(sim["title_game"][i]),
                expected_wins=float(sim["expected_wins"][i]),
                expected_losses=float(sim["expected_losses"][i]),
                selection_sensitivity=float(sim["alt"][i]),
                dependence_tiebreak_sensitivity=float(sensitivity["probability"][i]),
                mc_95_half_width=1.96 * np.sqrt(p * (1 - p) / args.n),
                seed_probabilities=sim["seed_probability"][i].tolist(),
                remaining=remaining,
            )
        )
    teams.sort(key=lambda t: -t["playoff"])
    bracket = [
        dict(
            seed=j + 1,
            team_id=int(m.IDS[i]),
            conference_champion=bool(sim["representative_champs"][i]),
        )
        for j, i in enumerate(sim["representative"])
    ]
    payload = dict(
        schema_version=1,
        season=2026,
        status="published",
        model="joint_scoring_v14",
        model_as_of=state.cutoff.isoformat(),
        results_as_of=cutoff.isoformat(),
        ap_week=ap_week,
        draws=args.n,
        selector="fitted_schedule_adjusted_record",
        teams=teams,
        bracket=bracket,
        rules_url=RULES_URL,
        bracket_method="One legal simulated field maximizing aggregate qualification support, then seed support.",
        limitations=[
            "Experimental CFP selection; historical validation covers only two 12-team seasons.",
            "AP ranks are display context, not an extra fitted selection weight.",
            "Weekly team strength is frozen; completed results update the playoff path.",
            "Conference computer-ranking tiebreaks use a resume proxy.",
            "Four Pac-12 flex opponents are provisional; games do not count in conference standings.",
            "Army-Navy is after selection and is excluded from qualification resumes.",
            "Reproduces the archived weekly model; published game forecasts remain unchanged.",
            "Sampling precision is not a confidence interval for real-world model accuracy.",
        ],
    )
    audit_body = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    (output / "forecast-audit.json").write_text(audit_body)
    body = json.dumps(compact(payload), separators=(",", ":"), allow_nan=False)
    (output / "forecast.json").write_text(body)
    sim["projections"].to_parquet(
        output / "remaining-game-forecasts.parquet", index=False
    )
    import gzip

    size = len(body.encode())
    checks = dict(
        weekly_replay_max_absolute_error=replay_delta,
        teams=len(teams),
        schedule_games=len(reg),
        completed_games=sum(np.isfinite(g["margin"]) for g in reg),
        future_games=len(sim["future"]),
        provisional_flex_games=len(additions),
        draws=args.n,
        probability_mass=sum(t["playoff"] for t in teams),
        bye_mass=sum(t["bye"] for t in teams),
        champion_mass=sum(t["champion"] for t in teams),
        max_game_probability_monte_carlo_z=sim["marginal_z"],
        max_split_half_difference_pp=100 * sim["half_difference"],
        max_selection_sensitivity_pp=100
        * max(abs(t["playoff"] - t["selection_sensitivity"]) for t in teams),
        max_dependence_tiebreak_sensitivity_pp=100
        * max(abs(t["playoff"] - t["dependence_tiebreak_sensitivity"]) for t in teams),
        audit_payload_bytes=len(audit_body.encode()),
        payload_bytes=size,
        gzip_bytes=len(gzip.compress(body.encode())),
        twenty_week_raw_payload_bytes=size * 20,
        elapsed_seconds=time.monotonic() - started,
        conference_tiebreak_counts=sim["tie_counts"],
    )
    assert sim["marginal_z"] < 6
    (output / "validation.json").write_text(json.dumps(checks, indent=2, default=int))
    files = [
        *inputs.glob("*.json"),
        *args.archive.glob("*.parquet"),
        Path(__file__),
        ROOT / "selection.py",
        ROOT / "game_model.py",
        ROOT / "season_2026.json",
        ROOT / "resume.py",
        ROOT / "rules.py",
        *Path("backend/model").glob("*.py"),
    ]
    manifest = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in checks.items()
                if k not in ["historical_assessment", "conference_tiebreak_counts"]
            },
            indent=2,
            default=int,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("output/cfb_playoff_preview")
    )
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--sensitivity-n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    if args.n < 1000 or args.sensitivity_n < 1000:
        parser.error("Use at least 1,000 draws per simulation")
    build(args)


if __name__ == "__main__":
    main()
