"""Frozen selector coefficients and conference championship projections."""

import numpy as np
import pandas as pd

from backend.playoffs import resume as m
from backend.playoffs import rules as e

SELECTION = m.DATA["selection"]
BETA = np.array(SELECTION["beta"])
RECORD_BETA = m.DATA["record_beta"]


def title_projections(state, year):
    records = []
    for conf in e.LEAGUES:
        teams = [m.IX[id] for id, row in m.MEM[year].items() if row["conf"] == conf]
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                hi, ai = int(m.IDS[h]), int(m.IDS[a])
                catalog = state.fit.teams.set_index("team_id")
                if hi not in catalog.index or ai not in catalog.index:
                    raise ValueError(f"Missing title team: {hi}, {ai}")
                neutral = conf in ["1", "4", "5", "8", "15"] or (
                    conf == "9" and year < 2026
                )
                records.append(
                    dict(
                        game_id=900000000 + len(records),
                        season=year,
                        week=16,
                        home_team_id=hi,
                        away_team_id=ai,
                        home_team=catalog.loc[hi, "team"],
                        away_team=catalog.loc[ai, "team"],
                        neutral_site=neutral,
                        h=h,
                        a=a,
                    )
                )
    schedule = pd.DataFrame(records)
    projections = state.project(schedule)
    state.validate_marginal_variance(projections)
    return projections.assign(
        h=schedule.h.to_numpy(), a=schedule.a.to_numpy()
    ).set_index(["h", "a"])


def corrected_pairs(year, reg, margins, proxy, wins):
    pairs, cw, counts = e.title_participants(year, reg, margins, proxy, wins)
    _, divisions, _, cg, _ = e.context(year, reg)
    # The archived engine resolved participants but not tied Sun Belt hosting.
    # Future composite computer rankings are approximated by the frozen resume score.
    if "37" in pairs and divisions["37"]:
        p = pairs["37"]
        rows = np.arange(len(p))
        h, a = p[:, 0], p[:, 1]
        tied = np.isclose(cw[rows, h] / cg[h], cw[rows, a] / cg[a])
        reverse = tied & (proxy[rows, a] > proxy[rows, h])
        p[reverse] = p[reverse, ::-1]
        counts["sun_belt_host_proxy_used"] = int(tied.sum())
    return pairs, counts
