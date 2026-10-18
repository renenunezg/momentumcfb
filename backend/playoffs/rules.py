"""Reviewed 2026 CFP access and conference tiebreak rules."""

import numpy as np

from backend.playoffs import resume as m

LEAGUES = m.DATA["leagues"]
CFG = m.DATA["config"]
INELIGIBLE = {}


def current_games():
    return m.DATA["flex"]


def resolve(candidates, cw, cg, result, opponents, score, league, capped, counts):
    cand = list(candidates)
    if len(cand) == 1:
        return cand[0]

    def keep(values):
        nonlocal cand
        best = max(values)
        new = [i for i, v in zip(cand, values) if abs(v - best) < 1e-9]
        changed = len(new) < len(cand)
        cand = new
        return changed

    # Direct meetings, then common opponents. A sweep can break an incomplete mini-league.
    allplayed = all(result[a, b] != 0 for a in cand for b in cand if a != b)
    if allplayed:
        if keep(
            [
                sum(result[i, j] > 0 for j in cand if j != i) / (len(cand) - 1)
                for i in cand
            ]
        ):
            return resolve(
                cand, cw, cg, result, opponents, score, league, capped, counts
            )
    else:
        sweep = [i for i in cand if all(result[i, j] > 0 for j in cand if j != i)]
        if sweep:
            return sweep[0]
        losers = (
            [i for i in cand if all(result[i, j] < 0 for j in cand if j != i)]
            if league == "ACC"
            else []
        )
        cand = [i for i in cand if i not in losers]
        if losers:
            return resolve(
                cand, cw, cg, result, opponents, score, league, capped, counts
            )
    if len(cand) == 1:
        return cand[0]
    if (league == "ACC" and counts["_year"] >= 2026) or league in (
        "American Athletic",
        "Mountain West",
    ):
        counts["ssa_proxy"] += 1
        return max(cand, key=lambda i: score[i])
    common = set.intersection(*(opponents[i] for i in cand))
    if common:
        if keep([sum(result[i, j] > 0 for j in common) / len(common) for i in cand]):
            return resolve(
                cand, cw, cg, result, opponents, score, league, capped, counts
            )
        if len(cand) == 1:
            return cand[0]
        for pct in sorted({cw[j] / cg[j] for j in common}, reverse=True):
            group = [j for j in common if abs(cw[j] / cg[j] - pct) < 1e-9]
            if keep([sum(result[i, j] > 0 for j in group) / len(group) for i in cand]):
                return resolve(
                    cand, cw, cg, result, opponents, score, league, capped, counts
                )
            if len(cand) == 1:
                return cand[0]
    keep(
        [
            sum(cw[j] for j in opponents[i]) / sum(cg[j] for j in opponents[i])
            for i in cand
        ]
    )
    if len(cand) == 1:
        return cand[0]
    if league == "SEC":
        keep([capped[i] for i in cand])
    if league == "Big 12":
        counts["big12_total_wins_used"] += 1
        if keep([counts["_eligible_wins"][i] for i in cand]):
            counts["big12_total_wins_resolved"] += 1
    if len(cand) == 1:
        return cand[0]
    counts["rating_fallback"] += 1
    return max(cand, key=lambda i: score[i])


def title_pair(members, cw, cg, result, opponents, score, league, capped, counts):
    left = list(members)
    pair = []
    for slot in range(2):
        top = max(cw[i] / cg[i] for i in left)
        best = [i for i in left if abs(cw[i] / cg[i] - top) < 1e-9]
        cand = best.copy()
        if league == "ACC" and counts["_year"] >= 2026:
            cand += [
                i
                for i in left
                if i not in best
                and any(
                    cg[i] != cg[j]
                    and (cw[i] == cw[j] or cg[i] - cw[i] == cg[j] - cw[j])
                    for j in best
                )
            ]
        pick = resolve(cand, cw, cg, result, opponents, score, league, capped, counts)
        pair.append(pick)
        left.remove(pick)
    return pair


def context(year, reg):
    mem = m.MEM[year]
    members = {
        c: [m.IX[id] for id, t in mem.items() if t["conf"] == c] for c in LEAGUES
    }
    divisions = {
        c: [
            [m.IX[id] for id, t in mem.items() if t["conf"] == c and t["division"] == v]
            for v in sorted(
                {
                    t["division"]
                    for t in mem.values()
                    if t["conf"] == c and t["division"]
                }
            )
        ]
        for c in LEAGUES
    }
    opponents = [set() for _ in range(m.T)]
    cg = np.zeros(m.T, int)
    cj = []
    for j, g in enumerate(reg):
        if g["conference"]:
            h, a = g["h"], g["a"]
            cg[h] += 1
            cg[a] += 1
            opponents[h].add(a)
            opponents[a].add(h)
            cj.append(j)
    return members, divisions, opponents, cg, cj


def title_participants(year, reg, margins, rating, regular_wins, random_ties=False):
    n = len(margins)
    members, divisions, opponents, cg, cj = context(year, reg)
    cw = np.zeros((n, m.T), int)
    results = np.zeros((n, m.T, m.T), np.int8)
    capped = np.zeros((n, m.T))
    counts = {
        "ssa_proxy": 0,
        "rating_fallback": 0,
        "big12_total_wins_used": 0,
        "big12_total_wins_resolved": 0,
    }
    for j in cj:
        g = reg[j]
        h, a = g["h"], g["a"]
        win = margins[:, j] > 0
        cw[:, h] += win
        cw[:, a] += ~win
        results[:, h, a] = np.where(win, 1, -1)
        results[:, a, h] = -results[:, h, a]
        capped[:, h] += np.clip(margins[:, j], -28, 28)
        capped[:, a] -= np.clip(margins[:, j], -28, 28)
    if year != 2026:
        raise ValueError("Only reviewed 2026 rules are supported")
    actualtitles = {}
    titleconfs = [
        c
        for c in LEAGUES
        if len(members[c]) > 2 and (year == 2026 or c in actualtitles)
    ]
    pairs = {c: np.zeros((n, 2), int) for c in titleconfs}
    fcswins = np.zeros((n, m.T), int)
    for j, g in enumerate(reg):
        if g["h"] < 0:
            fcswins[:, g["a"]] += margins[:, j] < 0
        if g["a"] < 0:
            fcswins[:, g["h"]] += margins[:, j] > 0
    eligiblewins = regular_wins - np.maximum(fcswins - 1, 0)
    tie_rng = np.random.default_rng(71717)
    scores = tie_rng.normal(size=rating.shape) if random_ties else rating
    for k in range(n):
        counts["_eligible_wins"] = eligiblewins[k]
        counts["_year"] = year
        for c in titleconfs:
            groups = divisions[c]
            allowed = members[c]
            # Transitioning teams could not win these conference championships in these seasons.
            exclusions = INELIGIBLE.get(year, [])
            allowed = [i for i in allowed if m.NAMES[i] not in exclusions]
            if groups:
                p = []
                for group in groups:
                    group = [i for i in group if i in allowed]
                    p.append(
                        title_pair(
                            group,
                            cw[k],
                            cg,
                            results[k],
                            opponents,
                            scores[k],
                            LEAGUES[c],
                            capped[k],
                            counts,
                        )[0]
                    )
                if cw[k, p[1]] / cg[p[1]] > cw[k, p[0]] / cg[p[0]]:
                    p = p[::-1]
            else:
                p = title_pair(
                    allowed,
                    cw[k],
                    cg,
                    results[k],
                    opponents,
                    scores[k],
                    LEAGUES[c],
                    capped[k],
                    counts,
                )
            pairs[c][k] = p
    counts.pop("_eligible_wins", None)
    counts.pop("_year", None)
    return pairs, cw, counts


def field(score, champs, eligible, year):
    n = len(score)
    score = score.copy()
    score[:, [m.NAMEIX[n] for n in INELIGIBLE.get(year, [])]] = -np.inf
    score[:, ~np.isin(np.arange(m.T), eligible)] = -np.inf
    order = np.argsort(-score, axis=1)
    rank = np.argsort(order, axis=1) + 1
    q = np.zeros_like(score, dtype=bool)
    if year <= 2023:
        q[np.arange(n)[:, None], order[:, :4]] = True
        return q, rank
    if year in [2024, 2025]:
        champion_order = np.argsort(-np.where(champs, score, -np.inf), axis=1)[:, :5]
        q[np.arange(n)[:, None], champion_order] = True
    else:
        p4 = [
            m.IX[id]
            for id, t in m.MEM[year].items()
            if t["conf"] in ["1", "4", "5", "8"]
        ]
        q[:, p4] = champs[:, p4]
        g6 = [
            m.IX[id]
            for id, t in m.MEM[year].items()
            if t["conf"] not in ["1", "4", "5", "8", "18"]
        ]
        best = np.array(g6)[np.argmax(score[:, g6], axis=1)]
        q[np.arange(n), best] = True
        nd = m.NAMEIX["Notre Dame"]
        q[:, nd] |= rank[:, nd] <= 12
    for k in range(n):
        q[k, [i for i in order[k] if not q[k, i]][: 12 - q[k].sum()]] = True
    assert np.all(q.sum(1) == 12)
    return q, rank
