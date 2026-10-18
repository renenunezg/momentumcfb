"""Protect the 2026 automatic-bid boundaries used by the weekly publisher."""

import numpy as np

from backend.playoffs import selection as c
from backend.playoffs.forecast import seeded_field


def test_reviewed_2026_automatic_bids_and_seeding():
    m = c.m
    eligible = np.array([m.IX[i] for i in m.MEM[2026]])
    scores = np.full((2, m.T), -1000.0)
    scores[:, eligible] = np.linspace(100, 0, len(eligible))
    champions = np.zeros_like(scores, bool)
    p4_champs = []
    for conf in ["1", "4", "5", "8"]:
        i = next(m.IX[id] for id, t in m.MEM[2026].items() if t["conf"] == conf)
        p4_champs.append(i)
        champions[:, i] = True
        scores[:, i] = -100
    nd = m.NAMEIX["Notre Dame"]
    g6 = next(m.IX[id] for id, t in m.MEM[2026].items() if t["conf"] == "9")
    scores[:, g6] = 200  # Highest G6 team qualifies even without a league title.
    scores[:, nd] = 90
    # Place ND exactly 12th in row zero; put ND 13th in row one.
    other = [i for i in eligible if i not in [nd, g6, *p4_champs]]
    scores[:, other] = 0
    scores[:, other[:10]] = np.arange(110, 120)
    scores[1, other[10]] = 95
    q, seeds = seeded_field(scores, champions, eligible)
    assert np.all(q[:, p4_champs]) and np.all(q[:, g6])
    assert q[0, nd] and not q[1, nd]
    assert np.all(np.isin(seeds[:, -4:], p4_champs))
    assert dict(
        p4_champions_outside_top12=True,
        highest_g6_nonchampion=True,
        notre_dame_rank12_guarantee=True,
        notre_dame_rank13_no_guarantee=True,
        _low_ranked_automatic_bids_seeded_last=True,
    )
