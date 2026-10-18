"""Frozen selection inputs and schedule-adjusted resume calculations."""

import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

DATA = json.loads(Path(__file__).with_name("season_2026.json").read_text())
IDS = DATA["ids"]
IX = {v: i for i, v in enumerate(IDS)}
T = len(IDS)
MEM = {2026: DATA["membership"]}
NAMES = {int(k): v for k, v in DATA["names"].items()}
NAMEIX = DATA["name_index"]
HFA = 3.0
FCS = -20.0


def prior_for(year, rho, ridge):
    if year != 2026 or rho != DATA["config"]["rho"] or ridge != DATA["config"]["ridge"]:
        raise ValueError("Selection priors require a reviewed season configuration")
    return np.array(DATA["prior"])


def matrix(games):
    A = np.zeros((len(games), T))
    offset = np.array([g["loc"] for g in games])
    for j, g in enumerate(games):
        if g["h"] >= 0:
            A[j, g["h"]] = 1
        else:
            offset[j] += FCS
        if g["a"] >= 0:
            A[j, g["a"]] = -1
        else:
            offset[j] -= FCS
    return A, offset


def features(games, margins, ratings, eligible, sd=16.0, extra=None):
    """Season resumes. Ratings come only from games available in this realization."""
    n = len(margins)
    bench = np.sort(ratings[:, eligible], axis=1)[:, -25:].mean(1)
    f = np.zeros((n, T, 9))
    wins = np.zeros((n, T))
    losses = np.zeros((n, T))
    quality = np.zeros((n, T))
    bad = np.zeros((n, T))
    sos = np.zeros((n, T))
    counts = np.zeros((n, T))
    for i in eligible:
        mass = np.ones((n, 1))
        w = np.zeros(n, int)
        meetings = []
        for j, g in enumerate(games):
            if i not in (g["h"], g["a"]):
                continue
            sign = 1 if i == g["h"] else -1
            o = g["a"] if sign == 1 else g["h"]
            op = ratings[:, o] if o >= 0 else np.full(n, FCS)
            win = sign * margins[:, j] > 0
            meetings.append((op, win, sign * g["loc"], np.ones(n, bool)))
        if extra is not None:
            played, opp, won, loc = extra
            op = ratings[np.arange(n), np.maximum(opp[:, i], 0)]
            meetings.append((op, won[:, i].astype(bool), loc[:, i], played[:, i]))
        for op, win, loc, mask in meetings:
            p = np.clip(ndtr((bench - op + loc) / sd), 1e-6, 1 - 1e-6)
            effective = np.where(mask, p, 0)
            nxt = np.zeros((n, mass.shape[1] + 1))
            nxt[:, :-1] += mass * (1 - effective[:, None])
            nxt[:, 1:] += mass * effective[:, None]
            mass = nxt
            w += mask * win
            losses[:, i] += mask * (~win)
            counts[:, i] += mask
            sos[:, i] += mask * op
            quality[:, i] += mask * win * (-np.log(p))
            bad[:, i] += mask * (~win) * np.log1p(-p)
        wins[:, i] = w
        tail = np.cumsum(mass[:, ::-1], axis=1)[:, ::-1]
        f[:, i, 0] = -np.log(np.clip(tail[np.arange(n), w], 1e-10, 1))
    f[:, :, 1] = ratings / 10
    f[:, :, 2] = quality
    f[:, :, 3] = bad
    f[:, :, 4] = wins / 4
    f[:, :, 5] = -losses
    f[:, :, 6] = (losses == 0) & (wins >= 5)
    f[:, :, 7] = sos / np.maximum(counts, 1) / 10
    # Weak records receive no bonus for being an undefeated team with no games played.
    f[:, :, 8] = np.maximum(wins - losses, 0) / 4
    return f, wins, losses
