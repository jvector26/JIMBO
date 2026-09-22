"""Rest-of-season simulation on the real schedule (results so far locked) + play-in and playoffs.
Adapted from code/simulate.py (preseason version: formula schedule)."""
import numpy as np
from scipy.stats import norm

DIV = {"Atlantic": "BOS BKN NYK PHI TOR", "Central": "CHI CLE DET IND MIL", "Southeast": "ATL CHA MIA ORL WAS",
       "Northwest": "DEN MIN OKC POR UTA", "Pacific": "GSW LAC LAL PHX SAC", "Southwest": "DAL HOU MEM NOP SAS"}
EAST = {"Atlantic", "Central", "Southeast"}
TEAMS = sorted(t for v in DIV.values() for t in v.split())
IDX = {t: i for i, t in enumerate(TEAMS)}
div_of = {t: d for d, v in DIV.items() for t in v.split()}
conf_of = {t: "E" if div_of[t] in EAST else "W" for t in TEAMS}
SIG, HCA = 12.8, 2.6


def fill_tbd(rem, played_n, rng):
    """Teams short of 82 scheduled games (NBA Cup knockout-week slots, TBD until December): add random
    same-conference games, home side random, until every team has 82."""
    need = 82 - (played_n + np.bincount(rem[:, 0], minlength=30) + np.bincount(rem[:, 1], minlength=30))
    extra = []
    for conf in "EW":
        ts = [IDX[t] for t in TEAMS if conf_of[t] == conf]
        pool = [t for t in ts for _ in range(max(int(need[t]), 0))]
        rng.shuffle(pool)
        while len(pool) >= 2:
            a = pool.pop()
            j = next((k for k in range(len(pool) - 1, -1, -1) if pool[k] != a), None)
            if j is None: break
            b = pool.pop(j); extra.append((a, b) if rng.random() < .5 else (b, a))
    return np.vstack([rem, np.array(extra, int).reshape(-1, 2)]) if extra else rem


def simulate_rest(margins, tau, rem, wins_now, played_n, n_sims=10000, seed=7):
    """margins: array(30) per-game neutral margins; rem: (G,2) remaining games [home_idx, away_idx];
    wins_now: array(30). Returns final wins W (n,30) and true strengths TRUE (n,30)."""
    rng = np.random.default_rng(seed)
    rem = fill_tbd(rem, played_n, rng)
    true = margins[None, :] + rng.normal(0, tau, (n_sims, 30))
    W = np.tile(wins_now, (n_sims, 1)).astype(int)
    if len(rem):
        for c0 in range(0, n_sims, 2000):
            t = true[c0:c0 + 2000]
            p = norm.cdf((t[:, rem[:, 0]] - t[:, rem[:, 1]] + HCA) / SIG)
            hw = rng.random(p.shape) < p
            W[c0:c0 + 2000] += np.stack([hw[:, rem[:, 0] == k].sum(1) + (~hw[:, rem[:, 1] == k]).sum(1) for k in range(30)], 1)
    return W, true, rng


def bo7_prob(pa, pb):
    home = [1, 1, 0, 0, 1, 0, 1]; dp = {(0, 0): np.ones_like(pa)}
    for _ in range(7):
        nd = {}
        for (x, y), pr in dp.items():
            if x == 4 or y == 4: nd[(x, y)] = nd.get((x, y), 0) + pr; continue
            p = pa if home[x + y] else pb
            nd[(x + 1, y)] = nd.get((x + 1, y), 0) + pr * p; nd[(x, y + 1)] = nd.get((x, y + 1), 0) + pr * (1 - p)
        dp = nd
    return sum(pr for (x, y), pr in dp.items() if x == 4)


def postseason(W, TRUE, rng):
    n = len(W)
    res = {k: np.zeros(30) for k in ["playoffs", "playin", "top6", "seed1", "r2", "cf", "finals", "title"]}
    jitter = rng.random(W.shape) * 0.5
    r = np.arange(n)
    pwin = lambda a, b, home: norm.cdf((TRUE[r, a] - TRUE[r, b] + (HCA if home else -HCA)) / SIG)
    champs = {}
    for conf in "EW":
        ci = np.array([IDX[t] for t in TEAMS if conf_of[t] == conf])
        order = ci[np.argsort(-(W[:, ci] + jitter[:, ci]), axis=1)]
        for k in range(6, 10): np.add.at(res["playin"], order[:, k], 1)
        for k in range(6): np.add.at(res["top6"], order[:, k], 1)
        np.add.at(res["seed1"], order[:, 0], 1)
        s7, s8, s9, s10 = order[:, 6], order[:, 7], order[:, 8], order[:, 9]
        g1 = rng.random(n) < pwin(s7, s8, True); seed7 = np.where(g1, s7, s8); l78 = np.where(g1, s8, s7)
        g2 = rng.random(n) < pwin(s9, s10, True); w910 = np.where(g2, s9, s10)
        g3 = rng.random(n) < pwin(l78, w910, True); seed8 = np.where(g3, l78, w910)
        seeds = np.column_stack([order[:, :6], seed7, seed8])
        for k in range(8): np.add.at(res["playoffs"], seeds[:, k], 1)
        def series(a, b):
            return np.where(rng.random(n) < bo7_prob(pwin(a, b, True), pwin(a, b, False)), a, b)
        def better(a, b):
            sw = W[r, b] > W[r, a]; return np.where(sw, b, a), np.where(sw, a, b)
        r1 = [series(*better(seeds[:, i], seeds[:, 7 - i])) for i in range(4)]
        for t in r1: np.add.at(res["r2"], t, 1)
        r2 = [series(*better(r1[0], r1[3])), series(*better(r1[1], r1[2]))]
        for t in r2: np.add.at(res["cf"], t, 1)
        cc = series(*better(r2[0], r2[1])); np.add.at(res["finals"], cc, 1); champs[conf] = cc
    a, b = champs["E"], champs["W"]
    hi, lo = np.where(W[r, b] > W[r, a], b, a), np.where(W[r, b] > W[r, a], a, b)
    ch = np.where(rng.random(n) < bo7_prob(pwin(hi, lo, True), pwin(hi, lo, False)), hi, lo)
    np.add.at(res["title"], ch, 1)
    return {k: v / n for k, v in res.items()}
