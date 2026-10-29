"""Judgment shrink (user, 2026-10-07): lineup residual x m, m = 1 - a*clip(1 - P/P1, 0, 1), P = raw window offensive
possessions (sum on_poss over the 3 window seasons = production poss_window). Symmetric. Game LOSO cost."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code")
from lineup_port_bt import G, B, y, seas, diffs, loso, fixed, ps
P = {}
for N in range(2014, 2025):
    P[N] = ps[ps.season.between(N - 2, N)].groupby("player").on_poss.sum()
G["Praw"] = [P[n].get(p, 0.0) for n, p in zip(G.N, G.player)]
base = diffs(np.ones(len(G))); b0, s0 = fixed(base)
print(f"baseline LOSO {b0:.4f}; share of player-game minutes with P<10k: {np.average(G.Praw < 10000, weights=G['min']):.2f}")
rows = []
for P1 in [6000, 8000, 10000, 12000]:
    ramp = np.clip(1 - G.Praw / P1, 0, 1)
    for a in [.2, .25, .33, .4, .5]:
        l, s = fixed(diffs(1 - a * ramp))
        h1 = np.sqrt(np.mean(s[:5]**2)) - np.sqrt(np.mean(s0[:5]**2)); h2 = np.sqrt(np.mean(s[5:]**2)) - np.sqrt(np.mean(s0[5:]**2))
        rows.append(dict(P1=P1, a=a, LOSO=l, d=l - b0, better=int((s < s0).sum()), h1=h1, h2=h2))
print(pd.DataFrame(rows).round(4).to_string(index=False))
