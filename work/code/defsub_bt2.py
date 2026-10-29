"""Variant: separate ridge for offensive controls (lam_o) and defenders (lam_d)."""
import sys, os, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import defsub as S, impact as I
LO = dict(r3=500, rrim=1000, rtov=1000, orb=1000, rfta=1000, prim=1000, rmid=500)
LD = [4000, 16000, 64000]
gm = pd.read_parquet(f"{I.D}/games_all.parquet")
for N in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    fn = f"{I.D}/defsub2_{N}.parquet"
    if os.path.exists(fn): continue
    seasons = [N-2, N-1, N]; r = S.rows(seasons); pace = (r.groupby("game").poss.sum() / 2).mean()
    dec = r.season.map(dict(zip(seasons, (.3, .6, 1.)))).to_numpy()
    X, players, nfe = S.design(r); P = len(players)
    nxt = pd.read_parquet(f"{I.D}/player_games_{N+1}.parquet")[["game", "player", "team", "min"]]
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]].copy()
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    rat = pd.DataFrame({"player": players})
    for name, lo in LO.items():
        y, w = S.outcome(r, name, dec)
        Xw = X.multiply(w[:, None]).tocsr(); A0 = (X.T @ Xw).tocsc(); bb = Xw.T @ y
        for ld in LD:
            pen = np.r_[np.full(P, lo), np.full(P, ld), np.full(nfe, 1e-6)]
            b = spsolve(A0 + sp.diags(pen, format="csc"), bb)[P:2*P]
            c = f"{name}_{ld}"; rat[c] = b
            v = nxt.player.map(pd.Series(b, players)).fillna(0) * nxt["min"] / 48
            t = v.groupby([nxt.game, nxt.team]).sum()
            g["ds_" + c] = (t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy()) * pace / 100
    rat["season"] = N; rat.to_parquet(f"{I.D}/defsub2_rat_{N}.parquet", index=False)
    g.to_parquet(fn, index=False); print("done", N, flush=True)
