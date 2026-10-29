"""Per window N: defender sub-outcome coefs (N-2..N) -> game diffs in N+1 with actual minutes (bt_split style)."""
import sys, os, numpy as np, pandas as pd, defsub as S, impact as I
LAMS = dict(r3=500, rmid=500, rrim=1000, rfta=1000, rtov=1000, orb=1000, prim=1000, p3=16000, pts=1000)
gm = pd.read_parquet(f"{I.D}/games_all.parquet")
for N in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    fn = f"{I.D}/defsub_{N}.parquet"
    if os.path.exists(fn): continue
    r = S.rows([N - 2, N - 1, N]); pace = (r.groupby("game").poss.sum() / 2).mean()
    o = S.fit_window(N, LAMS, r)
    o["season"] = N; o.to_parquet(f"{I.D}/defsub_rat_{N}.parquet", index=False)
    o = o.set_index("player")
    nxt = pd.read_parquet(f"{I.D}/player_games_{N+1}.parquet")[["game", "player", "team", "min"]]
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]].copy()
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    for c in LAMS:
        v = nxt.player.map(o[c]).fillna(0) * nxt["min"] / 48
        t = v.groupby([nxt.game, nxt.team]).sum()
        g["ds_" + c] = (t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy()) * pace / 100
    g.to_parquet(fn, index=False); print("done", N, flush=True)
