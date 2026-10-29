"""Refit lineup residuals with the dreb weight scaled (prior changes -> res_d re-absorbs). bt_split style."""
import sys, os, json, numpy as np, pandas as pd, impact as I
W = json.load(open(f"{I.D}/weights_bt.json")); Wo, Wd0 = pd.Series(W["Wo"]) * .5, pd.Series(W["Wd"]) * .5
MS = [0.25, 0.5, 1.0]
gm = pd.read_parquet(f"{I.D}/games_all.parquet")
for N in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    fn = f"{I.D}/dreb_refit_{N}.parquet"
    if os.path.exists(fn): continue
    rows = I.load_rows([N-2, N-1, N]); pace = (rows.groupby("game").poss.sum() / 2).mean()
    nxt = pd.read_parquet(f"{I.D}/player_games_{N+1}.parquet")[["game", "player", "team", "min"]]
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]].copy()
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    for m in MS:
        Wd = Wd0.copy(); Wd["d_dreb"] *= m
        o = I.fit_window(N, Wo, Wd, lam=2000, rows=rows).set_index("player")
        for c in ["res_o", "res_d"]:
            v = nxt.player.map(o[c]).fillna(0) * nxt["min"] / 48
            t = v.groupby([nxt.game, nxt.team]).sum()
            g[f"{c}_{m}"] = (t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy()) * pace / 100
    g.to_parquet(fn, index=False); print("done", N, flush=True)
