"""Refit lineup residuals with the foul-drawing weights scaled (prior changes -> res_o re-absorbs). Same as dreb_refit.py."""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code"); import impact as I
W = json.load(open(f"{I.D}/weights_bt.json")); Wo0, Wd = pd.Series(W["Wo"]) * .5, pd.Series(W["Wd"]) * .5
MS = [(1, 1), (.5, 1), (.5, 2.5), (.75, 1.75), (1, 2.5), (.5, 2)]
tag = lambda v, e: f"{v}_{e}"
gm = pd.read_parquet(f"{I.D}/games_all.parquet")
for N in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    fn = f"{I.D}/ft_refit_{N}.parquet"
    if os.path.exists(fn): continue
    rows = I.load_rows([N-2, N-1, N]); pace = (rows.groupby("game").poss.sum() / 2).mean()
    nxt = pd.read_parquet(f"{I.D}/player_games_{N+1}.parquet")[["game", "player", "team", "min"]]
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]].copy()
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    for v, e in MS:
        Wo = Wo0.copy(); Wo["o_ftvol"] *= v; Wo["o_fteff"] *= e
        o = I.fit_window(N, Wo, Wd, lam=2000, rows=rows).set_index("player")
        for c in ["res_o", "res_d"]:
            x = nxt.player.map(o[c]).fillna(0) * nxt["min"] / 48
            t = x.groupby([nxt.game, nxt.team]).sum()
            g[f"{c}_{tag(v, e)}"] = (t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy()) * pace / 100
    g.to_parquet(fn, index=False); print("done", N, flush=True)
