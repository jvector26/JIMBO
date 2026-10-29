"""Window decay test (session 12). Refit windows N-2..N with alternative season decays; box features AND lineup residuals
both use the decay (as fit_window does). Game diffs in N+1 (actual minutes) -> data/decay_bt_{N}.parquet:
columns <tag>:box_o, <tag>:box_d (sum of W-weighted features, o_3eff excluded: taken from backtest_components for all),
<tag>:res_o, <tag>:res_d. Usage: python3 decay_bt.py N0 N1"""
import sys, os, json, numpy as np, pandas as pd, impact as I
W = json.load(open(f"{I.D}/weights_bt.json")); Wo, Wd = pd.Series(W["Wo"]) * .5, pd.Series(W["Wd"]) * .5
DECAYS = {"d3": (.3, .6, 1.), "d1": (.1, .4, 1.), "d4": (.4, .7, 1.), "d5": (.5, .8, 1.), "d7": (.7, .9, 1.)}
gm = pd.read_parquet(f"{I.D}/games_all.parquet")
for N in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    fn = f"{I.D}/decay_bt_{N}.parquet"
    if os.path.exists(fn): continue
    rows = I.load_rows([N - 2, N - 1, N]); pace = (rows.groupby("game").poss.sum() / 2).mean()
    nxt = pd.read_parquet(f"{I.D}/player_games_{N+1}.parquet")[["game", "player", "team", "min"]]
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]].copy()
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    for tag, dec in DECAYS.items():
        o = I.fit_window(N, Wo, Wd, lam=2000, decay=dec, rows=rows).set_index("player")
        fo = [c for c in o.columns if c.startswith("f:o_") and c != "f:o_3eff"]; fd = [c for c in o.columns if c.startswith("f:d_")]
        V = pd.DataFrame({"box_o": o[fo].sum(axis=1), "box_d": o[fd].sum(axis=1), "res_o": o.res_o, "res_d": o.res_d})
        t = V.reindex(nxt.player).fillna(0).mul(nxt["min"].values / 48, axis=0)
        t.index = pd.MultiIndex.from_arrays([nxt.game, nxt.team]); t = t.groupby(level=[0, 1]).sum()
        d = (t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy()) * pace / 100
        for j, c in enumerate(V.columns): g[f"{tag}:{c}"] = d[:, j]
    g.to_parquet(fn, index=False); print("done", N, flush=True)
