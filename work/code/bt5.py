import json, numpy as np, pandas as pd, impact as I
TEST = range(2014, 2025); UNRATED = -2.0
W = json.load(open(f"{I.D}/weights_bt.json")); Wo, Wd = pd.Series(W["Wo"]) * .5, pd.Series(W["Wd"]) * .5
FEATS = ["f:" + x for x in I.OFF_FEATS + I.DEF_FEATS]
COMPS = FEATS + ["lineup_impact"]
pg = pd.read_parquet(f"{I.D}/player_games_all.parquet")[["season", "game", "player", "team", "min"]]
gm = pd.read_parquet(f"{I.D}/games_all.parquet"); gm["margin"] = gm.home_pts - gm.away_pts
out = []
for N in TEST:
    rows = I.load_rows([N - 2, N - 1, N]); pace = (rows.groupby("game").poss.sum() / 2).mean()
    o = I.fit_window(N, Wo, Wd, lam=2000, rows=rows).set_index("player")
    nxt = pg[pg.season == N + 1].copy()
    rated = nxt.player.isin(o.index)
    g = gm[gm.season == N + 1][["game", "home_team", "away_team", "margin"]].copy()
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    for c in COMPS + ["unrated"]:
        v = (nxt.player.map(o[c]).fillna(0) if c != "unrated" else (~rated) * UNRATED) * nxt["min"] / 48
        t = v.groupby([nxt.game, nxt.team]).sum()
        g[c] = (t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy()) * pace / 100
    g["season"] = N + 1; out.append(g); print("done", N + 1, flush=True)
B = pd.concat(out, ignore_index=True)
B.to_parquet(f"{I.D}/backtest_components.parquet", index=False)
pred = B[COMPS + ["unrated"]].sum(axis=1); X = np.c_[np.ones(len(B)), pred]
b = np.linalg.lstsq(X, B.margin, rcond=None)[0]; print("check RMSE", np.sqrt(((B.margin - X @ b) ** 2).mean()).round(3))
