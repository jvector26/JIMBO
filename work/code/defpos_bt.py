"""Defense #4: do defensive box features carry different value by position? Window N-2..N (decay .3/.6/1, poss_def
weighted, as fit_window) feature values x window position (pos_est - 3, or G/B dummies vs wings) -> game diffs in N+1
with actual minutes -> data/defpos_bt.parquet."""
import numpy as np, pandas as pd
D = "/home/claude/data"
FE = ["d_dreb", "d_blk", "d_opprim", "d_stl", "d_pf", "d_oppxq"]
f = pd.read_parquet(f"{D}/features.parquet")[["season", "player", "poss_def"] + FE]
po = pd.read_parquet(f"{D}/positions.parquet")[["season", "player", "pos_est"]]
x = f.merge(po, on=["season", "player"], how="left")
x["pos_est"] = x.pos_est.fillna(3.0)
gm = pd.read_parquet(f"{D}/games_all.parquet")
out = []
for N in range(2014, 2025):
    w = x[x.season.between(N - 2, N)].copy()
    w["wt"] = w.poss_def * w.season.map({N - 2: .3, N - 1: .6, N: 1.})
    v = w[FE + ["pos_est"]].mul(w.wt, axis=0).groupby(w.player).sum().div(w.groupby("player").wt.sum(), axis=0)
    pc = v.pos_est - 3
    G, B = (v.pos_est < 2.5).astype(float), (v.pos_est >= 3.9).astype(float)
    cols = {}
    for c in FE:
        cols[f"{c}_x_pos"] = v[c] * pc
        cols[f"{c}_x_G"] = v[c] * G
        cols[f"{c}_x_B"] = v[c] * B
    V = pd.DataFrame(cols)
    pg = pd.read_parquet(f"{D}/player_games_{N+1}.parquet")[["game", "player", "team", "min"]]
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]]
    t = V.reindex(pg.player).fillna(0).mul(pg["min"].values / 48, axis=0)
    t.index = pd.MultiIndex.from_arrays([pg.game, pg.team]); t = t.groupby(level=[0, 1]).sum()
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    dd = pd.DataFrame(t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy(), columns=["dp_" + c for c in V.columns])
    dd["game"] = g.game.values; out.append(dd)
pd.concat(out).to_parquet(f"{D}/defpos_bt.parquet", index=False); print("ok")
