"""Defense #3 backtest: window (N-2..N, decay .3/.6/1) player values of new defensive-event rates -> game diffs in N+1
(actual minutes) -> incremental value over current JIMBO (box + 0.6 res_o + 0.8 res_d), LOSO by season."""
import numpy as np, pandas as pd
D = "/home/claude/data"
f = pd.read_parquet(f"{D}/features.parquet")[["season", "player", "poss_def", "d_pf", "d_blk", "d_stl"]]
ev = pd.concat([pd.read_parquet(f"{D}/defev_{s}.parquet") for s in range(2012, 2026)])
x = f[f.season >= 2012].merge(ev, on=["season", "player"], how="left").fillna(0)
x["blk_non"] = x.blk_paint + x.blk_jump
x["pf_all"] = x.sfl + x.nsfl
x["offd_nc"] = x.offd - x.chg
EV = ["offd", "chg", "offd_nc", "sfl", "nsfl", "blk_rim", "blk_non", "blk_jump", "pf_all"]
KS = [300, 1000, 3000]
for c in EV:
    lg = x.groupby("season")[c].transform("sum") / x.groupby("season").poss_def.transform("sum")
    for k in KS:
        x[f"{c}_{k}"] = (x[c] - lg * x.poss_def) / (x.poss_def + k) * 100
cols = [f"{c}_{k}" for c in EV for k in KS]
gm = pd.read_parquet(f"{D}/games_all.parquet")
out = []
for N in range(2014, 2025):
    w = x[x.season.between(N - 2, N)].copy()
    w["wt"] = w.poss_def * w.season.map({N - 2: .3, N - 1: .6, N: 1.})
    v = w[cols].mul(w.wt, axis=0).groupby(w.player).sum().div(w.groupby("player").wt.sum(), axis=0)
    pg = pd.read_parquet(f"{D}/player_games_{N+1}.parquet")[["game", "player", "team", "min"]]
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]]
    pace = 100.0  # scale only; coefficient is free
    t = v.reindex(pg.player).fillna(0).mul(pg["min"].values / 48, axis=0)
    t.index = pd.MultiIndex.from_arrays([pg.game, pg.team]); t = t.groupby(level=[0, 1]).sum()
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    dd = pd.DataFrame(t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy(), columns=["ev_" + c for c in cols])
    dd["game"] = g.game.values
    out.append(dd)
pd.concat(out).to_parquet(f"{D}/defev_bt.parquet", index=False)
# window player values for inspection (2023-25)
N = 2025; w = x[x.season.between(N - 2, N)].copy(); w["wt"] = w.poss_def * w.season.map({N - 2: .3, N - 1: .6, N: 1.})
v = w[cols].mul(w.wt, axis=0).groupby(w.player).sum().div(w.groupby("player").wt.sum(), axis=0)
v["poss"] = w.groupby("player").poss_def.sum(); v.to_parquet(f"{D}/defev_window_2025.parquet")
print("ok")
