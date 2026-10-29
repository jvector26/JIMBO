"""Session 17: rim protection for small players. Game diffs N+1 (actual minutes, game_minutes.parquet) of
(a) window rim_protection / perimeter_def / dreb split by position group, (b) positional-baseline offsets
b(pos) = min-weighted quadratic fit of the component on window pos_est. Scaled to backtest_components units.
-> data/guard_rim_bt.parquet"""
import numpy as np, pandas as pd
D = "/home/claude/data"
R = pd.read_parquet(f"{D}/bt_split_ratings.parquet")
po = pd.read_parquet(f"{D}/positions.parquet")[["season", "player", "min", "pos_est"]]
gm = pd.read_parquet(f"{D}/game_minutes.parquet")[["season", "game", "player", "team", "min"]]
ga = pd.read_parquet(f"{D}/games_all.parquet"); ga = ga[ga.season.between(2015, 2025)][["game", "home_team", "away_team"]]
out = []
for N in range(2014, 2025):
    r = R[R.season == N].set_index("player").copy()
    w = po[po.season.between(N - 2, N)].copy(); w["wt"] = w["min"] * w.season.map({N - 2: .3, N - 1: .6, N: 1.})
    pe = (w.pos_est * w.wt).groupby(w.player).sum() / w.groupby("player").wt.sum()
    mw = w.groupby("player").wt.sum()
    r["pos"] = pe.reindex(r.index).fillna(3.0); r["mw"] = mw.reindex(r.index).fillna(0)
    r["boxd"] = r.def_rebounding + r.rim_protection + r.perimeter_def
    r["G"] = (r.pos < 2.5).astype(float); r["Bg"] = (r.pos >= 3.9).astype(float); r["Wg"] = 1 - r.G - r.Bg
    cols = {}
    for c in ["rim_protection", "perimeter_def", "def_rebounding", "boxd", "res_d"]:
        X = np.c_[np.ones(len(r)), r.pos, r.pos ** 2]; sw = np.sqrt(r.mw.values)
        b = np.linalg.lstsq(X * sw[:, None], r[c].values * sw, rcond=None)[0]
        base = X @ b
        cols[f"off_{c}"] = base                                   # positional baseline (to be subtracted if c=-1)
        for g in ["G", "Wg", "Bg"]:
            cols[f"{c}_{g}"] = r[c].values * r[g].values           # component carried by that position group
            cols[f"dev_{c}_{g}"] = (r[c].values - base) * r[g].values  # within-position deviation
    V = pd.DataFrame(cols, index=r.index)
    if N == 2024: print(r.groupby(pd.cut(r.pos, [0, 1.5, 2.5, 3.45, 4.4, 9]), observed=True)[["rim_protection", "perimeter_def", "def_rebounding"]].mean().round(2))
    nx = gm[gm.season == N + 1]
    t = V.reindex(nx.player).fillna(0).mul(nx["min"].values / 48, axis=0)
    t.index = pd.MultiIndex.from_arrays([nx.game.values, nx.team.values]); t = t.groupby(level=[0, 1]).sum()
    g = ga[ga.game.isin(nx.game)]
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    dd = pd.DataFrame(t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy(), columns=V.columns); dd["game"] = g.game.values
    out.append(dd)
pd.concat(out).to_parquet(f"{D}/guard_rim_bt.parquet", index=False); print("ok")
