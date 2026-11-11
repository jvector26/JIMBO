"""T6 transition add-on features (defev protocol): season rates vs league, shrunk poss/(poss+k), window N-2..N decay .3/.6/1,
game diffs in N+1 with actual minutes. Output fund/tr_bt.parquet, fund/tr_window_2025.parquet."""
import numpy as np, pandas as pd
D = "/home/claude/data"; FD = "/home/claude/fund"
XK = ["tr_plays", "tr_pts", "on_tro_pts", "on_tro_plays", "on_trd_pts", "on_trd_plays", "on_och", "on_dch"]
ev = pd.concat([pd.read_parquet(f"{FD}/xg_{s}.parquet").groupby("player")[XK].sum().reset_index().assign(season=s) for s in range(2012, 2026)])
f = pd.read_parquet(f"{FD}/features_all.parquet")[["season", "player", "poss_off", "poss_def"]]
x = f.merge(ev, on=["season", "player"], how="left").fillna(0)
# name -> (numerator, exposure)
DEF = {"o_trvol": ("tr_plays", "poss_off"), "o_tron": ("on_tro_pts", "poss_off"), "d_tron": ("on_trd_pts", "poss_def"),
       # robustness / decomposition: frequency (plays) on both ends
       "o_tronp": ("on_tro_plays", "poss_off"), "d_tronp": ("on_trd_plays", "poss_def")}
KS = [300, 1000, 3000]
for c, (num, ex) in DEF.items():
    lg = x.groupby("season")[num].transform("sum") / x.groupby("season")[ex].transform("sum")
    for k in KS: x[f"{c}_{k}"] = (x[num] - lg * x[ex]) / (x[ex] + k) * 100
cols = [f"{c}_{k}" for c in DEF for k in KS]
gm = pd.read_parquet(f"{D}/games_all.parquet"); out = []
def window(N):
    w = x[x.season.between(N - 2, N)].copy(); w["wt"] = (w.poss_off + w.poss_def) / 2 * w.season.map({N - 2: .3, N - 1: .6, N: 1.})
    return w[cols].mul(w.wt, axis=0).groupby(w.player).sum().div(w.groupby("player").wt.sum(), axis=0), w
for N in range(2014, 2025):
    v, _ = window(N)
    pg = pd.read_parquet(f"{D}/player_games_{N+1}.parquet")[["game", "player", "team", "min"]]
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]]
    t = v.reindex(pg.player).fillna(0).mul(pg["min"].values / 48, axis=0)
    t.index = pd.MultiIndex.from_arrays([pg.game, pg.team]); t = t.groupby(level=[0, 1]).sum()
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    dd = pd.DataFrame(t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy(), columns=["ev_" + c for c in cols]); dd["game"] = g.game.values; out.append(dd)
pd.concat(out).to_parquet(f"{FD}/tr_bt.parquet", index=False)
v, w = window(2025); v["poss"] = w.groupby("player").poss_off.sum()
nm = pd.read_parquet(f"{D}/features.parquet").drop_duplicates("player", keep="last").set_index("player").name
v["name"] = nm.reindex(v.index); v.to_parquet(f"{FD}/tr_window_2025.parquet")
big = v[v.poss > 6000]
for c in ["o_trvol_1000", "o_tron_1000", "d_tron_1000"]:
    s = big.sort_values(c)
    print(c, "sd", round(big[c].std(), 2), "| low:", ", ".join(f"{n} {a:+.1f}" for n, a in zip(s.name[:5], s[c][:5])), "| high:", ", ".join(f"{n} {a:+.1f}" for n, a in zip(s.name[-5:], s[c][-5:])))
# persistence: season-to-season correlation of the raw k=1000 rate (2000+ poss both years)
a = x[x.poss_off > 2000][["season", "player"] + [f"{c}_1000" for c in DEF]]; b = a.assign(season=a.season + 1)
m = a.merge(b, on=["season", "player"], suffixes=("", "_prev"))
print("year-to-year r:", {c: round(np.corrcoef(m[f"{c}_1000"], m[f"{c}_1000_prev"])[0, 1], 2) for c in DEF})
