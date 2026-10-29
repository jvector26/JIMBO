"""Per-season lineup effects for the player view: for each season N, (a) single-season fit (rows of N only) and
(b) the 3-season window through N exactly as production (decay .3/.6/1, lam 2000, W = 0.5 x prod). Each re-centered
so the minutes-weighted season-N player is 0 (off and def separately, as production.py). -> data/lineup_hist.parquet"""
import sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code"); import impact as I
W = json.load(open(f"{I.D}/weights_prod.json")); Wo = pd.Series(W["Wo"]) * 0.5; Wd = pd.Series(W["Wd"]) * 0.5
f = I.feats()
def centered(r, N):
    w = r.player.map(f[f.season == N].set_index("player")["min"]).fillna(0)
    oc = [c for c, v in I.COMPONENTS.items() if v[0].startswith("o_")]; dc = [c for c in I.COMPONENTS if c not in oc]
    # production re-centers total off (box + res) by moving res; replicate
    so = ((r[oc].sum(1) + r.res_o) * w).sum() / w.sum(); sd = ((r[dc].sum(1) + r.res_d) * w).sum() / w.sum()
    return r.res_o - so, r.res_d - sd
out = []
seasons = list(map(int, sys.argv[1:])) or list(range(1996, 2026))
for N in seasons:
    t = time.time()
    rows_all = I.load_rows([s for s in (N - 2, N - 1, N) if s >= 1996])
    rw = I.fit_window(N, Wo, Wd, lam=2000, rows=rows_all)
    rs = I.fit_window(N, Wo, Wd, lam=2000, rows=rows_all[rows_all.season == N].copy())
    wo, wd = centered(rw, N); so, sd = centered(rs, N)
    a = pd.DataFrame({"player": rw.player, "win_o": wo, "win_d": wd}); b = pd.DataFrame({"player": rs.player, "one_o": so, "one_d": sd})
    x = a.merge(b, on="player", how="left"); x["season"] = N
    x = x[x.player.isin(f[(f.season == N) & (f["min"] > 0)].player)]
    out.append(x); print(N, len(x), round(time.time() - t), "s", flush=True)
    pd.concat(out).to_parquet(f"{I.D}/lineup_hist_{seasons[0]}_{seasons[-1]}.parquet", index=False)
