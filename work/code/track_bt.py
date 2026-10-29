"""Defense #7: NBA.com tracking/hustle stats (gabriel1200/site_Data mirror) as added terms. Features per player-season
(rates per 100 def poss, season-centred, shrunk poss/(poss+k)); windows N-2..N decay .3/.6/1 over the seasons that
have data; game diffs in N+1 with actual minutes -> data/track_bt.parquet."""
import numpy as np, pandas as pd
D = "/home/claude/data"
f = pd.read_parquet(f"{D}/features.parquet")[["season", "player", "poss_def"]]
h = pd.read_csv(f"{D}/tracking/hustle.csv"); h["season"] = h.year - 1
h = h.groupby(["season", "PLAYER_ID"]).sum(numeric_only=True).reset_index().rename(columns={"PLAYER_ID": "player"})
H = h[["season", "player"]].assign(defl=h.DEFLECTIONS, cont=h.CONTESTED_SHOTS, cont2=h.CONTESTED_SHOTS_2PT,
                                   cont3=h.CONTESTED_SHOTS_3PT, lbr=h.DEF_LOOSE_BALLS_RECOVERED, dbox=h.DEF_BOXOUTS)
import re, unicodedata
def norm(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n); return re.sub(r"[^a-z]", "", n)
ps = pd.read_parquet(f"{D}/player_seasons.parquet")[["season", "player", "name"]]; ps["key"] = ps.name.map(norm)
R = []; unmatched = 0
for s in range(2013, 2025):
    ids = ps[ps.season == s].drop_duplicates("key").set_index("key").player
    for kind in ("rimdfg", "dfg"):
        t = pd.read_csv(f"{D}/tracking/{kind}_{s}.csv")
        t = t.rename(columns={"Diff%": "DIFF%"})
        pid = t.PLAYER.map(norm).map(ids)
        if "PLAYER_ID" in t: pid = t.PLAYER_ID.where(t.PLAYER_ID.notna(), pid)
        unmatched += pid.isna().sum(); t = t[pid.notna()].assign(pid=pid[pid.notna()].astype(int))
        t["sv"] = -(t["DIFF%"] / 100) * t.DFGA * 2  # points saved vs shooters' normal FG% (2 pts/shot approx)
        t = t.groupby("pid")[["sv", "DFGA"]].sum().reset_index()  # traded players: sum team rows
        R.append(pd.DataFrame({"season": s, "player": t.pid, "kind": kind, "v": t.sv, "a": t.DFGA}))
print("unmatched DFG rows", unmatched)
R = pd.concat(R).pivot_table(index=["season", "player"], columns="kind", values=["v", "a"]).fillna(0)
R.columns = [f"{b}_{a}" for a, b in R.columns]; R = R.reset_index()
R = R.rename(columns={"rimdfg_v": "rimsave", "dfg_v": "allsave", "rimdfg_a": "rimdfga", "dfg_a": "dfga"})
EV = {"defl": H, "cont": H, "cont2": H, "cont3": H, "lbr": H, "dbox": H, "rimsave": R, "allsave": R, "rimdfga": R}
x = f[f.season >= 2013].merge(H, on=["season", "player"], how="left").merge(R, on=["season", "player"], how="left")
hs = x.season.between(2017, 2025); rs = x.season.between(2013, 2024)
KS = [1000, 3000]
cols = []
for c, src in EV.items():
    have = hs if src is H else rs
    x[c] = x[c].fillna(0)
    lg = x[have].groupby("season")[c].sum() / x[have].groupby("season").poss_def.sum()
    for k in KS:
        v = (x[c] - x.season.map(lg) * x.poss_def) / (x.poss_def + k) * 100
        x[f"{c}_{k}"] = v.where(have); cols.append(f"{c}_{k}")
gm = pd.read_parquet(f"{D}/games_all.parquet"); out = []
for N in range(2014, 2025):
    w = x[x.season.between(N - 2, N)].copy(); w["wt0"] = w.poss_def * w.season.map({N - 2: .3, N - 1: .6, N: 1.})
    V = {}
    for c in cols:
        ok = w[c].notna(); ww = w.wt0.where(ok, 0)
        V[c] = (w[c].fillna(0) * ww).groupby(w.player).sum() / ww.groupby(w.player).sum().replace(0, np.nan)
    V = pd.DataFrame(V).fillna(0)
    pg = pd.read_parquet(f"{D}/player_games_{N+1}.parquet")[["game", "player", "team", "min"]]
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]]
    t = V.reindex(pg.player).fillna(0).mul(pg["min"].values / 48, axis=0)
    t.index = pd.MultiIndex.from_arrays([pg.game, pg.team]); t = t.groupby(level=[0, 1]).sum()
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    dd = pd.DataFrame(t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy(), columns=["tr_" + c for c in cols])
    dd["game"] = g.game.values; dd["N"] = N; out.append(dd)
pd.concat(out).to_parquet(f"{D}/track_bt.parquet", index=False)
N = 2025; w = x[x.season.between(N - 2, N)].copy(); w["wt0"] = w.poss_def * w.season.map({N - 2: .3, N - 1: .6, N: 1.})
V = {c: (w[c].fillna(0) * w.wt0.where(w[c].notna(), 0)).groupby(w.player).sum() / w.wt0.where(w[c].notna(), 0).groupby(w.player).sum().replace(0, np.nan) for c in cols}
pd.DataFrame(V).to_parquet(f"{D}/track_window_2025.parquet"); print("ok")
