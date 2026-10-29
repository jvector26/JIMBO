import json, numpy as np, pandas as pd, impact as I
W = json.load(open(f"{I.D}/weights_prod.json")); Wo, Wd = pd.Series(W["Wo"]) * .5, pd.Series(W["Wd"]) * .5
rows = I.load_rows([2023, 2024, 2025]); out = {}
for m in [1.0, 0.5]:
    wd = Wd.copy(); wd["d_dreb"] *= m
    r = I.fit_window(2025, Wo, wd, lam=2000, rows=rows)
    r["dv"] = r.def_rebounding + r.rim_protection + r.perimeter_def + 0.8 * r.res_d
    r["ov"] = r.three_pt*0 + r["off"] - r.res_o + 0.6 * r.res_o
    out[m] = r.set_index("player")
ps = pd.read_parquet(f"{I.D}/player_seasons.parquet"); last = ps[ps.season == 2025].groupby("player").agg(name=("name","last"), mn=("min","sum"))
hc = [c for c in ps.columns if "height" in c.lower() or c.lower() in ("pos","position")]; print("bio cols", hc)
M = last[last.mn > 800].copy()
for m in out:
    d = out[m].reindex(M.index); M[f"def_{m}"] = d.dv - (d.dv * M.mn).sum() / M.mn.sum(); M[f"tot_{m}"] = M[f"def_{m}"] + d.ov
    M[f"rk_{m}"] = M[f"def_{m}"].rank(ascending=False).astype(int)
h = ps[ps.season == 2025].groupby("player")[hc[0]].last() if hc else None
if h is not None:
    M["ht"] = h.reindex(M.index)
    M["grp"] = np.where(M.ht >= 81, "big 6'9+", np.where(M.ht >= 78, "wing 6'6-6'8", "guard <=6'5"))
    print(M.groupby("grp")[["def_1.0", "def_0.5"]].apply(lambda g: g.mean()).round(2))
print("top 15 before -> after:")
print(pd.DataFrame({"before": M.sort_values("rk_1.0").name.head(15).values, "after": M.sort_values("rk_0.5").name.head(15).values}).to_string())
who = ["Amen Thompson","Ausar Thompson","Luguentz Dort","Dyson Daniels","Herbert Jones","Toumani Camara","Alex Caruso","OG Anunoby","Derrick White","Jalen Suggs",
       "Domantas Sabonis","Nikola Jokic","Ivica Zubac","Andre Drummond","Jonas Valanciunas","Victor Wembanyama","Rudy Gobert","Jarrett Allen","Karl-Anthony Towns"]
print(M[M.name.isin(who)][["name","def_1.0","def_0.5","rk_1.0","rk_0.5"]].round(2).sort_values("rk_0.5").to_string(index=False))
print("players:", len(M))
