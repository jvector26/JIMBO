import numpy as np, pandas as pd
ps = pd.read_parquet("data/player_seasons.parquet")[["season", "player", "min", "gp", "gs", "age"]]
ng = pd.read_parquet("data/games_all.parquet").groupby("season").size() * 2 / 30
ps["mpg"] = ps["min"] / ps.gp; ps["avail"] = (ps.gp / ps.season.map(ng)).clip(upper=1); ps["gsr"] = ps.gs / ps.gp
def lag(k):
    x = ps[["season", "player", "mpg", "avail", "gsr", "age", "gp"]].copy(); x["season"] += k
    return x.rename(columns={c: f"{c}{k}" for c in ["mpg", "avail", "gsr", "age", "gp"]})
d = ps.merge(lag(1), on=["season", "player"]).merge(lag(2), on=["season", "player"], how="left").merge(lag(3), on=["season", "player"], how="left")
d = d.dropna(subset=["age1"]); d["age_now"] = d.age1 + 1
for k in (2, 3):
    d[f"has{k}"] = d[f"mpg{k}"].notna().astype(float)
    d[f"mpg{k}"] = d[f"mpg{k}"].fillna(d.mpg1); d[f"gsr{k}"] = d[f"gsr{k}"].fillna(d.gsr1)
    # availability only counts as evidence in seasons he was a real rotation player (mpg>=20); otherwise missing
    d[f"reg{k}"] = (d[f"mpg{k}"] >= 20) & d[f"avail{k}"].notna()
d["reg1"] = d.mpg1 >= 20
def Xm(df):  # minutes per game played
    a = df.age_now.clip(19, 40)
    return np.c_[np.ones(len(df)), df.mpg1, df.mpg2, df.mpg3, df.gsr1 * df.mpg1, df.avail1 * df.mpg1, np.clip(a - 32, 0, None), np.clip(24 - a, 0, None)]
def Xa(df):  # share of team games played, only regulars' seasons count
    a = df.age_now.clip(19, 40)
    cols = [np.ones(len(df))]
    for k in (1, 2, 3):
        r = df[f"reg{k}"]
        cols += [np.where(r, df[f"avail{k}"], 0.0), r.astype(float)]
    cols += [df.mpg1, np.clip(a - 30, 0, None)]
    return np.column_stack(cols)
tr = d[d.season.between(2000, 2018)]; te = d[(d.season >= 2021)]
bm = np.linalg.lstsq(Xm(tr) * np.sqrt(tr.gp.values)[:, None], tr.mpg * np.sqrt(tr.gp.values), rcond=None)[0]
ba = np.linalg.lstsq(Xa(tr), tr.avail, rcond=None)[0]
pred = (Xm(te) @ bm).clip(0, 40) * (Xa(te) @ ba).clip(0.05, 1) * te.season.map(ng)
r2 = 1 - ((te["min"] - pred) ** 2).sum() / ((te["min"] - te["min"].mean()) ** 2).sum()
print(f"out-of-sample R^2 2021-26: new {r2:.3f}  (previous hand-set model 0.515, pooled regression 0.500)")
# the cases in question: regulars coming off a short season, and role jumps
inj = te.reg1 & (te.avail1 < 0.5)
print(f"regulars who played <50% of games: next season actual avail {te.avail[inj].mean():.2f}, new model {(Xa(te[inj]) @ ba).mean():.2f}, old hand-set model {(0.4*te.avail1[inj]+0.25*te.avail2[inj].fillna(te.avail1[inj])+0.15*te.avail3[inj].fillna(te.avail1[inj])+0.2*.62).mean():.2f}")
jump = (te.mpg1 - te.mpg2 >= 10) & (te.has2 == 1)
print(f"players whose mpg jumped 10+: next season actual mpg {te.mpg[jump].mean():.1f}, new model {(Xm(te[jump]) @ bm).mean():.1f}, old model {(0.75*te.mpg1[jump]+0.25*te.mpg2[jump]).mean():.1f}")
bm = np.linalg.lstsq(Xm(d[d.season>=2000]) * np.sqrt(d[d.season>=2000].gp.values)[:, None], d[d.season>=2000].mpg * np.sqrt(d[d.season>=2000].gp.values), rcond=None)[0]
ba = np.linalg.lstsq(Xa(d[d.season>=2000]), d[d.season>=2000].avail, rcond=None)[0]
np.save("data/min_bm.npy", bm); np.save("data/min_ba.npy", ba)
print("availability weights (last, prior, 3 yrs ago):", ba[[1, 3, 5]].round(2))
# blend check (refit on train only)
bm_t = np.linalg.lstsq(Xm(tr) * np.sqrt(tr.gp.values)[:, None], tr.mpg * np.sqrt(tr.gp.values), rcond=None)[0]
ba_t = np.linalg.lstsq(Xa(tr), tr.avail, rcond=None)[0]
old_m = 0.75 * te.mpg1 + 0.25 * te.mpg2 - np.clip(te.age_now - 33, 0, None)
old_a = 0.4 * te.avail1 + 0.25 * te.avail2.fillna(te.avail1) + 0.15 * te.avail3.fillna(te.avail1) + 0.2 * 0.62
bl_m = 0.5 * (Xm(te) @ bm_t).clip(0, 40) + 0.5 * old_m
bl_a = 0.5 * (Xa(te) @ ba_t).clip(0.05, 1) + 0.5 * old_a
pb = bl_m * bl_a * te.season.map(ng)
print(f"blend R^2 {1 - ((te['min'] - pb) ** 2).sum() / ((te['min'] - te['min'].mean()) ** 2).sum():.3f}; injured regulars avail {bl_a[inj].mean():.2f} (actual {te.avail[inj].mean():.2f}); mpg jumpers {bl_m[jump].mean():.1f} (actual {te.mpg[jump].mean():.1f})")
