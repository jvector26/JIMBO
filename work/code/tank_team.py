"""Team-level: do teams coming off a tank season beat the preseason model / blend? (preseason_backtest, 2014-25 excl 2019)"""
import numpy as np, pandas as pd, json
import statsmodels.api as sm
D = "/home/claude/data"
A = pd.read_parquet(f"{D}/preseason_backtest.parquet").merge(pd.read_parquet(f"{D}/direction_features.parquet")[["season", "team", "mw_age"]], on=["season", "team"])
A = A[A.season != 2019].copy()
A["y"] = 82*A.wpct-41; A["mod"] = 82*A.model_wpct_agefix-41; A["mkt"] = 82*A.mkt_wpct-41
A["age_c"] = A.mw_age - A.groupby("season").mw_age.transform("mean")
TT = pd.read_parquet(f"{D}/tank_team_seasons.parquet")
TT["bad"] = (TT.wpct <= 0.33).astype(float); TT["late"] = ((TT.m_late <= -9) & (TT["drop"] <= -4)).astype(float)
TT["tank"] = np.maximum(TT.bad, TT.late); TT["season"] += 1
A = A.merge(TT[["season", "team", "tank", "bad", "late", "drop", "m_late"]].rename(columns={"drop": "pdrop", "m_late": "pm_late"}), on=["season", "team"], how="left")
print("unmatched prior:", A[A.tank.isna()][["season", "team"]].values.tolist())
A[["tank", "bad", "late"]] = A[["tank", "bad", "late"]].fillna(0)
# prior-season underperformance vs its own preseason expectation (market), seasons 2015+
P = pd.read_parquet(f"{D}/preseason_backtest.parquet")[["season", "team", "wpct", "mkt_wpct"]]; P["season"] += 1
P["under"] = 82*(P.wpct - P.mkt_wpct)
A = A.merge(P[["season", "team", "under"]], on=["season", "team"], how="left")
# gap year: prior season tank AND a returning star (missed prior season, projected rating >= 2)
R = pd.read_parquet(f"{D}/injury_return.parquet"); R = R[R.missed.astype(bool) & (R.rating >= 2)]
ret = R.groupby(["season", "team"]).size().rename("n_ret").reset_index()
A = A.merge(ret, on=["season", "team"], how="left"); A["n_ret"] = A.n_ret.fillna(0)
A["gap"] = A.tank * (A.n_ret > 0)
A.to_parquet(f"{D}/tank_team.parquet")
rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))

def loso(cols, df):
    Z = np.c_[df["mod"] - df.mkt, df.age_c, df[cols].to_numpy()] if cols else np.c_[df["mod"] - df.mkt, df.age_c]
    yz = (df.y - df.mkt).to_numpy(); p = np.empty(len(df))
    for s in df.season.unique():
        tr = (df.season != s).to_numpy(); p[~tr] = df.mkt.to_numpy()[~tr] + Z[~tr] @ np.linalg.lstsq(Z[tr], yz[tr], rcond=None)[0]
    return rmse(df.y - p)

def reg(df, cols, y):
    X = sm.add_constant(df[cols]); m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df.season.to_numpy()})
    return m

print("n team-seasons:", len(A), " prior tank:", int(A.tank.sum()), " bad:", int(A.bad.sum()), " late:", int(A.late.sum()), " gap:", int(A.gap.sum()))
for f in ["tank", "bad", "late", "gap"]:
    d = A
    em = d.y - d["mod"]; ev = d.y - d.mkt
    m1 = reg(d, [f], em); m2 = reg(d, [f], ev)
    mb = reg(d.assign(dm=d["mod"] - d.mkt), ["dm", "age_c", f], d.y - d.mkt)
    h = {k: round(float((g.y - g["mod"])[g[f] > 0].mean() - (g.y - g["mod"])[g[f] == 0].mean()), 2) for k, g in d.groupby(d.season >= 2020)}
    print(f"\n[{f}] n={int(d[f].sum())}: actual-model {m1.params[f]:+.2f} wins (t {m1.tvalues[f]:+.1f}) halves(14-18,20-25) {h} | actual-Vegas {m2.params[f]:+.2f} (t {m2.tvalues[f]:+.1f})"
          f" | in blend {mb.params[f]:+.2f} (t {mb.tvalues[f]:+.1f}); LOSO {loso([], d):.3f} -> {loso([f], d):.3f}")
d = A.dropna(subset=["under"])
for f in ["under", "pdrop"]:
    m1 = reg(d, [f], d.y - d["mod"]); mb = reg(d.assign(dm=d["mod"] - d.mkt), ["dm", "age_c", f], d.y - d.mkt)
    print(f"[{f}] (2015+, n={len(d)}) actual-model coef {m1.params[f]:+.3f} (t {m1.tvalues[f]:+.1f}) | in blend {mb.params[f]:+.3f} (t {mb.tvalues[f]:+.1f}); LOSO {loso([], d):.3f} -> {loso([f], d):.3f}")
pd.set_option("display.width", 200)
print("\nPrior-tank team-seasons (actual, model, Vegas, in wins):")
t = A[A.tank > 0].assign(act=41 + A.y, model=41 + A["mod"], vegas=41 + A.mkt)[["season", "team", "late", "n_ret", "act", "model", "vegas"]]
t["act-mod"] = t.act - t.model; t["act-veg"] = t.act - t.vegas
print(t.round(1).sort_values("season").to_string(index=False))
print("mean act-mod %.2f, act-veg %.2f" % (t["act-mod"].mean(), t["act-veg"].mean()))
