"""CHEMISTRY TEST 1: does the team's 'more than the sum of its parts' leftover follow the core or the individuals?
r_{T,N} = team-season mean game residual (actual margin - additive prediction with ACTUAL minutes, ratings through N-1;
backtest_components + bt_split, lineup .6/.8, calibrated a+b*pred). For season N+1 (team T):
  cont = share of T's N+1 minutes by 'stayers' (>= MINSTAY min for T in N)
  S    = cont * r_{T,N}                          (the old team leftover, carried by players who stayed)
  V    = sum over movers m_i * rbar_i,N         (movers' old teams' leftovers, rbar = minutes-weighted over N teams)
  F    = r_{T,N}                                 (franchise: coach/system/arena, regardless of who stayed)
Model r_{T,N+1} ~ F + S + V + cont + roster age (+ season FE). Readings:
  chemistry (team-specific, needs these players together): S > 0, V ~ 0 or < 0
  individual value the ratings missed (portable):          S ~ V > 0
  franchise / coach / system:                              F > 0, S ~ 0
"""
import numpy as np, pandas as pd, statsmodels.api as sm
D = "/home/claude/data"
MINSTAY = 500
bt = pd.read_parquet(f"{D}/backtest_components.parquet").merge(pd.read_parquet(f"{D}/bt_split.parquet"), on=["game", "home_team", "away_team"])
F_ = [c for c in bt.columns if c.startswith("f:")]
p = bt[F_].sum(axis=1) + 0.6*bt.res_o + 0.8*bt.res_d + bt.unrated
b, a = np.polyfit(p, bt.margin, 1); bt["e"] = bt.margin - (a + b*p)
tr = pd.concat([bt[["season", "home_team", "e"]].rename(columns={"home_team": "team"}),
                bt[["season", "away_team", "e"]].rename(columns={"away_team": "team"}).assign(e=lambda d: -d.e)])
R = tr.groupby(["season", "team"]).e.agg(["mean", "count"]).rename(columns={"mean": "r", "count": "ng"}).reset_index()
g = pd.read_parquet(f"{D}/game_minutes.parquet")
pm = g.groupby(["season", "team", "player"])["min"].sum().reset_index()
age = pd.read_parquet(f"{D}/player_seasons.parquet")[["season", "player", "age"]].drop_duplicates(["season", "player"])
pm = pm.merge(age, on=["season", "player"], how="left")
rows = []
Rd = R.set_index(["season", "team"]).r
for N in range(2015, 2025):
    cur = pm[pm.season == N+1]; prev = pm[pm.season == N]
    prev_tm = prev.set_index(["player", "team"])["min"]
    # movers' carried leftover: minutes-weighted over their N teams
    pv = prev.assign(r=[Rd.get((N, t), np.nan) for t in prev.team]).dropna(subset=["r"])
    carried = pv.groupby("player").apply(lambda d: np.average(d.r, weights=d["min"]), include_groups=False)
    totN = prev.groupby("player")["min"].sum()
    for T, d in cur.groupby("team"):
        if (N, T) not in Rd.index or (N+1, T) not in Rd.index: continue
        m = d["min"] / d["min"].sum()
        minT = np.array([prev_tm.get((pl, T), 0.0) for pl in d.player])
        stay = minT >= MINSTAY
        tot = totN.reindex(d.player).fillna(0).to_numpy()
        mover = (~stay) & (tot >= MINSTAY)
        V = float((m[mover] * carried.reindex(d.player[mover]).fillna(0).to_numpy()).sum())
        cont = float(m[stay].sum()); rT = Rd[(N, T)]
        rows.append(dict(season=N+1, team=T, y=Rd[(N+1, T)], F=rT, S=cont*rT, V=V, cont=cont, mov=float(m[mover].sum()),
                         age=float(np.average(d.age.fillna(d.age.mean()), weights=d["min"]))))
X = pd.DataFrame(rows)
X["age_c"] = X.age - X.groupby("season").age.transform("mean")
X.to_parquet(f"{D}/chem_team.parquet", index=False)
print(len(X), "team-seasons; residual sd", round(R.r.std(), 2), "; cont mean", round(X.cont.mean(), 2), "sd", round(X.cont.std(), 2), "; movers share", round(X.mov.mean(), 2))

def fit(df, cols, lab):
    Z = pd.get_dummies(df.season, prefix="s", drop_first=True).astype(float)
    Xm = sm.add_constant(pd.concat([df[cols], Z], axis=1))
    r = sm.OLS(df.y, Xm).fit(cov_type="cluster", cov_kwds={"groups": df.season})
    print(f"{lab:38s}", "  ".join(f"{c} {r.params[c]:+.2f} (t {r.tvalues[c]:+.1f})" for c in cols), f"  R2 {r.rsquared:.3f}")
    return r
for lab, sub in (("ALL 2016-2025", X), ("half 2016-2020", X[X.season <= 2020]), ("half 2021-2025", X[X.season >= 2021]),
                 ("excl 2019-20/2020-21", X[~X.season.isin([2019, 2020])])):
    print("==", lab, len(sub))
    fit(sub, ["F"], "persistence only")
    fit(sub, ["F", "S", "V", "cont", "age_c"], "franchise + stayers + movers")
    fit(sub, ["S", "V", "cont", "age_c"], "stayers + movers")
