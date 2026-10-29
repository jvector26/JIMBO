"""Test 1 follow-up: what drives the continuity effect? Split N+1 minutes into stayers / movers / rookies / other newcomers,
and the N+1 residual into first and second half of the team's games (familiarity -> gap closes; sell-offs -> gap widens)."""
import numpy as np, pandas as pd, statsmodels.api as sm
D = "/home/claude/data"; MINSTAY = 500
bt = pd.read_parquet(f"{D}/backtest_components.parquet").merge(pd.read_parquet(f"{D}/bt_split.parquet"), on=["game", "home_team", "away_team"])
F_ = [c for c in bt.columns if c.startswith("f:")]
p = bt[F_].sum(axis=1) + 0.6*bt.res_o + 0.8*bt.res_d + bt.unrated
b, a = np.polyfit(p, bt.margin, 1); bt["e"] = bt.margin - (a + b*p)
ga = pd.read_parquet(f"{D}/games_all.parquet")[["game", "date"]]; bt = bt.merge(ga, on="game")
tr = pd.concat([bt[["season", "game", "date", "home_team", "e", "unrated"]].rename(columns={"home_team": "team"}),
                bt[["season", "game", "date", "away_team", "e", "unrated"]].rename(columns={"away_team": "team"}).assign(e=lambda d: -d.e, unrated=lambda d: -d.unrated)])
tr = tr.sort_values(["season", "team", "date", "game"]); tr["gno"] = tr.groupby(["season", "team"]).cumcount()
tr["half"] = np.where(tr.gno < tr.groupby(["season", "team"]).gno.transform("max")/2, 1, 2)
R = tr.groupby(["season", "team"]).e.mean(); RH = tr.groupby(["season", "team", "half"]).e.mean().unstack()
g = pd.read_parquet(f"{D}/game_minutes.parquet").merge(tr[["game", "team", "half"]], on=["game", "team"], how="left")
ps = pd.read_parquet(f"{D}/player_seasons.parquet"); first = ps.groupby("player").season.min()
age = ps[["season", "player", "age"]].drop_duplicates(["season", "player"])
pm = g.groupby(["season", "team", "player"])["min"].sum().reset_index().merge(age, on=["season", "player"], how="left")
pm1 = g[g.half == 1].groupby(["season", "team", "player"])["min"].sum()   # first-half minutes (roster before the deadline)
rows = []
for N in range(2015, 2025):
    prev = pm[pm.season == N]; prev_tm = prev.set_index(["player", "team"])["min"]; totN = prev.groupby("player")["min"].sum()
    for T, d in pm[pm.season == N+1].groupby("team"):
        if (N, T) not in R.index or (N+1, T) not in R.index: continue
        def shares(mins):
            m = mins / mins.sum(); pl = d.player.to_numpy()
            minT = np.array([prev_tm.get((x, T), 0.0) for x in pl]); tot = totN.reindex(pl).fillna(0).to_numpy()
            stay = minT >= MINSTAY; mover = ~stay & (tot >= MINSTAY); rook = (first.reindex(pl).to_numpy() == N+1)
            other = ~stay & ~mover & ~rook
            return dict(cont=m[stay].sum(), mov=m[mover].sum(), rook=m[rook].sum(), oth=m[other].sum())
        s_all = shares(d["min"].to_numpy())
        m1 = np.array([pm1.get((N+1, T, x), 0.0) for x in d.player]); s1 = shares(m1) if m1.sum() > 0 else s_all
        rows.append(dict(season=N+1, team=T, y=R[(N+1, T)], y1=RH.loc[(N+1, T), 1], y2=RH.loc[(N+1, T), 2], F=R[(N, T)],
                         **s_all, **{k+"1": v for k, v in s1.items()},
                         age=float(np.average(d.age.fillna(d.age.mean()), weights=d["min"]))))
X = pd.DataFrame(rows); X["age_c"] = X.age - X.groupby("season").age.transform("mean")
X.to_parquet(f"{D}/chem_team2.parquet", index=False)
def fit(df, yv, cols, lab):
    Z = pd.get_dummies(df.season, prefix="s", drop_first=True).astype(float)
    r = sm.OLS(df[yv], sm.add_constant(pd.concat([df[cols], Z], axis=1))).fit(cov_type="cluster", cov_kwds={"groups": df.season})
    print(f"{lab:44s}", "  ".join(f"{c} {r.params[c]:+.2f} (t {r.tvalues[c]:+.1f})" for c in cols))
print("mean shares: stay %.2f mover %.2f rookie %.2f other %.2f" % tuple(X[["cont", "mov", "rook", "oth"]].mean()))
print("-- vs stayers as the base group (coef = pts/game if ALL minutes moved from stayers to that group)")
fit(X, "y", ["mov", "rook", "oth", "F", "age_c"], "full season")
fit(X, "y1", ["mov1", "rook1", "oth1", "F", "age_c"], "first half (first-half minutes)")
fit(X, "y2", ["mov", "rook", "oth", "F", "age_c"], "second half (full-season minutes)")
X["mvr"] = X.mov + X.rook + X.oth; X["mvr1"] = X.mov1 + X.rook1 + X.oth1
fit(X, "y1", ["mvr1", "F", "age_c"], "first half: all non-stayers")
fit(X, "y2", ["mvr1", "F", "age_c"], "second half, first-half roster")
for lab, sub in (("2016-2020", X[X.season <= 2020]), ("2021-2025", X[X.season >= 2021])):
    fit(sub, "y1", ["mov1", "rook1", "oth1", "F", "age_c"], "first half " + lab)
