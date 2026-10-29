"""Standard bar for a NEW-TEAM term. mover(S,p,T) = established (had a rating window before S) and < 100 possessions for T in
S-1 and S-2. yr2 = second season with T (joined during S-1, established then).
(1) Game LOSO (13,206 games 2015-25, actual minutes): margin ~ a + b*pred + k1*mv1 (+ k2*mv2), mv = (home - away) mover
    minutes / 48. Base = a + b*pred. Reported by season half.
(2) Preseason: pre_P ratings += d for projected movers (kind vet, opening team) -> write mvX_pre_P/T, score with age_eval.py."""
import numpy as np, pandas as pd, sys, subprocess, shutil
D = "/home/claude/data"
tp = []
for S in range(2011, 2026):
    st = pd.read_parquet(f"{D}/stints_{S}.parquet", columns=["teamA", "teamB", "lineupA", "lineupB", "fg2aA", "fg3aA", "orebA", "tovA", "ftaA", "fg2aB", "fg3aB", "orebB", "tovB", "ftaB"])
    st = st[(st.lineupA.str.count("-") == 4) & (st.lineupB.str.count("-") == 4)]
    p = sum(st[f"fg2a{m}"] + st[f"fg3a{m}"] - st[f"oreb{m}"] + st[f"tov{m}"] + .44*st[f"fta{m}"] for m in "AB")
    for m in "AB":
        ids = st[f"lineup{m}"].str.split("-", expand=True).astype(np.int64)
        for k in range(5): tp.append(pd.DataFrame({"season": S, "player": ids[k].to_numpy(), "team": st[f"team{m}"].to_numpy(), "poss": p.to_numpy()}))
PT = pd.concat(tp).groupby(["season", "player", "team"]).poss.sum().to_dict()
TOT = {}
for (s, p, t), v in PT.items(): TOT[(s, p)] = TOT.get((s, p), 0) + v
pt = lambda S, p, T: PT.get((S, p, T), 0.0)
rated = {S: set(pd.read_parquet(f"{D}/bigs_win_{S-1}.parquet").player) for S in range(2015, 2026)}
def st1(S, p, T, est=None): return (p in (rated[S] if est is None else est)) and pt(S-1, p, T) + pt(S-2, p, T) < 100
def st2(S, p, T): return pt(S-1, p, T) >= 100 and pt(S-2, p, T) + pt(S-3, p, T) < 100 and TOT.get((S-2, p), 0) >= 300
# ---------- (1) game LOSO
bt = pd.read_parquet(f"{D}/backtest_components.parquet").merge(pd.read_parquet(f"{D}/bt_split.parquet"), on=["game", "home_team", "away_team"])
F_ = [c for c in bt.columns if c.startswith("f:")]; bt["pred"] = bt[F_].sum(axis=1) + 0.6*bt.res_o + 0.8*bt.res_d + bt.unrated
g = pd.read_parquet(f"{D}/game_minutes.parquet"); g = g[g.season >= 2015]
g["m1"] = [st1(s, p, t) for s, p, t in zip(g.season, g.player, g.team)]
g["m2"] = [st2(s, p, t) for s, p, t in zip(g.season, g.player, g.team)]
agg = g.assign(v1=g["min"]*g.m1/48, v2=g["min"]*g.m2/48).groupby(["game", "team"])[["v1", "v2"]].sum()
for side in ("home", "away"):
    x = agg.reset_index().rename(columns={"team": f"{side}_team", "v1": f"v1_{side}", "v2": f"v2_{side}"})
    bt = bt.merge(x, on=["game", f"{side}_team"], how="left")
bt = bt.fillna({"v1_home": 0, "v1_away": 0, "v2_home": 0, "v2_away": 0})
bt["mv1"] = bt.v1_home - bt.v1_away; bt["mv2"] = bt.v2_home - bt.v2_away
print("mover full-games per team-game: yr1 %.2f  yr2 %.2f" % ((bt.v1_home.mean() + bt.v1_away.mean())/2, (bt.v2_home.mean() + bt.v2_away.mean())/2))
def loso(cols):
    X = np.c_[np.ones(len(bt)), bt[cols].to_numpy()]; y = bt.margin.to_numpy(); e = np.empty(len(bt)); S = bt.season.to_numpy()
    for s in np.unique(S):
        tr = S != s; e[~tr] = y[~tr] - X[~tr] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
    full = np.linalg.lstsq(X, y, rcond=None)[0]
    r = lambda m: float(np.sqrt(np.mean(e[m]**2)))
    return r(np.ones(len(e), bool)), r(S <= 2019), r(S >= 2020), full
b0 = loso(["pred"])
print("game LOSO RMSE  all / 2015-19 / 2020-25")
print("  base            %.4f / %.4f / %.4f" % b0[:3])
for cols in (["pred", "mv1"], ["pred", "mv1", "mv2"]):
    r = loso(cols); print("  + %-13s %.4f / %.4f / %.4f   (diff %+.4f / %+.4f / %+.4f)  coefs %s" % ("+".join(cols[1:]), *r[:3], r[0]-b0[0], r[1]-b0[1], r[2]-b0[2], np.round(r[3][2:], 3)))
for lab, m in (("2015-19", bt.season <= 2019), ("2020-25", bt.season >= 2020)):
    X = np.c_[np.ones(m.sum()), bt.loc[m, ["pred", "mv1", "mv2"]].to_numpy()]
    c = np.linalg.lstsq(X, bt.margin[m].to_numpy(), rcond=None)[0]; print("  half", lab, "k1 %.3f k2 %.3f" % (c[2], c[3]))
# season-clustered t for k1
import statsmodels.api as sm
r = sm.OLS(bt.margin, sm.add_constant(bt[["pred", "mv1", "mv2"]])).fit(cov_type="cluster", cov_kwds={"groups": bt.season})
print("  season-clustered: k1 %.3f (t %.1f)  k2 %.3f (t %.1f)" % (r.params.mv1, r.tvalues.mv1, r.params.mv2, r.tvalues.mv2))
bt[["game", "season", "mv1", "mv2"]].to_parquet(f"{D}/mover_games.parquet", index=False)
# ---------- (2) preseason team backtest
id2abbr = pd.read_parquet(f"{D}/games_2025.parquet").drop_duplicates("home_team").set_index("home_team").htm.to_dict()
abbr2id = {v: k for k, v in id2abbr.items()}
PP = {S: pd.read_parquet(f"{D}/pre_P_{S}.parquet") for S in range(2014, 2026)}
for S, P in PP.items():
    P["mover"] = [(k == "vet") and pt(S-1, p, abbr2id[t]) + pt(S-2, p, abbr2id[t]) < 100 for p, t, k in zip(P.player, P.team, P.kind)]
    P["yr2"] = [(k == "vet") and st2(S, p, abbr2id[t]) for p, t, k in zip(P.player, P.team, P.kind)]
sh = pd.concat(PP.values()); sh["w"] = sh.min_model
print("preseason: mover share of projected minutes %.2f, yr2 %.2f" % ((sh.w*sh.mover).sum()/sh.w.sum(), (sh.w*sh.yr2).sum()/sh.w.sum()))
for d1, d2 in ((-0.25, 0), (-0.5, 0), (-0.75, 0), (-0.5, -0.25), (-0.5, -0.5)):
    tag = f"mv{int(-d1*100)}_{int(-d2*100)}_"
    for S, P in PP.items():
        Q = P.copy(); Q["rating"] = Q.rating + d1*Q.mover + d2*Q.yr2
        Q.drop(columns=["mover", "yr2"]).to_parquet(f"{D}/{tag}pre_P_{S}.parquet", index=False)
        shutil.copy(f"{D}/pre_T_{S}.parquet", f"{D}/{tag}pre_T_{S}.parquet")
    print(subprocess.run(["python3", "/home/claude/code/age_eval.py", tag], capture_output=True, text=True).stdout.strip().splitlines()[-1], f"   (yr1 {d1}, yr2 {d2})", flush=True)
print(subprocess.run(["python3", "/home/claude/code/age_eval.py", ""], capture_output=True, text=True).stdout.strip().splitlines()[-1], "   (current)")
