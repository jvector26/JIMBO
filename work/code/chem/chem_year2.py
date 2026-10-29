"""Year-two check. Per player in a lineup (team T, season S), with PT = his possessions for T by season:
  yr1 = established (had a rating window before S) and PT(S-1)+PT(S-2) < 100   -> first season with T (as before)
  yr2 = PT(S-1) >= 100, PT(S-2)+PT(S-3) < 100, and he was established when he joined (window before S-1) -> second season
Ratings for season S use window S-3..S-1, where S-1 (year 1 with T, weight 1.0 of .3/.6/1.0) dominates. Predictions for yr2:
  familiarity (penalty fades once settled):   rating was dragged down by the year-1 penalty -> yr2 BEATS rating (> 0)
  adverse selection / real decline:           drop persists, only partly in the rating       -> yr2 still BELOW rating (< 0)
  old-team context stuck in the rating:       older seasons still in window                   -> yr2 below rating (< 0)"""
import numpy as np, pandas as pd, sys
sys.argv = ["x"]; sys.path.insert(0, "/home/claude/code/chem")
import chem_fit as C
D = "/home/claude/data"
R = pd.read_parquet(f"{D}/chem_stints_movers.parquet")
R[["age_o", "age_d"]] = R[["age_o", "age_d"]].fillna(R[["age_o", "age_d"]].mean()); R["cl"] = R.season*10000 + R.team_o % 10000
tp = []
for S in range(2011, 2026):
    st = pd.read_parquet(f"{D}/stints_{S}.parquet", columns=["teamA", "teamB", "lineupA", "lineupB", "fg2aA", "fg3aA", "orebA", "tovA", "ftaA", "fg2aB", "fg3aB", "orebB", "tovB", "ftaB"])
    st = st[(st.lineupA.str.count("-") == 4) & (st.lineupB.str.count("-") == 4)]
    p = sum(st[f"fg2a{m}"] + st[f"fg3a{m}"] - st[f"oreb{m}"] + st[f"tov{m}"] + .44*st[f"fta{m}"] for m in "AB")
    for m in "AB":
        ids = st[f"lineup{m}"].str.split("-", expand=True).astype(np.int64)
        for k in range(5): tp.append(pd.DataFrame({"season": S, "player": ids[k].to_numpy(), "team": st[f"team{m}"].to_numpy(), "poss": p.to_numpy()}))
PT = pd.concat(tp).groupby(["season", "player", "team"]).poss.sum().to_dict()
pt = lambda S, p, T: PT.get((S, p, T), 0.0)
rated = {S: set(pd.read_parquet(f"{D}/bigs_win_{S-1}.parquet").player) for S in range(2015, 2026)}
TOT = {}
for (S0, p0, T0), v in PT.items(): TOT[(S0, p0)] = TOT.get((S0, p0), 0) + v
def status(S, p, T):
    if p in rated[S] and pt(S-1, p, T) + pt(S-2, p, T) < 100: return 1
    if pt(S-1, p, T) >= 100 and pt(S-2, p, T) + pt(S-3, p, T) < 100 and TOT.get((S-2, p), 0) >= 300: return 2
    return 0
P = R[["p0", "p1", "p2", "p3", "p4"]].to_numpy(); Sx = R.season.to_numpy(); Tx = R.team_o.to_numpy()
cache = {}; y1 = np.zeros(len(R)); y2 = np.zeros(len(R))
for i in range(len(R)):
    for p in P[i]:
        k = (Sx[i], p, Tx[i])
        if k not in cache: cache[k] = status(*k)
        s = cache[k]; y1[i] += s == 1; y2[i] += s == 2
R["yr1_o"], R["yr2_o"] = y1, y2
key = ["game", "t_start", "period"]
R = R.merge(R[key + ["team_o", "yr1_o", "yr2_o"]].rename(columns={"team_o": "team_d", "yr1_o": "yr1_d", "yr2_o": "yr2_d"}), on=key + ["team_d"], how="left")
R[["yr1_d", "yr2_d"]] = R[["yr1_d", "yr2_d"]].fillna(0)
R.to_parquet(f"{D}/chem_stints_yr2.parquet", index=False)
print("per lineup: yr1 %.2f  yr2 %.2f" % (R.yr1_o.mean(), R.yr2_o.mean()))
C.BASE[:] = C.BASE + ["prog", "prog2", "prog3"]
A = ["yr1_o", "yr2_o", "yr1_d", "yr2_d"]
r = C.fit(R, A, "year 1 and year 2 on new team")
print("   net per player: yr1 %+.2f  yr2 %+.2f  (offense coef minus defense coef)" % (r.params.yr1_o - r.params.yr1_d, r.params.yr2_o - r.params.yr2_d))
r = C.fit(R, A, "  same, team-season FE", fe="two")
print("   net per player: yr1 %+.2f  yr2 %+.2f" % (r.params.yr1_o - r.params.yr1_d, r.params.yr2_o - r.params.yr2_d))
for lab, sub in (("2015-2019", R[R.season <= 2019]), ("2020-2025", R[R.season >= 2020])):
    r = C.fit(sub, A, "  " + lab); print("   net per player: yr1 %+.2f  yr2 %+.2f" % (r.params.yr1_o - r.params.yr1_d, r.params.yr2_o - r.params.yr2_d))
