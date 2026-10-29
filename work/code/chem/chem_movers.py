"""Stint-level mover test: established players (rated, i.e. had an NBA window) with no possessions for THIS team in the previous
2 seasons. Does their lineup under-perform, and does the gap close with time on the new team (days since his first game
for the team, as a share of the season; calendar time, not his minutes, to avoid coaches' playing-time choices)?"""
import numpy as np, pandas as pd, statsmodels.api as sm, sys
sys.argv = ["x"]; sys.path.insert(0, "/home/claude/code/chem")
import chem_fit as C
D = "/home/claude/data"; R = C.R
# player-team-season possessions from stints (both sides)
tp = []
for S in range(2012, 2026):
    st = pd.read_parquet(f"{D}/stints_{S}.parquet", columns=["game", "teamA", "teamB", "lineupA", "lineupB", "fg2aA", "fg3aA", "orebA", "tovA", "ftaA", "fg2aB", "fg3aB", "orebB", "tovB", "ftaB"])
    st = st[(st.lineupA.str.count("-") == 4) & (st.lineupB.str.count("-") == 4)]
    p = sum(st[f"fg2a{m}"] + st[f"fg3a{m}"] - st[f"oreb{m}"] + st[f"tov{m}"] + .44*st[f"fta{m}"] for m in "AB")
    for m in "AB":
        ids = st[f"lineup{m}"].str.split("-", expand=True).astype(np.int64)
        for k in range(5): tp.append(pd.DataFrame({"season": S, "player": ids[k].to_numpy(), "team": st[f"team{m}"].to_numpy(), "poss": p.to_numpy(), "game": st.game.to_numpy()}))
TP = pd.concat(tp); PT = TP.groupby(["season", "player", "team"]).poss.sum()
ga = pd.read_parquet(f"{D}/games_all.parquet")[["game", "date"]].set_index("game").date
TP["date"] = TP.game.map(ga); FIRST = TP.groupby(["season", "player", "team"]).date.min()
R["d"] = pd.to_datetime(R.date.astype(str))
s0 = R.groupby("season").d.transform("min"); slen = (R.groupby("season").d.transform("max") - s0).dt.days
rated = {S: set(pd.read_parquet(f"{D}/bigs_win_{S-1}.parquet").player) for S in range(2015, 2026)}
prevset = {}
for S in range(2015, 2026):
    a = PT.loc[[S-1, S-2]].reset_index(); prevset[S] = set(zip(a.player, a.team))
mov = np.zeros(len(R)); jt = np.zeros(len(R))
P = R[["p0", "p1", "p2", "p3", "p4"]].to_numpy(); Sx = R.season.to_numpy(); Tx = R.team_o.to_numpy(); dx = R.d.to_numpy()
fd = {k: np.datetime64(pd.Timestamp(str(v))) for k, v in FIRST.items()}
sl = slen.to_numpy()
for i in range(len(R)):
    S, T = Sx[i], Tx[i]; n = 0; t = 0.0
    for p in P[i]:
        if p in rated[S] and (p, T) not in prevset[S]:
            n += 1; t += (dx[i] - fd[(S, p, T)]).astype("timedelta64[D]").astype(int) / max(sl[i], 1)
    mov[i] = n; jt[i] = t / n if n else 0.0
R["mov_o"] = mov; R["movt_o"] = mov * jt          # movers x time on new team (share of season)
key = ["game", "t_start", "period"]
opp = R[key + ["team_o", "mov_o", "movt_o"]].rename(columns={"team_o": "team_d", "mov_o": "mov_d", "movt_o": "movt_d"})
R = R.merge(opp, on=key + ["team_d"], how="left"); R[["mov_d", "movt_d"]] = R[["mov_d", "movt_d"]].fillna(0)
print("rows", len(R), "; mean movers per lineup %.2f; share of mover-slots by time on team: <.25 %.2f" % (R.mov_o.mean(), ((R.movt_o/R.mov_o.replace(0, np.nan)) < .25).mean()))
d = R.d; s0 = d.groupby(R.season).transform("min"); s1 = d.groupby(R.season).transform("max")
R["prog"] = ((d - s0).dt.days/(s1 - s0).dt.days).clip(0, 1); R["prog2"] = R.prog**2; R["prog3"] = R.prog**3
C.BASE[:] = C.BASE + ["prog", "prog2", "prog3"]
C.fit(R, ["mov_o", "mov_d"], "movers (per player in lineup)")
C.fit(R, ["mov_o", "movt_o", "mov_d", "movt_d"], "movers + time on new team")
C.fit(R, ["mov_o", "movt_o", "mov_d", "movt_d"], "  same, team-season FE", fe="two")
C.fit(R, ["mov_o", "mov_d", "tenP_o", "tenP_d", "famP_o", "famP_d"], "movers + prior tenure + pairs")
for lab, sub in (("2015-2019", R[R.season <= 2019]), ("2020-2025", R[R.season >= 2020])):
    C.fit(sub, ["mov_o", "movt_o", "mov_d", "movt_d"], "movers + time " + lab)
R.drop(columns=["d"]).to_parquet(f"{D}/chem_stints_movers.parquet", index=False)
