"""CHEMISTRY TEST 2 data: stint sides 2015-16..2025-26 with familiarity measures, all computed from history BEFORE the game.
Per side (5 players of one team): famP = mean over the 10 pairs of log1p(shared possessions in the previous 2 seasons / 1000)
(any team: reunions count); famC = same for the current season before this game; tenP/tenC = mean over the 5 players of
log1p(own possessions with THIS team, previous 2 seasons / current season before this game, / 1000) = system/coach knowledge.
Ratings: window through N-1 (bigs_win_{N-1}: box offense/defense sums, res_o, res_d); unk = players without a window.
Output data/chem_stints.parquet: one row per stint side (offense y = luck-adjusted pts/100)."""
import numpy as np, pandas as pd, json
D = "/home/claude/data"
OFFB = ["f:o_3vol", "f:o_3eff", "f:o_rimvol", "f:o_rimeff", "f:o_oreb", "f:o_ftvol", "f:o_ast", "f:o_unast", "f:o_tov", "f:o_mideff", "f:o_midvol", "f:o_fteff"]
DEFB = ["f:d_dreb", "f:d_blk", "f:d_opprim", "f:d_stl", "f:d_pf", "f:d_oppxq"]
ga = pd.read_parquet(f"{D}/games_all.parquet")[["game", "date", "home_team"]].set_index("game")
K = 2_000_000
season_pairs, season_ten = {}, {}
out = []
for S in range(2011, 2026):
    st = pd.read_parquet(f"{D}/stints_{S}.parquet")
    st = st[(st.lineupA.str.count("-") == 4) & (st.lineupB.str.count("-") == 4)].copy()
    st["date"] = st.game.map(ga.date); st = st.sort_values(["date", "game", "t_start"]).reset_index(drop=True)

    for me in "AB":
        st[f"poss{me}"] = st[f"fg2a{me}"] + st[f"fg3a{me}"] - st[f"oreb{me}"] + st[f"tov{me}"] + 0.44*st[f"fta{me}"]
        st[f"adj{me}"] = 2*st[f"fg2m{me}"] + 3*st[f"x3{me}"] + st[f"xft{me}"]
    LA = np.array([list(map(int, s.split("-"))) for s in st.lineupA]); LB = np.array([list(map(int, s.split("-"))) for s in st.lineupB])
    tA = st.teamA.to_numpy(); tB = st.teamB.to_numpy(); gm = st.game.to_numpy(); tp = (st.possA + st.possB).to_numpy()
    prevP = {}; prevT = {}
    for s2 in (S-1, S-2):
        for k, v in season_pairs.get(s2, {}).items(): prevP[k] = prevP.get(k, 0) + v
        for k, v in season_ten.get(s2, {}).items(): prevT[k] = prevT.get(k, 0) + v
    curP, curT = {}, {}; bufP, bufT = {}, {}
    n = len(st); feats = np.zeros((n, 2, 4))   # side A/B x (famP, famC, tenP, tenC)
    last = None
    for i in range(n):
        if gm[i] != last:   # apply the finished game's increments
            for k, v in bufP.items(): curP[k] = curP.get(k, 0) + v
            for k, v in bufT.items(): curT[k] = curT.get(k, 0) + v
            bufP, bufT = {}, {}; last = gm[i]
        for j, (L, T) in enumerate(((LA[i], tA[i]), (LB[i], tB[i]))):
            L = sorted(L); fp = fc = 0.0
            for a in range(5):
                for b in range(a+1, 5):
                    k = L[a]*K + L[b]; fp += np.log1p(prevP.get(k, 0)/1000); fc += np.log1p(curP.get(k, 0)/1000)
                    bufP[k] = bufP.get(k, 0) + tp[i]
            tpv = tcv = 0.0
            for a in range(5):
                k = L[a]*K*10 + (T % 1000); tpv += np.log1p(prevT.get(k, 0)/1000); tcv += np.log1p(curT.get(k, 0)/1000)
                bufT[k] = bufT.get(k, 0) + tp[i]
            feats[i, j] = (fp/10, fc/10, tpv/5, tcv/5)
    for k, v in bufP.items(): curP[k] = curP.get(k, 0) + v
    for k, v in bufT.items(): curT[k] = curT.get(k, 0) + v
    season_pairs[S], season_ten[S] = curP, curT
    season_pairs.pop(S-3, None); season_ten.pop(S-3, None)
    if S < 2015: print(S, "history", flush=True); continue
    w = pd.read_parquet(f"{D}/bigs_win_{S-1}.parquet").set_index("player")
    w["bo"] = w[OFFB].sum(axis=1); w["bd"] = w[DEFB].sum(axis=1)
    lk = {c: w[c].to_dict() for c in ("bo", "bd", "res_o", "res_d", "age_N", "poss_window")}
    def sums(L, c): return np.array([sum(lk[c].get(p, 0.0) for p in row) for row in L])
    def unk(L): return np.array([sum(p not in lk["bo"] for p in row) for row in L])
    def mage(L): return np.array([np.mean([lk["age_N"].get(p, 22.0) for p in row]) for row in L])
    def lposs(L): return np.array([np.mean([np.log1p(lk["poss_window"].get(p, 0)/1000) for p in row]) for row in L])
    SA = {c: sums(LA, c) for c in ("bo", "bd", "res_o", "res_d")}; SB = {c: sums(LB, c) for c in ("bo", "bd", "res_o", "res_d")}
    uA, uB, aA, aB, pA, pB = unk(LA), unk(LB), mage(LA), mage(LB), lposs(LA), lposs(LB)
    m = st.margin_start.abs(); garbage = (((st.t_start >= 2160) & (m >= 20)) | ((st.t_start >= 2520) & (m >= 15))).to_numpy()
    home = st.game.map(ga.home_team).to_numpy()
    gno = st.groupby("teamA").game.transform(lambda g: pd.factorize(g)[0]).to_numpy()   # rough game index (for season-progress)
    for me, op, j, jo, S1, S2, u1, u2, a1, a2, p1, p2, T1, T2 in (("A", "B", 0, 1, SA, SB, uA, uB, aA, aB, pA, pB, tA, tB), ("B", "A", 1, 0, SB, SA, uB, uA, aB, aA, pB, pA, tB, tA)):
        poss = st[f"poss{me}"].to_numpy(); keep = (poss > 0.5) & ~garbage
        x = pd.DataFrame({"season": S, "game": gm, "date": st.date.to_numpy(), "t_start": st.t_start.to_numpy(), "poss": poss,
                          "y": 100*st[f"adj{me}"].to_numpy()/np.maximum(poss, 1e-9), "home_off": (T1 == home).astype(np.int8),
                          "team_o": T1, "team_d": T2,
                          "bo_o": S1["bo"], "ro_o": S1["res_o"], "bd_d": S2["bd"], "rd_d": S2["res_d"], "unk_o": u1, "unk_d": u2,
                          "age_o": a1, "age_d": a2, "lp_o": p1, "lp_d": p2,
                          "famP_o": feats[:, j, 0], "famC_o": feats[:, j, 1], "tenP_o": feats[:, j, 2], "tenC_o": feats[:, j, 3],
                          "famP_d": feats[:, jo, 0], "famC_d": feats[:, jo, 1], "tenP_d": feats[:, jo, 2], "tenC_d": feats[:, jo, 3],
                          "ids_o": list(map(tuple, (LA if me == "A" else LB))), "period": st.period.to_numpy()})
        out.append(x[keep])
    print(S, n, flush=True)
R = pd.concat(out, ignore_index=True)
ids = np.array(R.pop("ids_o").tolist()); 
for k in range(5): R[f"p{k}"] = ids[:, k]
R.to_parquet(f"{D}/chem_stints.parquet", index=False)
print(R.shape)
