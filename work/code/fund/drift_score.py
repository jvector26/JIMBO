"""T1 score: do rookies / second-year players drift up within the season beyond the running in-season estimate?"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code/learnspeed"); import ls_lib as L
L.LS = "/home/claude/fund/ls"; from ls_lib import *
N = L.next_frame(refresh=True); y = N.margin.to_numpy(float); seas = N.season.to_numpy(); tgn = N.tgn.to_numpy(); dates = N.date.to_numpy()
prog = ((N.tgn + N.tgx) / 2 / 82).to_numpy()
teams = np.unique(np.r_[N.home_team, N.away_team]); ti = {t: i for i, t in enumerate(teams)}; hi = N.home_team.map(ti).to_numpy(); ai = N.away_team.map(ti).to_numpy(); nT = len(teams)
def offsets(pcal, lam_t):
    off = np.zeros(len(N))
    for S in SEAS:
        idx = np.flatnonzero(seas == S); days = np.unique(dates[idx]); dg = [np.flatnonzero(dates[idx] == d) for d in days]
        for di in range(len(days)):
            past = np.concatenate(dg[:di]) if di else np.array([], int)
            if len(past) < 15: continue
            res = (y - pcal)[idx[past]]; hp, ap = hi[idx[past]], ai[idx[past]]; w = np.ones(len(past))
            C = np.zeros((nT, nT)); np.add.at(C, (hp, hp), w); np.add.at(C, (ap, ap), w); np.add.at(C, (hp, ap), -w); np.add.at(C, (ap, hp), -w)
            bv = np.zeros(nT); np.add.at(bv, hp, w * res); np.add.at(bv, ap, -w * res); o = np.linalg.solve(C + lam_t * np.eye(nT), bv); off[idx[dg[di]]] = o[hi[idx[dg[di]]]] - o[ai[idx[dg[di]]]]
    return off
ca = lambda k: N[f"a_{k}"].to_numpy(float)
pa = loso(ca("pre") + ca("bx_1.0") + ca("rp_1.0_6000"), y, seas, np.ones(len(N), bool)); o80 = offsets(pa, 80.0); b2 = N.b2b.to_numpy()
for mn, lab in (("a", "ACTUAL minutes"), ("t", "PREGAME (trailing-5) minutes  <- decides")):
    c = lambda k: N[f"{mn}_{k}"].to_numpy(float); m = np.isfinite(c("pre")) & (tgn >= 3)
    stack = c("pre") + c("bx_1.0") + c("rp_1.0_6000"); E = {k: c(f"e_{k}") for k in ["r1", "r2", "y3", "old"]}
    X0 = np.c_[stack, o80, b2]; p0 = loso(X0, y, seas, m)
    print(f"\n== {lab}: games {m.sum()}, production stack LOSO {rmse(p0, y, m):.4f}")
    print("   mean |exposure| per game: " + "  ".join(f"{k} {np.abs(E[k][m]).mean():.2f}" for k in E))
    Xl = np.c_[X0, E["r1"], E["r2"]]; pl = loso(Xl, y, seas, m); r = pair(pl, p0, y, seas, m)
    print(f"   level terms only (R1, R2):        {r[0]:+.4f} t {r[1]:+.1f} {r[2]}/{r[3]} halves {r[4]:+.4f}/{r[5]:+.4f}")
    Xd = np.c_[Xl, E["r1"] * prog, E["r2"] * prog]; pd_ = loso(Xd, y, seas, m); r = pair(pd_, pl, y, seas, m)
    print(f"   PRIMARY + drift terms (vs level): {r[0]:+.4f} t {r[1]:+.1f} {r[2]}/{r[3]} halves {r[4]:+.4f}/{r[5]:+.4f}")
    nm = ["stack", "offset", "b2b", "R1 level", "R2 level", "R1 drift", "R2 drift"]
    for hl, mm in (("all", m), ("2014-19", m & (seas <= 2019)), ("2020-25", m & (seas >= 2020))):
        R = creg(Xd, y, seas, mm, nm); print(f"   {hl:8s} " + "  ".join(f"{k}: {R.coef[k]:+.2f} (t {R.coef[k]/R.se[k]:+.1f})" for k in nm[3:]))
    print("   (level = pts/100 the group is mis-rated at the start of the season; drift = change over a full season, expected > 0)")
    # robustness: one group at a time, other groups
    for k, lb in (("r1", "rookies"), ("r2", "second-year"), ("y3", "under 23 with 2+ seasons"), ("old", "32 and over")):
        Xb = np.c_[X0, E[k]]; pb = loso(Xb, y, seas, m); pv = loso(np.c_[Xb, E[k] * prog], y, seas, m); r = pair(pv, pb, y, seas, m)
        R = creg(np.c_[Xb, E[k] * prog], y, seas, m, ["s", "o", "b", "lev", "dr"]); R1 = creg(np.c_[Xb, E[k] * prog], y, seas, m & (seas <= 2019), ["s", "o", "b", "lev", "dr"]); R2 = creg(np.c_[Xb, E[k] * prog], y, seas, m & (seas >= 2020), ["s", "o", "b", "lev", "dr"])
        print(f"   robustness {lb:26s} drift {R.coef['dr']:+.2f} (t {R.coef['dr']/R.se['dr']:+.1f}; halves {R1.coef['dr']/R1.se['dr']:+.1f}/{R2.coef['dr']/R2.se['dr']:+.1f})  level {R.coef['lev']:+.2f} (t {R.coef['lev']/R.se['lev']:+.1f})  dLOSO {r[0]:+.4f} t {r[1]:+.1f} {r[2]}/{r[3]}")
    # where: late-season stage effect of the primary
    for lo, h2 in [(3, 25), (25, 50), (50, 70), (70, 99)]:
        mm = m & (tgn >= lo) & (tgn < h2); print(f"   games {lo}-{h2}: primary vs level {rmse(pd_, y, mm) - rmse(pl, y, mm):+.4f}", end=" |")
    print()
