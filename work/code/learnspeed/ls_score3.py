"""Horizon check (rest-of-season, what the win forecast needs) + team-offset speed on top of the player layer."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code/learnspeed"); from ls_lib import *
from ls_preds import LAMS
pd.set_option("display.width", 250)
CP = [10, 20, 30, 41, 55]; fr = []
for S in SEAS:
    t = pd.read_parquet(f"{LS}/ls_{S}.parquet", columns=["cp", "game", "a_pre", "a_bx_1.0"] + [f"a_rp_1.0_{l}" for l in LAMS]); t = t[t.cp.isin(CP)]
    g = pd.read_parquet(f"{D}/games_{S}.parquet")[["game", "home_team", "away_team", "home_pts", "away_pts"]]; t = t.merge(g, on="game"); t["season"] = S; fr.append(t)
H = pd.concat(fr, ignore_index=True); H["margin"] = H.home_pts - H.away_pts
print("REST OF SEASON from a checkpoint (actual minutes): game RMSE and team rest-of-season mean-margin RMSE, change vs lam 6000")
for cp in CP:
    h = H[H.cp == cp].reset_index(drop=True); y = h.margin.to_numpy(float); seas = h.season.to_numpy(); m = np.ones(len(h), bool); res = {}
    for l in LAMS:
        p = loso((h.a_pre + h["a_bx_1.0"] + h[f"a_rp_1.0_{l}"]).to_numpy(float), y, seas, m); e = y - p
        tm = pd.concat([pd.DataFrame({"s": seas, "t": h.home_team, "e": e}), pd.DataFrame({"s": seas, "t": h.away_team, "e": -e})]).groupby(["s", "t"]).e.mean()
        res[l] = (p, np.sqrt((tm ** 2).mean()))
    pb = res[6000][0]
    print(f"cp {cp:2d}: base game {rmse(pb,y,m):.4f} team {res[6000][1]:.3f} | " + "  ".join(f"{l}: {rmse(res[l][0],y,m)-rmse(pb,y,m):+.4f} (t {pair(res[l][0],pb,y,seas,m)[1]:+.1f}) team {res[l][1]-res[6000][1]:+.3f}" for l in LAMS if l != 6000))
# team offsets on next-game predictions
N = next_frame(); y = N.margin.to_numpy(float); seas = N.season.to_numpy(); tgn = N.tgn.to_numpy(); dates = N.date.to_numpy()
teams = np.unique(np.r_[N.home_team, N.away_team]); ti = {t: i for i, t in enumerate(teams)}; hi = N.home_team.map(ti).to_numpy(); ai = N.away_team.map(ti).to_numpy(); nT = len(teams)
def offsets(pcal, lam_t, decay=None):
    off = np.zeros(len(N))
    for S in SEAS:
        idx = np.flatnonzero(seas == S); days = np.unique(dates[idx]); dg = [np.flatnonzero(dates[idx] == d) for d in days]
        for di in range(len(days)):
            past = np.concatenate(dg[:di]) if di else np.array([], int)
            if len(past) < 15: continue
            res = (y - pcal)[idx[past]]; hp, ap = hi[idx[past]], ai[idx[past]]; w = np.ones(len(past))
            if decay: w = decay ** (di - np.searchsorted(days, dates[idx[past]]))
            C = np.zeros((nT, nT)); np.add.at(C, (hp, hp), w); np.add.at(C, (ap, ap), w); np.add.at(C, (hp, ap), -w); np.add.at(C, (ap, hp), -w)
            bv = np.zeros(nT); np.add.at(bv, hp, w * res); np.add.at(bv, ap, -w * res); o = np.linalg.solve(C + lam_t * np.eye(nT), bv); off[idx[dg[di]]] = o[hi[idx[dg[di]]]] - o[ai[idx[dg[di]]]]
    return off
STAGES = [(3, 10), (10, 25), (25, 50), (50, 70), (70, 99)]
for mn in ["a", "t"]:
    c = lambda k: N[f"{mn}_{k}"].to_numpy(float); m = np.isfinite(c("pre")) & (tgn >= 3)
    # residuals for offsets always from actual-minutes predictions (as production: past games are scored with who played)
    ca = lambda k: N[f"a_{k}"].to_numpy(float); ma = np.isfinite(ca("pre"))
    pa = loso(ca("pre") + ca("bx_1.0") + ca("rp_1.0_6000"), y, seas, np.ones(len(N), bool))
    base = c("pre") + c("bx_1.0") + c("rp_1.0_6000"); o80 = offsets(pa, 80.0); pb = loso(np.c_[base, o80], y, seas, m); p0 = loso(base, y, seas, m)
    print(f"\n== minutes {mn}: player layer only {rmse(p0,y,m):.4f}; + team offsets lam 80 (production) {rmse(pb,y,m):.4f}; + B2B term: {rmse(loso(np.c_[base,o80,N.b2b.to_numpy()],y,seas,m),y,m):.4f}")
    r = creg(np.c_[base, o80], y, seas, m, ["player layer", "offset"]); print("   slopes:", "  ".join(f"{k} {v.coef:.3f} ({v.se:.3f})" for k, v in r.iterrows()))
    for lo, hi_ in STAGES:
        mm = m & (tgn >= lo) & (tgn < hi_); r = creg(np.c_[base, o80], y, seas, mm, ["p", "o"]); print(f"   games {lo}-{hi_}: offset slope {r.coef['o']:.2f} ({r.se['o']:.2f})  sd(offset) {o80[mm].std():.2f}")
    for lt in [20, 40, 160, 320]:
        pv = loso(np.c_[base, offsets(pa, float(lt))], y, seas, m); r = pair(pv, pb, y, seas, m)
        print(f"   team offset lam {lt:3d}: {r[0]:+.4f} t {r[1]:+.1f} {r[2]}/{r[3]} halves {r[4]:+.4f}/{r[5]:+.4f} | stages " + " ".join(f"{rmse(pv,y,m&(tgn>=lo)&(tgn<h2))-rmse(pb,y,m&(tgn>=lo)&(tgn<h2)):+.4f}" for lo, h2 in STAGES))
    for dc in [0.99, 0.98]:
        pv = loso(np.c_[base, offsets(pa, 80.0, dc)], y, seas, m); r = pair(pv, pb, y, seas, m)
        print(f"   offsets lam 80, daily decay {dc}: {r[0]:+.4f} t {r[1]:+.1f} {r[2]}/{r[3]} halves {r[4]:+.4f}/{r[5]:+.4f}")
    # player-layer speed re-checked with offsets + b2b in place
    b2 = N.b2b.to_numpy(); pbb = loso(np.c_[base, o80, b2], y, seas, m)
    for l in [3000, 12000]:
        v = c("pre") + c("bx_1.0") + c(f"rp_1.0_{l}"); pal = loso(ca("pre") + ca("bx_1.0") + ca(f"rp_1.0_{l}"), y, seas, np.ones(len(N), bool))
        pv = loso(np.c_[v, offsets(pal, 80.0), b2], y, seas, m); r = pair(pv, pbb, y, seas, m)
        print(f"   full stack with lam {l}: {r[0]:+.4f} t {r[1]:+.1f} {r[2]}/{r[3]} halves {r[4]:+.4f}/{r[5]:+.4f} | stages " + " ".join(f"{rmse(pv,y,m&(tgn>=lo)&(tgn<h2))-rmse(pbb,y,m&(tgn>=lo)&(tgn<h2)):+.4f}" for lo, h2 in STAGES))
    if mn == "t":
        ln = N.line.to_numpy(); ml = m & np.isfinite(ln)
        print("   gap to the line by stage (full stack, pregame minutes): " + " ".join(f"{lo}-{h2}: {rmse(pbb,y,ml&(tgn>=lo)&(tgn<h2))-rmse(ln,y,ml&(tgn>=lo)&(tgn<h2)):+.3f}" for lo, h2 in STAGES))
