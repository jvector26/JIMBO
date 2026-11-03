"""Pass 2 of the Elo test against the production in-season setup (session 44).
Base = M4 in-season ratings refit before every game day (elo_m4_preds.py) + actual minutes, LOSO-calibrated.
Team layers are refit before every game day from season-to-date residuals computed with the CURRENT ratings
(as production nightly.py does):
  R{lam}   ridge team offsets (production = lam 80, from 15 league games)
  ELO      residual Elo run through the season's past games on those residuals, step k, season start c * last end
  R80+ELO  Elo on what the ridge layer leaves
  ELOpre   classic Elo on PREGAME residuals (no re-scoring of past games with current ratings)
  +PELO    plain points Elo (results only, from 1996) as an extra regressor next to the M4 base, LOSO
k, c picked LOSO. Scored on 2014-15..2025-26 regular seasons."""
import numpy as np, pandas as pd

D = "/home/claude/data"
SEAS = np.arange(2014, 2026)
GA = pd.read_parquet(f"{D}/games_all.parquet"); GA["margin"] = GA.home_pts - GA.away_pts
G = GA[GA.season.isin(SEAS)].sort_values(["date", "game"]).reset_index(drop=True)
y = G.margin.to_numpy(float); seas = G.season.to_numpy(); dates = G.date.to_numpy()
teams = np.unique(np.r_[G.home_team, G.away_team]); ti = {t: i for i, t in enumerate(teams)}
hi = G.home_team.map(ti).to_numpy(); ai = G.away_team.map(ti).to_numpy(); nT = len(teams)

# pregame base and per-day current-rating predictions of past games
P = {}; pre_raw = np.empty(len(G)); cpd = {}
for S in SEAS:
    t = pd.read_parquet(f"{D}/elo_m4_preds_{S}.parquet")
    piv = t.pivot(index="cp_date", columns="game", values="pred").sort_index()
    idx = np.flatnonzero(seas == S); gm = G.game.to_numpy()[idx]
    cp_dates = piv.index.to_numpy()
    act = cp_dates[np.searchsorted(cp_dates, dates[idx], side="right") - 1]
    M = piv[gm].to_numpy()  # rows cp, cols games of S in G order
    row = {d: i for i, d in enumerate(cp_dates)}
    pre_raw[idx] = M[[row[a] for a in act], np.arange(len(idx))]
    P[S] = (idx, M, row, act)

# LOSO calibration of the base
X = np.c_[np.ones(len(y)), pre_raw]; cal = {}
for S in SEAS:
    b = np.linalg.lstsq(X[seas != S], y[seas != S], rcond=None)[0]; cal[S] = b
base = np.array([cal[s][0] + cal[s][1] * p for s, p in zip(seas, pre_raw)])
rm = lambda p, m=None: np.sqrt(((y - p)[m if m is not None else slice(None)] ** 2).mean())
print(f"M4 base (refit nightly) LOSO {rm(base):.4f}   raw uncalibrated {rm(pre_raw):.4f}  cal slope ~{np.mean([c[1] for c in cal.values()]):.3f}")


def layers(lams=(20, 40, 80, 160), KS=(0.005, 0.01, 0.015, 0.02, 0.03, 0.045), CS=(0.0, 0.25, 0.5)):
    out = {("R", l): np.zeros(len(y)) for l in lams}
    for k in KS:
        for c in CS:
            out[("E", k, c)] = np.zeros(len(y)); out[("RE", k, c)] = np.zeros(len(y)); out[("EP", k, c)] = np.zeros(len(y))
    carry = {key: np.zeros(nT) for key in out if key[0] != "R"}
    for S in SEAS:
        idx, M, row, act = P[S]; a0, b0 = cal[S]
        days = np.unique(dates[idx]); start = {key: carry[key][0] * 0 + key[2] * carry[key] for key in carry}
        ep = {key: start[key].copy() for key in carry if key[0] == "EP"}
        dg = [np.flatnonzero(dates[idx] == d) for d in days]  # positions within idx
        for di, d in enumerate(days):
            today = dg[di]; past = np.concatenate(dg[:di]) if di else np.array([], int)
            rows_cur = row[act[today[0]]]
            if len(past):
                res = y[idx[past]] - (a0 + b0 * M[rows_cur, past])
                hp, ap = hi[idx[past]], ai[idx[past]]
            th, ta = hi[idx[today]], ai[idx[today]]
            # ridge
            off80 = np.zeros(nT)
            if len(past) >= 15:
                C = np.zeros((nT, nT)); np.add.at(C, (hp, hp), 1); np.add.at(C, (ap, ap), 1)
                np.add.at(C, (hp, ap), -1); np.add.at(C, (ap, hp), -1)
                bv = np.zeros(nT); np.add.at(bv, hp, res); np.add.at(bv, ap, -res)
                for l in lams:
                    off = np.linalg.solve(C + l * np.eye(nT), bv)
                    out[("R", l)][idx[today]] = off[th] - off[ta]
                    if l == 80: off80 = off
            # Elo on current-rating residuals, rerun through past days
            for key in carry:
                if key[0] == "EP": continue
                k = key[1]; r = start[key].copy()
                if len(past):
                    rr = res - ((off80[hp] - off80[ap]) if key[0] == "RE" else 0)
                    pos = 0
                    for dj in range(di):
                        n = len(dg[dj]); h, a, e0 = hp[pos:pos + n], ap[pos:pos + n], rr[pos:pos + n]
                        e = e0 - (r[h] - r[a]); np.add.at(r, h, k * e); np.add.at(r, a, -k * e); pos += n
                out[key][idx[today]] = r[th] - r[ta] + ((off80[th] - off80[ta]) if key[0] == "RE" else 0)
                if di == len(days) - 1:
                    rf = r.copy(); e = y[idx[today]] - base[idx[today]] - (rf[th] - rf[ta])
                    np.add.at(rf, th, k * e); np.add.at(rf, ta, -k * e); carry[key] = rf
            # classic Elo on pregame residuals (sequential)
            for key, r in ep.items():
                lay = r[th] - r[ta]; out[key][idx[today]] = lay
                e = y[idx[today]] - base[idx[today]] - lay
                np.add.at(r, th, key[1] * e); np.add.at(r, ta, -key[1] * e)
        for key in ep: carry[key] = ep[key]
    return out


def pick_loso(runs):
    n = np.array([(seas == s).sum() for s in SEAS])
    M = {k: np.array([((y - p)[seas == s] ** 2).mean() for s in SEAS]) for k, p in runs.items()}
    p_out = np.empty(len(y)); ch = []
    for i, s in enumerate(SEAS):
        o = np.arange(len(SEAS)) != i
        best = min(M, key=lambda k: (M[k][o] * n[o]).sum()); ch.append(best); p_out[seas == s] = runs[best][seas == s]
    return p_out, ch


def compare(p, ref, name, refname, rows):
    d = (y - p) ** 2 - (y - ref) ** 2
    S = np.array([d[seas == s].sum() for s in SEAS]); n = np.array([(seas == s).sum() for s in SEAS]); m = d.mean()
    se = np.sqrt(((S - n * m) ** 2).sum()) / len(d)
    h1, h2 = seas <= 2019, seas >= 2020
    dr = lambda mk: rm(p, mk) - rm(ref, mk)
    print(f"  {name:20s} {rm(p):.4f}  vs {refname:8s} {rm(p)-rm(ref):+.4f}  t {m/se:+.2f}  better {(S<0).sum()}/{len(SEAS)}"
          f"  halves {dr(h1):+.4f} / {dr(h2):+.4f}")
    rows.append(dict(name=name, ref=refname, rmse=rm(p), d=rm(p) - rm(ref), t=m / se, better=int((S < 0).sum()),
                     h1=dr(h1), h2=dr(h2)))


L = layers()
runs = {k: base + v for k, v in L.items()}
r80 = runs[("R", 80)]
print("ridge lam:", {k[1]: round(rm(v), 4) for k, v in runs.items() if k[0] == "R"})
for typ, nm in [("E", "Elo on current-rating residuals"), ("RE", "Elo on top of ridge 80"), ("EP", "classic Elo on pregame residuals")]:
    tab = pd.DataFrame({c: [rm(runs[(typ, k, c)]) for k in (0.005, 0.01, 0.015, 0.02, 0.03, 0.045)] for c in (0.0, 0.25, 0.5)},
                       index=(0.005, 0.01, 0.015, 0.02, 0.03, 0.045))
    print(f"in-sample grid, {nm} (rows k, cols c):"); print(tab.round(4).to_string())

rows = []
elo, ch1 = pick_loso({k: v for k, v in runs.items() if k[0] == "E"})
both, ch2 = pick_loso({k: v for k, v in runs.items() if k[0] == "RE"})
elop, ch3 = pick_loso({k: v for k, v in runs.items() if k[0] == "EP"})
ridge, ch4 = pick_loso({k: v for k, v in runs.items() if k[0] == "R"})
print("LOSO picks: ELO", sorted(set(ch1)), " R80+ELO", sorted(set(ch2)), " ELOpre", sorted(set(ch3)), " ridge", sorted(set(ch4)))
print("\nAGAINST THE PRODUCTION IN-SEASON SETUP (M4 nightly ratings + ridge 80 team offsets)")
compare(r80, base, "R80 (production)", "M4 only", rows)
compare(ridge, r80, "ridge, lam LOSO", "R80", rows)
compare(elo, r80, "ELO instead of R80", "R80", rows)
compare(both, r80, "R80 + ELO", "R80", rows)
compare(elop, r80, "ELOpre instead", "R80", rows)

# plain points Elo next to M4 (+ ridge)
GA = GA.sort_values(["date", "game"]).reset_index(drop=True)
at = np.unique(np.r_[GA.home_team, GA.away_team]); gi = {t: i for i, t in enumerate(at)}
gh = GA.home_team.map(gi).to_numpy(); ga = GA.away_team.map(gi).to_numpy(); gm = GA.margin.to_numpy(float)
gs = GA.season.to_numpy(); gd = GA.date.to_numpy()
r = np.zeros(len(at)); pel = np.zeros(len(GA)); last = None
for d in np.unique(gd):
    g = np.flatnonzero(gd == d); s = gs[g[0]]
    if s != last and last is not None: r = 0.6 * r
    last = s; pel[g] = r[gh[g]] - r[ga[g]]
    e = gm[g] - (2.5 + pel[g]); np.add.at(r, gh[g], 0.04 * e); np.add.at(r, ga[g], -0.04 * e)
pel = pd.Series(pel, index=GA.game).reindex(G.game).to_numpy()


def loso(Xm):
    Xm = np.c_[np.ones(len(y)), Xm]; p = np.empty(len(y))
    for s in SEAS:
        b = np.linalg.lstsq(Xm[seas != s], y[seas != s], rcond=None)[0]; p[seas == s] = Xm[seas == s] @ b
    return p


compare(loso(np.c_[pre_raw, pel]), base, "M4 + plain Elo reg", "M4 only", rows)
pr = loso(np.c_[pre_raw, r80 - base, pel])
compare(pr, r80, "M4+R80+plain Elo", "R80", rows)
bb = np.linalg.lstsq(np.c_[np.ones(len(y)), pre_raw, r80 - base, pel], y, rcond=None)[0]
print(f"  joint coefs: M4 raw {bb[1]:.3f}, ridge layer {bb[2]:.3f}, plain Elo {bb[3]:.3f}")

# by stage of season
tg = G.groupby("season").cumcount() * 2 / 30
lab = pd.cut(tg, [0, 10, 25, 45, 65, 83], right=False)
tab = pd.DataFrame({"games": lab.value_counts(sort=False)})
for nm, p in [("M4", base), ("R80", r80), ("ELO", elo), ("R80+ELO", both), ("ELOpre", elop)]:
    tab[nm] = pd.Series((y - p) ** 2).groupby(lab).mean() ** .5
print("\nRMSE by team games played:"); print(tab.round(3).to_string())
pd.DataFrame(rows).to_csv(f"{D}/elo_m4_results.csv", index=False); tab.to_csv(f"{D}/elo_m4_by_stage.csv")

# isolate the plain-Elo part: same free-coefficient LOSO regression without Elo
pr0 = loso(np.c_[pre_raw, r80 - base])
print("\nfree-coefficient check:")
compare(pr0, r80, "M4+R80 free coefs", "R80", rows)
compare(pr, pr0, "+ plain Elo", "M4+R80 fr", rows)
lab2 = pd.cut(tg, [0, 10, 25, 45, 65, 83], right=False)
print((pd.Series((y - pr) ** 2 - (y - pr0) ** 2).groupby(lab2).mean()).round(3).to_string())
for s in SEAS:
    m = seas == s; print(s, round(rm(pr, m) - rm(pr0, m), 4), end=" | ")
print()
