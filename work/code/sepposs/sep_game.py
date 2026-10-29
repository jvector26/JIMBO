"""Game backtest: shrink lineup residuals by SEPARATED possessions instead of total possessions (session 24).
Baseline = production judgment shrink: m = 1 - A*clip(1 - P/P1, 0, 1), A=1/3, P1=10k, P = raw window on_poss.
Variants replace P by P * c / k, where c = collinearity factor (neff/naive per side, 1-sh1, or sep2) and k = its
minutes-weighted mean, so the average shrink level is unchanged and only WHO gets shrunk changes.
Test stat: margin ~ const + base_pred + g*(variant_pred - base_pred), season-clustered t on g; LOSO RMSE; halves."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, "/home/claude/code")
from lineup_port_bt import G, B, y, seas, loso, ps
D = "/home/claude/data"
P = {N: ps[ps.season.between(N - 2, N)].groupby("player").on_poss.sum() for N in range(2014, 2025)}
G["Praw"] = [P[n].get(p, 0.0) for n, p in zip(G.N, G.player)]
S = pd.read_parquet(f"{D}/sep_poss.parquet")
S["c_neff_o"] = (S.neff_o / S.nnaive_o).clip(0, 1); S["c_neff_d"] = (S.neff_d / S.nnaive_d).clip(0, 1)
S["c_sh1"] = 1 - S.sh1; S["c_sep2"] = S.sep2
G = G.merge(S[["player", "N", "c_neff_o", "c_neff_d", "c_sh1", "c_sep2"]], on=["player", "N"], how="left")
for c in ["c_neff_o", "c_neff_d", "c_sh1", "c_sep2"]:
    k = np.average(G[c].fillna(G[c].median()), weights=G["min"])
    G[c] = (G[c] / k).fillna(1.0)
print("missing sep rows:", int(G.c_sh1.isna().sum()), " rows", len(G))

def diffs2(mo, md):
    v = G.sgn * G.w48
    return (pd.DataFrame({"o": v * mo * G.res_o, "d": v * md * G.res_d, "game": G.game}).groupby("game")[["o", "d"]].sum()
            .reindex(B.index).fillna(0))

def mult(Pv, a, P1):
    return 1 - a * np.clip(1 - Pv / P1, 0, 1)

def pred(dd):
    return B.box.values + .6 * dd.o.values + .8 * dd.d.values

def score(pv, p0):
    l, s = loso(np.c_[pv]); l0, s0 = loso(np.c_[p0])
    X = sm.add_constant(np.c_[p0, pv - p0]); f = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": seas})
    h1 = np.sqrt(np.mean(s[:5] ** 2)) - np.sqrt(np.mean(s0[:5] ** 2)); h2 = np.sqrt(np.mean(s[5:] ** 2)) - np.sqrt(np.mean(s0[5:] ** 2))
    return dict(LOSO=l, d=l - l0, better=int((s < s0).sum()), g=f.params[2], t=f.tvalues[2], h1=h1, h2=h2)

A0, P10 = 1 / 3, 10000.0
m0 = mult(G.Praw, A0, P10); base = pred(diffs2(m0, m0))
noshr = pred(diffs2(np.ones(len(G)), np.ones(len(G))))
print(f"current shrink LOSO {loso(np.c_[base])[0]:.4f}   no shrink {loso(np.c_[noshr])[0]:.4f}")
rows = []
variants = {"neff (o/d)": ("c_neff_o", "c_neff_d"), "1 - top1 share": ("c_sh1", "c_sh1"), "sep from top2": ("c_sep2", "c_sep2")}
for name, (co, cd) in variants.items():
    for P1 in [8000, 10000, 12000, 15000]:
        for a in [.25, 1 / 3, .5]:
            mo, md = mult(G.Praw * G[co], a, P1), mult(G.Praw * G[cd], a, P1)
            r = score(pred(diffs2(mo, md)), base); r.update(var=name, P1=P1, a=round(a, 2)); rows.append(r)
# same grid on total possessions (so a variant's best is compared with total's best)
for P1 in [8000, 10000, 12000, 15000]:
    for a in [.25, 1 / 3, .5]:
        m = mult(G.Praw, a, P1); r = score(pred(diffs2(m, m)), base); r.update(var="total (current)", P1=P1, a=round(a, 2)); rows.append(r)
R = pd.DataFrame(rows)[["var", "P1", "a", "LOSO", "d", "better", "g", "t", "h1", "h2"]]
pd.set_option("display.width", 200)
print(R.round(4).to_string(index=False))
R.to_csv(f"{D}/sep_game_grid.csv", index=False)
