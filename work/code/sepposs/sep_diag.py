"""Direct test (session 24): with NO shrink as the starting point, does collinearity predict how much of a residual carries
over, beyond total possessions?  y ~ const + base + X_ramp + X_c   (season-clustered)
  base   = box + .6 sum(res_o) + .8 sum(res_d)          (signed, minutes/48 x pace weighted, as in the backtest)
  X_ramp = .6/.8 residual sums x clip(1 - Praw/10k, 0, 1)   (negative coef = thin samples overstated -> total-poss shrink)
  X_c    = .6/.8 residual sums x (c - 1), c = normalised collinearity factor (1 = average separation)
           positive coef = well-separated players' residuals carry over MORE (stapled players' carry over less).
Also: average shrink of each variant at A=1/3, P1=10k (minutes-weighted), to separate 'targeting' from 'less shrink'."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, "/home/claude/code/sepposs"); sys.path.insert(0, "/home/claude/code")
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import sep_game as SG
G, B, y, seas, mult = SG.G, SG.B, SG.y, SG.seas, SG.mult

def agg(col):
    v = G.sgn * G.w48 * col
    return (pd.DataFrame({"o": v * G.res_o, "d": v * G.res_d, "game": G.game}).groupby("game")[["o", "d"]].sum().reindex(B.index).fillna(0))

def comb(dd): return .6 * dd.o.values + .8 * dd.d.values
ramp = np.clip(1 - G.Praw / 10000, 0, 1)
base = B.box.values + comb(agg(np.ones(len(G)))); Xr = comb(agg(ramp))
print("mean shrink (minutes-weighted 1-m) at A=1/3, P1=10k:")
m0 = mult(G.Praw, 1 / 3, 10000); print(f"  total poss     {np.average(1 - m0, weights=G['min']):.4f}")
for nm, c in [("neff", (G.c_neff_o + G.c_neff_d) / 2), ("1-top1", G.c_sh1), ("sep2", G.c_sep2)]:
    m = mult(G.Praw * c, 1 / 3, 10000); print(f"  {nm:14s} {np.average(1 - m, weights=G['min']):.4f}")
rows = []
for nm, co, cd in [("neff", "c_neff_o", "c_neff_d"), ("1-top1", "c_sh1", "c_sh1"), ("sep2", "c_sep2", "c_sep2")]:
    v = G.sgn * G.w48
    dd = (pd.DataFrame({"o": v * G.res_o * (G[co] - 1), "d": v * G.res_d * (G[cd] - 1), "game": G.game})
          .groupby("game")[["o", "d"]].sum().reindex(B.index).fillna(0))
    Xc = comb(dd)
    for label, X in [("c only", np.c_[base, Xc]), ("c + ramp", np.c_[base, Xr, Xc])]:
        out = {}
        for half, msk in [("all", np.ones(len(y), bool)), ("2015-19", seas <= 2019), ("2020-25", seas >= 2020)]:
            f = sm.OLS(y[msk], sm.add_constant(X[msk])).fit(cov_type="cluster", cov_kwds={"groups": seas[msk]})
            out[half] = (f.params[-1], f.tvalues[-1])
            if half == "all" and X.shape[1] == 3: out["ramp"] = (f.params[2], f.tvalues[2])
        rows.append(dict(measure=nm, model=label, coef=out["all"][0], t=out["all"][1], c1519=out["2015-19"][0], t1519=out["2015-19"][1],
                         c2025=out["2020-25"][0], t2025=out["2020-25"][1], ramp_coef=out.get("ramp", (np.nan,))[0], ramp_t=out.get("ramp", (np.nan, np.nan))[1]))
f = sm.OLS(y, sm.add_constant(np.c_[base, Xr])).fit(cov_type="cluster", cov_kwds={"groups": seas})
print(f"ramp alone: coef {f.params[2]:.3f} t {f.tvalues[2]:.2f}   (base coef {f.params[1]:.3f})")
print("sd of c (minutes-weighted):", {c: round(float(np.sqrt(np.average((G[c] - 1) ** 2, weights=G['min']))), 3) for c in ["c_neff_o", "c_sh1", "c_sep2"]})
print(pd.DataFrame(rows).round(3).to_string(index=False))
