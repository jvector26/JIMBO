import numpy as np, pandas as pd
from scipy.stats import norm
D = "/home/claude/data"
gm = pd.read_parquet(f"{D}/games_all.parquet")
id2abbr = pd.read_parquet(f"{D}/games_2025.parquet").drop_duplicates("home_team").set_index("home_team").htm.to_dict()
res = pd.concat([gm.assign(t=gm.home_team, w=gm.home_pts > gm.away_pts), gm.assign(t=gm.away_team, w=gm.away_pts > gm.home_pts)])
act = res.groupby(["season", "t"]).agg(wpct=("w", "mean"), n=("w", "size")).reset_index()
act["team"] = act.t.map(id2abbr)
T = pd.concat([pd.read_parquet(f"{D}/pre_T_{s}.parquet") for s in range(2014, 2026)])
L = pd.read_csv(f"{D}/hist_lines.csv")
L["games"] = np.where(L.season == 2020, 72, 82); L["mkt_wpct"] = L.line / L.games
A = T.merge(act[["season", "team", "wpct", "n"]], on=["season", "team"]).merge(L[["season", "team", "mkt_wpct"]], on=["season", "team"])
prev = act.assign(season=act.season + 1)[["season", "team", "wpct"]].rename(columns={"wpct": "prev_wpct"})
A = A.merge(prev, on=["season", "team"])
# regress prev-season record halfway to .500 (standard naive baseline)
A["prev_reg"] = 0.5 + 0.5 * (A.prev_wpct - 0.5)
assert len(A) == 360, len(A)
W = lambda x: 82 * x  # express everything in wins per 82 games
A.to_parquet(f"{D}/preseason_backtest.parquet", index=False)

def rmse(e): return float(np.sqrt(np.mean(np.square(e))))
def report(df, label):
    y = W(df.wpct)
    print(f"\n== {label}: {len(df)} team-seasons ==")
    for name, col in [("Vegas line", "mkt_wpct"), ("Model", "model_wpct"), ("Prev record, half-regressed", "prev_reg")]:
        p = W(df[col]); b = np.polyfit(p, y, 1)
        print(f"  {name:28s} RMSE {rmse(y - p):5.2f} wins | MAE {np.mean(np.abs(y - p)):5.2f} | corr {np.corrcoef(p, y)[0, 1]:.3f} | slope(actual on pred) {b[0]:.2f} | sd {p.std():4.1f}")
    print(f"  {'Everyone 41 wins':28s} RMSE {rmse(y - 41):5.2f} | actual sd {y.std():.1f}")

report(A, "All seasons 2014-15..2025-26")
B = A[A.season != 2019]  # 2019-20 cut short (bets voided)
report(B, "Excluding 2019-20")

# Calibrated model: does stretching the model help, and what's the best blend with Vegas? (leave-one-season-out)
print("\n== Leave-one-season-out: fit on 11 seasons, predict the 12th ==")
rows = []
for s in sorted(B.season.unique()):
    tr, te = B[B.season != s], B[B.season == s]
    X = lambda d: np.c_[np.ones(len(d)), W(d.model_wpct - .5)]
    b = np.linalg.lstsq(X(tr), W(tr.wpct - .5), rcond=None)[0]
    Xb = lambda d: np.c_[W(d.model_wpct - .5), W(d.mkt_wpct - .5)]
    c = np.linalg.lstsq(Xb(tr), W(tr.wpct - .5), rcond=None)[0]
    rows.append(pd.DataFrame({"season": s, "y": W(te.wpct), "mkt": W(te.mkt_wpct), "raw": W(te.model_wpct),
                              "stretched": 41 + X(te) @ b, "blend": 41 + Xb(te) @ c, "w_model": c[0], "w_mkt": c[1], "stretch": b[1]}))
CV = pd.concat(rows)
for k in ["mkt", "raw", "stretched", "blend"]:
    print(f"  {k:10s} RMSE {rmse(CV.y - CV[k]):.2f}")
print("  blend weights (median over folds): model %.2f, Vegas %.2f;  model stretch factor %.2f" %
      (CV.groupby("season").w_model.first().median(), CV.groupby("season").w_mkt.first().median(), CV.groupby("season").stretch.first().median()))

# Full-sample blend + significance of the model's extra information
y = W(B.wpct - .5); Xf = np.c_[W(B.model_wpct - .5), W(B.mkt_wpct - .5)]
c, *_ = np.linalg.lstsq(Xf, y, rcond=None); e = y - Xf @ c
cov = np.linalg.inv(Xf.T @ Xf) * e.var()
print("\n== Full-sample blend (no intercept): actual-41 = %.2f*(model-41) + %.2f*(Vegas-41)  [t = %.1f, %.1f]" % (c[0], c[1], c[0] / np.sqrt(cov[0, 0]), c[1] / np.sqrt(cov[1, 1])))

# Per-season view
ps_ = B.groupby("season").apply(lambda d: pd.Series({"model": rmse(W(d.wpct - d.model_wpct)), "vegas": rmse(W(d.wpct - d.mkt_wpct))}))
print("\n== RMSE by season (wins) =="); print(ps_.round(2).T.to_string())

# Uncertainty: how much of the error is true-talent projection error vs game-to-game luck?
luck_var = (82 ** 2) * (B.wpct * (1 - B.wpct) / B.n)  # binomial variance of win% given the season's true p (approx)
for name, col in [("Vegas", "mkt_wpct"), ("Model", "model_wpct")]:
    tot = np.mean(np.square(W(B.wpct - B[col])))
    print(f"\n{name}: total error var {tot:.1f} wins^2, game-luck part {luck_var.mean():.1f} -> projection sd {np.sqrt(max(tot - luck_var.mean(), 0)):.1f} wins"
          f" = {np.sqrt(max(tot - luck_var.mean(), 0)) / 2.8:.2f} pts of margin")

# Where the model misses relative to Vegas
B = B.assign(gap=W(B.model_wpct - B.mkt_wpct), m_err=W(B.wpct - B.model_wpct), v_err=W(B.wpct - B.mkt_wpct))
print("\n== When model and Vegas disagree by 6+ wins, who was closer? ==")
big = B[B.gap.abs() >= 6]
print(f"  {len(big)} cases; Vegas closer in {(big.v_err.abs() < big.m_err.abs()).mean():.0%}; actual landed on model's side of the line in {((big.wpct - big.mkt_wpct) * big.gap > 0).mean():.0%}")
print("  avg |error|: model %.1f, Vegas %.1f" % (big.m_err.abs().mean(), big.v_err.abs().mean()))
for lo, hi, lab in [(0, .366, "Vegas <30 wins"), (.366, .634, "Vegas 30-52"), (.634, 1, "Vegas >52 wins")]:
    d = B[(B.mkt_wpct >= lo) & (B.mkt_wpct < hi)]
    print(f"  {lab:15s} n={len(d):3d}  model bias {(-d.m_err).mean():+5.1f}  Vegas bias {(-d.v_err).mean():+5.1f}  (positive = projected too many wins)")
