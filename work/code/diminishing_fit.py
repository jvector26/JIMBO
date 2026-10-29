import numpy as np, pandas as pd
R = pd.read_parquet("/home/claude/data/dr_rows.parquet")
OFF = [c for c in R.columns if c.startswith("off_") and c not in ("off_unk",)]; DEF = [c for c in R.columns if c.startswith("dfn_") and c != "dfn_unk"]
for c in OFF + DEF:
    mu = np.average(R[c], weights=R.poss); R[c + "^2"] = (R[c] - mu) ** 2
R["oreb_x_dreb"] = (R.off_off_rebounding - R.off_off_rebounding.mean()) * (R.dfn_def_rebounding - R.dfn_def_rebounding.mean())
R["three_x_rim"] = (R.off_three_pt - R.off_three_pt.mean()) * (R.off_rim_finishing - R.off_rim_finishing.mean())
R["three_x_play"] = (R.off_three_pt - R.off_three_pt.mean()) * (R.off_playmaking - R.off_playmaking.mean())
SFE = pd.get_dummies(R.season, prefix="s", drop_first=True).astype(float)
def fit(cols, d=R):
    X = np.c_[np.ones(len(d)), d[["home_off", "off_unk", "dfn_unk"] + cols].to_numpy(float), SFE.loc[d.index].loc[:, SFE.loc[d.index].sum() > 0].iloc[:, 1:].to_numpy()]
    w = d.poss.to_numpy(); y = d.y.to_numpy(); Xw = X * w[:, None]
    XtWX = X.T @ Xw; b = np.linalg.solve(XtWX, Xw.T @ y); e = y - X @ b
    Xi = np.linalg.inv(XtWX); M = (Xw * e[:, None]).T @ (Xw * e[:, None]); se = np.sqrt(np.diag(Xi @ M @ Xi))
    k = slice(4, 4 + len(cols)); return pd.DataFrame({"coef": b[k], "t": b[k] / se[k]}, index=cols)
LIN = OFF + DEF; SQ = [c + "^2" for c in LIN]; INT = ["oreb_x_dreb", "three_x_rim", "three_x_play"]
full = fit(LIN + SQ + INT)
e, l = fit(LIN + SQ + INT, R[R.season <= 2018]), fit(LIN + SQ + INT, R[R.season >= 2019])
out = full.join(e.add_suffix("_12-18")).join(l.add_suffix("_19-25"))
sd = {c: np.sqrt(np.cov(R[c], aweights=R.poss)) for c in LIN}
out["lineup_sd"] = [sd.get(c.replace("^2", ""), np.nan) for c in out.index]
print(out.round(3).to_string())

# ---- size of the effect: nonlinear part per team-season (actual lineups), pts/100 net
import scipy
b = full.coef; NL = SQ
adj = sum(b[c] * R[c] for c in NL)  # added to offense y
R["nl"] = adj - np.average(adj, weights=R.poss)
