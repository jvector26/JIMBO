"""CHEMISTRY TEST 2 fits on data/chem_stints.parquet. WLS (possession weights), cluster = offense team-season.
Base: y ~ season FE + home + offense ratings (box, lineup res) + defense ratings + unrated counts + age + sample size.
Added: familiarity (pairs) and tenure (players with this team), prior seasons (P) and this season so far (C), both sides.
Signs: offense familiarity should RAISE y (+), defense familiarity should LOWER y (-)."""
import numpy as np, pandas as pd, statsmodels.api as sm, sys
D = "/home/claude/data"
R = pd.read_parquet(f"{D}/chem_stints.parquet")
R[["age_o","age_d"]] = R[["age_o","age_d"]].fillna(R[["age_o","age_d"]].mean())
R["cl"] = R.season*10000 + R.team_o % 10000
BASE = ["home_off", "bo_o", "ro_o", "bd_d", "rd_d", "unk_o", "unk_d", "age_o", "age_d", "lp_o", "lp_d"]
def demean(df, cols, groups, w, it=30):
    X = df[cols].to_numpy(float).copy()
    for _ in range(it):
        for g in groups:
            ws = np.bincount(g, weights=w); 
            for c in range(X.shape[1]):
                X[:, c] -= (np.bincount(g, weights=w*X[:, c]) / ws)[g]
    return pd.DataFrame(X, columns=cols, index=df.index)
def fit(df, add, lab, fe="season"):
    cols = BASE + add; w = df.poss.to_numpy()
    if fe == "season":
        Z = pd.get_dummies(df.season, prefix="s", drop_first=True).astype(float)
        X = sm.add_constant(pd.concat([df[cols], Z], axis=1)); y = df.y
    else:   # offense team-season and defense team-season fixed effects
        go = pd.factorize(df.season*10000 + df.team_o % 10000)[0]; gd = pd.factorize(df.season*10000 + df.team_d % 10000)[0]
        M = demean(df, cols + ["y"], [go, gd], w); X = M[cols]; y = M.y
    r = sm.WLS(y, X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": df.cl})
    sd = df[add].std()
    print(f"{lab:34s}", "  ".join(f"{c} {r.params[c]:+.2f} (t {r.tvalues[c]:+.1f}; 1sd {r.params[c]*sd[c]:+.2f})" for c in add), flush=True)
    return r
if __name__ == "__main__":
    print(len(R), "stint sides;", "sd famP %.2f famC %.2f tenP %.2f tenC %.2f" % tuple(R[["famP_o", "famC_o", "tenP_o", "tenC_o"]].std()))
    r0 = fit(R, [], "base")
    print("base coefs:", "  ".join(f"{c} {r0.params[c]:+.2f}" for c in BASE[:7]))
    PO, PD = ["famP_o", "tenP_o"], ["famP_d", "tenP_d"]
    fit(R, ["famP_o", "famP_d"], "pairs, prior seasons")
    fit(R, ["tenP_o", "tenP_d"], "tenure, prior seasons")
    fit(R, PO + PD, "pairs + tenure, prior")
    fit(R, PO + PD + ["famC_o", "tenC_o", "famC_d", "tenC_d"], "prior + this season")
    for lab, sub in (("2015-2019", R[R.season <= 2019]), ("2020-2025", R[R.season >= 2020])):
        fit(sub, PO + PD, "pairs + tenure prior " + lab)
    fit(R, PO + PD, "prior, team-season FE", fe="two")
    fit(R, PO + PD + ["famC_o", "tenC_o", "famC_d", "tenC_d"], "prior + this season, team-season FE", fe="two")
