"""Free decomposition: lineup term (0.6*res_o + 0.8*res_d) split into player groups, each with its own coefficient."""
import numpy as np, pandas as pd
from lineup_port_bt import G, B, y, seas, diffs, loso
lin = lambda mask: (lambda dd: 0.6 * dd.o + 0.8 * dd.d)(diffs(mask.astype(float)))
def clus(X):
    X = np.c_[np.ones(len(y)), X]; b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b
    XtXi = np.linalg.inv(X.T @ X); meat = sum(np.outer(X[seas == s].T @ r[seas == s], X[seas == s].T @ r[seas == s]) for s in np.unique(seas))
    return b, np.sqrt(np.diag(XtXi @ meat @ XtXi))
for cut in [2500, 4000]:
    thin = G.P < cut; mv = G.moved == 1
    groups = {"stay,thick": ~thin & ~mv, "stay,thin": thin & ~mv, "moved,thick": ~thin & mv, "moved,thin": thin & mv}
    Xg = np.column_stack([lin(m) for m in groups.values()])
    for nm, msk in [("all", np.ones(len(y), bool)), ("2015-19", seas <= 2019), ("2020-25", seas >= 2020)]:
        yy = y
        X = np.c_[np.ones(msk.sum()), B.box.values[msk], Xg[msk]]; b = np.linalg.lstsq(X, y[msk], rcond=None)[0]
        r = y[msk] - X @ b; ss = seas[msk]; XtXi = np.linalg.inv(X.T @ X)
        meat = sum(np.outer(X[ss == s].T @ r[ss == s], X[ss == s].T @ r[ss == s]) for s in np.unique(ss)); se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
        rel = b[2:] / b[1]
        print(f"cut {cut} {nm:8s} box {b[1]:.3f} | " + " | ".join(f"{k} {rel[i]:.2f} (t {b[2+i]/se[2+i]:.1f})" for i, k in enumerate(groups)))
    # share of lineup minutes
    print("   minute shares:", {k: round(G['min'][m].sum() / G['min'].sum(), 3) for k, m in groups.items()})
    print("   LOSO pooled", round(loso(np.c_[B.box, lin(np.ones(len(G), bool))])[0], 4), " split", round(loso(np.c_[B.box, Xg])[0], 4))
