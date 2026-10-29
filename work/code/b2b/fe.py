import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import lsqr
def fe_fit(df, extra):
    """margin ~ team-season FE + HCA + extra. Returns coef on extras, season-clustered SE."""
    ts = pd.Categorical(np.r_[df.season.astype(str) + df.htm, df.season.astype(str) + df.vtm])
    n = len(df); codes = ts.codes; nt = len(ts.categories)
    rows = np.r_[np.arange(n), np.arange(n)]; vals = np.r_[np.ones(n), -np.ones(n)]
    A = sp.csr_matrix((vals, (rows, codes)), shape=(n, nt))
    X = df[extra].to_numpy(float)
    # partial out FE: residualize y and X on A (+ constant via HCA column)
    Z = sp.hstack([A, sp.csr_matrix(np.ones((n, 1)))]).tocsr()
    def res(v): return v - Z @ lsqr(Z, v, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
    y = res(df.margin.to_numpy(float)); Xr = np.column_stack([res(X[:, j]) for j in range(X.shape[1])])
    b = np.linalg.lstsq(Xr, y, rcond=None)[0]; e = y - Xr @ b
    XtXi = np.linalg.inv(Xr.T @ Xr); s = df.season.to_numpy()
    meat = sum(np.outer(Xr[s == q].T @ e[s == q], Xr[s == q].T @ e[s == q]) for q in np.unique(s))
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    return b, se

