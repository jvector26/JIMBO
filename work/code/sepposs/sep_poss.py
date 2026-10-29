"""Separated-possessions measures for the lineup-shrink test (user question 2026-10-15, session 24).
For each window N (seasons N-2..N), per player:
  P_raw   = raw offensive possessions on court (same as production poss_window)
  sh1     = share of his possessions shared with his most frequent teammate
  sep2    = share of his possessions with NEITHER of his top-2 teammates on
  neff_o/neff_d = ridge-posterior effective sample: 1/[(X'WX + lam I)^-1]_ii - lam (same design, decay and lam as the RAPM
            fit), i.e. how many 'clean' possessions the residual estimate is worth after collinearity with everyone.
  nnaive_o/_d = decayed own possessions (what neff would be with no collinearity at all)
Writes data/sep_poss.parquet (player, N, ...)."""
import sys, numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, "/home/claude/code")
import impact as I
D = "/home/claude/data"; LAM = 2000.0; DEC = (0.3, 0.6, 1.0)
Ns = [int(a) for a in sys.argv[1:]] or list(range(2014, 2026))
out = []
for N in Ns:
    seasons = [N - 2, N - 1, N]
    r = I.load_rows(seasons)
    w = r.poss.to_numpy() * r.season.map(dict(zip(seasons, DEC))).to_numpy()
    players = pd.unique(np.concatenate([np.concatenate(r.off.str.split("-").values),
                                        np.concatenate(r.dfn.str.split("-").values)]).astype(np.int64))
    idx = {p: i for i, p in enumerate(players)}; P = len(players)
    Xo, ids_o = I.lineup_matrix(r.off.values, idx); Xd, _ = I.lineup_matrix(r.dfn.values, idx)
    # --- teammate overlap (offense rows: raw possessions, no garbage) ---
    raw = r.poss.to_numpy()
    Xw = Xo.multiply(raw[:, None]).tocsr()
    S = (Xo.T @ Xw).toarray()                      # S[i,j] = raw poss i and j on offense together; diag = own poss
    own = np.diag(S).copy(); np.fill_diagonal(S, 0)
    top = np.argsort(-S, axis=1)[:, :2]
    sh1 = S[np.arange(P), top[:, 0]] / np.maximum(own, 1)
    # possessions with neither top-2 teammate: own - poss(with j1) - poss(with j2) + poss(with both)
    # poss(with both) from rows: compute via triple overlap on offense rows
    Xc = Xo.tocsc()
    both = np.zeros(P)
    for i in range(P):
        j1, j2 = top[i]
        ci = Xc[:, i].toarray().ravel(); c1 = Xc[:, j1].toarray().ravel(); c2 = Xc[:, j2].toarray().ravel()
        both[i] = (raw * ci * c1 * c2).sum()
    sep2 = (own - S[np.arange(P), top[:, 0]] - S[np.arange(P), top[:, 1]] + both) / np.maximum(own, 1)
    # --- ridge posterior effective sample (same design as impact.fit_window) ---
    seas = pd.get_dummies(r.season).to_numpy(float)
    X = sp.hstack([Xo, Xd, sp.csr_matrix(np.hstack([seas, r.home_off.to_numpy(float)[:, None]]))]).tocsr()
    pen = np.concatenate([np.full(2 * P, LAM), np.zeros(seas.shape[1] + 1)])
    A = (X.T @ X.multiply(w[:, None]).tocsr()).toarray() + np.diag(pen)
    V = np.diag(np.linalg.inv(A))
    nn = np.asarray(X.T @ w).ravel()
    df = pd.DataFrame({"player": players, "N": N, "P_raw": own, "sh1": sh1, "sep2": sep2,
                       "neff_o": 1 / V[:P] - LAM, "neff_d": 1 / V[P:2 * P] - LAM, "nnaive_o": nn[:P], "nnaive_d": nn[P:2 * P]})
    out.append(df)
    q = df[df.P_raw > 3000]
    print(N, P, f"sh1 {q.sh1.median():.2f} sep2 {q.sep2.median():.2f} neff/naive o {np.median(q.neff_o / q.nnaive_o):.2f}"
          f" d {np.median(q.neff_d / q.nnaive_d):.2f}", flush=True)
res = pd.concat(out, ignore_index=True)
fn = f"{D}/sep_poss.parquet"
try:
    old = pd.read_parquet(fn); res = pd.concat([old[~old.N.isin(Ns)], res], ignore_index=True)
except FileNotFoundError:
    pass
res.to_parquet(fn, index=False)
