import numpy as np, pandas as pd, time, defsub as S
GRID = [250, 1000, 4000, 16000, 64000, 256000, 1e6]
res = []
for N in [2018, 2024]:
    t = time.time(); r = S.rows([N - 2, N - 1, N]); dec = r.season.map(dict(zip([N-2,N-1,N], (.3,.6,1.)))).to_numpy()
    X, players, nfe = S.design(r); P = len(players)
    half = (r.game.to_numpy() % 2 == 0)
    for name in S.OUTS:
        y, w = S.outcome(r, name, dec)
        for tr, te in ((half, ~half), (~half, half)):
            Xt = X[tr]
            for lam in GRID + [1e12]:
                b = S.fit(Xt, P, nfe, y[tr], w[tr], lam)
                e = y[te] - X[te] @ b
                res.append((N, name, lam, (w[te] * e**2).sum() / w[te].sum()))
    print(N, "done", round(time.time() - t), flush=True)
R = pd.DataFrame(res, columns=["N", "out", "lam", "mse"]).groupby(["out", "lam"]).mse.mean().unstack()
base = R[1e12]
print(((R.drop(columns=1e12).sub(base, axis=0)).div(base, axis=0) * 1e4).round(1).to_string())  # bp change vs FE-only
