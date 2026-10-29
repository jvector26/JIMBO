import numpy as np, pandas as pd
D = "/home/claude/data"
A = pd.read_parquet(f"{D}/preseason_backtest.parquet").merge(pd.read_parquet(f"{D}/direction_features.parquet"), on=["season", "team"])
A = A[A.season != 2019].copy()
A["y"] = 82 * A.wpct - 41; A["mod"] = 82 * A.model_wpct - 41; A["mkt"] = 82 * A.mkt_wpct - 41
A["prev"] = 82 * A.prev_wpct - 41
A["below"] = (-A["mod"]).clip(lower=0)                        # projected wins below .500
A["seller"] = A.vet_quality * (A["mod"] < 0)                 # good vets on a sub-.500 projected team
A["seller_x"] = A.vet_quality * A.below / 10                 # scaled by how bad
A["young_bad"] = A.young_share * (A["mod"] < 0)              # youth movement on bad teams (tank + development)
FEATS = ["prev", "mw_age", "young_share", "young_quality", "vet_quality", "seller", "seller_x", "young_bad", "continuity", "rookie_share"]
for f in FEATS: A[f + "_c"] = A[f] - A.groupby("season")[f].transform("mean")
rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))

def loso(cols, base=("mod",)):
    pred = pd.Series(index=A.index, dtype=float); coefs = []
    for s in A.season.unique():
        tr, te = A.season != s, A.season == s
        X = A[list(base) + cols].to_numpy()
        b = np.linalg.lstsq(X[tr], A.y[tr], rcond=None)[0]; coefs.append(b)
        pred[te] = X[te] @ b
    return rmse(A.y - pred), np.median(coefs, axis=0), pred

def full_t(cols, base=("mod",)):
    X = A[list(base) + cols].to_numpy(); b = np.linalg.lstsq(X, A.y, rcond=None)[0]; e = A.y - X @ b
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * e.var()); return b / se

r0, _, _ = loso([]); rb0, _, _ = loso([], ("mod", "mkt"))
print(f"baseline  model-only LOSO RMSE {r0:.3f} | blend(model+Vegas) {rb0:.3f}\n")
print(f"{'feature':15s} {'model+feat':>11s} {'gain':>6s} {'coef':>7s} {'t':>5s} | {'blend+feat':>10s} {'gain':>6s} {'t(blend)':>8s}")
for f in FEATS:
    c = [f + "_c"]
    r, b, _ = loso(c); rb, bb, _ = loso(c, ("mod", "mkt"))
    print(f"{f:15s} {r:11.3f} {r0 - r:+6.3f} {b[-1]:7.2f} {full_t(c)[-1]:5.1f} | {rb:10.3f} {rb0 - rb:+6.3f} {full_t(c, ('mod','mkt'))[-1]:8.1f}")
