import numpy as np, pandas as pd
B = pd.read_parquet("data/backtest_components.parquet")
S = pd.read_parquet("data/bt_split.parquet")
B = B.merge(S, on=["game", "home_team", "away_team"])
print("rows", len(B), " max |res_o+res_d - lineup_impact|:", (B.res_o + B.res_d - B.lineup_impact).abs().max().round(4),
      " corr:", np.corrcoef(B.res_o + B.res_d, B.lineup_impact)[0, 1].round(4))
F = [c for c in B.columns if c.startswith("f:")]
box = B[F].sum(axis=1) + B.unrated
y = B.margin.values; seas = B.season.values

def rmse(pred, mask=None):
    X = np.c_[np.ones(len(pred)), pred]
    if mask is None: mask = np.ones(len(y), bool)
    b = np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]
    return np.sqrt(((y[mask] - X[mask] @ b) ** 2).mean())

def loso(pred):
    X = np.c_[np.ones(len(pred)), pred]; e = []
    for s in np.unique(seas):
        tr, te = seas != s, seas == s
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; e.append((y[te] - X[te] @ b) ** 2)
    return np.sqrt(np.concatenate(e).mean())

print("\nbaseline single weight (lineup_impact):")
for w in [0, .6, 1.0]:
    print(f"  w={w}: RMSE {rmse(box + w * B.lineup_impact):.4f}")
grid = np.round(np.arange(0, 1.61, .1), 2)
R = pd.DataFrame([[loso(box + wo * B.res_o + wd * B.res_d) for wd in grid] for wo in grid], index=grid, columns=grid)
R.index.name = "w_off \\ w_def"
print("\nLOSO RMSE grid (rows w_off, cols w_def):"); print(R.round(4).iloc[::2, ::2].to_string())
i, j = np.unravel_index(R.values.argmin(), R.shape)
print(f"\nbest: w_off {grid[i]}, w_def {grid[j]}, LOSO {R.values.min():.4f};  (0.6,0.6) = {R.loc[.6, .6]:.4f}")
# free coefficients (with box sum as one regressor), per half
for name, m in [("all", np.ones(len(y), bool)), ("2015-19", seas <= 2019), ("2020-25", seas >= 2020)]:
    X = np.c_[np.ones(m.sum()), box[m], B.res_o[m], B.res_d[m]]
    b, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    r = y[m] - X @ b; s2 = (r ** 2).sum() / (m.sum() - 4)
    # season-clustered SE
    XtXi = np.linalg.inv(X.T @ X); meat = sum(np.outer(X[seas[m] == s].T @ r[seas[m] == s], X[seas[m] == s].T @ r[seas[m] == s]) for s in np.unique(seas[m]))
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    rel = b[2:] / b[1]  # relative to box scale
    print(f"{name:8s} box {b[1]:.3f}  res_o {b[2]:.3f} (t {b[2]/se[2]:.1f})  res_d {b[3]:.3f} (t {b[3]/se[3]:.1f})"
          f"  -> relative w_off {rel[0]:.2f}, w_def {rel[1]:.2f};  diff t {(b[3]-b[2])/np.sqrt(se[2]**2+se[3]**2):.1f}")
R.to_csv("data/bt_split_grid.csv")
