import numpy as np, pandas as pd
B = pd.read_parquet("data/backtest_components.parquet").merge(pd.read_parquet("data/bt_split.parquet"), on=["game", "home_team", "away_team"])
E = pd.read_parquet("data/guard_rim_bt.parquet"); B = B.merge(E, on="game", how="left")
print("rows", len(B), "missing", B.filter(like="off_").isna().any(axis=1).sum())
F = [c for c in B.columns if c.startswith("f:")]
box = B[F].sum(axis=1) + B.unrated
base = (box + .6 * B.res_o + .8 * B.res_d).to_numpy()
y = B.margin.values; seas = B.season.values
# scale my (min/48, no pace) diffs to backtest units per season
mine = B[["rim_protection_G", "rim_protection_Wg", "rim_protection_Bg"]].sum(axis=1)
theirs = B["f:d_blk"] + B["f:d_opprim"]
sc = {}
for s in np.unique(seas):
    m = seas == s; sc[s] = (mine[m] @ theirs[m]) / (mine[m] @ mine[m])
print("corr mine vs backtest rim:", np.corrcoef(mine, theirs)[0, 1].round(4), " scale", np.round(list(sc.values()), 3))
scale = pd.Series(seas).map(sc).values
for c in E.columns:
    if c != "game": B[c] = B[c] * scale
def design(X): return np.c_[np.ones(len(y)), X]
def loso(X):
    X = design(X); e = []
    for s in np.unique(seas):
        tr, te = seas != s, seas == s
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; e.append((y[te] - X[te] @ b) ** 2)
    return np.sqrt(np.concatenate(e).mean())
def coefs(X, m):
    X = design(X)[m]; yy = y[m]; ss = seas[m]
    b = np.linalg.lstsq(X, yy, rcond=None)[0]; r = yy - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[ss == s].T @ r[ss == s], X[ss == s].T @ r[ss == s]) for s in np.unique(ss))
    return b, np.sqrt(np.diag(XtXi @ meat @ XtXi))
H1, H2, ALL = seas <= 2019, seas >= 2020, np.ones(len(y), bool)
b0 = loso(base[:, None]); print(f"base LOSO {b0:.4f}")
def test(name, cols, fixed=None):
    X = np.c_[base, B[cols].to_numpy()]
    b, se = coefs(X, ALL); b1, s1 = coefs(X, H1); b2, s2 = coefs(X, H2)
    rel = lambda bb, i: bb[2 + i] / bb[1]   # relative to base coefficient
    line = f"{name:44s} dLOSO {loso(X) - b0:+.4f} | " + " ; ".join(
        f"{c}: {rel(b, i):+.2f} (se {se[2 + i] / b[1]:.2f}; halves {rel(b1, i):+.2f}/{rel(b2, i):+.2f})" for i, c in enumerate(cols))
    print(line)
    # fixed hypothesis: add term with given coefficient (no fitting)
    if fixed is not None:
        Xf = (base + sum(fixed[i] * B[c].to_numpy() for i, c in enumerate(cols)))[:, None]
        print(f"{'':44s} fixed {fixed}: dLOSO {loso(Xf) - b0:+.4f}")
print("--- extra weight on each group's rim protection (0 = current; -1 = don't count it) ---")
test("guards' rim_protection", ["rim_protection_G"], fixed=[-1])
test("wings' rim_protection", ["rim_protection_Wg"], fixed=[-1])
test("bigs' rim_protection", ["rim_protection_Bg"])
test("guards' rim prot, deviation from guard avg", ["dev_rim_protection_G"], fixed=[-1])
print("--- positional baseline offsets (0 = current; -1 = judge each player vs his position) ---")
for c in ["rim_protection", "perimeter_def", "def_rebounding", "boxd"]:
    test(f"baseline {c}", [f"off_{c}"], fixed=[-1])
print("--- perimeter_def for bigs ---")
test("bigs' perimeter_def", ["perimeter_def_Bg"], fixed=[-1])
test("guards' perimeter_def", ["perimeter_def_G"])
