"""Refit the age curve. Game-level design (age_design.parquet: per game, home-minus-away possession share of minutes
by exact age, plus rookie/unrated shares) regressed on game margin together with the rating base, as aging.py did,
but comparing shapes: current bins, freely refit bins, a smooth linear spline, and the player-level shape (age_check.py)
scaled by one fitted parameter. Scored by leave-one-season-out game RMSE and by split halves."""
import numpy as np, pandas as pd
D = "data"
A = pd.read_parquet(f"{D}/age_design.parquet")
B = pd.read_parquet(f"{D}/backtest_components.parquet")
A = A.merge(B.drop(columns=["season"]), on="game")
comp = [c for c in B.columns if c.startswith("f:")]
A["base"] = A[comp].sum(axis=1) + 0.6 * A["lineup_impact"]
AGES = [int(c[1:]) for c in A.columns if c[0] == "A" and c[1:].isdigit()]
X_age = A[[f"A{a}" for a in AGES]].to_numpy()                      # minute share by exact age (home - away)
CTRL = ["base", "rookie_1_3", "rookie_4_10", "rookie_11_30", "other_unrated"]
BINS = [(0, 21, "<=21"), (22, 23, "22-23"), (24, 25, "24-25"), (26, 27, "26-27"), (28, 29, "28-29"),
        (30, 31, "30-31"), (32, 33, "32-33"), (34, 35, "34-35"), (36, 99, "36+")]
cur = pd.read_csv(f"{D}/aging.csv").set_index("bin").adj
# player-level shape (age_check.py gaps, 26-27 = 0)
PLAYER = {"<=21": 0.91, "22-23": 0.59, "24-25": 0.23, "26-27": 0.03, "28-29": -0.13, "30-31": -0.34,
          "32-33": -0.50, "34-35": -0.46, "36+": -0.80}
def binof(a): return next(l for lo, hi, l in BINS if lo <= a <= hi)
def bin_basis(drop="26-27"):
    labs = [l for *_, l in BINS if l != drop]
    return np.column_stack([X_age[:, [i for i, a in enumerate(AGES) if binof(a) == l]].sum(1) for l in labs]), labs
def spline_basis(knots=(22, 25, 28, 31, 34, 37)):
    f = [np.array([a - 26.0 for a in AGES])] + [np.array([max(0.0, a - k) for a in AGES]) for k in knots]
    return np.column_stack([X_age @ v for v in f]), ["lin"] + [f"h{k}" for k in knots]
def fixed_shape(vals):   # one column: minute share weighted by a known shape
    v = np.array([vals[binof(a)] for a in AGES]); return (X_age @ v)[:, None], ["scale"]
def design(kind):
    if kind == "bins": Z, nm = bin_basis()
    elif kind == "spline": Z, nm = spline_basis()
    elif kind == "player": Z, nm = fixed_shape(PLAYER)
    elif kind == "current": Z, nm = fixed_shape({l: float(cur[l]) for *_, l in BINS})
    elif kind == "none": Z, nm = np.zeros((len(A), 0)), []
    return np.c_[np.ones(len(A)), A[CTRL].to_numpy(), Z], CTRL + nm
def curve_from(kind, b):
    k = 1 + len(CTRL)
    if kind == "bins":
        labs = [l for *_, l in BINS if l != "26-27"]; d = dict(zip(labs, b[k:])); d["26-27"] = 0.0
        return {l: d[l] for *_, l in BINS}
    if kind == "spline":
        _, nm = spline_basis(); knots = [int(n[1:]) for n in nm[1:]]
        f = lambda a: b[k] * (a - 26) + sum(b[k + 1 + i] * max(0.0, a - kn) for i, kn in enumerate(knots))
        w = {l: [] for *_, l in BINS}
        for a in AGES: w[binof(a)].append(f(a))
        c = {l: float(np.mean(v)) for l, v in w.items()}; base = c["26-27"]
        return {l: v - base for l, v in c.items()}
    if kind in ("player", "current"):
        src = PLAYER if kind == "player" else {l: float(cur[l]) for *_, l in BINS}
        return {l: b[k] * v for l, v in src.items()}
    return {l: 0.0 for *_, l in BINS}
def loso(kind):
    X, nm = design(kind); y = A.margin.to_numpy(); p = np.empty(len(A)); seasons = A.season.unique()
    for s in seasons:
        tr = (A.season != s).to_numpy()
        p[~tr] = X[~tr] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    half = {h: float(np.sqrt(np.mean((y[m] - p[m]) ** 2))) for h, m in
            (("15-19", (A.season <= 2018).to_numpy()), ("20-25", (A.season >= 2019).to_numpy()))}
    return float(np.sqrt(np.mean((y - p) ** 2))), half, b, nm
print("LOSO game RMSE (13,206 games)")
res = {}
for kind in ("none", "current", "bins", "spline", "player"):
    r, half, b, nm = loso(kind); res[kind] = (r, b, nm)
    extra = "" if kind in ("none",) else "  scale %.2f" % b[1 + len(CTRL)] if kind in ("player", "current") else ""
    print("  %-8s %.4f   halves %.3f / %.3f%s" % (kind, r, half["15-19"], half["20-25"], extra))
print("\ncurves (pts/100, 26-27 = 0)")
rows = {k: curve_from(k, res[k][1]) for k in ("bins", "spline", "player")}
rows["current"] = {l: float(cur[l]) for *_, l in BINS}
print("  %-9s" % "bin", "  ".join("%7s" % l for *_, l in BINS))
for k in ("current", "bins", "spline", "player"):
    print("  %-9s" % k, "  ".join("%+7.2f" % rows[k][l] for *_, l in BINS))
pd.DataFrame(rows).to_csv(f"{D}/age_curves.csv")
