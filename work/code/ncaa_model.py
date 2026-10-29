"""Stage A: predict NBA rookie-season box rating from college stats (+ pick, age, height), leave-one-class-out."""
import numpy as np, pandas as pd, json
D = "/home/claude/data"
m = pd.read_parquet(f"{D}/ncaa_draft_match.parquet")
C = pd.read_parquet(f"{D}/ncaa_player_seasons.parquet")
W = json.load(open(f"{D}/weights_bt.json")); f = pd.read_parquet(f"{D}/features.parquet")
f["box"] = sum(f[c]*v for c, v in {**W["Wo"], **W["Wd"]}.items()) * 0.5
ht = lambda s: (int(s.split("-")[0])*12 + int(s.split("-")[1])) if isinstance(s, str) and "-" in s else np.nan
m["h_in"] = m.height.map(ht); m["lp"] = np.log(m.pick); m["age"] = m.age.fillna(m.age.median())
# career FT% / 3P% across all college seasons up to draft
car = C.merge(m[["athlete_id", "year"]], on="athlete_id"); car = car[car.season <= car.year]
car = car.groupby("athlete_id")[["ftm", "fta", "tpm", "tpa", "min"]].sum().add_prefix("c_")
car["n_seasons"] = C.groupby("athlete_id").season.nunique()
m = m.join(car, on="athlete_id")
p = m.pon.clip(lower=1) / 100
F = pd.DataFrame(index=m.index)
F["pts"] = m.pts / p; F["ts"] = m.pts / (2*(m.fga + 0.44*m.fta)).clip(lower=1)
F["tpa"] = m.tpa / p; F["tp"] = (m.c_tpm + 0.33*100) / (m.c_tpa + 100)
F["ft"] = (m.c_ftm + 0.70*50) / (m.c_fta + 50); F["ftr"] = m.fta / m.fga.clip(lower=1)
for c in ["orb", "drb", "ast", "stl", "blk", "tov", "pf"]: F[c] = m[c] / p
F["usg"] = (m.fga + 0.44*m.fta + m.tov) / p; F["mpg"] = m["min"] / m.g.clip(lower=1)
F["srs"] = m.srs; F["nseas"] = m.n_seasons
F["h"] = m.h_in; F["age"] = m.age; F["lp"] = m.lp
NCAA = [c for c in F.columns if c not in ("h", "age", "lp")]
# target: NBA rookie-season box (season label = draft year)
fb = f.groupby(["season", "player"]).agg(box=("box", "first"), mn=("min", "first"))
key = pd.MultiIndex.from_arrays([m.year, m.player.fillna(-1).astype(np.int64)])
m["y"] = fb.box.reindex(key).to_numpy(); m["wmin"] = fb.mn.reindex(key).to_numpy()
M = m.college & m.year.between(2006, 2025) & (m.wmin >= 200) & F[NCAA].notna().all(axis=1)
print("training rows (college first-rounders with 200+ rookie min):", M.sum())
F = F.fillna(F.median())
mu, sd = F[M].mean(), F[M].std(); Z = (F - mu) / sd

def ridge(cols, lam, tr):
    X = np.c_[np.ones(tr.sum()), Z.loc[tr, cols]]; w = np.sqrt(m.wmin[tr].clip(upper=2000).to_numpy())
    P = np.eye(X.shape[1]) * lam; P[0, 0] = 0
    return np.linalg.solve((X*w[:, None]**2).T @ X + P, (X*w[:, None]**2).T @ m.y[tr])
def loco(cols, lam):
    pred = pd.Series(np.nan, index=m.index)
    for yr in sorted(m.year[M].unique()):
        tr = M & (m.year != yr); b = ridge(cols, lam, tr)
        te = m.college & (m.year == yr); pred[te] = np.c_[np.ones(te.sum()), Z.loc[te, cols]] @ b
    e = (m.y - pred)[M]; w = m.wmin[M].clip(upper=2000)
    return float(np.sqrt((w*e**2).sum()/w.sum())), pred
base = ["age", "h"]
for lab, cols in [("pick+age+height", base), ("college only", NCAA + ["h", "age"]), ("pick+age+height+college", base + NCAA)]:
    for lam in (50, 5000, 50000, 300000):
        r, _ = loco(cols, lam); print(f"{lab:26s} lam {lam:4d}  LOCO wRMSE {r:.3f}")
lam = 50000; b = ridge(base + NCAA, lam, M)
print("\ncoefficients (per sd):", dict(zip(["c"] + base + NCAA, np.round(b, 2))))
_, pred = loco(base + NCAA, lam); m["cscore"] = pred
_, pb = loco(base, lam); m["pscore"] = pb
# full-data score for the 2026 class
b = ridge(base + NCAA, lam, M); t26 = m.college & (m.year == 2026)
m.loc[t26, "cscore"] = np.c_[np.ones(t26.sum()), Z.loc[t26, base + NCAA]] @ b
b0 = ridge(base, lam, M); m.loc[t26, "pscore"] = np.c_[np.ones(t26.sum()), Z.loc[t26, base]] @ b0
m.join(F.add_prefix("x_"))[["year", "pick", "name", "player", "college", "cscore", "pscore", "y", "wmin"] + ["x_" + c for c in F.columns]] \
    .to_parquet(f"{D}/ncaa_scores.parquet", index=False)

def explain(name, yr=2026):
    i = m[(m.year == yr) & (m.name == name)].index[0]; bb = ridge(base + NCAA, lam, M)
    z = Z.loc[i, base + NCAA]; contrib = pd.Series(bb[1:] * z.values, index=base + NCAA)
    raw = F.loc[i, base + NCAA]; ref = F[M & (m.pick <= 5)][base + NCAA].median()
    return pd.DataFrame({"value": raw, "top5_median": ref, "contrib": contrib}).round(3)
if __name__ == "__main__":
    import sys
    for n in sys.argv[1:]: e = explain(n); print(n, "total", round(e.contrib.sum() + ridge(base + NCAA, lam, M)[0], 2)); print(e.sort_values("contrib").to_string())
