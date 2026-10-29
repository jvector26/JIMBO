"""Market-bias hunt: does an added term improve the LOSO blend (model + Vegas + roster age)?"""
import numpy as np, pandas as pd, json
D = "/home/claude/data"
A = pd.read_parquet(f"{D}/preseason_backtest.parquet").merge(pd.read_parquet(f"{D}/direction_features.parquet"), on=["season", "team"])
A["y"] = 82*A.wpct - 41; A["mod"] = 82*A.model_wpct_agefix - 41; A["mkt"] = 82*A.mkt_wpct - 41

# (a) big market
A["bigmkt"] = A.team.isin(["LAL", "NYK", "GSW", "BOS", "CHI"]).astype(float)
# (b) prior-year close-game luck: wins above point-differential expectation (per 82)
G = pd.read_parquet(f"{D}/games_all.parquet"); G = G[G.score_ok]
fix = {"NOH": "NOP", "NJN": "BKN", "CHA": "CHA"}
rows = []
for side, opp, tm in [("home_pts", "away_pts", "htm"), ("away_pts", "home_pts", "vtm")]:
    rows.append(pd.DataFrame({"season": G.season, "team": G[tm].replace(fix), "m": G[side] - G[opp]}))
T = pd.concat(rows).groupby(["season", "team"]).m.agg(mov="mean", wp=lambda s: (s > 0).mean()).reset_index()
k = np.polyfit(T.mov, T.wp, 1); T["luck"] = 82*(T.wp - np.polyval(k, T.mov))
T["season"] += 1; A = A.merge(T[["season", "team", "luck", "mov"]].rename(columns={"mov": "prev_mov"}), on=["season", "team"], how="left")
print(f"pyth: wp = {k[1]:.3f} + {k[0]:.4f}*MOV   (luck sd {A.luck.std():.2f} wins)")
# (c) headline moves: top-30 box-rated players (>=1500 min) arriving / leaving
W = json.load(open(f"{D}/weights_bt.json")); f = pd.read_parquet(f"{D}/features.parquet")
f["box"] = sum(f[c]*v for c, v in {**W["Wo"], **W["Wd"]}.items())
f["first"] = f.teams.str.split("/").str[0].replace(fix); f["last"] = f.teams.str.split("/").str[-1].replace(fix)
arr, dep = {}, {}
for S in range(2014, 2026):
    prev = f[(f.season == S-1) & (f["min"] >= 1500)].nlargest(30, "box").set_index("player")
    cur = f[f.season == S].set_index("player")
    for p, r in prev.iterrows():
        if p in cur.index and cur.at[p, "first"] != r["last"]:
            arr[(S, cur.at[p, "first"])] = arr.get((S, cur.at[p, "first"]), 0) + 1
            dep[(S, r["last"])] = dep.get((S, r["last"]), 0) + 1
A["star_in"] = [arr.get((s, t), 0) for s, t in zip(A.season, A.team)]
A["star_out"] = [dep.get((s, t), 0) for s, t in zip(A.season, A.team)]
A["turnover"] = 1 - A.continuity
# (d) conference finalists previous season (prev season start year -> teams)
CF = {2013: "SAS OKC MIA IND", 2014: "GSW HOU CLE ATL", 2015: "GSW OKC CLE TOR", 2016: "GSW SAS CLE BOS",
      2017: "GSW HOU CLE BOS", 2018: "TOR MIL GSW POR", 2019: "LAL DEN MIA BOS", 2020: "MIL ATL PHX LAC",
      2021: "GSW DAL BOS MIA", 2022: "DEN LAL MIA BOS", 2023: "BOS IND DAL MIN", 2024: "OKC MIN IND NYK"}
A["cf_prev"] = [float(t in CF[s-1].split()) for s, t in zip(A.season, A.team)]

A = A[A.season != 2019].copy()
A["age_c"] = A.mw_age - A.groupby("season").mw_age.transform("mean")
FEATS = ["bigmkt", "luck", "star_in", "star_out", "turnover", "cf_prev"]
for c in FEATS: A[c+"_c"] = A[c] - A.groupby("season")[c].transform("mean")
rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))
BASE = ["mod", "mkt", "age_c"]

def loso(cols, d=A):
    X = d[BASE+cols].to_numpy(); y = d.y.to_numpy(); pred = np.empty(len(d))
    for s in d.season.unique():
        tr = (d.season != s).to_numpy(); b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; pred[~tr] = X[~tr] @ b
    return rmse(y - pred)

def fit(cols, d=A):
    X = d[BASE+cols].to_numpy(); y = d.y.to_numpy(); b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X@b
    # season-clustered SE
    XtXi = np.linalg.inv(X.T@X); M = sum(np.outer(X[g].T@e[g], X[g].T@e[g]) for g in [(d.season == s).to_numpy() for s in d.season.unique()])
    se = np.sqrt(np.diag(XtXi@M@XtXi)); return b, b/se

r0 = loso([])
print(f"baseline LOSO RMSE (model+Vegas+age) {r0:.3f}\n")
print(f"{'term':10s} {'mean|1':>7s} {'LOSO':>7s} {'gain':>6s} {'coef':>6s} {'t':>5s} | {'coef14-18':>9s} {'coef20-25':>9s} | raw Vegas miss coef")
early, late = A[A.season < 2019], A[A.season > 2019]
for c in FEATS:
    cc = [c+"_c"]; b, t = fit(cc); b1, _ = fit(cc, early); b2, _ = fit(cc, late)
    rv = np.polyfit(A[c+"_c"], A.y - A.mkt, 1)[0]
    print(f"{c:10s} {A[c].mean():7.2f} {loso(cc):7.3f} {r0-loso(cc):+6.3f} {b[-1]:6.2f} {t[-1]:5.1f} | {b1[-1]:9.2f} {b2[-1]:9.2f} | {rv:+.2f}")
A.to_parquet(f"{D}/market_bias_features.parquet", index=False)
