"""Same game-level design as aging.py (minute shares by age bin + rookie groups vs game margin), but with
season-block bootstrap standard errors for each age-bin coefficient, to see how well identified the bins are."""
import numpy as np, pandas as pd, json
D = "data"
B = pd.read_parquet(f"{D}/backtest_components.parquet")
ps = pd.read_parquet(f"{D}/player_seasons.parquet")[["season", "player", "age"]]
gm = pd.read_parquet(f"{D}/games_all.parquet")
BINS = [(0, 21.99, "<=21"), (22, 23.99, "22-23"), (24, 25.99, "24-25"), (26, 27.99, "26-27"), (28, 29.99, "28-29"),
        (30, 31.99, "30-31"), (32, 33.99, "32-33"), (34, 35.99, "34-35"), (36, 99, "36+")]
dr = pd.read_parquet(f"{D}/draft_r1.parquet")[["year", "pick", "player"]]
pace = {}
rows = []
for S in sorted(B.season.unique()):
    N = S - 1
    pg = pd.concat([pd.read_parquet(f"{D}/player_games_{y}.parquet")[["game", "player", "team", "min"]].assign(season=y)
                    for y in range(N - 2, S + 1)])
    st = pd.read_parquet(f"{D}/stints_{N}.parquet")
    poss = sum(st[f"fg2a{z}"] + st[f"fg3a{z}"] - st[f"oreb{z}"] + st[f"tov{z}"] + 0.44 * st[f"fta{z}"] for z in "AB")
    pace[N] = poss.sum() / 2 / st.game.nunique()
    rated = set(pg[(pg.season.between(N - 2, N)) & (pg["min"] > 0)].player)
    x = pg[pg.season == S].merge(ps[ps.season == S], on=["season", "player"], how="left")
    x["rated"] = x.player.isin(rated); x["w"] = x["min"] / 48 * pace[N] / 100
    g = gm[gm.season == S][["game", "home_team", "away_team"]]
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    out = pd.DataFrame({"game": g.game.values, "season": S})
    for lo, hi, lab in [(a, a + 0.99, f"A{a}") for a in range(18, 45)]:
        t = (x.w * (x.rated & x.age.between(lo, hi))).groupby([x.game, x.team]).sum()
        out[lab] = t.reindex(hk).fillna(0).to_numpy() - t.reindex(ak).fillna(0).to_numpy()
    rk = x.player.map(dr[dr.year == S].set_index("player").pick)
    import json as _j
    RV = _j.load(open(f"{D}/rookie_value.json")); SC = pd.read_parquet(f"{D}/ncaa_scores.parquet")
    sc = SC[SC.year == S].drop_duplicates("player").set_index("player")
    cs = x.player.map(sc.cscore); col = x.player.map(sc.college).fillna(False).astype(bool)
    clip = (cs - RV["col_mean"]).clip(-RV["col_cap"], RV["col_cap"])
    pickval = RV["rk"] + RV["rk_lp"] * (np.log(rk) - RV["center_lp"]) + RV["rk_age"] * (x.age.fillna(20) - RV["center_age"])
    isrk = (~x.rated) & rk.notna()
    groups = {"rk_col": isrk & col & cs.notna(), "rk_col_cs": (isrk & col & cs.notna()).astype(float) * clip.fillna(0),
              "rk_non": isrk & (~(col & cs.notna())), "other_unrated": ~x.rated & rk.isna()}
    groups["rk_non_val"] = (isrk & (~(col & cs.notna()))).astype(float) * pickval.fillna(0)
    for gname, gmk in groups.items():
        t = (x.w * (np.asarray(gmk, dtype=float))).groupby([x.game, x.team]).sum()
        out[gname] = t.reindex(hk).fillna(0).to_numpy() - t.reindex(ak).fillna(0).to_numpy()
    rows.append(out)
pd.concat(rows).to_parquet(f"{D}/age_design2.parquet", index=False)
A = pd.concat(rows).merge(B.drop(columns=[c for c in ("season",) if c in B]), on="game")
comp = [c for c in B.columns if c.startswith("f:")]
A["base"] = A[comp].sum(axis=1) + 0.6 * A["lineup_impact"]
age_cols = [c for c in A.columns if c.startswith("A") and c[1:].isdigit()]
cols = ["base", "rookie_1_3", "rookie_4_10", "rookie_11_30", "other_unrated"] + age_cols
def fit(df):
    X = np.c_[np.ones(len(df)), df[cols].to_numpy()]
    return np.linalg.lstsq(X, df.margin.to_numpy(), rcond=None)[0]
import sys; sys.exit(0)
b = fit(A); seasons = sorted(A.season.unique()); rng = np.random.default_rng(0)
bs = np.array([fit(pd.concat([A[A.season == s] for s in rng.choice(seasons, len(seasons))])) for _ in range(200)])
se = bs.std(0)
print("coef (season-block bootstrap SE), n games", A.game.nunique())
for i, c in enumerate(["const"] + cols):
    print("  %-14s %+0.3f  se %.3f  t %+0.1f" % (c, b[i], se[i], b[i] / se[i] if se[i] else 0))
