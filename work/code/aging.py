import numpy as np, pandas as pd
D = "data"
B = pd.read_parquet(f"{D}/backtest_components.parquet")
pg = pd.read_parquet(f"{D}/player_games_all.parquet")[["season", "game", "player", "team", "min"]]
ps = pd.read_parquet(f"{D}/player_seasons.parquet")[["season", "player", "age"]]
gm = pd.read_parquet(f"{D}/games_all.parquet")
# pace by season (team possessions per game, all minutes)
pace = {}
for s in range(2013, 2026):
    st = pd.read_parquet(f"{D}/stints_{s}.parquet")
    poss = sum(st[f"fg2a{z}"] + st[f"fg3a{z}"] - st[f"oreb{z}"] + st[f"tov{z}"] + 0.44 * st[f"fta{z}"] for z in "AB")
    pace[s] = poss.sum() / 2 / st.game.nunique()
BINS = [(0, 21.99, "<=21"), (22, 23.99, "22-23"), (24, 25.99, "24-25"), (26, 27.99, "26-27"), (28, 29.99, "28-29"),
        (30, 31.99, "30-31"), (32, 33.99, "32-33"), (34, 35.99, "34-35"), (36, 99, "36+")]
dr = pd.read_parquet(f"{D}/draft_r1.parquet")[["year", "pick", "player"]]
rows = []
for S in sorted(B.season.unique()):
    N = S - 1
    rated = set(pg[(pg.season.between(N - 2, N)) & (pg["min"] > 0)].player)
    x = pg[pg.season == S].merge(ps[ps.season == S], on=["season", "player"], how="left")
    x["rated"] = x.player.isin(rated)
    x["w"] = x["min"] / 48 * pace[N] / 100
    g = gm[gm.season == S][["game", "home_team", "away_team"]]
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    out = pd.DataFrame({"game": g.game.values})
    for lo, hi, lab in BINS:
        m = x.rated & x.age.between(lo, hi)
        t = (x.w * m).groupby([x.game, x.team]).sum()
        out[lab] = t.reindex(hk).fillna(0).to_numpy() - t.reindex(ak).fillna(0).to_numpy()
    rk = x.player.map(dr[dr.year == S].set_index("player").pick)
    groups = {"rookie_1_3": rk.between(1, 3), "rookie_4_10": rk.between(4, 10), "rookie_11_30": rk.between(11, 30)}
    groups["other_unrated"] = ~x.rated & rk.isna()
    for gname, gm_ in groups.items():
        t = (x.w * (gm_ & ~x.rated)).groupby([x.game, x.team]).sum()
        out[gname] = t.reindex(hk).fillna(0).to_numpy() - t.reindex(ak).fillna(0).to_numpy()
    out["unrated_share"] = out[list(groups)].sum(axis=1)
    rows.append(out)
A = pd.concat(rows).merge(B, on="game")
comp_cols = [c for c in B.columns if c.startswith("f:")] + ["lineup_impact"]
LW = 0.6  # lineup-impact weight chosen by backtest + persistence tests
A["base"] = A[[c for c in comp_cols if c != "lineup_impact"]].sum(axis=1) + LW * A["lineup_impact"]
age_cols = [lab for *_, lab in BINS if lab != "26-27"]

def fit(df, cols):
    X = np.c_[np.ones(len(df)), df[cols].to_numpy()]
    b, *_ = np.linalg.lstsq(X, df.margin.to_numpy(), rcond=None)
    return b
def rmse(df, cols, b):
    X = np.c_[np.ones(len(df)), df[cols].to_numpy()]; return np.sqrt(((df.margin - X @ b) ** 2).mean())

tr, te = A[A.season <= 2020], A[A.season >= 2021]
for name, cols in [("rating only (unrated=-2 fixed)", ["base", "unrated"]),
                   ("+ learned rookie/unrated value", ["base", "unrated_share"]),
                   ("+ aging curve", ["base", "unrated_share"] + age_cols)]:
    b = fit(tr, cols); print(f"{name:34s} train {rmse(tr, cols, b):.3f}  test(2021-26) {rmse(te, cols, b):.3f}")
RK = ["rookie_1_3", "rookie_4_10", "rookie_11_30", "other_unrated"]
b_tr = fit(tr, ["base"] + RK + age_cols)
print(f"{'+ rookie value by draft slot':34s} train {rmse(tr, ['base'] + RK + age_cols, b_tr):.3f}  test(2021-26) {rmse(te, ['base'] + RK + age_cols, b_tr):.3f}")
cols = ["base"] + RK + age_cols
b = fit(A, cols)
rv = pd.Series(b[2:6], index=RK) / b[1]
print("value by group (rating pts/100, pre-calibration):", rv.round(2).to_dict())
b = np.r_[b[:2], 0.0, b[6:]]  # keep layout [intercept, base, (placeholder), ages...] for code below
slope = b[1]
aging = pd.Series(b[3:], index=age_cols) / slope  # convert to rating points per 100 (same units as player ratings)
aging["26-27"] = 0.0
aging = aging[[lab for *_, lab in BINS]]
print("calibration slope on ratings:", round(slope, 3))

print("one-year aging adjustment by age (pts/100, relative to 26-27):"); print(aging.round(2).to_string())
pd.DataFrame({"bin": aging.index, "adj": aging.values}).to_csv(f"{D}/aging.csv", index=False)
pd.Series({"slope": slope, "unrated": rv["other_unrated"], "intercept": b[0], "rookie_1_3": rv["rookie_1_3"],
           "rookie_4_10": rv["rookie_4_10"], "rookie_11_30": rv["rookie_11_30"]}).to_csv(f"{D}/calib.csv")
